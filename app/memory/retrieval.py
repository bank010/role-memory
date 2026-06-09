"""记忆检索。

两路召回，互补：
1. 情节记忆（压缩）：relevance × recency × importance 三维打分 —— 管全局剧情/重要事件
2. 逐字原话（verbatim）：向量 + 关键词「混合检索」—— 管精确细节（名字/数字/专名）

为什么要混合检索：纯向量对"我的猫叫煤球"这类精确细节召回率差，
叠加字面/关键词匹配后，精确事实的召回率大幅提升。
"""

import math
import re
from typing import Dict, List

from app import config, embeddings, normalizer, rerank
from app.memory import stores


# ---------------- 情节记忆：三维打分 ----------------
async def retrieve_episodes(session: str, query: str, top_k: int = None) -> List[Dict]:
    top_k = top_k or config.RETRIEVE_TOP_K

    qvec = await embeddings.embed(await normalizer.to_base_lang(query))
    # 候选预筛：postgres 走 pgvector 原生 KNN 取 top-N，sqlite 取全部；再做三维重排
    episodes = stores.candidate_episodes(session, qvec, max(top_k * 4, 20))
    if not episodes:
        return []

    now_turn = stores.max_turn(session)
    w_rel, w_rec, w_imp = config.SCORE_WEIGHTS

    scored = []
    for ep in episodes:
        relevance = max(0.0, embeddings.cosine(qvec, ep["vec"]))
        recency = math.exp(-config.RECENCY_DECAY * max(0, now_turn - ep["turn"]))
        importance = ep["importance"] / 10.0
        score = w_rel * relevance + w_rec * recency + w_imp * importance
        scored.append({
            "id": ep["id"], "event": ep["event"], "emotion": ep["emotion"],
            "importance": ep["importance"], "turn": ep["turn"],
            "sensitive": ep.get("sensitive", False),
            "relevance": round(relevance, 3), "recency": round(recency, 3),
            "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # 二阶段精排：对三维打分的前 N 个候选用 reranker 重排，取 top_k
    if rerank.enabled() and len(scored) > 1:
        cand = scored[: max(top_k * config.RERANK_CANDIDATE_MULT, top_k)]
        ranked = await rerank.rerank(query, [c["event"] for c in cand])
        order = [i for i, _ in ranked]
        if order:
            reordered = [cand[i] for i in order if i < len(cand)]
            for c, (_, rs) in zip(reordered, ranked):
                c["rerank"] = round(rs, 4)
            top = reordered[:top_k]
        else:
            top = scored[:top_k]
    else:
        top = scored[:top_k]

    if top:
        stores.mark_recalled([t["id"] for t in top], now_turn)
    return top


# ---------------- 逐字原话：混合检索 ----------------
_CN = re.compile(r"[\u4e00-\u9fff]")


# 常见停用字，匹配上也几乎没有区分度，权重压到最低
_STOP_CHARS = set("的了吗呢吧啊哦我你他她它们之是不在有和跟与去来还叫做想要这那个会就都也很")


def _weighted_tokens(text: str) -> Dict[str, float]:
    """返回 token->权重。双字词/英文词权重高（区分度强），单字权重低。"""
    text = text.lower()
    toks: Dict[str, float] = {}
    for w in re.findall(r"[a-z0-9]+", text):
        toks[w] = 2.0  # 英文/数字词，强信号（名字、专名）
    cn = "".join(_CN.findall(text))
    for ch in cn:
        toks[ch] = max(toks.get(ch, 0.0), 0.15 if ch in _STOP_CHARS else 0.4)
    for i in range(len(cn) - 1):
        bg = cn[i:i + 2]
        toks[bg] = max(toks.get(bg, 0.0), 2.0)  # 双字词，强信号（周六/青岛/煤球）
    return toks


def _lexical_score(q_weighted: Dict[str, float], text: str) -> float:
    """绝对命中权重 + 平滑：命中几个区分度高的 token（双字词/专名）即可得高分，
    不被 query 长度稀释——这对"问一句、命中一个关键事实"的细节召回至关重要。"""
    if not q_weighted:
        return 0.0
    t_tokens = set(_weighted_tokens(text).keys())
    matched = sum(w for t, w in q_weighted.items() if t in t_tokens)
    return matched / (matched + 2.0)  # squash 到 0~1


async def retrieve_verbatim(session: str, query: str, top_k: int = None,
                            exclude_recent: int = 0) -> List[Dict]:
    """逐字召回：vector 0.5 + lexical 0.5 融合。exclude_recent 用于剔除已在工作窗口里的轮次。"""
    top_k = top_k or (config.RETRIEVE_TOP_K + 2)
    chunks = stores.all_chunks(session)
    if not chunks:
        return []

    now_turn = stores.max_turn(session)
    # 多语言：query 也归一到基准语言，再做向量 + 关键词匹配（对齐 chunk.norm）
    qnorm = await normalizer.to_base_lang(query)
    qvec = await embeddings.embed(qnorm)
    q_weighted = _weighted_tokens(qnorm)

    scored = []
    for c in chunks:
        if exclude_recent and c["turn"] > now_turn - exclude_recent:
            continue  # 最近 N 轮已逐字进了工作窗口，不重复
        vec_s = max(0.0, embeddings.cosine(qvec, c["vec"]))
        lex_s = _lexical_score(q_weighted, c.get("norm") or c["text"])
        score = 0.5 * vec_s + 0.5 * lex_s
        if score <= 0.01:
            continue
        scored.append({
            "turn": c["turn"], "role": c["role"], "text": c["text"],
            "vec": round(vec_s, 3), "lex": round(lex_s, 3), "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # 二阶段精排：逐字片段同样用 reranker 重排前 N，提升精确细节命中
    if rerank.enabled() and len(scored) > 1:
        cand = scored[: max(top_k * config.RERANK_CANDIDATE_MULT, top_k)]
        ranked = await rerank.rerank(query, [c["text"] for c in cand])
        order = [i for i, _ in ranked]
        if order:
            reordered = [cand[i] for i in order if i < len(cand)]
            for c, (_, rs) in zip(reordered, ranked):
                c["rerank"] = round(rs, 4)
            return reordered[:top_k]

    return scored[:top_k]
