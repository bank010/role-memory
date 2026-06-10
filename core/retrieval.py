"""记忆检索（在线读路径，毫秒级预算）。

两路召回，互补：
1. 情节记忆（压缩）：relevance × recency × importance 三维打分 —— 管全局剧情/重要事件
2. 逐字原话（verbatim）：向量 + 关键词「混合检索」—— 管精确细节（名字/数字/专名）

为什么要混合检索：纯向量对"我的猫叫煤球"这类精确细节召回率差，
叠加字面/关键词匹配后，精确事实的召回率大幅提升。

性能要点：
- query 向量由调用方（assembler）算一次后传入，两路共享，不重复 embed；
- 候选集走存储层 KNN 预筛（postgres），不全量加载向量到内存；
- rerank 有严格延迟预算（RERANK_TIMEOUT_MS），超时立即降级，绝不拖垮读路径；
- mark_recalled 是统计性写操作，丢进后台 task，不占读路径时延。
"""

import asyncio
import math
import re
import time
from typing import Dict, List, Optional

import numpy as np

from . import config, embeddings, rerank, stores


def _time_decay(ts) -> float:
    """真实时间衰减因子：exp(-RECENCY_TIME_DECAY × 距今天数)。无 ts 时不衰减。"""
    if not ts:
        return 1.0
    days = max(0.0, (time.time() - float(ts)) / 86400.0)
    return math.exp(-config.RECENCY_TIME_DECAY * days)

# mark_recalled 后台任务强引用，防 GC
_bg_tasks: set = set()


def _spawn(coro) -> None:
    t = asyncio.ensure_future(coro)
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)


async def _rerank_top(query: str, scored: List[Dict], text_key: str, top_k: int) -> List[Dict]:
    """二阶段精排：对粗排前 N 个候选用 reranker 重排，取 top_k。失败/未启用走原顺序。"""
    if not rerank.enabled() or len(scored) <= 1:
        return scored[:top_k]
    cand = scored[: max(top_k * config.RERANK_CANDIDATE_MULT, top_k)]
    ranked = await rerank.rerank(query, [c[text_key] for c in cand])
    order = [i for i, _ in ranked]
    if not order:
        return scored[:top_k]
    reordered = [cand[i] for i in order if i < len(cand)]
    for c, (_, rs) in zip(reordered, ranked):
        c["rerank"] = round(rs, 4)
    return reordered[:top_k]


# ---------------- 情节记忆：三维打分 ----------------
async def retrieve_episodes(session: str, query: str, top_k: int = None,
                            qvec: Optional[np.ndarray] = None) -> List[Dict]:
    top_k = top_k or config.RETRIEVE_TOP_K

    if qvec is None:
        from . import normalizer
        qvec = await embeddings.embed_query(await normalizer.to_base_lang(query))
    # 候选预筛：postgres 走 pgvector 原生 KNN 取 top-N，sqlite 取全部；再做三维重排
    episodes = await stores.candidate_episodes(session, qvec, max(top_k * 4, 20))
    if not episodes:
        return []

    now_turn = await stores.max_turn(session)
    w_rel, w_rec, w_imp = config.SCORE_WEIGHTS

    scored = []
    for ep in episodes:
        relevance = max(0.0, embeddings.cosine(qvec, ep["vec"]))
        # 新近度 = 轮次衰减 × 真实时间衰减（用户离开两周回来，旧记忆不再"鲜活如昨"）
        recency = (math.exp(-config.RECENCY_DECAY * max(0, now_turn - ep["turn"]))
                   * _time_decay(ep.get("ts")))
        importance = ep["importance"] / 10.0
        score = w_rel * relevance + w_rec * recency + w_imp * importance
        scored.append({
            "id": ep["id"], "event": ep["event"], "emotion": ep["emotion"],
            "importance": ep["importance"], "turn": ep["turn"], "ts": ep.get("ts"),
            "sensitive": ep.get("sensitive", False),
            "relevance": round(relevance, 3), "recency": round(recency, 3),
            "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = await _rerank_top(query, scored, "event", top_k)

    if top:
        # 统计性写操作：后台执行，不占读路径时延
        _spawn(stores.mark_recalled([t["id"] for t in top], now_turn))
    return top


# ---------------- 逐字原话：混合检索 ----------------
# 无空格分词的文字（按字符 bigram 匹配）：汉字 / 日文假名 / 泰文。
# 其余文字（拉丁/西里尔/阿拉伯/韩文/希腊等）按空格分出的词匹配，天然语言无关。
_UNSEG_RUN = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf"   # CJK 统一汉字 + 扩展A
    r"\u3040-\u309f\u30a0-\u30ff"    # 日文平假名 / 片假名
    r"\u0e00-\u0e7f]+"               # 泰文
)

# 常见停用字（高频虚词，匹配上也几乎没有区分度），权重压到最低。
# 中文虚词 + 日文高频助词假名。
_STOP_CHARS = set("的了吗呢吧啊哦我你他她它们之是不在有和跟与去来还叫做想要这那个会就都也很"
                  "はがをにのでとへもよねかな")


def _weighted_tokens(text: str) -> Dict[str, float]:
    """返回 token->权重，语言无关：
    - 空格分词语言（英/俄/韩/阿拉伯等）：整词，权重 2.0（强信号：名字、专名）
    - 无空格语言（中/日/泰）：字符 bigram 权重 2.0，单字权重低（停用字更低）
    """
    text = text.lower()
    toks: Dict[str, float] = {}
    runs = _UNSEG_RUN.findall(text)
    rest = _UNSEG_RUN.sub(" ", text)
    for w in re.findall(r"[^\W_]+", rest, re.UNICODE):
        toks[w] = 2.0
    for run in runs:
        for ch in run:
            toks[ch] = max(toks.get(ch, 0.0), 0.15 if ch in _STOP_CHARS else 0.4)
        for i in range(len(run) - 1):
            bg = run[i:i + 2]
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
                            exclude_recent: int = 0,
                            qvec: Optional[np.ndarray] = None) -> List[Dict]:
    """逐字召回：vector 0.5 + lexical 0.5 融合。exclude_recent 用于剔除已在工作窗口里的轮次。"""
    top_k = top_k or (config.RETRIEVE_TOP_K + 2)

    # 多语言：query 归一到基准语言再向量化（NORMALIZE 关闭时即原文，推荐多语言 embedding）
    if qvec is None:
        from . import normalizer
        qvec = await embeddings.embed_query(await normalizer.to_base_lang(query))

    # 候选预筛：postgres 走 pgvector KNN 取 top-N（避免全量加载），sqlite 取全部
    chunks = await stores.candidate_chunks(session, qvec, config.VERBATIM_CANDIDATES)
    if not chunks:
        return []

    now_turn = await stores.max_turn(session)
    q_weighted = _weighted_tokens(query)

    scored = []
    for c in chunks:
        if exclude_recent and c["turn"] > now_turn - exclude_recent:
            continue  # 最近 N 轮已逐字进了工作窗口，不重复
        vec_s = max(0.0, embeddings.cosine(qvec, c["vec"]))
        # 词法匹配同时看归一化文本和原文：保证同语言精确命中（原文）与跨语言（norm）都有效
        lex_s = max(_lexical_score(q_weighted, c.get("norm") or ""),
                    _lexical_score(q_weighted, c["text"]))
        score = 0.5 * vec_s + 0.5 * lex_s
        if score <= 0.01:
            continue
        scored.append({
            "turn": c["turn"], "role": c["role"], "text": c["text"], "ts": c.get("ts"),
            "vec": round(vec_s, 3), "lex": round(lex_s, 3), "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return await _rerank_top(query, scored, "text", top_k)


# ---------------- 画像相关性选取 ----------------
async def select_facts(session: str, qvec: np.ndarray, facts: List[Dict],
                       top_k: int = None) -> List[Dict]:
    """facts 超过注入上限时，按"与当前 query 相关性 + 置信度 + 新近度"选 top-K。

    单值字段（姓名/年龄/职业等核心身份，无实体向量）始终保留——它们数量少且每轮都该在场。
    多值偏好按分数取 top，避免重度用户的画像撑爆 system prompt。
    """
    from . import profile_schema

    top_k = top_k or config.FACTS_INJECT_TOP_K
    if len(facts) <= top_k:
        return facts

    with_vec = {f["key"]: f.get("vec") for f in await stores.facts_with_vec(session)}
    now_turn = await stores.max_turn(session)

    keep, scored = [], []
    for f in facts:
        if profile_schema.is_single_value_key(f.get("key", "")):
            keep.append(f)  # 核心身份字段始终注入
            continue
        vec = with_vec.get(f.get("key"))
        relevance = max(0.0, embeddings.cosine(qvec, vec)) if vec is not None else 0.0
        age = max(0, now_turn - (f.get("updated_turn") or 0))
        recency = math.exp(-config.RECENCY_DECAY * age)
        score = 0.5 * relevance + 0.3 * (f.get("confidence") or 0.5) + 0.2 * recency
        scored.append((score, f))

    scored.sort(key=lambda x: x[0], reverse=True)
    quota = max(0, top_k - len(keep))
    return keep + [f for _, f in scored[:quota]]
