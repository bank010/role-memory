"""上下文拼装器：把四层记忆组装成最终发给 LLM 的 messages。

注入优先级（也是裁剪时的保留优先级）：
  人设 > 关系状态 > 语义画像 > 滚动摘要 > 情节召回 > 工作窗口 > 本轮输入

性能（在线读路径，毫秒级预算）：
- query 只 embed 一次（带 Redis 缓存），两路召回共享向量；
- facts / relationship / 情节召回 / 逐字召回 / 工作窗口 / max_turn 全部并行 gather；
- debug 里带分段计时（embed_ms / retrieve_ms / total_ms），方便压 SLO。
"""

import asyncio
import re
import time
from typing import Dict, List, Tuple

from .. import config
from ..client import embeddings
from ..util import normalizer
from . import retrieval, stores

_CHAR_PAT = re.compile(r"\{\{\s*char\s*\}\}", re.I)
_USER_PAT = re.compile(r"\{\{\s*user\s*\}\}", re.I)

# 记忆使用规则：拼在每轮 system prompt 末尾，只约束「如何使用上面注入的记忆」。
# 这是记忆系统自身的契约——没有它，注入的事实只是参考资料，模型照样会编造用户信息。
# 语言/回复格式/剧情风格等产品级规则不在这里写，由角色卡(persona)自行定义。
# {user} 会被替换成用户名。
_MEMORY_RULES = """[How to use the memory above]
- Facts about {user} (name/job/preferences/history) must come ONLY from the memory above. If not in memory, say you don't know yet — never invent facts about them. (Your own persona details and in-scene improvisation are free.)
- Weave remembered facts/episodes into the reply naturally; do not list them out or mention "memory"."""

# 接口显式指定回复语言时注入（优先级高于角色卡语言）；未指定则不注入，由角色卡决定。
_LANGUAGE_RULE = ("[Language - MANDATORY]\n"
                  "- You MUST reply in {language} ONLY, no matter what language "
                  "the persona card or earlier replies used.")


def memory_user_name(facts: List[Dict]) -> str:
    """从记忆里取用户名（identity:nickname / name）；取不到返回空串。"""
    for f in facts:
        if f.get("key") in ("identity:nickname", "name") and f.get("value"):
            return str(f["value"]).strip()
    return ""


def _memory_user_name(facts: List[Dict]) -> str:
    """注入用：取用户名，未知则用第二人称 you。"""
    return memory_user_name(facts) or "you"


def _fill_placeholders(text: str, char_name: str, user_name: str) -> str:
    """把记忆/人设里的 {{char}}/{{user}} 占位符替换成真实名字。

    记忆统一用占位符存储（见 pipeline 抽取/摘要 prompt），注入时在这里替换，
    保证视角一致、可移植：换角色名/用户名同一份记忆即可复用。
    """
    if not text:
        return text
    text = _CHAR_PAT.sub(char_name, text)
    text = _USER_PAT.sub(user_name, text)
    return text


def _rel_turn_label(turn: int, now: int) -> str:
    d = now - turn
    if d <= 0:
        return "just now"
    if d == 1:
        return "last turn"
    return f"{d} turns ago"


def _age_label(ts, turn: int, now_turn: int) -> str:
    """记忆年龄标签：优先用真实时间（"3 days ago"比"5 turns ago"对模型更有语义），
    无 ts 时回退轮次。用户离开两周回来，角色能自然地说"我们两周没聊了"。"""
    if not ts:
        return _rel_turn_label(turn, now_turn)
    sec = max(0.0, time.time() - float(ts))
    if sec < 120:
        return "just now"
    if sec < 3600:
        return f"{int(sec // 60)} minutes ago"
    if sec < 86400:
        return f"{int(sec // 3600)} hours ago"
    days = int(sec // 86400)
    if days < 14:
        return f"{days} day{'s' if days > 1 else ''} ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


_CJK_CHAR = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\u0e00-\u0e7f]")


def _effective_len(text: str) -> int:
    """信息量长度：CJK/泰文一个字符的信息量约等于两个拉丁字符，按 2 计。"""
    cjk = len(_CJK_CHAR.findall(text))
    return len(text) + cjk


def _build_retrieval_query(user_msg: str, window: List[Dict]) -> str:
    """检索 query 增强：消息很短（"那它呢？"这类指代）时拼上一轮对话再检索。

    纯 embedding 对指代消解无能为力——"后来怎么样了"的向量和任何记忆都不相似。
    用最近一轮的原文补全语境，让向量与词法两路都有信号可用。
    """
    msg = (user_msg or "").strip()
    if _effective_len(msg) >= config.QUERY_AUGMENT_MAX_LEN or not window:
        return msg
    last = window[-1]
    prev_user = (last.get("user_msg") or "").strip()
    prev_ai = (last.get("ai_reply") or "").strip()
    return " ".join(p for p in [prev_user[:200], prev_ai[:200], msg] if p)


async def build_context(session: str, persona: str, user_msg: str,
                        char_name: str = "Character",
                        user_name: str = None,
                        language: str = None) -> Tuple[List[Dict], Dict]:
    """返回 (messages, debug)。debug 用于前端可视化本轮检索过程。

    {{char}}/{{user}} 用前端传入的 char_name/user_name 替换；
    user_name 未传时回退到记忆里抽到的用户名，再回退到第二人称。
    """
    t0 = time.perf_counter()

    # 1) 先并行取轻量数据：facts / relationship / 工作窗口 / max_turn / 首轮时间（毫秒级 DB 读）
    facts, rel, window, now_turn, first_ts = await asyncio.gather(
        stores.all_facts(session),
        stores.get_relationship(session),
        stores.recent_turns(session, config.WORKING_WINDOW),
        stores.max_turn(session),
        stores.first_turn_ts(session),
    )

    # 2) 检索 query 增强（短消息/指代补全语境）后只 embed 一次（Redis 缓存），两路召回共享
    rquery = _build_retrieval_query(user_msg, window)
    qvec = await embeddings.embed_query(await normalizer.to_base_lang(rquery))
    t_embed = time.perf_counter()

    # 3) 两路召回并行；facts 超注入上限时按相关性选取 top-K
    episodes, verbatim, facts = await asyncio.gather(
        retrieval.retrieve_episodes(session, rquery, qvec=qvec),
        retrieval.retrieve_verbatim(session, rquery,
                                    exclude_recent=config.WORKING_WINDOW, qvec=qvec),
        retrieval.select_facts(session, qvec, facts),
    )
    t_retrieve = time.perf_counter()

    user_name = (user_name or "").strip() or _memory_user_name(facts)
    persona = _fill_placeholders(persona, char_name, user_name)

    # NSFW 关闭：从注入上下文中过滤掉高敏感事实与情节
    if not config.NSFW_ENABLED:
        facts = [f for f in facts if not f.get("sensitive")]
        episodes = [e for e in episodes if not e.get("sensitive")]

    parts = [persona.strip(), ""]

    parts.append("[Current relationship state]")
    parts.append(f"- The person you are talking to is named: {user_name}")
    parts.append(f"- Intimacy: {rel['intimacy']:.2f} / Trust: {rel['trust']:.2f} / Stage: {rel['stage']}")
    parts.append(f"- Current emotional tone: {rel['mood']}")
    if first_ts:
        parts.append(f"- You first talked to {user_name}: {_age_label(first_ts, 0, 0)} (total {now_turn} messages so far)")
    else:
        parts.append(f"- This is your very first conversation with {user_name}. You have NO shared history yet — do not invent any.")
    parts.append("")

    if facts:
        parts.append(f"[Facts you remember about {user_name}]")
        for f in facts:
            val = _fill_placeholders(f['value'], char_name, user_name)
            parts.append(f"- Fact: {val} (confidence {f['confidence']:.1f})")
        parts.append("")

    if rel.get("summary"):
        parts.append("[Recent story summary]")
        parts.append(_fill_placeholders(rel["summary"], char_name, user_name))
        parts.append("")

    if episodes:
        parts.append("[Relevant past episodes (recalled by importance + relevance + recency)]")
        for ep in episodes:
            label = _age_label(ep.get("ts"), ep["turn"], now_turn)
            ev = _fill_placeholders(ep['event'], char_name, user_name)
            parts.append(f"- Episode: ({label}, emotion: {ep['emotion']}) {ev}")
        parts.append("")

    if verbatim:
        parts.append("[Verbatim snippets they said (exact-detail recall; use these for precise facts)]")
        for v in verbatim:
            label = _age_label(v.get("ts"), v["turn"], now_turn)
            who = user_name if v["role"] == "user" else char_name
            parts.append(f"- Quote: ({label}, {who} said) {v['text']}")
        parts.append("")

    parts.append(_MEMORY_RULES.replace("{user}", user_name))
    if (language or "").strip():
        parts.append("")
        parts.append(_LANGUAGE_RULE.replace("{language}", language.strip()))

    messages = [{"role": "system", "content": "\n".join(parts)}]
    for t in window:
        if t["user_msg"]:
            messages.append({"role": "user", "content": t["user_msg"]})
        if t["ai_reply"]:
            messages.append({"role": "assistant", "content": t["ai_reply"]})
    messages.append({"role": "user", "content": user_msg})

    debug = {
        "system_prompt": messages[0]["content"],
        "retrieved_episodes": episodes,
        "retrieved_verbatim": verbatim,
        "facts_injected": facts,
        "relationship": rel,
        "window_turns": len(window),
        "user_name": user_name,
        "char_name": char_name,
        "timing_ms": {
            "embed": round((t_embed - t0) * 1000, 1),
            "retrieve": round((t_retrieve - t_embed) * 1000, 1),
            "total": round((time.perf_counter() - t0) * 1000, 1),
        },
    }
    return messages, debug
