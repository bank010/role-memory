"""上下文拼装器：把四层记忆组装成最终发给 LLM 的 messages。

注入优先级（也是裁剪时的保留优先级）：
  人设 > 关系状态 > 语义画像 > 滚动摘要 > 情节召回 > 工作窗口 > 本轮输入
"""

import re
from typing import Dict, List, Tuple

from app import config
from app.memory import retrieval, stores

_CHAR_PAT = re.compile(r"\{\{\s*char\s*\}\}", re.I)
_USER_PAT = re.compile(r"\{\{\s*user\s*\}\}", re.I)


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


async def build_context(session: str, persona: str, user_msg: str,
                        char_name: str = "Character",
                        user_name: str = None) -> Tuple[List[Dict], Dict]:
    """返回 (messages, debug)。debug 用于前端可视化本轮检索过程。

    {{char}}/{{user}} 用前端传入的 char_name/user_name 替换；
    user_name 未传时回退到记忆里抽到的用户名，再回退到第二人称。
    """
    facts = stores.all_facts(session)
    rel = stores.get_relationship(session)
    user_name = (user_name or "").strip() or _memory_user_name(facts)
    persona = _fill_placeholders(persona, char_name, user_name)
    episodes = await retrieval.retrieve_episodes(session, user_msg)

    # NSFW 关闭：从注入上下文中过滤掉高敏感事实与情节
    if not config.NSFW_ENABLED:
        facts = [f for f in facts if not f.get("sensitive")]
        episodes = [e for e in episodes if not e.get("sensitive")]
    # 逐字召回：剔除已在工作窗口里的最近轮次，避免重复
    verbatim = await retrieval.retrieve_verbatim(
        session, user_msg, exclude_recent=config.WORKING_WINDOW
    )
    window = stores.recent_turns(session, config.WORKING_WINDOW)
    now_turn = stores.max_turn(session)

    parts = [persona.strip(), ""]

    parts.append("[Current relationship state]")
    parts.append(f"- Intimacy: {rel['intimacy']:.2f} / Trust: {rel['trust']:.2f} / Stage: {rel['stage']}")
    parts.append(f"- Current emotional tone: {rel['mood']}")
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
            label = _rel_turn_label(ep["turn"], now_turn)
            ev = _fill_placeholders(ep['event'], char_name, user_name)
            parts.append(f"- Episode: ({label}, emotion: {ep['emotion']}) {ev}")
        parts.append("")

    if verbatim:
        parts.append("[Verbatim snippets they said (exact-detail recall; use these for precise facts)]")
        for v in verbatim:
            label = _rel_turn_label(v["turn"], now_turn)
            who = user_name if v["role"] == "user" else char_name
            parts.append(f"- Quote: ({label}, {who} said) {v['text']}")
        parts.append("")

    parts.append(
        "[Reply requirements]\n"
        "- Always stay consistent with your persona and the relationship state above; "
        "show what you remember naturally, do not list out settings.\n"
        f"- Objective facts about {user_name} (name/job/preferences/history) must rely ONLY on the "
        "information present in the memory above.\n"
        f"  If {user_name} asks about something not in memory, honestly say you don't know yet / "
        "they haven't told you; NEVER fabricate facts about them.\n"
        "  (Note: this only constrains \"facts about the user\"; your own persona details and "
        "in-scene improvisation are not affected.)\n"
        "\n"
        "[Drive the plot forward - IMPORTANT]\n"
        "- You are a co-driver of the story, not a passive responder. Every reply should move the story FORWARD a step.\n"
        "- Actively create plot: introduce new actions / scene details, express your own desires or emotional shifts, "
        "recall and echo past episodes, rather than only answering what they said or waiting for instructions.\n"
        "- End with a CLEAR hook for them: an action, a question, a choice, or a tease, so they know what they can do next.\n"
        "  Avoid vague endings like \"what do you want to do?\"; be specific, vivid, and fit the current situation and your persona.\n"
        "- Keep narrative tension: depending on the relationship stage and emotional tone, proactively escalate, "
        "create a turn, or introduce a small conflict so the conversation has dynamics.\n"
        "- If the role setting (persona above) specifies reply format / paragraphs / point of view, follow it strictly."
    )

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
    }
    return messages, debug
