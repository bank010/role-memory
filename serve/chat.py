"""LLM 对话服务 — 单文件，只负责调模型拿回复。

记忆相关的事全部交给 MemoryBox；这里只做：
1. 从 MemoryBox 拿到拼好的 messages
2. 调 LLM 生成回复
3. 把本轮对话存回 MemoryBox（后台异步加工）
"""

import logging
import re
import time
from typing import Dict, Optional

from core import MemoryBox
from core.client import llm
from personas import get_persona, get_char_name

log = logging.getLogger("serve.chat")

_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """只剥 markdown 标题符号；保留 *action* 星号（角色扮演通用格式）。"""
    if not text or "#" not in text:
        return text
    return _MD_HEADER.sub("", text)


async def chat_with_memory(
    box: MemoryBox,
    user_id: str,
    role_id: str,
    message: str,
    persona_text: Optional[str] = None,
    persona_id: Optional[str] = None,
    char_name: Optional[str] = None,
    user_name: Optional[str] = None,
    language: Optional[str] = None,
) -> Dict:
    """完整的一轮对话：读记忆 → 调 LLM → 存记忆，返回结构化结果。"""
    persona = (persona_text or "").strip() or get_persona(persona_id)
    char_name = (char_name or "").strip() or get_char_name(persona_id)

    t0 = time.perf_counter()

    messages, debug = await box.build_prompt(
        user_id=user_id, role_id=role_id,
        persona=persona, user_msg=message,
        char_name=char_name, user_name=user_name, language=language,
    )
    t_ctx = time.perf_counter()

    reply = _strip_markdown(await llm.chat(messages))
    t_llm = time.perf_counter()

    mem_user_name = (user_name or "").strip() or debug.get("user_name") or ""
    turn = await box.save_turn(
        user_id=user_id, role_id=role_id,
        user_msg=message, ai_reply=reply,
        char_name=char_name, user_name=mem_user_name,
    )

    timing = dict(debug.get("timing_ms") or {})
    timing["context"] = round((t_ctx - t0) * 1000, 1)
    timing["llm"] = round((t_llm - t_ctx) * 1000, 1)
    timing["total"] = round((t_llm - t0) * 1000, 1)

    return {
        "reply": reply,
        "turn": turn,
        "debug": {
            "retrieved_episodes": debug["retrieved_episodes"],
            "retrieved_verbatim": debug["retrieved_verbatim"],
            "facts_injected": debug["facts_injected"],
            "relationship": debug["relationship"],
            "window_turns": debug["window_turns"],
            "system_prompt": debug["system_prompt"],
            "timing_ms": timing,
        },
    }
