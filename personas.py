"""人设记忆（L0，静态）。

要点：人设/系统提示词是 **L0 静态层**，每轮原样 100% 注入 system prompt，
不进可变记忆库——保证人设永不漂移、永不被错误抽取覆盖。

角色完全由前端创建/管理（提示词存在浏览器 localStorage），通过 /api/chat 的
`persona_text` 字段随每轮请求传入。后端不再内置任何角色，也不加载外部 role_*.py。
"""

import logging

log = logging.getLogger("personas")

# 前端未传 persona_text 时的兜底人设（保证空角色也能起对话）
_FALLBACK_PERSONA = (
    "你是一个友好、自然的角色扮演伙伴。请保持人设一致，与用户自然对话。"
)


def list_personas() -> list:
    """不再内置角色；角色由前端 localStorage 管理。"""
    return []


def get_persona(persona_id: str = None) -> str:
    """后端无内置角色，返回兜底人设（正常情况下前端会直传 persona_text）。"""
    return _FALLBACK_PERSONA


def get_char_name(persona_id: str = None) -> str:
    return "Character"
