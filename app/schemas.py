from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    # 记忆隔离维度：每个 (user_id, role_id) 组合是一段完全独立的记忆。
    # 同一用户对不同角色、不同用户对同一角色，记忆互不干扰（1对多 / 多对多）。
    user_id: Optional[str] = None
    role_id: Optional[str] = None
    # 兼容旧调用：直接传 session 也可（不传 user_id/role_id 时用它）。
    session: Optional[str] = None
    message: str
    persona_id: Optional[str] = None
    # 角色提示词正文：由前端创建/管理（localStorage 持久化），随每轮请求直传。
    persona_text: Optional[str] = None
    # 前端传入用于替换角色设定里的 {{char}} / {{user}}（对应主项目的 bot_name / user_name）
    char_name: Optional[str] = None
    user_name: Optional[str] = None
    # 回复语言：传了就在 system prompt 里强制用该语言回复（如 "中文"/"English"）；
    # 不传则不注入语言规则，由角色卡自行约定。
    language: Optional[str] = None


class ResetRequest(BaseModel):
    user_id: Optional[str] = None
    role_id: Optional[str] = None
    session: Optional[str] = None
    char_name: Optional[str] = None
    user_name: Optional[str] = None
