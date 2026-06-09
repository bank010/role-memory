"""会话标识：记忆隔离的核心维度。

记忆按 (user_id, role_id) 组合隔离 —— 每个用户对每个角色都是一段完全独立的记忆
（独立画像、独立关系、独立事件）。这是面向"全球多用户 × 多角色"产品的隔离模型。

内部仍以单一字符串 `session = f"{user_id}\x1f{role_id}"` 作为存储主键，
便于复用现有大量 `WHERE session=?` 逻辑；同时各表另存 user_id/role_id 独立列建索引，
支持运营查询（某用户的全部角色、某角色的全部用户等）与更快的检索过滤。
"""

from typing import Optional, Tuple

# 用不可见分隔符，避免 user_id/role_id 里出现 ':' 造成歧义
_SEP = "\x1f"
_DEFAULT_USER = "anon"
_DEFAULT_ROLE = "default"


def make_session(user_id: Optional[str], role_id: Optional[str],
                 fallback_session: Optional[str] = None) -> Tuple[str, str, str]:
    """组装 (session, user_id, role_id)。

    优先用 user_id/role_id；都没有则尝试解析 fallback_session（兼容旧的单 session 调用）；
    再不行用默认值。返回标准化后的三元组。
    """
    u = (user_id or "").strip()
    r = (role_id or "").strip()
    if not u and not r and fallback_session:
        # 兼容：旧 session 可能是 "user\x1frole" 或任意单串
        if _SEP in fallback_session:
            u, r = fallback_session.split(_SEP, 1)
        else:
            u, r = fallback_session.strip(), _DEFAULT_ROLE
    u = u or _DEFAULT_USER
    r = r or _DEFAULT_ROLE
    return f"{u}{_SEP}{r}", u, r


def split_session(session: str) -> Tuple[str, str]:
    """从 session 还原 (user_id, role_id)。"""
    if _SEP in session:
        u, r = session.split(_SEP, 1)
        return u, r
    return session, _DEFAULT_ROLE
