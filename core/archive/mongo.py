"""MongoDB 对话归档（异步，旁路写入，绝不阻塞回复）。

定位：与核心记忆（store/）完全解耦的"对话流水 + 训练数据湖"。
每轮对话把可复现训练样本的全部信息落一份到 MongoDB，供：
- 偏好训练：点赞/点踩/重新生成（后续接入 feedback 接口）
- 外部系统集成：按 user_id/role_id 拉取完整对话记录

设计原则（对齐 util/cache.py）：
- 全异步（motor）：归档操作不阻塞事件循环。
- 旁路优化，绝不能成为故障点。未配置 / Mongo 不可用时全部降级为 no-op，对话无感。
- 文档结构灵活：训练字段（feedback / regenerations）后续可平滑追加。
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .. import config

log = logging.getLogger("archive")

_client = None
_collection = None
_init_done = False
_init_lock = asyncio.Lock()


async def _get_collection():
    global _client, _collection, _init_done
    if _init_done:
        return _collection
    async with _init_lock:
        if _init_done:
            return _collection
        if not config.ARCHIVE_ENABLED:
            _init_done = True
            return None
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            c = AsyncIOMotorClient(
                config.MONGO_URL,
                maxPoolSize=config.MONGO_MAX_POOL,
                serverSelectionTimeoutMS=config.MONGO_TIMEOUT_MS,
                connectTimeoutMS=config.MONGO_TIMEOUT_MS,
            )
            await c.admin.command("ping")
            coll = c[config.MONGO_DB][config.MONGO_ARCHIVE_COLLECTION]
            await coll.create_index([("user_id", 1), ("role_id", 1), ("turn", 1)])
            await coll.create_index([("created_at", -1)])
            await coll.create_index([("feedback", 1)])
            _client = c
            _collection = coll
            log.info("MongoDB 对话归档已启用: %s/%s.%s",
                     config.MONGO_URL, config.MONGO_DB, config.MONGO_ARCHIVE_COLLECTION)
        except Exception as e:
            log.warning("MongoDB 不可用，对话归档降级为 no-op: %s", e)
            _collection = None
        _init_done = True
        return _collection


def enabled() -> bool:
    """同步快速判断（已初始化后准确；未初始化时按配置判断）。"""
    return _collection is not None if _init_done else config.ARCHIVE_ENABLED


async def archive_turn(
    user_id: str,
    role_id: str,
    session: str,
    turn: int,
    user_msg: str,
    reply: str,
    system_prompt: str,
    messages: List[Dict],
    char_name: str = "",
    model: str = "",
    timing_ms: Optional[Dict] = None,
) -> Optional[str]:
    """归档一轮对话，返回文档 id（供后续反馈接口引用）。失败返回 None（静默降级）。

    存的是【可复现训练样本】：messages 是当时实际发给 LLM 的完整输入
    （含注入的记忆/人设），reply 是产出，后续 feedback/regenerations 字段挂在同一文档上。
    """
    coll = await _get_collection()
    if coll is None:
        return None
    try:
        doc = {
            "user_id": user_id,
            "role_id": role_id,
            "session": session,
            "turn": turn,
            "user_msg": user_msg,
            "reply": reply,
            "system_prompt": system_prompt,
            "messages": messages,
            "char_name": char_name,
            "model": model,
            "timing_ms": timing_ms or {},
            "created_at": datetime.now(timezone.utc),
            "feedback": None,        # 后续 /api/feedback 填 "up"/"down"
            "regenerations": [],     # 后续重新生成的候选回复
        }
        res = await coll.insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        log.debug("对话归档写入失败(忽略): %s", e)
        return None


async def get_conversations(
    user_id: str = "",
    role_id: str = "",
    limit: int = 20,
    skip: int = 0,
    full: bool = False,
    order: str = "desc",
) -> Dict:
    """分页查询归档的历史对话。

    - user_id / role_id：过滤条件，留空则不限。
    - full=False（默认）：剔除 messages / system_prompt 两个大字段，列表更轻。
      full=True：返回完整训练样本（含发给 LLM 的完整 messages 和注入的 system_prompt）。
    - order：created_at 排序，desc=最新在前（默认），asc=按时间正序。
    返回 {"total": 总数, "items": [...]}；Mongo 不可用时返回空集（静默降级）。
    """
    coll = await _get_collection()
    if coll is None:
        return {"total": 0, "items": [], "enabled": False}
    try:
        q: Dict = {}
        if user_id:
            q["user_id"] = user_id
        if role_id:
            q["role_id"] = role_id
        projection = None if full else {"messages": 0, "system_prompt": 0}
        direction = 1 if order == "asc" else -1

        total = await coll.count_documents(q)
        cursor = (coll.find(q, projection)
                  .sort("created_at", direction)
                  .skip(max(skip, 0))
                  .limit(max(min(limit, 200), 1)))
        items = []
        async for d in cursor:
            d["id"] = str(d.pop("_id"))
            ca = d.get("created_at")
            if ca is not None and hasattr(ca, "isoformat"):
                d["created_at"] = ca.isoformat()
            items.append(d)
        return {"total": total, "items": items, "enabled": True}
    except Exception as e:
        log.debug("归档查询失败(降级): %s", e)
        return {"total": 0, "items": [], "enabled": True, "error": str(e)}


async def aclose():
    global _client, _collection, _init_done
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _collection = None
    _init_done = False
