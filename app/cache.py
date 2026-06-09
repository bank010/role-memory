"""Redis 热缓存（读穿透 + 写失效）。

只缓存"读多写少、且读在对话主链路上"的热数据：
- relationship（每轮拼上下文都要读）
- facts（每轮拼上下文都要读）

设计原则：
- 缓存是旁路优化，绝不能成为故障点。Redis 不可用 / 未配置时全部降级为直查后端，业务无感。
- 写操作（pipeline 异步加工）走 invalidate，避免脏读；同时设 TTL 兜底。
- 逐字/情节召回涉及向量与打分，不进缓存（命中率低、收益小）。
"""

import json
import logging
from typing import Callable, List, Optional

from app import config

log = logging.getLogger("cache")

_client = None
_init_done = False


def _get_client():
    global _client, _init_done
    if _init_done:
        return _client
    _init_done = True
    if not config.CACHE_ENABLED:
        return None
    try:
        import redis
        c = redis.from_url(config.REDIS_URL, decode_responses=True,
                           socket_connect_timeout=1, socket_timeout=1)
        c.ping()
        _client = c
        log.info("Redis 缓存已启用: %s", config.REDIS_URL)
    except Exception as e:
        log.warning("Redis 不可用，降级为直查后端: %s", e)
        _client = None
    return _client


def enabled() -> bool:
    return _get_client() is not None


def _key(session: str, kind: str) -> str:
    return f"rm:{kind}:{session}"


def get_json(session: str, kind: str):
    c = _get_client()
    if not c:
        return None
    try:
        raw = c.get(_key(session, kind))
        return json.loads(raw) if raw else None
    except Exception as e:
        log.debug("缓存读失败(降级): %s", e)
        return None


def set_json(session: str, kind: str, value) -> None:
    c = _get_client()
    if not c:
        return
    try:
        c.setex(_key(session, kind), config.CACHE_TTL, json.dumps(value, ensure_ascii=False))
    except Exception as e:
        log.debug("缓存写失败(忽略): %s", e)


def invalidate(session: str, *kinds: str) -> None:
    c = _get_client()
    if not c:
        return
    try:
        c.delete(*[_key(session, k) for k in (kinds or ("facts", "relationship"))])
    except Exception as e:
        log.debug("缓存失效失败(忽略): %s", e)


def cached(session: str, kind: str, loader: Callable):
    """读穿透：命中返回缓存，未命中回源并回填。"""
    hit = get_json(session, kind)
    if hit is not None:
        return hit
    value = loader()
    set_json(session, kind, value)
    return value
