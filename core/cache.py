"""Redis 热缓存（异步，读穿透 + 写失效）。

职责：
- relationship / facts 热读缓存（每轮拼上下文都要读，读多写少）
- 查询 embedding 缓存（重复/相近问法免一次远程 embed 调用，读路径毫秒级的关键之一）
- 分布式加工锁（多 worker / 多实例下保证同一 session 的记忆加工不并发）

设计原则：
- 全异步（redis.asyncio）：缓存操作绝不阻塞事件循环。
- 缓存是旁路优化，绝不能成为故障点。Redis 不可用 / 未配置时全部降级为直查后端，业务无感。
- 写操作（pipeline 异步加工）走 invalidate，避免脏读；同时设 TTL 兜底。
"""

import asyncio
import base64
import hashlib
import json
import logging
from typing import Awaitable, Callable, Optional

from . import config

log = logging.getLogger("cache")

_client = None
_init_done = False
_init_lock = asyncio.Lock()


async def _get_client():
    global _client, _init_done
    if _init_done:
        return _client
    async with _init_lock:
        if _init_done:
            return _client
        if not config.CACHE_ENABLED:
            _init_done = True
            return None
        try:
            import redis.asyncio as aioredis
            c = aioredis.from_url(
                config.REDIS_URL,
                decode_responses=True,
                max_connections=config.REDIS_MAX_CONNECTIONS,
                socket_connect_timeout=config.REDIS_SOCKET_CONNECT_TIMEOUT,
                socket_timeout=config.REDIS_SOCKET_TIMEOUT,
            )
            await c.ping()
            _client = c
            log.info("Redis 缓存已启用: %s (pool=%d)", config.REDIS_URL, config.REDIS_MAX_CONNECTIONS)
        except Exception as e:
            log.warning("Redis 不可用，降级为直查后端: %s", e)
            _client = None
        _init_done = True
        return _client


def enabled() -> bool:
    """同步快速判断（仅在已初始化后准确；未初始化时按配置判断）。"""
    return _client is not None if _init_done else config.CACHE_ENABLED


def _key(session: str, kind: str) -> str:
    return f"rm:{kind}:{session}"


async def get_json(session: str, kind: str):
    c = await _get_client()
    if not c:
        return None
    try:
        raw = await c.get(_key(session, kind))
        return json.loads(raw) if raw else None
    except Exception as e:
        log.debug("缓存读失败(降级): %s", e)
        return None


async def set_json(session: str, kind: str, value) -> None:
    c = await _get_client()
    if not c:
        return
    try:
        await c.setex(_key(session, kind), config.CACHE_TTL,
                      json.dumps(value, ensure_ascii=False))
    except Exception as e:
        log.debug("缓存写失败(忽略): %s", e)


async def invalidate(session: str, *kinds: str) -> None:
    c = await _get_client()
    if not c:
        return
    try:
        await c.delete(*[_key(session, k) for k in (kinds or ("facts", "relationship"))])
    except Exception as e:
        log.debug("缓存失效失败(忽略): %s", e)


async def cached(session: str, kind: str, loader: Callable[[], Awaitable]):
    """读穿透：命中返回缓存，未命中回源（async loader）并回填。"""
    hit = await get_json(session, kind)
    if hit is not None:
        return hit
    value = await loader()
    await set_json(session, kind, value)
    return value


# ---------------- 查询 embedding 缓存 ----------------
def _emb_key(model: str, text: str) -> str:
    h = hashlib.sha1(f"{model}\x1f{text}".encode("utf-8")).hexdigest()
    return f"rm:emb:{h}"


async def get_embedding(model: str, text: str) -> Optional[bytes]:
    """命中返回 float32 原始字节，未命中返回 None。"""
    c = await _get_client()
    if not c:
        return None
    try:
        raw = await c.get(_emb_key(model, text))
        return base64.b64decode(raw) if raw else None
    except Exception as e:
        log.debug("embedding 缓存读失败(降级): %s", e)
        return None


async def set_embedding(model: str, text: str, blob: bytes) -> None:
    c = await _get_client()
    if not c:
        return
    try:
        await c.setex(_emb_key(model, text), config.EMBED_CACHE_TTL,
                      base64.b64encode(blob).decode("ascii"))
    except Exception as e:
        log.debug("embedding 缓存写失败(忽略): %s", e)


# ---------------- 分布式加工锁 ----------------
async def acquire_lock(name: str, ttl_sec: int = 120) -> Optional[str]:
    """SET NX 抢锁。成功返回锁 token（释放时校验），失败返回 None。
    Redis 未启用时返回 'local'（让上层退化为仅靠进程内锁）。"""
    c = await _get_client()
    if not c:
        return "local"
    import uuid
    token = uuid.uuid4().hex
    try:
        ok = await c.set(f"rm:lock:{name}", token, nx=True, ex=ttl_sec)
        return token if ok else None
    except Exception as e:
        log.debug("抢锁失败(降级放行): %s", e)
        return "local"


async def release_lock(name: str, token: str) -> None:
    if token == "local":
        return
    c = await _get_client()
    if not c:
        return
    try:
        # 校验 token 再删，避免误删他人持有的锁（过期后被重抢的场景）
        lua = ("if redis.call('get', KEYS[1]) == ARGV[1] then "
               "return redis.call('del', KEYS[1]) else return 0 end")
        await c.eval(lua, 1, f"rm:lock:{name}", token)
    except Exception as e:
        log.debug("释放锁失败(忽略，TTL 兜底): %s", e)


async def aclose():
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception:
            pass
        _client = None
