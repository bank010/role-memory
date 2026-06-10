"""向量化。

- 有 key → 调真实 embedding 接口
- 无 key → 本地哈希向量（char n-gram 散列到固定维度），离线也能演示语义检索的"形"
本地向量精度远不如真实模型，但足够展示"相关性召回"的机制。
"""

import asyncio
import hashlib
import logging
from typing import List

import httpx
import numpy as np

from app import config

log = logging.getLogger("embeddings")

_LOCAL_DIM = 256

_sem = asyncio.Semaphore(config.EMBED_CONCURRENCY)

_client = httpx.AsyncClient(
    base_url=config.EMBED_BASE_URL,
    headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
    timeout=httpx.Timeout(
        connect=config.HTTPX_CONNECT_TIMEOUT,
        read=config.HTTPX_READ_TIMEOUT,
        write=10.0, pool=10.0,
    ),
    limits=httpx.Limits(
        max_connections=config.HTTPX_MAX_CONNECTIONS,
        max_keepalive_connections=config.HTTPX_MAX_KEEPALIVE,
    ),
)


def _local_embed(text: str) -> np.ndarray:
    vec = np.zeros(_LOCAL_DIM, dtype=np.float32)
    text = text.lower()
    tokens = list(text) + [text[i:i + 2] for i in range(len(text) - 1)]
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % _LOCAL_DIM] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


async def _embed_remote(text: str) -> np.ndarray:
    """带信号量 + 重试的远程 embedding 调用。"""
    last_err = None
    for attempt in range(config.API_RETRIES + 1):
        async with _sem:
            try:
                resp = await _client.post(
                    "/embeddings", json={"model": config.EMBED_MODEL, "input": text[:4000]}
                )
                resp.raise_for_status()
                arr = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(arr)
                return arr / norm if norm > 0 else arr
            except Exception as e:
                last_err = e
                if attempt < config.API_RETRIES:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    log.debug("embed 重试 %d/%d err=%s", attempt + 1, config.API_RETRIES, e)
    raise last_err


async def embed(text: str) -> np.ndarray:
    if not config.EMBED_REAL:
        return _local_embed(text)
    return await _embed_remote(text)


async def embed_query(text: str) -> np.ndarray:
    """读路径专用：带 Redis 缓存的查询向量化。

    重复/相同问法直接命中缓存，省掉一次远程 embedding 调用（几十毫秒级），
    是在线检索做到毫秒级的关键一环。写路径（记忆入库）仍走 embed()。
    """
    from app import cache

    cached_blob = await cache.get_embedding(config.EMBED_MODEL, text)
    if cached_blob is not None:
        vec = from_blob(cached_blob)
        if vec.size:
            return vec
    vec = await embed(text)
    await cache.set_embedding(config.EMBED_MODEL, text, to_blob(vec))
    return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    return float(np.dot(a, b))  # 已归一化


def to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    if not blob:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


async def aclose():
    await _client.aclose()
