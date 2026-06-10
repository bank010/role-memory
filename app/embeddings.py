"""向量化。

- 有 key → 调真实 embedding 接口
- 无 key → 本地哈希向量（char n-gram 散列到固定维度），离线也能演示语义检索的"形"
本地向量精度远不如真实模型，但足够展示"相关性召回"的机制。
"""

import hashlib
from typing import List

import httpx
import numpy as np

from app import config

_LOCAL_DIM = 256

_client = httpx.AsyncClient(
    base_url=config.EMBED_BASE_URL,
    headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
    timeout=httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0),
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


async def embed(text: str) -> np.ndarray:
    if not config.EMBED_REAL:
        return _local_embed(text)
    resp = await _client.post(
        "/embeddings", json={"model": config.EMBED_MODEL, "input": text[:4000]}
    )
    resp.raise_for_status()
    arr = np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


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
