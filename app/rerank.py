"""Rerank 精排（两阶段检索的第二阶段）。

Qwen3-Reranker via vLLM 的 /v1/rerank 端点。关键坑：
- Qwen3-Reranker 强依赖官方 chat 模板，裸文本打分近乎随机/偏向短文本。
  必须把 query / document 套进 <|im_start|>...<|im_end|> 模板，分差才会拉开（实测正确答案≈0.98、无关≈0.00）。
- reranker 在 query 与文档【同语言】时分更高，正好补强 embedding 跨语言粗召回。

设计：
- enabled() 由 config.RERANK_ENABLED 决定；任何异常都降级（返回原顺序），绝不阻断检索。
- rerank() 输入 (query, docs)，输出按相关性降序的 [(原始index, score), ...]。
- 严格延迟预算（RERANK_TIMEOUT_MS）：在线读路径是毫秒级 SLO，精排超时立即降级
  回粗排顺序，绝不让一次网络抖动把整条读路径拖到秒级。
"""

import asyncio
import logging
from typing import List, Tuple

import httpx

from app import config

log = logging.getLogger("rerank")

# 官方推荐模板（来自 vLLM issue #21681 / Qwen3-Reranker 文档）
_INSTRUCT = "Given a query about the user, retrieve memory entries relevant to answering it"
_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
           'based on the Query and the Instruct provided. Note that the answer can '
           'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_client = httpx.AsyncClient(
    base_url=config.RERANK_BASE_URL or "http://localhost",
    headers={"Authorization": f"Bearer {config.RERANK_API_KEY}"},
    timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0),
)


def enabled() -> bool:
    return config.RERANK_ENABLED


def _fmt_query(q: str) -> str:
    return f"{_PREFIX}<Instruct>: {_INSTRUCT}\n<Query>: {q}\n"


def _fmt_doc(d: str) -> str:
    return f"<Document>: {d}{_SUFFIX}"


async def _call(query: str, docs: List[str]) -> List[Tuple[int, float]]:
    resp = await _client.post("/rerank", json={
        "model": config.RERANK_MODEL,
        "query": _fmt_query(query),
        "documents": [_fmt_doc(d) for d in docs],
    })
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [(r["index"], float(r.get("relevance_score", 0.0))) for r in results]


async def rerank(query: str, docs: List[str]) -> List[Tuple[int, float]]:
    """返回按相关性降序的 (原始下标, 分数)。失败/超预算/未启用则降级为原顺序（分数置 0）。"""
    if not config.RERANK_ENABLED or not docs:
        return [(i, 0.0) for i in range(len(docs))]
    fallback = [(i, 0.0) for i in range(len(docs))]
    try:
        if config.RERANK_TIMEOUT_MS > 0:
            ranked = await asyncio.wait_for(_call(query, docs),
                                            timeout=config.RERANK_TIMEOUT_MS / 1000.0)
        else:
            ranked = await _call(query, docs)
        return ranked or fallback
    except asyncio.TimeoutError:
        log.warning("rerank 超出延迟预算 %dms，降级为粗排顺序", config.RERANK_TIMEOUT_MS)
        return fallback
    except Exception as e:
        log.warning("rerank 调用失败，降级为原顺序 err=%s", e)
        return fallback


async def aclose():
    await _client.aclose()
