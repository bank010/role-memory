"""语言归一化层（多语言记忆的关键）。

设计：记忆层统一用一种基准语言（默认英文）做 embedding / 关键词 / 事实抽取，
原文照常保留显示。查询时也先把 query 归一到基准语言，从而实现跨语言一致召回。

- 真实模式：用 LLM 翻译到基准语言（也可换成你自己的翻译服务，见 NORMALIZE_URL 钩子）
- mock 模式：无法翻译，原样返回（多语言一致性需真实模式或外接翻译服务）

带一个轻量 LRU 缓存，避免重复翻译同一句（高并发友好）。
"""

import logging
from collections import OrderedDict

from app import config, llm

log = logging.getLogger("normalizer")

BASE_LANG = "English"
_CACHE_MAX = 2000
_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_get(key: str):
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_put(key: str, val: str):
    _cache[key] = val
    _cache.move_to_end(key)
    if len(_cache) > _CACHE_MAX:
        _cache.popitem(last=False)


async def to_base_lang(text: str) -> str:
    """把任意语言文本归一化到基准语言，用于 embedding / 关键词匹配。"""
    text = (text or "").strip()
    if not text:
        return ""
    if not config.NORMALIZE_ENABLED:
        return text  # 多语言 embedding 直接处理原文，无需翻译归一（推荐）
    if config.MOCK_MODE:
        return text  # mock 无法翻译；跨语言一致性需真实模式
    cached = _cache_get(text)
    if cached is not None:
        return cached
    try:
        out = await llm.chat(
            [
                {"role": "system",
                 "content": f"Translate the user's text to {BASE_LANG}. "
                            f"Output ONLY the translation, no quotes, no explanation. "
                            f"If it is already {BASE_LANG}, return it unchanged."},
                {"role": "user", "content": text[:1000]},
            ],
            temperature=0.0, max_tokens=300,
        )
        out = (out or text).strip()
        _cache_put(text, out)
        return out
    except Exception as e:
        log.warning("归一化失败，回退原文 err=%s", e)
        return text
