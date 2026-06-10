"""语言归一化层（默认关闭，仅作兜底）。

当前策略：记忆直接用用户的语言存储和 embedding，不做任何翻译——
绝大多数用户只说一种语言，原文存储读起来自然、不丢小语种细节，
且省掉翻译这一跳的 token 成本；跨语言召回交给多语言 embedding（Qwen3）解决。

NORMALIZE_ENABLED=1 仅在你用的 embedding 模型不支持多语言时作为兜底开启：
把文本翻译到基准语言后再 embed，换取跨语言一致性（多一次 LLM 调用）。

带一个轻量 LRU 缓存，避免重复翻译同一句（高并发友好）。
"""

import logging
from collections import OrderedDict

from . import config, llm

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
