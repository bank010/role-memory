"""LLM 客户端。

- 有 OPENAI_API_KEY → 调真实的 OpenAI 兼容接口
- 没有 key → mock 模式：模板回复 + 规则抽取，让记忆机制在离线下也能完整演示
"""

import asyncio
import json
import re
from typing import Dict, List, Optional

import httpx

from app import config

_chat_sem = asyncio.Semaphore(config.LLM_CONCURRENCY)

_pool_limits = httpx.Limits(
    max_connections=config.HTTPX_MAX_CONNECTIONS,
    max_keepalive_connections=config.HTTPX_MAX_KEEPALIVE,
)
_timeout = httpx.Timeout(
    connect=config.HTTPX_CONNECT_TIMEOUT,
    read=config.HTTPX_READ_TIMEOUT,
    write=10.0, pool=10.0,
)

_client = httpx.AsyncClient(
    base_url=config.CHAT_BASE_URL,
    headers={"Authorization": f"Bearer {config.CHAT_API_KEY}"},
    timeout=_timeout, limits=_pool_limits,
)

_extract_client = httpx.AsyncClient(
    base_url=config.EXTRACT_BASE_URL,
    headers={"Authorization": f"Bearer {config.EXTRACT_API_KEY}"},
    timeout=_timeout, limits=_pool_limits,
)


async def chat(messages: List[Dict], model: Optional[str] = None,
               temperature: float = 0.95, max_tokens: int = 800,
               use_extract_endpoint: bool = False) -> str:
    """对话生成。use_extract_endpoint=True 时走抽取端点（用于摘要等非角色扮演任务）。"""
    if config.MOCK_MODE:
        return _mock_chat(messages)
    client = _extract_client if use_extract_endpoint else _client
    default_model = config.EXTRACT_MODEL if use_extract_endpoint else config.CHAT_MODEL
    async with _chat_sem:
        resp = await client.post(
            "/chat/completions",
            json={
                "model": model or default_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


async def extract_json(prompt: str, model: Optional[str] = None) -> dict:
    """让模型返回 JSON。mock 模式走规则抽取。

    注意：不强制 response_format=json_object —— 部分 OpenAI 兼容端点（如 BytePlus Ark 上的
    某些模型）不支持该参数会直接 400。这里靠 system 指令 + _safe_json 正则兜底解析。
    可用 EXTRACT_JSON_MODE=1 在支持的端点上启用原生 JSON 模式。
    """
    if config.MOCK_MODE:
        return _mock_extract(prompt)
    payload = {
        "model": model or config.EXTRACT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise information extractor. Output ONLY valid JSON, no markdown, no explanation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
    }
    if config.EXTRACT_JSON_MODE:
        payload["response_format"] = {"type": "json_object"}
    resp = await _extract_client.post("/chat/completions", json=payload)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return _safe_json(raw)


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}


# ============================================================
# Mock 实现（离线演示用）
# ============================================================
def _last_user_msg(messages: List[Dict]) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return m["content"]
    return ""


def _mock_chat(messages: List[Dict]) -> str:
    user = _last_user_msg(messages)
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    hints = []
    for line in system.splitlines():
        if line.startswith("- 事实:") or line.startswith("- 情节:"):
            hints.append(line)
    hint = ("（我记得：" + "；".join(h.split(":", 1)[1].strip() for h in hints[:2]) + "）") if hints else ""
    return f"[mock 角色] 嗯，关于「{user[:40]}」，我听到了。{hint}"


# 排除问句词，避免把「我叫什么」抽成名字
_STOP = {"什么", "谁", "啥", "哪", "几"}
_NAME_PAT = re.compile(r"(?:我叫|我的名字是|我名字叫)\s*([A-Za-z\u4e00-\u9fff]{1,12})")
_LIKE_PAT = re.compile(r"我(?:也|还|很|超|特别|最|真的)*(?:喜欢|爱)\s*([^，。!?,.\n]{1,20})")
_HATE_PAT = re.compile(r"我(?:也|还|很|超|特别|真的)*(?:讨厌|害怕|怕|不喜欢)\s*([^，。!?,.\n]{1,20})")
_JOB_PAT = re.compile(r"(?:我是(?:一名|一个)?|我的工作是|我职业是)\s*([^，。!?,.\n]{1,15}?(?:师|员|生|家|手|长|工))")
_PET_PAT = re.compile(r"(?:我养了?|我有(?:一只|只)?)\s*([^，。!?,.\n]{1,12}?(?:猫|狗|鸟|鱼|兔|仓鼠))")


def _dialogue_only(prompt: str) -> str:
    """mock 抽取只看真正的对话部分，避免把 prompt 模板当成内容。"""
    if "Dialogue:" in prompt and "JSON format:" in prompt:
        return prompt.split("Dialogue:", 1)[1].split("JSON format:", 1)[0].strip()
    if "对话：" in prompt and "输出 JSON" in prompt:
        return prompt.split("对话：", 1)[1].split("输出 JSON", 1)[0].strip()
    return prompt.strip()


def _mock_extract(prompt: str) -> dict:
    dialogue = _dialogue_only(prompt)
    user_lines = []
    for line in dialogue.splitlines():
        s = line.strip()
        for prefix in ("User:", "用户:"):
            if s.startswith(prefix):
                user_lines.append(s.split(prefix, 1)[1].strip())
                break
    # 没有可识别的对话行（如 reflect 等其他 JSON 调用）：不要把 prompt 模板当对话抽取
    if not user_lines:
        return {"facts": [], "episode": None, "relationship": {}}
    user_text = "\n".join(user_lines)

    # key 用 schema 规范的 module:field 两段式：多值字段由 pipeline 按 value 自动派生
    # entity 子键（喜欢A/喜欢B 各存一条，不互相覆盖）
    facts, importance = [], 3
    for m in _NAME_PAT.finditer(user_text):
        if m.group(1) not in _STOP:
            facts.append({"key": "identity:nickname", "value": m.group(1), "confidence": 0.9})
    for m in _JOB_PAT.finditer(user_text):
        facts.append({"key": "identity:job", "value": m.group(1), "confidence": 0.8})
    for m in _PET_PAT.finditer(user_text):
        facts.append({"key": "relationship:pet", "value": f"养了{m.group(1)}", "confidence": 0.8})
    for m in _LIKE_PAT.finditer(user_text):
        facts.append({"key": "interest:other", "value": f"喜欢{m.group(1)}", "confidence": 0.7})
    for m in _HATE_PAT.finditer(user_text):
        facts.append({"key": "emotional:trigger", "value": f"害怕/讨厌{m.group(1)}", "confidence": 0.7})
        importance = 6

    # 记忆用用户的语言存储；情绪枚举值保持英文标识（neutral/expectant/tired）便于程序判断
    emotion = "neutral"
    if any(k in user_text for k in ["约定", "答应", "承诺", "一起去", "约好"]):
        importance, emotion = 8, "expectant"
    elif any(k in user_text for k in ["累", "难过", "伤心", "烦"]):
        emotion = "tired"

    episode = None
    first_user = user_text.splitlines()[0].strip() if user_text.splitlines() else ""
    if first_user:
        episode = {"event": first_user[:80], "emotion": emotion, "importance": importance}
    return {
        "facts": facts,
        "episode": episode,
        "relationship": {
            "intimacy_delta": round(0.02 + 0.03 * len(facts), 3),
            "trust_delta": 0.02 if facts else 0.0,
            "mood": emotion if emotion != "neutral" else "calm",
            "stage": "",
        },
    }


async def aclose():
    await _client.aclose()
    await _extract_client.aclose()
