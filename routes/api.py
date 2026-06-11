"""API 路由 — 纯 HTTP 接口，不含业务逻辑。"""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core import config
from core.archive import mongo
from core.client import rerank
from core.util import cache
from personas import list_personas
from schemas import ChatRequest, ResetRequest
from serve.chat import chat_with_memory

log = logging.getLogger("routes.api")

router = APIRouter(prefix="/api")

# box 由 main.py 注入
box = None


def set_box(memory_box):
    global box
    box = memory_box


@router.get("/health")
async def health():
    return {
        "mock_mode": config.MOCK_MODE,
        "chat_model": config.CHAT_MODEL,
        "window": config.WORKING_WINDOW,
        "process_every": config.PROCESS_EVERY,
        "store_backend": config.STORE_BACKEND,
        "cache_enabled": cache.enabled(),
        "rerank_enabled": rerank.enabled(),
        "personas": list_personas(),
    }


@router.post("/chat")
async def chat(req: ChatRequest):
    role_id = req.role_id or req.persona_id
    try:
        result = await chat_with_memory(
            box,
            user_id=req.user_id or "anon",
            role_id=role_id or "default",
            message=req.message,
            persona_text=req.persona_text,
            persona_id=req.persona_id,
            char_name=req.char_name,
            user_name=req.user_name,
            language=req.language,
        )
    except Exception as e:
        log.error("对话失败 err=%r", e)
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable",
                     "message": "服务暂时不可用，请稍后重试。",
                     "detail": str(e)},
        )
    return JSONResponse(result)


@router.get("/memory")
async def memory(session: str = None, user_id: str = None, role_id: str = None):
    return await box.get_memory(
        user_id=user_id or "anon",
        role_id=role_id or "default",
        session=session,
    )


@router.get("/history")
async def history(session: str = None, user_id: str = None, role_id: str = None, n: int = 40):
    turns = await box.get_history(
        user_id=user_id or "anon",
        role_id=role_id or "default",
        n=n, session=session,
    )
    return {"turns": turns}


@router.get("/conversations")
async def conversations(
    user_id: str = None,
    role_id: str = None,
    limit: int = 20,
    skip: int = 0,
    full: bool = False,
    order: str = "desc",
):
    """从 MongoDB 归档库分页拉取历史对话（训练数据 / 外部集成用）。

    与 /history 的区别：/history 读 SQLite/PG 的 turns 表（在线记忆，最近 N 轮原文）；
    本接口读 Mongo 归档（含完整 prompt、模型、耗时、feedback，支持分页与全量训练样本）。
    """
    return await mongo.get_conversations(
        user_id=(user_id or "").strip(),
        role_id=(role_id or "").strip(),
        limit=limit, skip=skip, full=full, order=order,
    )


@router.post("/reprocess")
async def reprocess(req: ResetRequest):
    result = await box.reprocess(
        user_id=req.user_id or "anon",
        role_id=req.role_id or "default",
        char_name=(req.char_name or "").strip() or "Character",
        user_name=(req.user_name or "").strip(),
        session=req.session,
    )
    return {"ok": True, **result}


@router.post("/reset")
async def reset(req: ResetRequest):
    await box.reset(
        user_id=req.user_id or "anon",
        role_id=req.role_id or "default",
        session=req.session,
    )
    return {"ok": True}
