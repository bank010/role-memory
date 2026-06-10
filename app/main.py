import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import cache, config, embeddings, llm, personas, rerank
from app import session as session_mod
from app.memory import assembler, pipeline, stores
from app.schemas import ChatRequest, ResetRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("app")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# 后台任务强引用集合：防止 asyncio 只持弱引用导致 task 被中途 GC。
_bg_tasks: set = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await stores.init()
    log.info("启动完成 | mock=%s | chat=%s @ %s | embed_real=%s | rerank=%s | store=%s | cache=%s",
             config.MOCK_MODE, config.CHAT_MODEL, config.CHAT_BASE_URL, config.EMBED_REAL,
             rerank.enabled(), config.STORE_BACKEND, cache.enabled())
    yield
    # 优雅停机：先排空在途的后台索引/加工任务，再关闭各客户端与存储
    if _bg_tasks:
        await asyncio.gather(*list(_bg_tasks), return_exceptions=True)
    await llm.aclose()
    await embeddings.aclose()
    await rerank.aclose()
    await cache.aclose()
    await stores.close()


app = FastAPI(title="角色扮演记忆系统 Demo", lifespan=lifespan)

# 前后端分离：允许跨域请求。
# CORS_ORIGINS 环境变量可填逗号分隔的前端域名白名单（如 https://app.example.com,https://x.com）；
# 留空（默认）则放行所有来源，方便本地/演示。生产建议显式配白名单。
_origins = [o.strip() for o in config.CORS_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"mock_mode": config.MOCK_MODE, "chat_model": config.CHAT_MODEL,
            "window": config.WORKING_WINDOW, "process_every": config.PROCESS_EVERY,
            "store_backend": config.STORE_BACKEND, "cache_enabled": cache.enabled(),
            "rerank_enabled": rerank.enabled(),
            "personas": personas.list_personas()}


_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """只剥 markdown 标题符号；保留 *action* 星号（角色扮演通用格式）。"""
    if not text or "#" not in text:
        return text
    return _MD_HEADER.sub("", text)


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # 角色提示词优先用前端直传的 persona_text；没有则回退到兜底人设
    persona = (req.persona_text or "").strip() or personas.get_persona(req.persona_id)
    # 优先用前端传入的名字，未传则回退
    char_name = (req.char_name or "").strip() or personas.get_char_name(req.persona_id)

    # 记忆隔离：(user_id, role_id) → session；role_id 未传时回退用 persona_id
    role_id = req.role_id or req.persona_id
    session, user_id, role_id = session_mod.make_session(req.user_id, role_id, req.session)

    t0 = time.perf_counter()

    # 1) 读路径（在线，快）：拼装上下文（含 embedding 调用，可能超时）
    try:
        messages, debug = await assembler.build_context(
            session, persona, req.message, char_name, req.user_name, req.language
        )
    except Exception as e:
        log.error("记忆读路径失败 session=%s err=%r", session, e)
        return JSONResponse(
            status_code=503,
            content={"error": "memory_unavailable",
                     "message": "记忆服务暂时不可用（embedding 超时或网络波动），请稍后重试。",
                     "detail": str(e)},
        )
    t_ctx = time.perf_counter()

    # 2) 生成回复（LLM 端点可能超时/报错：返回结构化 JSON 错误，前端能正常解析，不再裸 500）
    try:
        reply = _strip_markdown(await llm.chat(messages))
    except Exception as e:
        log.error("LLM 对话生成失败 session=%s err=%r", session, e)
        return JSONResponse(
            status_code=503,
            content={"error": "llm_unavailable",
                     "message": "模型暂时不可用（超时或网络波动），请稍后重试。",
                     "detail": str(e)},
        )

    # 3) 写路径（落日志立即返回；逐字索引 + 记忆加工后台异步，不阻塞）
    turn = await stores.append_turn(session, req.message, reply, user_id, role_id)

    # 记忆直接用真实名字存储：用前端传入的名字（user_name 未传则用 build_context 解析出的）
    mem_user_name = (req.user_name or "").strip() or debug.get("user_name") or ""

    async def _index_and_process():
        # 逐字索引这一轮原话（细节 100% 进可检索库）。
        # index 与加工解耦：逐字索引失败也不能阻断记忆加工（否则画像永远不更新）。
        try:
            await stores.index_chunk(session, turn, "user", req.message, user_id, role_id)
            await stores.index_chunk(session, turn, "assistant", reply, user_id, role_id)
        except Exception as e:
            log.error("逐字索引失败 session=%s err=%r", session, e)
        try:
            await pipeline.maybe_process(session, user_id, role_id, char_name, mem_user_name)
        except Exception as e:
            log.error("后台记忆加工失败 session=%s err=%r", session, e)

    # 保存 task 强引用，防止事件循环只持弱引用时被 GC 中途回收（官方已知坑）。
    _bg_tasks.add(t := asyncio.create_task(_index_and_process()))
    t.add_done_callback(_bg_tasks.discard)

    # 整链路分段计时：context = 记忆读路径（embed+召回+拼装），llm = 模型生成，
    # 前端按气泡展示，方便定位"慢在哪"
    t_end = time.perf_counter()
    timing = dict(debug.get("timing_ms") or {})
    timing["context"] = round((t_ctx - t0) * 1000, 1)
    timing["llm"] = round((t_end - t_ctx) * 1000, 1)
    timing["total"] = round((t_end - t0) * 1000, 1)

    return JSONResponse({
        "reply": reply,
        "turn": turn,
        "debug": {
            "retrieved_episodes": debug["retrieved_episodes"],
            "retrieved_verbatim": debug["retrieved_verbatim"],
            "facts_injected": debug["facts_injected"],
            "relationship": debug["relationship"],
            "window_turns": debug["window_turns"],
            "system_prompt": debug["system_prompt"],
            "timing_ms": timing,
        },
    })


@app.get("/api/memory")
async def memory(session: str = None, user_id: str = None, role_id: str = None):
    session, _, _ = session_mod.make_session(user_id, role_id, session)
    facts, eps, rel, mt, lp = await asyncio.gather(
        stores.all_facts(session),
        stores.all_episodes(session),
        stores.get_relationship(session),
        stores.max_turn(session),
        stores.get_last_processed(session),
    )
    for e in eps:
        e.pop("vec", None)
    eps.sort(key=lambda x: x["turn"], reverse=True)
    return {
        "facts": facts,
        "episodes": eps,
        "relationship": rel,
        "max_turn": mt,
        "last_processed": lp,
    }


@app.post("/api/reprocess")
async def reprocess(req: ResetRequest):
    """重置加工进度到 0，让 pipeline 重新从头抽取所有事实/情节。
    用于 prompt 改动后补救已有 session，或手动触发补全漏掉的记忆。"""
    session, user_id, role_id = session_mod.make_session(req.user_id, req.role_id, req.session)
    char_name = (req.char_name or "").strip() or personas.get_char_name(req.role_id)
    user_name = (req.user_name or "").strip() \
        or assembler.memory_user_name(await stores.all_facts(session))
    await stores.set_last_processed(session, 0)
    await pipeline.maybe_process(session, user_id, role_id, char_name, user_name)
    return {"ok": True, "max_turn": await stores.max_turn(session),
            "last_processed": await stores.get_last_processed(session)}


@app.get("/api/history")
async def history(session: str = None, user_id: str = None, role_id: str = None, n: int = 40):
    """返回最近 n 轮对话，供前端刷新后恢复聊天记录。"""
    session, _, _ = session_mod.make_session(user_id, role_id, session)
    turns = await stores.recent_turns(session, n)
    return {"turns": turns}


@app.post("/api/reset")
async def reset(req: ResetRequest):
    session, _, _ = session_mod.make_session(req.user_id, req.role_id, req.session)
    await stores.reset_session(session)
    return {"ok": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
