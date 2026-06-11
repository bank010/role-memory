"""FastAPI 入口。

职责：创建 app、挂载路由、管理生命周期。
业务逻辑在 serve/，记忆引擎在 core/，路由在 routes/。
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import MemoryBox, config
from core.archive import mongo
from core.client import llm
from routes import api
from serve import chat as chat_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("main")

STATIC_DIR = Path(__file__).resolve().parent / "static"

box = MemoryBox()


@asynccontextmanager
async def lifespan(application: FastAPI):
    await box.init()
    api.set_box(box)
    await mongo._get_collection()  # 提前建连+建索引，启动日志即可见归档状态（连不上自动降级）
    log.info("启动完成 | mock=%s | chat=%s @ %s | archive=%s",
             config.MOCK_MODE, config.CHAT_MODEL, config.CHAT_BASE_URL, mongo.enabled())
    yield
    # 排空归档后台任务，再关闭连接，避免丢数据
    if chat_service._archive_tasks:
        import asyncio
        await asyncio.gather(*list(chat_service._archive_tasks), return_exceptions=True)
    await box.close()
    await llm.aclose()
    await mongo.aclose()


app = FastAPI(title="角色扮演记忆系统", lifespan=lifespan)

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


app.include_router(api.router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
