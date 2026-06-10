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

from core import MemoryBox, config, llm
from routes import api

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("main")

STATIC_DIR = Path(__file__).resolve().parent / "static"

box = MemoryBox()


@asynccontextmanager
async def lifespan(application: FastAPI):
    await box.init()
    api.set_box(box)
    log.info("启动完成 | mock=%s | chat=%s @ %s",
             config.MOCK_MODE, config.CHAT_MODEL, config.CHAT_BASE_URL)
    yield
    await box.close()
    await llm.aclose()


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
