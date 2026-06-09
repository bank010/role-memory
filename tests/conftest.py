"""测试夹具：每个测试用独立的临时 SQLite store，互不污染。

测试全程离线运行（不调任何 LLM / embedding 网络）：
- embedding 走本地确定性哈希向量（EMBED_REAL=False）
- normalizer 在 NORMALIZE_ENABLED=0 时直通原文
所以无需 mock 网络即可测纯逻辑：key 归一、entity 派生、追加vs覆盖、三维打分、去重、隔离、并发锁。
"""

import os

import pytest

# 关掉归一化（直通原文），保证 embedding 输入确定
os.environ.setdefault("NORMALIZE_ENABLED", "0")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """强制所有测试离线：embedding 走本地确定性哈希、归一化直通原文。

    .env 里配了真实 embed key 时，EMBED_REAL=True 会让 embeddings.embed 走网络；
    测试必须拦掉，否则跨事件循环关闭 httpx 客户端会报 'Event loop is closed'，且慢、不稳定。
    """
    import app.embeddings as emb_mod
    import app.normalizer as norm_mod

    async def _local(text):
        return emb_mod._local_embed(text)

    async def _passthrough(text):
        return text

    monkeypatch.setattr(emb_mod, "embed", _local)
    monkeypatch.setattr(norm_mod, "to_base_lang", _passthrough)


# ---------------- 基础 store 夹具 ----------------
@pytest.fixture()
def store(tmp_path):
    """裸 SQLite store（不接编排层），用于直连存储层的测试。"""
    from app.store.sqlite_store import SqliteStore
    s = SqliteStore(str(tmp_path / "test.db"))
    s.init()
    yield s
    s.close()


@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """把编排层 stores 的后端替换成临时 store，并关闭缓存。返回该 store。

    （历史测试通过 stores.* 间接读写，依赖这个夹具把 _store 指到临时库。）
    """
    from app.store.sqlite_store import SqliteStore
    from app.memory import stores as mem_stores
    from app import cache
    s = SqliteStore(str(tmp_path / "wired.db"))
    s.init()
    monkeypatch.setattr(mem_stores, "_store", s)
    monkeypatch.setattr(cache, "enabled", lambda: False)
    yield s
    s.close()


@pytest.fixture()
def wired_store(tmp_db):
    """别名：返回已接好临时后端的编排层 stores 模块（新测试用）。"""
    from app.memory import stores as mem_stores
    return mem_stores


@pytest.fixture()
def session():
    """统一的测试会话标识（user\x1frole 组合键格式）。"""
    return "testuser\x1ftestrole"


# ---------------- 可选：Postgres 集成测试跳过标记 ----------------
def _pg_available() -> bool:
    if os.getenv("RUN_PG_TESTS") != "1":
        return False
    try:
        import psycopg  # noqa: F401
        from pgvector.psycopg import register_vector  # noqa: F401
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="需要 PostgreSQL+pgvector，且设置 RUN_PG_TESTS=1 才运行",
)
