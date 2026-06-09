"""存储后端工厂。

按 config.STORE_BACKEND 选择具体实现：
- sqlite  : 零依赖，开箱即跑（demo 默认）
- postgres: pgvector，生产级

上层 app.memory.stores 只依赖 BaseStore 接口，不关心底层是谁。
切换后端只改一个环境变量，业务代码零改动。
"""

from app import config
from app.store.base import BaseStore

_backend: BaseStore = None


def get_store() -> BaseStore:
    global _backend
    if _backend is not None:
        return _backend

    if config.STORE_BACKEND == "postgres":
        from app.store.postgres_store import PostgresStore
        _backend = PostgresStore(config.PG_DSN, config.EMBED_DIM)
    else:
        from app.store.sqlite_store import SqliteStore
        _backend = SqliteStore(str(config.DB_PATH))
    return _backend
