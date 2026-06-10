"""PostgreSQL + pgvector 后端：生产级存储（全异步）。

相比 SQLite 的关键升级：
- AsyncConnectionPool 连接池：真异步、真并发，不阻塞事件循环（毫秒级读路径的前提）；
- 向量进 pgvector 的 vector 列，episodes/chunks 都建 HNSW 索引，原生 KNN 预筛；
- append_turn 用单语句 INSERT..SELECT..RETURNING + 唯一键冲突重试，并发下 turn 序号原子；
- 标准 SQL，便于水平扩展 / 读写分离 / 备份。

依赖：psycopg[binary,pool]、pgvector。本机没有 PG 时，保持 STORE_BACKEND=sqlite 即可，
本文件不会被 import（工厂按需加载）。
"""

import logging
import time
from typing import Dict, List

import numpy as np

from app.store.base import BaseStore

log = logging.getLogger("store.postgres")

_APPEND_TURN_RETRIES = 5


def _split(session: str):
    if "\x1f" in session:
        u, r = session.split("\x1f", 1)
        return u, r
    return session, ""


class PostgresStore(BaseStore):
    def __init__(self, dsn: str, dim: int, pool_min: int = 2, pool_max: int = 20):
        self._dsn = dsn
        self._dim = dim
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool = None

    async def init(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector_async
        from psycopg_pool import AsyncConnectionPool

        # 扩展要先于 register_vector（vector 类型需存在），用独立短连接建
        async with await psycopg.AsyncConnection.connect(self._dsn, autocommit=True) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        async def _configure(conn):
            await conn.set_autocommit(True)
            await register_vector_async(conn)

        self._pool = AsyncConnectionPool(
            self._dsn, min_size=self._pool_min, max_size=self._pool_max,
            configure=_configure, open=False,
        )
        await self._pool.open(wait=True)

        ddl = f"""
        CREATE TABLE IF NOT EXISTS turns (
            session TEXT NOT NULL, user_id TEXT, role_id TEXT,
            turn INT NOT NULL, ts DOUBLE PRECISION NOT NULL,
            user_msg TEXT, ai_reply TEXT, PRIMARY KEY (session, turn)
        );
        CREATE TABLE IF NOT EXISTS facts (
            session TEXT NOT NULL, user_id TEXT, role_id TEXT, key TEXT NOT NULL, value TEXT,
            confidence REAL DEFAULT 0.5, updated_turn INT,
            embedding vector({self._dim}), PRIMARY KEY (session, key)
        );
        CREATE TABLE IF NOT EXISTS episodes (
            id BIGSERIAL PRIMARY KEY, session TEXT NOT NULL, user_id TEXT, role_id TEXT,
            event TEXT, emotion TEXT,
            importance INT DEFAULT 3, ts DOUBLE PRECISION, turn INT,
            embedding vector({self._dim}),
            last_recalled_turn INT DEFAULT 0, recall_count INT DEFAULT 0,
            sensitive BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id BIGSERIAL PRIMARY KEY, session TEXT NOT NULL, user_id TEXT, role_id TEXT,
            turn INT, role TEXT,
            text TEXT, norm TEXT, ts DOUBLE PRECISION, embedding vector({self._dim})
        );
        CREATE TABLE IF NOT EXISTS relationship (
            session TEXT PRIMARY KEY, user_id TEXT, role_id TEXT,
            intimacy REAL DEFAULT 0.1, trust REAL DEFAULT 0.1,
            stage TEXT DEFAULT 'new', mood TEXT DEFAULT 'calm', summary TEXT DEFAULT '',
            updated_turn INT DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS meta (
            session TEXT PRIMARY KEY, user_id TEXT, role_id TEXT, last_processed_turn INT DEFAULT 0
        );
        ALTER TABLE turns ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE turns ADD COLUMN IF NOT EXISTS role_id TEXT;
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS role_id TEXT;
        ALTER TABLE episodes ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE episodes ADD COLUMN IF NOT EXISTS role_id TEXT;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS role_id TEXT;
        ALTER TABLE relationship ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE relationship ADD COLUMN IF NOT EXISTS role_id TEXT;
        ALTER TABLE meta ADD COLUMN IF NOT EXISTS user_id TEXT;
        ALTER TABLE meta ADD COLUMN IF NOT EXISTS role_id TEXT;
        CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session);
        CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session);
        CREATE INDEX IF NOT EXISTS idx_turns_ur ON turns(user_id, role_id);
        CREATE INDEX IF NOT EXISTS idx_facts_ur ON facts(user_id, role_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_ur ON episodes(user_id, role_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_ur ON chunks(user_id, role_id);
        CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id);
        CREATE INDEX IF NOT EXISTS idx_episodes_hnsw ON episodes
            USING hnsw (embedding vector_cosine_ops);
        CREATE INDEX IF NOT EXISTS idx_chunks_hnsw ON chunks
            USING hnsw (embedding vector_cosine_ops);
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS embedding vector({self._dim});
        ALTER TABLE episodes ADD COLUMN IF NOT EXISTS sensitive BOOLEAN DEFAULT FALSE;
        ALTER TABLE relationship ALTER COLUMN stage SET DEFAULT 'new';
        ALTER TABLE relationship ALTER COLUMN mood SET DEFAULT 'calm';
        UPDATE relationship SET stage='new' WHERE stage='初识';
        UPDATE relationship SET mood='calm' WHERE mood='平静';
        """
        async with self._pool.connection() as conn:
            await conn.execute(ddl)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def _q(self, sql, params=()):
        async with self._pool.connection() as conn:
            cur = await conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            rows = await cur.fetchall()
            return [dict(zip(cols, row)) for row in rows]

    async def _one(self, sql, params=()):
        rows = await self._q(sql, params)
        return rows[0] if rows else None

    async def _exec(self, sql, params=()):
        async with self._pool.connection() as conn:
            await conn.execute(sql, params)

    # ---- turns ----
    async def append_turn(self, session, user_msg, ai_reply, user_id="", role_id=""):
        """单语句原子取号 + 插入；并发撞号时唯一键冲突，重试即可。"""
        import psycopg

        sql = """INSERT INTO turns(session, user_id, role_id, turn, ts, user_msg, ai_reply)
                 SELECT %s,%s,%s, COALESCE(MAX(turn),0)+1, %s,%s,%s FROM turns WHERE session=%s
                 RETURNING turn"""
        last_err = None
        for _ in range(_APPEND_TURN_RETRIES):
            try:
                async with self._pool.connection() as conn:
                    cur = await conn.execute(
                        sql, (session, user_id, role_id, time.time(), user_msg, ai_reply, session))
                    row = await cur.fetchone()
                    return row[0]
            except psycopg.errors.UniqueViolation as e:
                last_err = e  # 并发撞号，重试
        raise last_err

    async def recent_turns(self, session, n):
        rows = await self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=%s ORDER BY turn DESC LIMIT %s",
            (session, n),
        )
        return list(reversed(rows))

    async def turns_after(self, session, after_turn):
        return await self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=%s AND turn>%s ORDER BY turn",
            (session, after_turn),
        )

    async def max_turn(self, session):
        row = await self._one(
            "SELECT COALESCE(MAX(turn),0) AS m FROM turns WHERE session=%s", (session,))
        return row["m"] or 0

    async def first_turn_ts(self, session):
        row = await self._one(
            "SELECT MIN(ts) AS ts FROM turns WHERE session=%s", (session,))
        return row["ts"] if row else None

    # ---- facts ----
    async def upsert_fact(self, session, key, value, confidence, turn, vec=None,
                          user_id="", role_id=""):
        arr = (np.asarray(vec, dtype=np.float32)
               if vec is not None and getattr(vec, "size", 0) else None)
        await self._exec(
            """INSERT INTO facts(session, user_id, role_id, key, value, confidence,
                                 updated_turn, embedding)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(session, key) DO UPDATE SET
                 value=excluded.value,
                 confidence=GREATEST(facts.confidence, excluded.confidence),
                 updated_turn=excluded.updated_turn,
                 embedding=COALESCE(excluded.embedding, facts.embedding)""",
            (session, user_id, role_id, key, value, confidence, turn, arr),
        )

    async def all_facts(self, session):
        return await self._q(
            "SELECT key, value, confidence, updated_turn FROM facts WHERE session=%s "
            "ORDER BY updated_turn DESC",
            (session,),
        )

    async def facts_with_vec(self, session):
        rows = await self._q(
            "SELECT key, value, confidence, updated_turn, embedding FROM facts WHERE session=%s",
            (session,),
        )
        for r in rows:
            emb = r.pop("embedding", None)
            r["vec"] = (np.asarray(emb, dtype=np.float32)
                        if emb is not None else np.zeros(0, dtype=np.float32))
        return rows

    async def count_facts(self, session):
        row = await self._one("SELECT COUNT(*) AS n FROM facts WHERE session=%s", (session,))
        return row["n"] if row else 0

    async def delete_facts(self, session, keys):
        if not keys:
            return
        await self._exec("DELETE FROM facts WHERE session=%s AND key = ANY(%s)",
                         (session, list(keys)))

    # ---- episodes ----
    async def insert_episode(self, session, event, emotion, importance, ts, turn, vec,
                             sensitive=False, user_id="", role_id=""):
        await self._exec(
            """INSERT INTO episodes(session, user_id, role_id, event, emotion, importance, ts, turn,
                                    embedding, last_recalled_turn, sensitive)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s)""",
            (session, user_id, role_id, event, emotion, int(importance), ts, turn,
             np.asarray(vec, dtype=np.float32), bool(sensitive)),
        )

    async def update_episode(self, episode_id, importance, turn, ts, event=None, vec=None):
        if event is not None:
            await self._exec(
                "UPDATE episodes SET importance=%s, turn=%s, ts=%s, event=%s, embedding=%s "
                "WHERE id=%s",
                (int(importance), turn, ts, event,
                 np.asarray(vec, dtype=np.float32) if vec is not None else None, episode_id),
            )
        else:
            await self._exec(
                "UPDATE episodes SET importance=%s, turn=%s, ts=%s WHERE id=%s",
                (int(importance), turn, ts, episode_id),
            )

    @staticmethod
    def _rows_to_vec(rows):
        for r in rows:
            emb = r.pop("embedding", None)
            r["vec"] = (np.asarray(emb, dtype=np.float32)
                        if emb is not None else np.zeros(0, dtype=np.float32))
            if "sensitive" in r:
                r["sensitive"] = bool(r.get("sensitive"))
        return rows

    async def all_episodes(self, session):
        rows = await self._q(
            "SELECT id, event, emotion, importance, turn, ts, embedding, last_recalled_turn, recall_count, sensitive "
            "FROM episodes WHERE session=%s",
            (session,),
        )
        return self._rows_to_vec(rows)

    async def candidate_episodes(self, session, qvec, limit):
        # pgvector 原生 KNN 预筛 top-N，上层再做三维重排
        rows = await self._q(
            "SELECT id, event, emotion, importance, turn, ts, embedding, last_recalled_turn, recall_count, sensitive "
            "FROM episodes WHERE session=%s ORDER BY embedding <=> %s LIMIT %s",
            (session, np.asarray(qvec, dtype=np.float32), max(limit, 1)),
        )
        return self._rows_to_vec(rows)

    async def mark_recalled(self, ids, turn):
        if not ids:
            return
        await self._exec(
            "UPDATE episodes SET last_recalled_turn=%s, recall_count=recall_count+1 "
            "WHERE id = ANY(%s)",
            (turn, list(ids)),
        )

    async def count_episodes(self, session):
        row = await self._one("SELECT COUNT(*) AS n FROM episodes WHERE session=%s", (session,))
        return row["n"] if row else 0

    async def delete_episodes(self, ids):
        if not ids:
            return
        await self._exec("DELETE FROM episodes WHERE id = ANY(%s)", (list(ids),))

    # ---- chunks ----
    async def insert_chunk(self, session, turn, role, text, norm, ts, vec, user_id="", role_id=""):
        await self._exec(
            "INSERT INTO chunks(session, user_id, role_id, turn, role, text, norm, ts, embedding) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (session, user_id, role_id, turn, role, text, norm, ts,
             np.asarray(vec, dtype=np.float32)),
        )

    async def all_chunks(self, session):
        rows = await self._q(
            "SELECT id, turn, role, text, norm, ts, embedding FROM chunks WHERE session=%s",
            (session,),
        )
        return self._rows_to_vec(rows)

    async def candidate_chunks(self, session, qvec, limit):
        # pgvector KNN 预筛 top-N：避免把整个会话的 chunk 向量全量拉回内存
        rows = await self._q(
            "SELECT id, turn, role, text, norm, ts, embedding FROM chunks "
            "WHERE session=%s ORDER BY embedding <=> %s LIMIT %s",
            (session, np.asarray(qvec, dtype=np.float32), max(limit, 1)),
        )
        return self._rows_to_vec(rows)

    async def count_chunks(self, session):
        row = await self._one("SELECT COUNT(*) AS n FROM chunks WHERE session=%s", (session,))
        return row["n"] if row else 0

    async def evict_oldest_chunks(self, session, keep_n):
        """保留最新 keep_n 条（按 turn 降序），删除其余。"""
        await self._exec(
            """DELETE FROM chunks WHERE session=%s AND id NOT IN (
                 SELECT id FROM chunks WHERE session=%s ORDER BY turn DESC, id DESC LIMIT %s
               )""",
            (session, session, keep_n),
        )

    # ---- relationship ----
    async def get_relationship(self, session):
        row = await self._one("SELECT * FROM relationship WHERE session=%s", (session,))
        if row:
            return row
        u, r = _split(session)
        # 默认值显式写入（不依赖 DDL DEFAULT：老库的 DEFAULT 可能还是中文旧值）
        await self._exec(
            "INSERT INTO relationship(session, user_id, role_id, stage, mood) "
            "VALUES(%s,%s,%s,'new','calm') ON CONFLICT DO NOTHING", (session, u, r))
        return await self._one("SELECT * FROM relationship WHERE session=%s", (session,))

    async def save_relationship(self, session, intimacy, trust, stage, mood, summary, turn,
                                user_id="", role_id=""):
        await self._exec(
            """UPDATE relationship SET intimacy=%s, trust=%s, stage=%s, mood=%s, summary=%s,
                 updated_turn=%s, user_id=COALESCE(NULLIF(%s,''),user_id),
                 role_id=COALESCE(NULLIF(%s,''),role_id)
               WHERE session=%s""",
            (intimacy, trust, stage, mood, summary, turn, user_id, role_id, session),
        )

    # ---- meta ----
    async def get_last_processed(self, session):
        row = await self._one(
            "SELECT last_processed_turn FROM meta WHERE session=%s", (session,))
        return row["last_processed_turn"] if row else 0

    async def set_last_processed(self, session, turn, user_id="", role_id=""):
        await self._exec(
            """INSERT INTO meta(session, user_id, role_id, last_processed_turn) VALUES(%s,%s,%s,%s)
               ON CONFLICT(session) DO UPDATE SET
                 last_processed_turn=excluded.last_processed_turn""",
            (session, user_id, role_id, turn),
        )

    async def reset_session(self, session):
        for t in ("turns", "facts", "episodes", "relationship", "meta", "chunks"):
            await self._exec(f"DELETE FROM {t} WHERE session=%s", (session,))
