"""PostgreSQL + pgvector 后端：生产级存储。

相比 SQLite 的关键升级：
- 向量进 pgvector 的 vector 列，建 HNSW 索引，支持原生 KNN（candidate_episodes 做 ANN 预筛）；
- 真正的并发写（连接池可替换这里的单连接 + 锁）；
- 标准 SQL，便于水平扩展 / 读写分离 / 备份。

依赖：psycopg[binary]、pgvector。本机没有 PG 时，保持 STORE_BACKEND=sqlite 即可，
本文件不会被 import（工厂按需加载）。
"""

import threading
import time
from typing import Dict, List

import numpy as np

from app.store.base import BaseStore


def _split(session: str):
    if "\x1f" in session:
        u, r = session.split("\x1f", 1)
        return u, r
    return session, ""


class PostgresStore(BaseStore):
    def __init__(self, dsn: str, dim: int):
        import psycopg
        from pgvector.psycopg import register_vector

        self._dim = dim
        self._lock = threading.Lock()
        self._conn = psycopg.connect(dsn, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._conn)

    def init(self) -> None:
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
            stage TEXT DEFAULT '初识', mood TEXT DEFAULT '平静', summary TEXT DEFAULT '',
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
        ALTER TABLE facts ADD COLUMN IF NOT EXISTS embedding vector({self._dim});
        ALTER TABLE episodes ADD COLUMN IF NOT EXISTS sensitive BOOLEAN DEFAULT FALSE;
        """
        with self._lock, self._conn.cursor() as cur:
            cur.execute(ddl)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _q(self, sql, params=()):
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _one(self, sql, params=()):
        rows = self._q(sql, params)
        return rows[0] if rows else None

    def _exec(self, sql, params=()):
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)

    # ---- turns ----
    def append_turn(self, session, user_msg, ai_reply, user_id="", role_id=""):
        # 单连接 + 行级原子；高并发可改用序列或 advisory lock
        with self._lock, self._conn.cursor() as cur:
            cur.execute("SELECT COALESCE(MAX(turn),0)+1 FROM turns WHERE session=%s", (session,))
            turn = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO turns(session, user_id, role_id, turn, ts, user_msg, ai_reply) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s)",
                (session, user_id, role_id, turn, time.time(), user_msg, ai_reply),
            )
            return turn

    def recent_turns(self, session, n):
        rows = self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=%s ORDER BY turn DESC LIMIT %s",
            (session, n),
        )
        return list(reversed(rows))

    def turns_after(self, session, after_turn):
        return self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=%s AND turn>%s ORDER BY turn",
            (session, after_turn),
        )

    def max_turn(self, session):
        row = self._one("SELECT COALESCE(MAX(turn),0) AS m FROM turns WHERE session=%s", (session,))
        return row["m"] or 0

    # ---- facts ----
    def upsert_fact(self, session, key, value, confidence, turn, vec=None,
                    user_id="", role_id=""):
        arr = (np.asarray(vec, dtype=np.float32)
               if vec is not None and getattr(vec, "size", 0) else None)
        self._exec(
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

    def all_facts(self, session):
        return self._q(
            "SELECT key, value, confidence, updated_turn FROM facts WHERE session=%s "
            "ORDER BY updated_turn DESC",
            (session,),
        )

    def facts_with_vec(self, session):
        rows = self._q(
            "SELECT key, value, confidence, updated_turn, embedding FROM facts WHERE session=%s",
            (session,),
        )
        for r in rows:
            emb = r.pop("embedding", None)
            r["vec"] = (np.asarray(emb, dtype=np.float32)
                        if emb is not None else np.zeros(0, dtype=np.float32))
        return rows

    # ---- episodes ----
    def insert_episode(self, session, event, emotion, importance, ts, turn, vec, sensitive=False,
                       user_id="", role_id=""):
        self._exec(
            """INSERT INTO episodes(session, user_id, role_id, event, emotion, importance, ts, turn,
                                    embedding, last_recalled_turn, sensitive)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,0,%s)""",
            (session, user_id, role_id, event, emotion, int(importance), ts, turn,
             np.asarray(vec, dtype=np.float32), bool(sensitive)),
        )

    def update_episode(self, episode_id, importance, turn, ts):
        self._exec(
            "UPDATE episodes SET importance=%s, turn=%s, ts=%s WHERE id=%s",
            (int(importance), turn, ts, episode_id),
        )

    def _rows_to_episodes(self, rows):
        for r in rows:
            emb = r.pop("embedding", None)
            r["vec"] = (np.asarray(emb, dtype=np.float32)
                        if emb is not None else np.zeros(0, dtype=np.float32))
            r["sensitive"] = bool(r.get("sensitive"))
        return rows

    def all_episodes(self, session):
        rows = self._q(
            "SELECT id, event, emotion, importance, turn, embedding, last_recalled_turn, recall_count, sensitive "
            "FROM episodes WHERE session=%s",
            (session,),
        )
        return self._rows_to_episodes(rows)

    def candidate_episodes(self, session, qvec, limit):
        # pgvector 原生 KNN 预筛 top-N（生产级核心收益），上层再做三维重排
        rows = self._q(
            "SELECT id, event, emotion, importance, turn, embedding, last_recalled_turn, recall_count, sensitive "
            "FROM episodes WHERE session=%s ORDER BY embedding <=> %s LIMIT %s",
            (session, np.asarray(qvec, dtype=np.float32), max(limit, 1)),
        )
        return self._rows_to_episodes(rows)

    def mark_recalled(self, ids, turn):
        if not ids:
            return
        self._exec(
            "UPDATE episodes SET last_recalled_turn=%s, recall_count=recall_count+1 "
            "WHERE id = ANY(%s)",
            (turn, list(ids)),
        )

    def count_episodes(self, session):
        row = self._one("SELECT COUNT(*) AS n FROM episodes WHERE session=%s", (session,))
        return row["n"] if row else 0

    def delete_episodes(self, ids):
        if not ids:
            return
        self._exec("DELETE FROM episodes WHERE id = ANY(%s)", (list(ids),))

    # ---- chunks ----
    def insert_chunk(self, session, turn, role, text, norm, ts, vec, user_id="", role_id=""):
        self._exec(
            "INSERT INTO chunks(session, user_id, role_id, turn, role, text, norm, ts, embedding) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (session, user_id, role_id, turn, role, text, norm, ts,
             np.asarray(vec, dtype=np.float32)),
        )

    def all_chunks(self, session):
        rows = self._q(
            "SELECT id, turn, role, text, norm, embedding FROM chunks WHERE session=%s",
            (session,),
        )
        for r in rows:
            emb = r.pop("embedding", None)
            r["vec"] = (np.asarray(emb, dtype=np.float32)
                        if emb is not None else np.zeros(0, dtype=np.float32))
        return rows

    def count_chunks(self, session):
        row = self._one("SELECT COUNT(*) AS n FROM chunks WHERE session=%s", (session,))
        return row["n"] if row else 0

    def evict_oldest_chunks(self, session, keep_n):
        """保留最新 keep_n 条（按 turn 降序），删除其余。"""
        self._exec(
            """DELETE FROM chunks WHERE session=%s AND id NOT IN (
                 SELECT id FROM chunks WHERE session=%s ORDER BY turn DESC, id DESC LIMIT %s
               )""",
            (session, session, keep_n),
        )

    # ---- relationship ----
    def get_relationship(self, session):
        row = self._one("SELECT * FROM relationship WHERE session=%s", (session,))
        if row:
            return row
        u, r = _split(session)
        self._exec("INSERT INTO relationship(session, user_id, role_id) VALUES(%s,%s,%s) "
                   "ON CONFLICT DO NOTHING", (session, u, r))
        return self._one("SELECT * FROM relationship WHERE session=%s", (session,))

    def save_relationship(self, session, intimacy, trust, stage, mood, summary, turn,
                          user_id="", role_id=""):
        self._exec(
            """UPDATE relationship SET intimacy=%s, trust=%s, stage=%s, mood=%s, summary=%s,
                 updated_turn=%s, user_id=COALESCE(NULLIF(%s,''),user_id),
                 role_id=COALESCE(NULLIF(%s,''),role_id)
               WHERE session=%s""",
            (intimacy, trust, stage, mood, summary, turn, user_id, role_id, session),
        )

    # ---- meta ----
    def get_last_processed(self, session):
        row = self._one("SELECT last_processed_turn FROM meta WHERE session=%s", (session,))
        return row["last_processed_turn"] if row else 0

    def set_last_processed(self, session, turn, user_id="", role_id=""):
        self._exec(
            """INSERT INTO meta(session, user_id, role_id, last_processed_turn) VALUES(%s,%s,%s,%s)
               ON CONFLICT(session) DO UPDATE SET
                 last_processed_turn=excluded.last_processed_turn""",
            (session, user_id, role_id, turn),
        )

    def reset_session(self, session):
        for t in ("turns", "facts", "episodes", "relationship", "meta", "chunks"):
            self._exec(f"DELETE FROM {t} WHERE session=%s", (session,))
