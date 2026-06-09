"""SQLite 后端：零外部依赖，开箱即跑（demo 默认）。

向量以 float32 blob 落盘；情节召回不做原生 ANN，candidate_episodes 直接返回全部
（单会话情节量级很小，上层在内存里做三维重排即可）。
"""

import sqlite3
import threading
import time
from typing import Dict, List

import numpy as np

from app.store.base import BaseStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    session TEXT NOT NULL, user_id TEXT, role_id TEXT, turn INTEGER NOT NULL, ts REAL NOT NULL,
    user_msg TEXT, ai_reply TEXT, PRIMARY KEY (session, turn)
);
CREATE TABLE IF NOT EXISTS facts (
    session TEXT NOT NULL, user_id TEXT, role_id TEXT, key TEXT NOT NULL, value TEXT,
    confidence REAL DEFAULT 0.5, updated_turn INTEGER, embedding BLOB,
    PRIMARY KEY (session, key)
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, user_id TEXT, role_id TEXT,
    event TEXT, emotion TEXT, importance INTEGER DEFAULT 3, ts REAL, turn INTEGER,
    embedding BLOB, last_recalled_turn INTEGER DEFAULT 0, recall_count INTEGER DEFAULT 0,
    sensitive INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, user_id TEXT, role_id TEXT,
    turn INTEGER, role TEXT, text TEXT, norm TEXT, ts REAL, embedding BLOB
);
CREATE TABLE IF NOT EXISTS relationship (
    session TEXT PRIMARY KEY, user_id TEXT, role_id TEXT,
    intimacy REAL DEFAULT 0.1, trust REAL DEFAULT 0.1,
    stage TEXT DEFAULT '初识', mood TEXT DEFAULT '平静', summary TEXT DEFAULT '',
    updated_turn INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    session TEXT PRIMARY KEY, user_id TEXT, role_id TEXT, last_processed_turn INTEGER DEFAULT 0
);
"""
# 注：user_id/role_id 索引在 init() 的迁移阶段创建（老库需先 ALTER 加列，才能建索引）


def _to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _from_blob(blob) -> np.ndarray:
    if not blob:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(blob, dtype=np.float32)


def _split(session: str):
    """从 session(user_id\\x1frole_id) 还原 (user_id, role_id)，兼容旧单串。"""
    if "\x1f" in session:
        u, r = session.split("\x1f", 1)
        return u, r
    return session, ""


class SqliteStore(BaseStore):
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def init(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            cols = [r[1] for r in self._conn.execute("PRAGMA table_info(chunks)").fetchall()]
            if "norm" not in cols:
                try:
                    self._conn.execute("ALTER TABLE chunks ADD COLUMN norm TEXT")
                except Exception:
                    pass
            fcols = [r[1] for r in self._conn.execute("PRAGMA table_info(facts)").fetchall()]
            if "embedding" not in fcols:
                try:
                    self._conn.execute("ALTER TABLE facts ADD COLUMN embedding BLOB")
                except Exception:
                    pass
            ecols = [r[1] for r in self._conn.execute("PRAGMA table_info(episodes)").fetchall()]
            if "sensitive" not in ecols:
                try:
                    self._conn.execute("ALTER TABLE episodes ADD COLUMN sensitive INTEGER DEFAULT 0")
                except Exception:
                    pass
            # 迁移：给所有表补 user_id/role_id 独立列（隔离 + 运营查询 + 加速过滤）
            for tbl in ("turns", "facts", "episodes", "chunks", "relationship", "meta"):
                tcols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({tbl})").fetchall()]
                for col in ("user_id", "role_id"):
                    if col not in tcols:
                        try:
                            self._conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} TEXT")
                        except Exception:
                            pass
            # 索引（建表语句里已含，迁移老库时补建）
            for idx_sql in (
                "CREATE INDEX IF NOT EXISTS idx_turns_ur ON turns(user_id, role_id)",
                "CREATE INDEX IF NOT EXISTS idx_facts_ur ON facts(user_id, role_id)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_ur ON episodes(user_id, role_id)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_ur ON chunks(user_id, role_id)",
                "CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)",
                "CREATE INDEX IF NOT EXISTS idx_episodes_user ON episodes(user_id)",
            ):
                try:
                    self._conn.execute(idx_sql)
                except Exception:
                    pass
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _q(self, sql, params=()):
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql, params=()):
        rows = self._q(sql, params)
        return rows[0] if rows else None

    def _exec(self, sql, params=()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur.lastrowid

    # ---- turns ----
    def append_turn(self, session, user_msg, ai_reply, user_id="", role_id=""):
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(turn),0) AS m FROM turns WHERE session=?", (session,)
            ).fetchone()
            turn = (row["m"] or 0) + 1
            self._conn.execute(
                "INSERT INTO turns(session, user_id, role_id, turn, ts, user_msg, ai_reply) "
                "VALUES(?,?,?,?,?,?,?)",
                (session, user_id, role_id, turn, time.time(), user_msg, ai_reply),
            )
            self._conn.commit()
            return turn

    def recent_turns(self, session, n):
        rows = self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=? ORDER BY turn DESC LIMIT ?",
            (session, n),
        )
        return list(reversed(rows))

    def turns_after(self, session, after_turn):
        return self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=? AND turn>? ORDER BY turn",
            (session, after_turn),
        )

    def max_turn(self, session):
        row = self._one("SELECT COALESCE(MAX(turn),0) AS m FROM turns WHERE session=?", (session,))
        return row["m"] or 0

    # ---- facts ----
    def upsert_fact(self, session, key, value, confidence, turn, vec=None,
                    user_id="", role_id=""):
        blob = _to_blob(vec) if vec is not None and getattr(vec, "size", 0) else None
        self._exec(
            """INSERT INTO facts(session, user_id, role_id, key, value, confidence,
                                 updated_turn, embedding)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(session, key) DO UPDATE SET
                 value=excluded.value,
                 confidence=MAX(facts.confidence, excluded.confidence),
                 updated_turn=excluded.updated_turn,
                 embedding=COALESCE(excluded.embedding, facts.embedding)""",
            (session, user_id, role_id, key, value, confidence, turn, blob),
        )

    def all_facts(self, session):
        return self._q(
            "SELECT key, value, confidence, updated_turn FROM facts WHERE session=? "
            "ORDER BY updated_turn DESC",
            (session,),
        )

    def facts_with_vec(self, session):
        rows = self._q(
            "SELECT key, value, confidence, updated_turn, embedding FROM facts WHERE session=?",
            (session,),
        )
        for r in rows:
            r["vec"] = _from_blob(r.pop("embedding"))
        return rows

    # ---- episodes ----
    def insert_episode(self, session, event, emotion, importance, ts, turn, vec, sensitive=False,
                       user_id="", role_id=""):
        self._exec(
            """INSERT INTO episodes(session, user_id, role_id, event, emotion, importance, ts, turn,
                                    embedding, last_recalled_turn, sensitive)
               VALUES(?,?,?,?,?,?,?,?,?,0,?)""",
            (session, user_id, role_id, event, emotion, int(importance), ts, turn,
             _to_blob(vec), 1 if sensitive else 0),
        )

    def update_episode(self, episode_id, importance, turn, ts):
        self._exec(
            "UPDATE episodes SET importance=?, turn=?, ts=? WHERE id=?",
            (int(importance), turn, ts, episode_id),
        )

    def all_episodes(self, session):
        rows = self._q(
            "SELECT id, event, emotion, importance, turn, embedding, last_recalled_turn, recall_count, sensitive "
            "FROM episodes WHERE session=?",
            (session,),
        )
        for r in rows:
            r["vec"] = _from_blob(r.pop("embedding"))
            r["sensitive"] = bool(r.get("sensitive"))
        return rows

    def candidate_episodes(self, session, qvec, limit):
        # SQLite 无原生向量检索：单会话情节量级小，返回全部交上层重排
        return self.all_episodes(session)

    def mark_recalled(self, ids, turn):
        for eid in ids:
            self._exec(
                "UPDATE episodes SET last_recalled_turn=?, recall_count=recall_count+1 WHERE id=?",
                (turn, eid),
            )

    def count_episodes(self, session):
        row = self._one("SELECT COUNT(*) AS n FROM episodes WHERE session=?", (session,))
        return row["n"] if row else 0

    def delete_episodes(self, ids):
        if not ids:
            return
        ph = ",".join("?" * len(ids))
        self._exec(f"DELETE FROM episodes WHERE id IN ({ph})", tuple(ids))

    # ---- chunks ----
    def insert_chunk(self, session, turn, role, text, norm, ts, vec, user_id="", role_id=""):
        self._exec(
            "INSERT INTO chunks(session, user_id, role_id, turn, role, text, norm, ts, embedding) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (session, user_id, role_id, turn, role, text, norm, ts, _to_blob(vec)),
        )

    def all_chunks(self, session):
        rows = self._q(
            "SELECT id, turn, role, text, norm, embedding FROM chunks WHERE session=?",
            (session,),
        )
        for r in rows:
            r["vec"] = _from_blob(r.pop("embedding"))
        return rows

    def count_chunks(self, session):
        row = self._one("SELECT COUNT(*) AS n FROM chunks WHERE session=?", (session,))
        return row["n"] if row else 0

    def evict_oldest_chunks(self, session, keep_n):
        """保留最新 keep_n 条（按 turn 降序），删除其余。"""
        self._exec(
            """DELETE FROM chunks WHERE session=? AND id NOT IN (
                 SELECT id FROM chunks WHERE session=? ORDER BY turn DESC, id DESC LIMIT ?
               )""",
            (session, session, keep_n),
        )

    # ---- relationship ----
    def get_relationship(self, session):
        row = self._one("SELECT * FROM relationship WHERE session=?", (session,))
        if row:
            return row
        u, r = _split(session)
        self._exec("INSERT INTO relationship(session, user_id, role_id) VALUES(?,?,?)",
                   (session, u, r))
        return self._one("SELECT * FROM relationship WHERE session=?", (session,))

    def save_relationship(self, session, intimacy, trust, stage, mood, summary, turn,
                          user_id="", role_id=""):
        self._exec(
            """UPDATE relationship SET intimacy=?, trust=?, stage=?, mood=?, summary=?,
                 updated_turn=?, user_id=COALESCE(NULLIF(?,''),user_id),
                 role_id=COALESCE(NULLIF(?,''),role_id)
               WHERE session=?""",
            (intimacy, trust, stage, mood, summary, turn, user_id, role_id, session),
        )

    # ---- meta ----
    def get_last_processed(self, session):
        row = self._one("SELECT last_processed_turn FROM meta WHERE session=?", (session,))
        return row["last_processed_turn"] if row else 0

    def set_last_processed(self, session, turn, user_id="", role_id=""):
        self._exec(
            """INSERT INTO meta(session, user_id, role_id, last_processed_turn) VALUES(?,?,?,?)
               ON CONFLICT(session) DO UPDATE SET
                 last_processed_turn=excluded.last_processed_turn""",
            (session, user_id, role_id, turn),
        )

    def reset_session(self, session):
        for t in ("turns", "facts", "episodes", "relationship", "meta", "chunks"):
            self._exec(f"DELETE FROM {t} WHERE session=?", (session,))
