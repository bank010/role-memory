"""SQLite 后端：零外部依赖，开箱即跑（demo / 开发默认）。

向量以 float32 blob 落盘；无原生 ANN，candidate_episodes / candidate_chunks 直接返回
全部（单会话量级小，上层在内存里重排即可）。

接口为 async 以对齐 BaseStore；SQLite 操作本身微秒级，直接在协程内同步执行。
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
    stage TEXT DEFAULT 'new', mood TEXT DEFAULT 'calm', summary TEXT DEFAULT '',
    updated_turn INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    session TEXT PRIMARY KEY, user_id TEXT, role_id TEXT, last_processed_turn INTEGER DEFAULT 0
);
"""
# 注：user_id/role_id 索引在 init() 的迁移阶段创建（老库需先 ALTER 加列，才能建索引）
# 注：relationship 默认 stage/mood 为英文（记忆 canonical 语言统一为英文，展示层再本地化）


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
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row

    async def init(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
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
            # 迁移：老库 relationship 默认值是中文，统一为英文 canonical（展示层再本地化）
            try:
                self._conn.execute("UPDATE relationship SET stage='new' WHERE stage='初识'")
                self._conn.execute("UPDATE relationship SET mood='calm' WHERE mood='平静'")
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

    async def close(self) -> None:
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
    async def append_turn(self, session, user_msg, ai_reply, user_id="", role_id=""):
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

    async def recent_turns(self, session, n):
        rows = self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=? ORDER BY turn DESC LIMIT ?",
            (session, n),
        )
        return list(reversed(rows))

    async def turns_after(self, session, after_turn):
        return self._q(
            "SELECT turn, user_msg, ai_reply FROM turns WHERE session=? AND turn>? ORDER BY turn",
            (session, after_turn),
        )

    async def max_turn(self, session):
        row = self._one("SELECT COALESCE(MAX(turn),0) AS m FROM turns WHERE session=?", (session,))
        return row["m"] or 0

    async def first_turn_ts(self, session):
        row = self._one("SELECT MIN(ts) AS ts FROM turns WHERE session=?", (session,))
        return row["ts"] if row else None

    # ---- facts ----
    async def upsert_fact(self, session, key, value, confidence, turn, vec=None,
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

    async def all_facts(self, session):
        return self._q(
            "SELECT key, value, confidence, updated_turn FROM facts WHERE session=? "
            "ORDER BY updated_turn DESC",
            (session,),
        )

    async def facts_with_vec(self, session):
        rows = self._q(
            "SELECT key, value, confidence, updated_turn, embedding FROM facts WHERE session=?",
            (session,),
        )
        for r in rows:
            r["vec"] = _from_blob(r.pop("embedding"))
        return rows

    async def count_facts(self, session):
        row = self._one("SELECT COUNT(*) AS n FROM facts WHERE session=?", (session,))
        return row["n"] if row else 0

    async def delete_facts(self, session, keys):
        if not keys:
            return
        ph = ",".join("?" * len(keys))
        self._exec(f"DELETE FROM facts WHERE session=? AND key IN ({ph})",
                   (session, *keys))

    # ---- episodes ----
    async def insert_episode(self, session, event, emotion, importance, ts, turn, vec,
                             sensitive=False, user_id="", role_id=""):
        self._exec(
            """INSERT INTO episodes(session, user_id, role_id, event, emotion, importance, ts, turn,
                                    embedding, last_recalled_turn, sensitive)
               VALUES(?,?,?,?,?,?,?,?,?,0,?)""",
            (session, user_id, role_id, event, emotion, int(importance), ts, turn,
             _to_blob(vec), 1 if sensitive else 0),
        )

    async def update_episode(self, episode_id, importance, turn, ts, event=None, vec=None):
        if event is not None:
            self._exec(
                "UPDATE episodes SET importance=?, turn=?, ts=?, event=?, embedding=? WHERE id=?",
                (int(importance), turn, ts, event,
                 _to_blob(vec) if vec is not None else None, episode_id),
            )
        else:
            self._exec(
                "UPDATE episodes SET importance=?, turn=?, ts=? WHERE id=?",
                (int(importance), turn, ts, episode_id),
            )

    async def all_episodes(self, session):
        rows = self._q(
            "SELECT id, event, emotion, importance, turn, ts, embedding, last_recalled_turn, recall_count, sensitive "
            "FROM episodes WHERE session=?",
            (session,),
        )
        for r in rows:
            r["vec"] = _from_blob(r.pop("embedding"))
            r["sensitive"] = bool(r.get("sensitive"))
        return rows

    async def candidate_episodes(self, session, qvec, limit):
        # SQLite 无原生向量检索：单会话情节量级小，返回全部交上层重排
        return await self.all_episodes(session)

    async def mark_recalled(self, ids, turn):
        for eid in ids:
            self._exec(
                "UPDATE episodes SET last_recalled_turn=?, recall_count=recall_count+1 WHERE id=?",
                (turn, eid),
            )

    async def count_episodes(self, session):
        row = self._one("SELECT COUNT(*) AS n FROM episodes WHERE session=?", (session,))
        return row["n"] if row else 0

    async def delete_episodes(self, ids):
        if not ids:
            return
        ph = ",".join("?" * len(ids))
        self._exec(f"DELETE FROM episodes WHERE id IN ({ph})", tuple(ids))

    # ---- chunks ----
    async def insert_chunk(self, session, turn, role, text, norm, ts, vec, user_id="", role_id=""):
        self._exec(
            "INSERT INTO chunks(session, user_id, role_id, turn, role, text, norm, ts, embedding) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (session, user_id, role_id, turn, role, text, norm, ts, _to_blob(vec)),
        )

    async def all_chunks(self, session):
        rows = self._q(
            "SELECT id, turn, role, text, norm, ts, embedding FROM chunks WHERE session=?",
            (session,),
        )
        for r in rows:
            r["vec"] = _from_blob(r.pop("embedding"))
        return rows

    async def candidate_chunks(self, session, qvec, limit):
        # SQLite 无原生向量检索：返回全部交上层混合重排（单会话 chunk 上限有限）
        return await self.all_chunks(session)

    async def count_chunks(self, session):
        row = self._one("SELECT COUNT(*) AS n FROM chunks WHERE session=?", (session,))
        return row["n"] if row else 0

    async def evict_oldest_chunks(self, session, keep_n):
        """保留最新 keep_n 条（按 turn 降序），删除其余。"""
        self._exec(
            """DELETE FROM chunks WHERE session=? AND id NOT IN (
                 SELECT id FROM chunks WHERE session=? ORDER BY turn DESC, id DESC LIMIT ?
               )""",
            (session, session, keep_n),
        )

    # ---- relationship ----
    async def get_relationship(self, session):
        row = self._one("SELECT * FROM relationship WHERE session=?", (session,))
        if row:
            return row
        u, r = _split(session)
        # 默认值显式写入（不依赖 DDL DEFAULT：老库的 DEFAULT 可能还是中文旧值）
        self._exec(
            "INSERT OR IGNORE INTO relationship(session, user_id, role_id, stage, mood) "
            "VALUES(?,?,?,'new','calm')",
            (session, u, r))
        return self._one("SELECT * FROM relationship WHERE session=?", (session,))

    async def save_relationship(self, session, intimacy, trust, stage, mood, summary, turn,
                                user_id="", role_id=""):
        self._exec(
            """UPDATE relationship SET intimacy=?, trust=?, stage=?, mood=?, summary=?,
                 updated_turn=?, user_id=COALESCE(NULLIF(?,''),user_id),
                 role_id=COALESCE(NULLIF(?,''),role_id)
               WHERE session=?""",
            (intimacy, trust, stage, mood, summary, turn, user_id, role_id, session),
        )

    # ---- meta ----
    async def get_last_processed(self, session):
        row = self._one("SELECT last_processed_turn FROM meta WHERE session=?", (session,))
        return row["last_processed_turn"] if row else 0

    async def set_last_processed(self, session, turn, user_id="", role_id=""):
        self._exec(
            """INSERT INTO meta(session, user_id, role_id, last_processed_turn) VALUES(?,?,?,?)
               ON CONFLICT(session) DO UPDATE SET
                 last_processed_turn=excluded.last_processed_turn""",
            (session, user_id, role_id, turn),
        )

    async def reset_session(self, session):
        for t in ("turns", "facts", "episodes", "relationship", "meta", "chunks"):
            self._exec(f"DELETE FROM {t} WHERE session=?", (session,))
