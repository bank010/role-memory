"""SQLite 后端 CRUD 测试。

覆盖：turns / facts / episodes / chunks / relationship / meta / reset。
"""

import time

import numpy as np
import pytest

from app.store.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(str(tmp_path / "store.db"))
    s.init()
    return s


SID = "s1"


# ---------- turns ----------
def test_append_and_recent_turns(store):
    t1 = store.append_turn(SID, "hello", "hi")
    t2 = store.append_turn(SID, "how are you", "fine")
    assert t1 == 1
    assert t2 == 2
    rows = store.recent_turns(SID, 10)
    assert len(rows) == 2
    assert rows[0]["user_msg"] == "hello"
    assert rows[1]["user_msg"] == "how are you"


def test_recent_turns_limit(store):
    for i in range(10):
        store.append_turn(SID, f"u{i}", f"a{i}")
    rows = store.recent_turns(SID, 3)
    assert len(rows) == 3
    assert rows[-1]["user_msg"] == "u9"


def test_turns_after(store):
    for i in range(5):
        store.append_turn(SID, f"u{i}", f"a{i}")
    rows = store.turns_after(SID, 2)
    assert [r["turn"] for r in rows] == [3, 4, 5]


def test_max_turn_empty(store):
    assert store.max_turn("no_such") == 0


# ---------- facts ----------
def test_upsert_fact_insert_and_update(store):
    store.upsert_fact(SID, "name", "Alice", 0.9, 1)
    facts = store.all_facts(SID)
    assert facts[0]["key"] == "name"
    assert facts[0]["value"] == "Alice"

    store.upsert_fact(SID, "name", "Alice Smith", 0.95, 2)
    facts = store.all_facts(SID)
    assert len(facts) == 1
    assert facts[0]["value"] == "Alice Smith"
    assert facts[0]["confidence"] == pytest.approx(0.95)


def test_upsert_fact_keeps_higher_confidence(store):
    store.upsert_fact(SID, "job", "doctor", 0.8, 1)
    store.upsert_fact(SID, "job", "doctor confirmed", 0.6, 2)
    fact = store.all_facts(SID)[0]
    assert fact["confidence"] == pytest.approx(0.8)  # MAX 保留较高值


# ---------- episodes ----------
def make_vec(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(256).astype(np.float32)
    return v / np.linalg.norm(v)


def test_insert_and_all_episodes(store):
    v = make_vec()
    store.insert_episode(SID, "met Alice", "happy", 7, time.time(), 1, v)
    eps = store.all_episodes(SID)
    assert len(eps) == 1
    assert eps[0]["event"] == "met Alice"
    assert eps[0]["importance"] == 7
    assert eps[0]["vec"].shape == (256,)


def test_update_episode(store):
    v = make_vec()
    store.insert_episode(SID, "old event", "neutral", 3, time.time(), 1, v)
    eid = store.all_episodes(SID)[0]["id"]
    store.update_episode(eid, 9, 5, time.time())
    ep = store.all_episodes(SID)[0]
    assert ep["importance"] == 9
    assert ep["turn"] == 5


def test_count_and_delete_episodes(store):
    for i in range(5):
        store.insert_episode(SID, f"event {i}", "neutral", i + 1, time.time(), i, make_vec(i))
    assert store.count_episodes(SID) == 5
    ids = [ep["id"] for ep in store.all_episodes(SID)[:3]]
    store.delete_episodes(ids)
    assert store.count_episodes(SID) == 2


def test_delete_episodes_empty_list(store):
    store.delete_episodes([])  # 不应抛异常


def test_mark_recalled(store):
    v = make_vec()
    store.insert_episode(SID, "event", "ok", 5, time.time(), 1, v)
    eid = store.all_episodes(SID)[0]["id"]
    store.mark_recalled([eid], turn=3)
    ep = store.all_episodes(SID)[0]
    assert ep["last_recalled_turn"] == 3
    assert ep["recall_count"] == 1


# ---------- chunks ----------
def test_insert_and_all_chunks(store):
    v = make_vec()
    store.insert_chunk(SID, 1, "user", "猫叫煤球", "cat named meiiu", time.time(), v)
    chunks = store.all_chunks(SID)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "猫叫煤球"
    assert chunks[0]["vec"].shape == (256,)


def test_count_chunks(store):
    for i in range(4):
        store.insert_chunk(SID, i, "user", f"text{i}", f"norm{i}", time.time(), make_vec(i))
    assert store.count_chunks(SID) == 4


def test_evict_oldest_chunks(store):
    for i in range(10):
        store.insert_chunk(SID, i, "user", f"text{i}", f"norm{i}", time.time(), make_vec(i))
    store.evict_oldest_chunks(SID, keep_n=4)
    remaining = store.all_chunks(SID)
    assert len(remaining) == 4
    turns_kept = sorted(c["turn"] for c in remaining)
    assert turns_kept == [6, 7, 8, 9]  # 保留最新 4 条


# ---------- relationship ----------
def test_get_relationship_creates_default(store):
    rel = store.get_relationship(SID)
    assert rel["intimacy"] == pytest.approx(0.1)
    assert rel["stage"] == "初识"


def test_save_relationship(store):
    store.get_relationship(SID)  # 确保行存在
    store.save_relationship(SID, 0.5, 0.6, "熟悉", "excited", "summary text", 10)
    rel = store.get_relationship(SID)
    assert rel["intimacy"] == pytest.approx(0.5)
    assert rel["mood"] == "excited"
    assert rel["summary"] == "summary text"


# ---------- meta ----------
def test_last_processed_default_zero(store):
    assert store.get_last_processed(SID) == 0


def test_set_and_get_last_processed(store):
    store.set_last_processed(SID, 7)
    assert store.get_last_processed(SID) == 7
    store.set_last_processed(SID, 15)
    assert store.get_last_processed(SID) == 15


# ---------- reset ----------
def test_reset_session_clears_all(store):
    store.append_turn(SID, "hi", "hello")
    store.upsert_fact(SID, "name", "Bob", 0.8, 1)
    v = make_vec()
    store.insert_episode(SID, "event", "ok", 5, time.time(), 1, v)
    store.insert_chunk(SID, 1, "user", "text", "norm", time.time(), v)
    store.get_relationship(SID)
    store.set_last_processed(SID, 1)

    store.reset_session(SID)
    assert store.max_turn(SID) == 0
    assert store.all_facts(SID) == []
    assert store.all_episodes(SID) == []
    assert store.all_chunks(SID) == []
    assert store.get_last_processed(SID) == 0
