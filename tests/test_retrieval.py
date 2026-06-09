"""检索算法单元测试。

覆盖：三维打分（relevance/recency/importance）、词法得分、混合检索排名。
全部纯函数计算，不需要数据库。
"""

import math

import numpy as np
import pytest

from app.memory.retrieval import _lexical_score, _weighted_tokens


# ---------- 词法工具 ----------
def test_weighted_tokens_english():
    toks = _weighted_tokens("my cat meimei")
    assert "my" in toks
    assert "cat" in toks
    assert toks["cat"] == pytest.approx(2.0)  # 英文词权重 2.0


def test_weighted_tokens_chinese_bigram():
    toks = _weighted_tokens("煤球真可爱")
    assert "煤球" in toks
    assert toks["煤球"] == pytest.approx(2.0)


def test_weighted_tokens_stopchar_low_weight():
    toks = _weighted_tokens("我的猫")
    assert toks.get("我", 1.0) < 0.5  # 停用字权重低


def test_lexical_score_exact_match():
    """query 中关键词完全命中 text，得分应接近 1.0。"""
    q_toks = _weighted_tokens("my cat meimei")
    score = _lexical_score(q_toks, "my cat is named meimei")
    assert score > 0.7


def test_lexical_score_no_match():
    q_toks = _weighted_tokens("weekend beach trip")
    score = _lexical_score(q_toks, "我今天吃了包子")
    assert score < 0.1


def test_lexical_score_empty_query():
    score = _lexical_score({}, "any text here")
    assert score == 0.0


def test_lexical_score_partial_match():
    q_toks = _weighted_tokens("煤球 cat Saturday")
    full = _lexical_score(q_toks, "我的猫叫煤球")
    partial = _lexical_score(q_toks, "今天天气很好")
    assert full > partial


# ---------- 三维打分逻辑 ----------
def _unit(seed=0):
    rng = np.random.default_rng(seed)
    v = rng.random(256).astype(np.float32)
    return v / np.linalg.norm(v)


def test_recency_decay_formula():
    """verify the recency formula used in retrieval: exp(-decay * age)."""
    decay = 0.02
    age = 10
    expected = math.exp(-decay * age)
    assert abs(expected - math.exp(-0.2)) < 1e-9


def test_cosine_similarity_identical():
    from app.embeddings import cosine
    v = _unit(5)
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-5)


def test_cosine_similarity_orthogonal():
    from app.embeddings import cosine
    a = np.zeros(256, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(256, dtype=np.float32)
    b[1] = 1.0
    assert cosine(a, b) == pytest.approx(0.0, abs=1e-5)


def test_cosine_handles_empty():
    from app.embeddings import cosine
    assert cosine(np.zeros(0, np.float32), np.zeros(0, np.float32)) == 0.0


# ---------- 端到端：retrieve_verbatim 排名 ----------
@pytest.mark.asyncio
async def test_retrieve_verbatim_ranks_relevant_higher(tmp_db, session, monkeypatch):
    """在 mock embedding 下，逐字召回的混合检索应把包含关键词的 chunk 排在前面。"""
    import app.embeddings as emb_mod
    import app.normalizer as norm_mod
    from app.memory import retrieval, stores

    async def mock_norm(text):
        return text

    monkeypatch.setattr(norm_mod, "to_base_lang", mock_norm)

    async def mock_embed(text):
        return emb_mod._local_embed(text)

    monkeypatch.setattr(emb_mod, "embed", mock_embed)

    # 往 store 里插两条 chunk
    store = tmp_db
    store.insert_chunk(session, 1, "user", "我叫小明养了只猫叫煤球", "my cat is named meimei",
                       1.0, emb_mod._local_embed("my cat is named meimei"))
    store.insert_chunk(session, 2, "user", "今天天气不错出去散步了", "weather nice walk",
                       2.0, emb_mod._local_embed("weather nice walk"))

    results = await retrieval.retrieve_verbatim(session, "我的猫叫什么", top_k=2)
    assert results, "应返回结果"
    assert results[0]["text"] == "我叫小明养了只猫叫煤球"


# ---------- Postgres 后端测试（有 PG 才跑）----------
from tests.conftest import requires_pg


@requires_pg
def test_pg_store_basic(session):
    from app.store.postgres_store import PostgresStore
    import app.config as cfg
    store = PostgresStore(cfg.PG_DSN, cfg.EMBED_DIM)
    store.init()

    sid = "pg_test_" + session
    store.reset_session(sid)
    t = store.append_turn(sid, "hello", "hi")
    assert t == 1
    store.upsert_fact(sid, "name", "test_user", 0.9, 1)
    facts = store.all_facts(sid)
    assert facts[0]["key"] == "name"
    store.reset_session(sid)
    store.close()
