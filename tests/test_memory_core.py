"""核心回归测试：保证记忆系统的关键逻辑不被后续改动碰坏。

覆盖：
- key 归一化（_normalize_key / _clean_key）
- 多条目 entity 派生（默认追加，单值覆盖）
- fact 追加 vs 覆盖（upsert_fact 语义合并）
- 三维打分排序（relevance × recency × importance）
- 情节去重（高相似合并、不同概念并存）
- (user_id, role_id) 记忆隔离
- session 级加工锁（同会话串行，防并发重复加工）
"""

import asyncio

import pytest

from app.memory import pipeline
from app.memory import profile_schema as ps


# ---------------- key 归一化 ----------------
def test_clean_key_basic():
    assert pipeline._clean_key("Identity:Job") == "identity:job"
    assert pipeline._clean_key("  nsfw:xp  ") == "nsfw:xp"
    # 非字符串 / 纯非法字符应被丢弃
    assert pipeline._clean_key(None) is None
    assert pipeline._clean_key("！！！") is None
    # 含非 ascii 但有合法片段：清洗后保留合法部分（不强求丢弃）
    assert pipeline._clean_key("非法 key!!") == "key"


def test_normalize_key_interest_consolidation():
    # gaming/hobby 等收口到 interest:<domain>:<entity>
    assert pipeline._normalize_key("gaming:genre:rimworld") == "interest:game:rimworld"
    assert pipeline._normalize_key("hobby:painting") == "interest:other:painting"


def test_normalize_key_gaming_nsfw_to_xp():
    assert pipeline._normalize_key("gaming:xp_fetish:yandere") == "nsfw:xp:yandere"


# ---------------- entity 派生：默认追加 vs 单值覆盖 ----------------
def test_multi_field_appends_distinct_entities():
    k1 = pipeline._ensure_appendable_entity("nsfw:xp", "the user enjoys voyeurism")
    k2 = pipeline._ensure_appendable_entity("nsfw:xp", "the user likes oral sex")
    assert k1 != k2, "同类不同值必须落到不同 key 才能并存"
    assert k1.startswith("nsfw:xp:") and k2.startswith("nsfw:xp:")


def test_single_value_field_overwrites():
    # 天然单值字段不补 entity → 同 key 覆盖
    assert pipeline._ensure_appendable_entity("identity:age", "26") == "identity:age"
    assert pipeline._ensure_appendable_entity("nsfw:orientation", "bisexual") == "nsfw:orientation"


def test_unknown_field_is_appendable():
    # schema 之外的自创字段也应可追加（不写死类型）
    k = pipeline._ensure_appendable_entity("pet:dog", "has a husky")
    assert k.startswith("pet:dog:")


def test_existing_entity_preserved():
    assert pipeline._ensure_appendable_entity("nsfw:xp:sm", "whatever") == "nsfw:xp:sm"


def test_single_value_whitelist_sane():
    # 单值白名单里的都是合法 schema 前缀，避免拼错
    valid = {f.key_prefix for f in ps.SCHEMA}
    for p in ps.SINGLE_VALUE_PREFIXES:
        assert p in valid, f"{p} 不在 schema 中，单值白名单拼写错误"


# ---------------- fact 追加 vs 覆盖（编排层语义合并）----------------
@pytest.mark.asyncio
async def test_upsert_fact_append_and_overwrite(wired_store):
    s = "u1\x1froleA"
    await wired_store.upsert_fact(s, "nsfw:xp:voyeurism", "the user enjoys voyeurism", 0.9, 1)
    await wired_store.upsert_fact(s, "nsfw:xp:oral_sex", "the user likes oral sex", 0.9, 2)
    keys = {f["key"] for f in wired_store.all_facts(s)}
    assert keys == {"nsfw:xp:voyeurism", "nsfw:xp:oral_sex"}, "两个不同 XP 必须并存"

    # 单值字段覆盖
    await wired_store.upsert_fact(s, "identity:job", "programmer", 0.8, 3)
    await wired_store.upsert_fact(s, "identity:job", "designer", 0.8, 4)
    jobs = [f["value"] for f in wired_store.all_facts(s) if f["key"] == "identity:job"]
    assert jobs == ["designer"], "单值字段应被新值覆盖"


# ---------------- 三维打分排序 ----------------
@pytest.mark.asyncio
async def test_retrieval_scores_recent_and_important_higher(wired_store, monkeypatch):
    from app.memory import retrieval
    from app import rerank
    monkeypatch.setattr(rerank, "enabled", lambda: False)  # 关精排，单测三维打分

    s = "u1\x1froleA"
    # 老但低重要度
    await wired_store.add_episode(s, "we talked about the weather", "neutral", 2, 1)
    # 新且高重要度，且与 query 强相关
    await wired_store.add_episode(s, "the user confessed deep love to me", "love", 9, 10)
    # 推进 turn
    wired_store.append_turn(s, "x", "y", "u1", "roleA")
    for _ in range(9):
        wired_store.append_turn(s, "x", "y", "u1", "roleA")

    top = await retrieval.retrieve_episodes(s, "the user confessed love", top_k=2)
    assert top, "应有召回"
    assert "love" in top[0]["event"], "高相关+高重要+新近的情节应排第一"


# ---------------- 情节去重 ----------------
@pytest.mark.asyncio
async def test_episode_dedup_merges_near_duplicate(wired_store):
    s = "u1\x1froleA"
    await wired_store.add_episode(s, "the user and I went hiking together", "happy", 5, 1)
    n1 = wired_store._store.count_episodes(s)
    # 几乎相同的描述应被去重合并，而不是新增
    await wired_store.add_episode(s, "the user and I went hiking together", "happy", 6, 2)
    n2 = wired_store._store.count_episodes(s)
    assert n2 == n1, "近重复情节应合并而非新增"

    # 完全不同的情节应新增
    await wired_store.add_episode(s, "the user shouted at me in anger about money", "anger", 7, 3)
    n3 = wired_store._store.count_episodes(s)
    assert n3 == n2 + 1, "不同情节应各存一条"


# ---------------- (user, role) 隔离 ----------------
@pytest.mark.asyncio
async def test_memory_isolation_by_user_and_role(wired_store):
    sa, sb, sc = "u1\x1fA", "u1\x1fB", "u2\x1fA"
    await wired_store.upsert_fact(sa, "identity:job", "programmer", 0.9, 1, "u1", "A")
    await wired_store.upsert_fact(sb, "identity:job", "artist", 0.9, 1, "u1", "B")
    await wired_store.upsert_fact(sc, "identity:job", "doctor", 0.9, 1, "u2", "A")

    def job(s):
        return [f["value"] for f in wired_store.all_facts(s)]

    assert job(sa) == ["programmer"]
    assert job(sb) == ["artist"]
    assert job(sc) == ["doctor"]


# ---------------- session 级加工锁 ----------------
@pytest.mark.asyncio
async def test_maybe_process_serialized_per_session(wired_store, monkeypatch):
    from app import config
    s = "u1\x1froleA"
    # 制造足够的轮次触发加工
    for i in range(config.PROCESS_EVERY + 1):
        wired_store.append_turn(s, f"msg{i}", f"reply{i}", "u1", "roleA")

    calls = {"running": 0, "max_concurrent": 0, "total": 0}

    async def fake_process(*args, **kwargs):
        calls["running"] += 1
        calls["total"] += 1
        calls["max_concurrent"] = max(calls["max_concurrent"], calls["running"])
        await asyncio.sleep(0.05)
        calls["running"] -= 1

    monkeypatch.setattr(pipeline, "_process", fake_process)

    # 并发触发多次，锁应保证不并行、且只加工一次（其余被二次确认拦下）
    await asyncio.gather(*[
        pipeline.maybe_process(s, "u1", "roleA", "Vivica", "Leo") for _ in range(5)
    ])
    assert calls["max_concurrent"] == 1, "同会话加工必须串行，不能并发"
    assert calls["total"] == 1, "并发触发只应实际加工一次（其余被锁内二次确认拦下）"
