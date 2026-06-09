"""pipeline 校验兜底测试。

覆盖：非法 key 丢弃、value 截断、confidence clamp、importance clamp、
      delta clamp、LLM 返回 None/空/非法 JSON 时不崩溃。
"""

import pytest

from app.memory.pipeline import (
    _clean_key,
    _clean_str,
    _clamp_float,
    _clamp_int,
    _normalize_key,
)


# ---------- _normalize_key（类别别名归一）----------
class TestNormalizeKey:
    def test_venue_to_place(self):
        assert _normalize_key("pref:venue:internet_cafe") == "pref:place:internet_cafe"

    def test_cuisine_to_food(self):
        assert _normalize_key("pref:cuisine:sushi") == "pref:food:sushi"

    def test_unknown_category_unchanged(self):
        assert _normalize_key("pref:color:blue") == "pref:color:blue"

    def test_two_segment_unchanged(self):
        assert _normalize_key("pref:coffee") == "pref:coffee"

    def test_single_segment_unchanged(self):
        assert _normalize_key("name") == "name"

    def test_clean_key_applies_normalization(self):
        # _clean_key 内部应调用 _normalize_key
        assert _clean_key("pref:venue:bar") == "pref:place:bar"


# ---------- _clean_key ----------
class TestCleanKey:
    def test_normal(self):
        assert _clean_key("pref:music") == "pref:music"

    def test_uppercase_lowercased(self):
        assert _clean_key("Name") == "name"

    def test_spaces_stripped(self):
        assert _clean_key("  job  ") == "job"

    def test_special_chars_replaced(self):
        k = _clean_key("user's name!")
        assert k is not None
        assert "'" not in k and "!" not in k

    def test_too_long_truncated(self):
        k = _clean_key("a" * 60)
        assert len(k) == 40

    def test_none_returns_none(self):
        assert _clean_key(None) is None

    def test_int_returns_none(self):
        assert _clean_key(123) is None

    def test_empty_string_returns_none(self):
        assert _clean_key("") is None

    def test_only_special_chars_returns_none(self):
        assert _clean_key("!!!") is None

    def test_chinese_key_is_rejected(self):
        # 中文字符全部转为下划线后会被 strip，最终为空 → None（key 必须是英文标识符）
        assert _clean_key("用户名字") is None

    def test_colon_and_hyphen_allowed(self):
        assert _clean_key("pref:jazz-rock") == "pref:jazz-rock"


# ---------- _clean_str ----------
class TestCleanStr:
    def test_normal(self):
        assert _clean_str("hello world", 100) == "hello world"

    def test_truncation(self):
        s = _clean_str("a" * 600, 500)
        assert len(s) == 500

    def test_empty_returns_none(self):
        assert _clean_str("", 100) is None

    def test_whitespace_only_returns_none(self):
        assert _clean_str("   ", 100) is None

    def test_non_string_returns_none(self):
        assert _clean_str(42, 100) is None
        assert _clean_str(None, 100) is None
        assert _clean_str(["list"], 100) is None

    def test_strips_leading_trailing_whitespace(self):
        assert _clean_str("  hello  ", 100) == "hello"


# ---------- _clamp_float ----------
class TestClampFloat:
    def test_normal(self):
        assert _clamp_float(0.5, 0.0, 1.0, 0.0) == pytest.approx(0.5)

    def test_above_hi(self):
        assert _clamp_float(2.0, 0.0, 1.0, 0.0) == pytest.approx(1.0)

    def test_below_lo(self):
        assert _clamp_float(-1.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_string_number(self):
        assert _clamp_float("0.7", 0.0, 1.0, 0.0) == pytest.approx(0.7)

    def test_invalid_returns_default(self):
        assert _clamp_float("abc", 0.0, 1.0, 0.5) == pytest.approx(0.5)
        assert _clamp_float(None, 0.0, 1.0, 0.5) == pytest.approx(0.5)


# ---------- _clamp_int ----------
class TestClampInt:
    def test_normal(self):
        assert _clamp_int(5, 1, 10, 3) == 5

    def test_float_rounds(self):
        assert _clamp_int(7.9, 1, 10, 3) == 7

    def test_above_clamp(self):
        assert _clamp_int(99, 1, 10, 3) == 10

    def test_below_clamp(self):
        assert _clamp_int(0, 1, 10, 3) == 1

    def test_invalid_returns_default(self):
        assert _clamp_int("bad", 1, 10, 3) == 3
        assert _clamp_int(None, 1, 10, 3) == 3


# ---------- pipeline._process 端到端校验 ----------
@pytest.mark.asyncio
async def test_process_skips_bad_facts(tmp_db, session, monkeypatch):
    """LLM 返回非法 key/value 时，只保存合法条目，非法的静默跳过。"""
    import app.llm as llm_mod
    import app.normalizer as norm_mod
    from app.memory import pipeline, stores

    async def mock_extract(prompt, **kw):
        return {
            "facts": [
                {"key": "name", "value": "Alice", "confidence": 0.9},       # 合法
                {"key": None, "value": "should skip", "confidence": 0.5},   # key=None，丢弃
                {"key": "job", "value": "", "confidence": 0.8},              # value 空，丢弃
                {"key": "!!!", "value": "bad key", "confidence": 0.7},       # key 非法，丢弃
                {"key": "pref:food", "value": "x" * 600, "confidence": 1.5}, # 值过长截断+conf超出clamp
            ],
            "episode": {"event": "", "emotion": "ok", "importance": 5},
            "relationship": {"intimacy_delta": 0.1, "trust_delta": 0.05},
        }

    async def mock_norm(text):
        return text

    async def mock_chat(*a, **k):
        return "rolling summary"

    monkeypatch.setattr(llm_mod, "extract_json", mock_extract)
    monkeypatch.setattr(norm_mod, "to_base_lang", mock_norm)
    monkeypatch.setattr(llm_mod, "chat", mock_chat)  # 避免 _update_summary 走真实网络

    # 假造两轮
    stores.append_turn(session, "I'm Alice", "Nice to meet you Alice")
    stores.append_turn(session, "I like sushi", "Great!")

    await pipeline._process(session, 0, 2)

    facts = {f["key"]: f for f in stores.all_facts(session)}
    assert "name" in facts
    assert facts["name"]["value"] == "Alice"
    assert facts["name"]["confidence"] == pytest.approx(0.9)

    assert "job" not in facts         # value 空被丢弃
    # pref:food 是多值偏好字段：默认追加，会按 value 自动补 entity 子键（pref:food:<slug>）
    food_keys = [k for k in facts if k.startswith("pref:food")]
    assert len(food_keys) == 1, "合法 food 偏好应保存一条"
    food = facts[food_keys[0]]
    assert len(food["value"]) == 500                       # value 截断到 500
    assert food["confidence"] == pytest.approx(1.0)        # 1.5 clamp 到 1.0


@pytest.mark.asyncio
async def test_process_handles_none_data(tmp_db, session, monkeypatch):
    """LLM 返回 None 时，_process 不应崩溃。"""
    import app.llm as llm_mod
    from app.memory import pipeline, stores

    async def mock_extract(prompt, **kw):
        return None

    monkeypatch.setattr(llm_mod, "extract_json", mock_extract)

    stores.append_turn(session, "hello", "hi")
    await pipeline._process(session, 0, 1)  # 不应抛异常


@pytest.mark.asyncio
async def test_process_clamps_importance(tmp_db, session, monkeypatch):
    """importance 超出 [1,10] 应被 clamp，不应写入非法值。"""
    import app.llm as llm_mod
    import app.embeddings as emb_mod
    import app.normalizer as norm_mod
    from app.memory import pipeline, stores

    async def mock_extract(prompt, **kw):
        return {
            "facts": [],
            "episode": {"event": "something happened", "emotion": "ok", "importance": 99},
            "relationship": {},
        }

    async def mock_embed(text):
        return emb_mod._local_embed(text)

    async def mock_norm(text):
        return text

    async def mock_chat(*a, **k):
        return "summary"

    monkeypatch.setattr(llm_mod, "extract_json", mock_extract)
    monkeypatch.setattr(emb_mod, "embed", mock_embed)
    monkeypatch.setattr(norm_mod, "to_base_lang", mock_norm)
    monkeypatch.setattr(llm_mod, "chat", mock_chat)

    stores.append_turn(session, "something", "happened")
    await pipeline._process(session, 0, 1)

    eps = stores.all_episodes(session)
    assert eps[0]["importance"] == 10  # 99 被 clamp 到 10


@pytest.mark.asyncio
async def test_process_clamps_relationship_delta(tmp_db, session, monkeypatch):
    """极端 delta 应被 clamp 到 [-0.2, 0.3]，不应使亲密度无限增长。"""
    import app.llm as llm_mod
    from app.memory import pipeline, stores

    async def mock_extract(prompt, **kw):
        return {
            "facts": [],
            "episode": {"event": "", "emotion": "ok", "importance": 3},
            "relationship": {"intimacy_delta": 999.0, "trust_delta": -999.0},
        }

    async def mock_chat(*a, **k):
        return "summary"

    monkeypatch.setattr(llm_mod, "extract_json", mock_extract)
    monkeypatch.setattr(llm_mod, "chat", mock_chat)

    stores.append_turn(session, "hi", "hello")
    await pipeline._process(session, 0, 1)

    rel = stores.get_relationship(session)
    assert rel["intimacy"] <= 1.0
    assert rel["trust"] >= 0.0
