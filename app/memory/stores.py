"""四类记忆的读写【编排层】。

本层只做业务编排——向量化、语言归一化、情节去重、关系值 clamp、缓存读写穿透；
真正的持久化委托给 app.store 的后端（SQLite / Postgres+pgvector），面向接口编程。

- turns       : 原始对话日志（真相源）
- facts       : 语义记忆 / 用户画像（结构化、可覆盖）
- episodes    : 情节记忆（事件 + 向量）
- relationship: 关系/情绪状态（每会话一条，滚动更新）
- chunks      : 逐字记忆（原话向量化，精确细节召回）
"""

import logging
import math
import time
from typing import Dict, List, Optional

import numpy as np

from app import cache, config, embeddings
from app.store import get_store

log = logging.getLogger("memory.stores")

_store = get_store()


def init() -> None:
    _store.init()


# ---------------- 原始对话日志 ----------------
def append_turn(session: str, user_msg: str, ai_reply: str,
                user_id: str = "", role_id: str = "") -> int:
    turn = _store.append_turn(session, user_msg, ai_reply, user_id, role_id)
    cache.invalidate(session, "relationship")  # max_turn 变化会影响 recency
    return turn


def recent_turns(session: str, n: int) -> List[Dict]:
    return _store.recent_turns(session, n)


def turns_after(session: str, after_turn: int) -> List[Dict]:
    return _store.turns_after(session, after_turn)


def max_turn(session: str) -> int:
    return _store.max_turn(session)


# ---------------- 逐字记忆（verbatim chunks）----------------
async def index_chunk(session: str, turn: int, role: str, text: str,
                      user_id: str = "", role_id: str = "") -> None:
    """把一条原话向量化入库，用于逐字精确召回（细节记忆的关键）。

    多语言：embedding 基于归一化到基准语言的文本(norm)，原文(text)保留显示。
    """
    from app import normalizer

    text = (text or "").strip()
    if not text:
        return
    norm = await normalizer.to_base_lang(text)
    vec = await embeddings.embed(norm)
    _store.insert_chunk(session, turn, role, text, norm, time.time(), vec, user_id, role_id)
    evict_chunks_if_needed(session)


def all_chunks(session: str) -> List[Dict]:
    return _store.all_chunks(session)


# ---------------- 语义记忆 / 画像 ----------------
def _split_key(key: str):
    """拆出 (category, entity)。三段及以上才有 entity，否则 entity=None（单一属性）。
    例: pref:food:cilantro -> ('pref:food', 'cilantro'); name -> ('name', None)
    """
    parts = key.split(":")
    if len(parts) >= 3:
        return ":".join(parts[:-1]), parts[-1]
    return key, None


async def upsert_fact(session: str, key: str, value: str, confidence: float, turn: int,
                      user_id: str = "", role_id: str = "") -> None:
    """写入事实，带 category+entity 双重语义合并：

    - 单一属性（name/job/pref:coffee 等无 entity 的 key）：按 key 精确覆盖（本就该覆盖）。
    - 带 entity 的偏好（pref:food:cilantro）：在【同类别】已有事实里找实体向量最相似的，
      超过阈值 → 复用其 key 更新（同一条目的不同表述/改口），否则新增（同类不同条目各存一条）。
    这样既不会把"香菜/虾"误合并，又能把"cilantro/coriander/香菜"归并到一条。
    """
    from app import normalizer

    category, entity = _split_key(key)

    if entity is None:
        _store.upsert_fact(session, key, value, confidence, turn, None, user_id, role_id)
        cache.invalidate(session, "facts")
        return

    evec = await embeddings.embed(await normalizer.to_base_lang(entity))

    best_key, best_sim = None, 0.0
    for f in _store.facts_with_vec(session):
        fvec = f.get("vec")
        if fvec is None or getattr(fvec, "size", 0) == 0:
            continue
        f_cat, f_ent = _split_key(f["key"])
        if f_ent is None or f_cat != category:
            continue  # 只在同类别里比较实体
        sim = embeddings.cosine(evec, fvec)
        if sim > best_sim:
            best_key, best_sim = f["key"], sim

    target_key = best_key if (best_key and best_sim >= config.FACT_MERGE_THRESHOLD) else key
    _store.upsert_fact(session, target_key, value, confidence, turn, evec, user_id, role_id)
    cache.invalidate(session, "facts")


def all_facts(session: str) -> List[Dict]:
    # 热读：拼上下文每轮都要，走 Redis 读穿透
    facts = cache.cached(session, "facts", lambda: _store.all_facts(session))
    # 按 schema 标注高敏感字段（不进缓存键，读出时即时标注，避免缓存里混存语义）
    from app.memory import profile_schema
    for f in facts:
        f["sensitive"] = profile_schema.is_sensitive_key(f.get("key", ""))
    return facts


# ---------------- 情节记忆 ----------------
_EPISODE_DEDUP_THRESHOLD = 0.88  # 余弦相似度超过此值视为重复，合并而非新增


async def add_episode(session: str, event: str, emotion: str, importance: int, turn: int,
                      sensitive: bool = False, user_id: str = "", role_id: str = "") -> None:
    """新增情节；若与已有情节高度相似则合并（去重），避免记忆库越攒越脏。

    sensitive: 亲密/敏感事件（如"发生了亲密关系"）打标，便于后续隔离/开关控制注入。
    """
    from app import normalizer
    vec = await embeddings.embed(await normalizer.to_base_lang(event))

    best_id, best_sim, best_imp = None, 0.0, 0
    for r in _store.all_episodes(session):
        sim = embeddings.cosine(vec, r["vec"])
        if sim > best_sim:
            best_id, best_sim, best_imp = r["id"], sim, r["importance"]

    if best_id is not None and best_sim >= _EPISODE_DEDUP_THRESHOLD:
        _store.update_episode(best_id, max(best_imp, int(importance)), turn, time.time())
        return

    _store.insert_episode(session, event, emotion, int(importance), time.time(), turn, vec,
                          sensitive, user_id, role_id)
    evict_episodes_if_needed(session)


def all_episodes(session: str) -> List[Dict]:
    return _store.all_episodes(session)


def candidate_episodes(session: str, qvec: np.ndarray, limit: int) -> List[Dict]:
    """召回候选：postgres 走 pgvector 原生 KNN 预筛，sqlite 返回全部。"""
    return _store.candidate_episodes(session, qvec, limit)


def mark_recalled(ids: List[int], turn: int) -> None:
    _store.mark_recalled(ids, turn)


def evict_episodes_if_needed(session: str) -> None:
    """情节数超过 MAX_EPISODES 时，按"重要度 × 新近度"打分，淘汰尾部。

    淘汰标准：importance × exp(-RECENCY_DECAY × age_in_turns)
    最低分的情节最应该被遗忘（重要度低 + 很久没提到）。
    [insight] 洞察条目重要度本身已设为 8+，自然被保留。
    """
    count = _store.count_episodes(session)
    if count <= config.MAX_EPISODES:
        return
    now = _store.max_turn(session)
    episodes = _store.all_episodes(session)
    scored = []
    for ep in episodes:
        age = max(0, now - ep.get("turn", 0))
        recency = math.exp(-config.RECENCY_DECAY * age)
        score = ep.get("importance", 3) * recency
        scored.append((score, ep["id"]))
    scored.sort()  # 最低分在前
    to_evict = [eid for _, eid in scored[: count - config.MAX_EPISODES]]
    if to_evict:
        _store.delete_episodes(to_evict)
        log.info("情节淘汰 %d 条 session=%s（上限 %d）", len(to_evict), session, config.MAX_EPISODES)


def evict_chunks_if_needed(session: str) -> None:
    """chunk 数超过 MAX_CHUNKS 时，删除最旧的（按 turn 最小的），保留最近的。"""
    count = _store.count_chunks(session)
    if count <= config.MAX_CHUNKS:
        return
    keep = config.MAX_CHUNKS
    _store.evict_oldest_chunks(session, keep)
    log.info("逐字记忆淘汰至 %d 条 session=%s", keep, session)


# ---------------- 关系/情绪状态 ----------------
def get_relationship(session: str) -> Dict:
    return cache.cached(session, "relationship", lambda: dict(_store.get_relationship(session)))


def update_relationship(session: str, intimacy_delta: float = 0.0, trust_delta: float = 0.0,
                        stage: Optional[str] = None, mood: Optional[str] = None,
                        summary: Optional[str] = None, turn: int = 0,
                        user_id: str = "", role_id: str = "") -> None:
    rel = _store.get_relationship(session)  # 直查后端，避免基于缓存做累加
    intimacy = min(1.0, max(0.0, rel["intimacy"] + intimacy_delta))
    trust = min(1.0, max(0.0, rel["trust"] + trust_delta))
    _store.save_relationship(
        session, intimacy, trust,
        stage or rel["stage"],
        mood or rel["mood"],
        summary if summary is not None else rel["summary"],
        turn, user_id, role_id,
    )
    cache.invalidate(session, "relationship")


# ---------------- 加工进度 ----------------
def get_last_processed(session: str) -> int:
    return _store.get_last_processed(session)


def set_last_processed(session: str, turn: int, user_id: str = "", role_id: str = "") -> None:
    _store.set_last_processed(session, turn, user_id, role_id)


def reset_session(session: str) -> None:
    _store.reset_session(session)
    cache.invalidate(session, "facts", "relationship")
