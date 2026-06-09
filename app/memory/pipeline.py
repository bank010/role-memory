"""异步记忆加工管线 —— 系统真正的"大脑"，与在线对话解耦。

触发：每累计 PROCESS_EVERY 轮跑一次。
动作：事实抽取 → 情节归纳 → 关系/情绪更新 →（更长周期）反思。
全部在后台 task 中执行，异常只记日志，绝不影响对话。
"""

import asyncio
import logging
import re
import traceback
import weakref
from typing import Dict, List, Optional

from app import config, llm
from app.memory import stores

log = logging.getLogger("memory.pipeline")

# ---------------------------------------------------------------------------
# 校验辅助：LLM 输出格式不稳定，任何字段都需要 clamp / 过滤，防止脏数据污染记忆库
# ---------------------------------------------------------------------------
_KEY_RE = re.compile(r"^[a-z0-9:_-]+$")


# 类别别名归一（方案 A）：把模型可能用的近义类别词统一到标准类别，
# 让 pref:venue:x 与 pref:place:x 落到同一类别，后续语义合并才能命中。
_CATEGORY_ALIASES = {
    # 食物/地点/饮品
    "venue": "place", "location": "place", "spot": "place", "site": "place",
    "cuisine": "food", "dish": "food", "meal": "food", "snack": "food",
    "drink": "beverage", "beverages": "beverage",
    # 兴趣大类 interest:<domain> 的 domain 同义词归一
    "gaming": "game", "games": "game", "videogame": "game",
    "song": "music", "artist": "music", "band": "music",
    "movie": "film", "show": "film", "tv": "film", "drama": "film", "cinema": "film",
    "animation": "anime", "manga": "anime", "comic": "anime",
    "reading": "book", "books": "book", "novel": "book",
    "sports": "sport", "fitness": "sport", "exercise": "sport", "workout": "sport",
    "hobby": "other", "interest": "other", "misc": "other",
}


def _clean_key(raw) -> Optional[str]:
    """规范化 fact key：小写、只保留合法字符、最长 40。返回 None 表示非法丢弃。"""
    if not isinstance(raw, str):
        return None
    k = raw.strip().lower()[:40]
    k = re.sub(r"[^a-z0-9:_-]", "_", k).strip("_-:")
    return _normalize_key(k) if k and _KEY_RE.match(k) else None


# interest 的合法 domain（第二段）
_INTEREST_DOMAINS = {"game", "music", "film", "anime", "book", "sport", "other"}
# gaming 模块旧字段里属于"成人性癖"的，应改投 nsfw:xp，不能并进兴趣
_GAMING_NSFW_FIELDS = {"xp_fetish", "adult_genre", "adult_content"}
# 会被收口到 interest 大类的模块名
_INTEREST_MODULES = {"gaming", "game", "hobby"}


def _normalize_key(key: str) -> str:
    """归一 key，把模型可能用的细碎/近义 key 收口到 schema 大类。

    规则（按优先级）：
    1) gaming 模块里的成人性癖字段 -> nsfw:xp:<具体>（先拦截，避免误并入兴趣）；
    2) 类别段（倒数第二段）按别名表归一，如 venue->place、movie->film；
    3) gaming/hobby/game 模块 + preference 的兴趣领域 -> interest:<domain>:<具体>；
    4) interest 的 domain 非法时兜底为 other，且不丢失末段具体项。
    例: gaming:genre:rimworld -> interest:game:rimworld
        hobby:painting        -> interest:other:painting
        gaming:xp_fetish:yandere -> nsfw:xp:yandere
    """
    parts = key.split(":")
    mod = parts[0]

    # 1) 成人性癖优先拦截
    if mod == "gaming" and len(parts) >= 2 and parts[1] in _GAMING_NSFW_FIELDS:
        tail = parts[2] if len(parts) >= 3 else parts[1]
        return f"nsfw:xp:{tail}"

    # 2) 类别段别名归一
    if len(parts) >= 3:
        parts[-2] = _CATEGORY_ALIASES.get(parts[-2], parts[-2])

    # 3) 收口到 interest 大类
    if mod in _INTEREST_MODULES or (mod == "preference" and len(parts) >= 2
            and parts[1] in _INTEREST_DOMAINS | {"content", "entertainment"}):
        # 末段“具体项”：三段取第三段，两段取第二段
        tail = parts[2] if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else "")
        # domain：三段取已归一的第二段，两段无明确 domain
        domain = parts[1] if len(parts) >= 3 else ""
        domain = _CATEGORY_ALIASES.get(domain, domain)
        if domain not in _INTEREST_DOMAINS:
            # gaming 模块下任何未知类别段（如 genre）都归属 game，否则兜底 other
            domain = "game" if mod in ("gaming", "game") else "other"
        return ":".join(p for p in ["interest", domain, tail] if p)

    return ":".join(parts)


def _slug_from_value(value: str) -> str:
    """从英文 value 派生一个稳定的 entity slug（小写、下划线、最长 24）。

    抽取的 value 是英文（canonical），如 "user enjoys voyeurism" -> "voyeurism"。
    取信息量最高的尾部词，去掉停用词，保证同一概念每次得到相同 slug。
    """
    v = (value or "").strip().lower()
    v = re.sub(r"[^a-z0-9\s_-]", " ", v)
    tokens = [t for t in re.split(r"[\s_-]+", v) if t and t not in _STOPWORDS]
    if not tokens:
        return "item"
    # 取最后两个有意义的词，覆盖 "anal sex" / "role play" 这类短语
    slug = "_".join(tokens[-2:]) if len(tokens) >= 2 else tokens[-1]
    return slug[:24].strip("_") or "item"


_STOPWORDS = {
    "user", "the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "on",
    "likes", "like", "liked", "enjoys", "enjoy", "loves", "love", "prefers",
    "prefer", "into", "having", "have", "has", "with", "for", "user's", "their",
    "being", "doing", "wants", "want", "kink", "fetish", "play",
}


def _ensure_appendable_entity(key: str, value: str) -> str:
    """默认让事实“可追加并存”：对任何 2 段 key（module:field），若不是天然单值字段，
    就用 value 自动补一个 entity 子键，使同类的多个值各存一条、而非互相覆盖。

    设计原则（用户要求）：有新信息就追加，不靠预先写死的类型决定能否并存。
    只有 SINGLE_VALUE 白名单里的本质单值属性（年龄/性取向/当前职业/作息等）才保持覆盖。

    例:
      key="nsfw:xp",       value="user enjoys voyeurism" -> "nsfw:xp:voyeurism"   (追加)
      key="identity:age",  value="26"                    -> "identity:age"        (覆盖)
    """
    from app.memory import profile_schema
    parts = key.split(":")
    if len(parts) == 2 and not profile_schema.is_single_value_key(key):
        return f"{key}:{_slug_from_value(value)}"
    return key


def _clean_str(raw, max_len: int) -> Optional[str]:
    """截断并清洗字符串；空/非字符串返回 None。"""
    if not isinstance(raw, str):
        return None
    v = raw.strip()[:max_len]
    return v if v else None


def _clamp_float(v, lo: float, hi: float, default: float) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _clamp_int(v, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(float(v))))
    except (TypeError, ValueError):
        return default

_EXTRACT_PROMPT = """You maintain memory for an AI character chatting with a human user.
Extract long-term memory from the dialogue below (which may be in any language).

POINT OF VIEW — write ALL memory in clear third-person using the REAL NAMES below (consistent POV, natural for display and retrieval):
- Refer to the human user as: {user_tok}
- Refer to the character as: {char_tok}
- Use these exact names. Do NOT use "I/me", "the user/the character" generic words, or pronouns when a name fits; prefer the names for clarity.
- Example episode: "{user_tok} made {char_tok} perform oral sex; {char_tok} complied reluctantly, with shame and tears."
- Example fact: "{user_tok} likes SM."

WHAT GOES WHERE — facts and episodes are NOT mutually exclusive. Often a single event yields BOTH:
- "episode" = the ONE-TIME EVENT that happened — "what HAPPENED between {char_tok} and {user_tok}".
- "facts" = STABLE ATTRIBUTES / PREFERENCES of {user_tok} revealed by the dialogue — "who {user_tok} IS / what they LIKE".
- CRITICAL: when an event reveals a stable preference/trait, output the episode AND ALSO distill the
  preference into a fact. Do NOT skip facts just because it happened during a shared moment.
    Example: the dialogue shows {user_tok} ordering {char_tok} into rope bondage and dominating them.
      → episode: "{user_tok} tied {char_tok} up and dominated them during sex."
      → facts: [{{"key":"nsfw:xp:bondage","value":"{user_tok} likes bondage"}},
                {{"key":"nsfw:content_pref:dominance","value":"{user_tok} likes to take a dominant role"}}]
    Example: {user_tok} insists on chatting in Chinese.
      → fact: {{"key":"identity:language","value":"{user_tok} prefers Chinese over English"}}
- Be GENEROUS with facts: extract every stable preference, trait, kink, interest, habit, or boundary
  that {user_tok} expresses or demonstrates. It is better to capture a real preference than to miss it.

FACTS RULES:
- Describe ONLY the HUMAN USER ({user_tok}). NEVER store {char_tok}'s own persona traits.
- Always start a fact value with "{user_tok} ..." for clarity.
- Only return "facts": [] if {user_tok} truly revealed nothing about their preferences/traits this batch.
- Write "value"/"event"/"mood" in ENGLISH (canonical memory language), using {user_tok}/{char_tok} tokens.
- Choose "key" from the FIELD CATALOG below. Use exactly "module:field" for single-value fields,
  or "module:field:entity" (entity in lowercase english) for multi-item fields so they don't overwrite.
    GOOD: "preference:content:cyberpunk", "nsfw:xp:sm", "identity:job"
- If unsure which catalog key fits, pick the closest module and add a short english entity
  (e.g. "nsfw:xp:roleplay", "interest:other:worldbuilding") rather than dropping the fact.
- Reuse an EXISTING key (listed under known facts) when the new info is about the same item.
- If {user_tok} changes their mind, REUSE THE SAME KEY and update the value
  (e.g. "{user_tok} no longer likes jazz (previously liked it)").

EPISODES RULES:
- Write the event using {user_tok}/{char_tok} tokens (see POV rule above).
- Set "sensitive": true for intimate/sexual/NSFW events; otherwise false.
- Intimate/high-emotion events should get higher importance (7-10) so they persist as shared memory.

FIELD CATALOG (pick fact keys from here):
{field_catalog}

Currently known facts about the user (reuse these keys when relevant):
{known_facts}

Dialogue:
{dialogue}

JSON format:
{{
  "facts": [{{"key": "module:field or module:field:entity", "value": "english fact about the user", "confidence": 0.0-1.0}}],
  "episode": {{"event": "one english sentence of what happened; empty string if nothing worth keeping", "emotion": "english emotion word", "importance": integer 1-10, "sensitive": true/false}},
  "relationship": {{"intimacy_delta": -0.1 to 0.2, "trust_delta": -0.1 to 0.2, "stage": "relationship stage (optional)", "mood": "character's current mood in english"}}
}}
Output JSON only, no explanation."""

_SUMMARY_PROMPT = """Maintain a rolling summary of an ongoing role-play between a character and a user.
Merge the previous summary with the new dialogue into an updated, concise summary in ENGLISH.

POINT OF VIEW — write in clear third-person using the REAL NAMES below:
- Refer to the character as: {char_tok}
- Refer to the user as: {user_tok}
- Use these exact names (avoid bare pronouns when a name is clearer).
- Example: "{user_tok} pressured {char_tok} to kneel; {char_tok} complied reluctantly, ashamed and in tears."

Keep it under 120 words. Preserve key plot points, decisions, promises, emotional shifts and unresolved threads.
Drop trivial small talk. Write as a flowing recap, not bullet points.

Previous summary (may be empty):
{prev}

New dialogue:
{dialogue}

Output ONLY the updated summary text."""

_REFLECT_PROMPT = """Based on these recent episodes, derive ONE high-level insight about the user {user_tok}
(what they truly care about, their personality, the interaction pattern between {char_tok} and {user_tok}).
Write it in ENGLISH using the real names {user_tok} and {char_tok}.
This insight will be stored as an important memory.

Recent episodes:
{episodes}

Output JSON: {{"insight": "one english sentence insight about {user_tok}", "importance": integer 7-10}}
JSON only."""


def _format_dialogue(turns: List[Dict]) -> str:
    lines = []
    for t in turns:
        if t.get("user_msg"):
            lines.append(f"User: {t['user_msg']}")
        if t.get("ai_reply"):
            lines.append(f"Character: {t['ai_reply']}")
    return "\n".join(lines)


# 每个 session 一把异步锁，保证同会话的加工串行执行，避免并发请求重复抽取/进度错乱。
# 用 WeakValueDictionary 让空闲 session 的锁能被 GC 回收，不会无限增长。
_session_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = weakref.WeakValueDictionary()
_locks_guard = asyncio.Lock()


async def _get_session_lock(session: str) -> asyncio.Lock:
    async with _locks_guard:
        lock = _session_locks.get(session)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[session] = lock
        return lock


async def maybe_process(session: str, user_id: str = "", role_id: str = "",
                        char_name: str = "", user_name: str = "") -> None:
    """判断是否到触发点，到了就加工。

    char_name/user_name：当前会话的真实角色名/用户名，直接写进记忆（情节/画像/摘要），
    使记忆里就是真名而非占位符，展示与检索都自然。未传则回退到通用称谓。

    并发安全：同一 session 串行加工（session 级锁）；快速预检在锁外，真正加工在锁内二次确认，
    避免多个并发请求重复抽取同一批对话、或把 last_processed 推乱。
    """
    # 锁外快速预检：未到触发点直接返回，避免无谓抢锁
    if stores.max_turn(session) - stores.get_last_processed(session) < config.PROCESS_EVERY:
        return

    lock = await _get_session_lock(session)
    async with lock:
        # 锁内二次确认：可能在等锁期间已被另一协程加工过
        last = stores.get_last_processed(session)
        now = stores.max_turn(session)
        if now - last < config.PROCESS_EVERY:
            return
        try:
            await _process(session, last, now, user_id, role_id, char_name, user_name)
            # 仅在加工全程成功后才推进进度；任何异常都不推进，下次重试不丢这批记忆
            stores.set_last_processed(session, now, user_id, role_id)
        except Exception as e:
            log.error("记忆加工失败 session=%s err=%s\n%s", session, e, traceback.format_exc())


async def _process(session: str, after: int, now: int,
                   user_id: str = "", role_id: str = "",
                   char_name: str = "", user_name: str = "") -> None:
    new_turns = stores.turns_after(session, after)
    if not new_turns:
        return
    dialogue = _format_dialogue(new_turns)

    # 记忆直接用真实名字存储；未传名字时回退到通用称谓，保证语义清晰
    user_tok = (user_name or "").strip() or "the user"
    char_tok = (char_name or "").strip() or "the character"

    known = stores.all_facts(session)
    known_facts = "\n".join(f"- {f['key']}: {f['value']}" for f in known) or "(none yet)"

    from app.memory import profile_schema
    field_catalog = profile_schema.prompt_field_catalog(include_sensitive=config.NSFW_ENABLED)

    data = await llm.extract_json(
        _EXTRACT_PROMPT.format(dialogue=dialogue, known_facts=known_facts,
                               field_catalog=field_catalog,
                               user_tok=user_tok, char_tok=char_tok)
    )

    if not data:
        return

    # ---- facts：key 规范 + value 截断 + confidence clamp ----
    raw_facts = data.get("facts") or []
    log.info("抽取返回 facts=%d episode=%s session=%s | raw_keys=%s",
             len(raw_facts), bool((data.get("episode") or {}).get("event")), session,
             [f.get("key") for f in raw_facts])
    skipped_facts = 0
    for f in raw_facts:
        raw_key = f.get("key")
        key = _clean_key(raw_key)
        value = _clean_str(f.get("value"), 500)
        if not key or not value:
            skipped_facts += 1
            log.warning("丢弃 fact: raw_key=%r -> clean=%r value=%r", raw_key, key, value)
            continue
        # 默认可追加：非单值字段缺 entity 时按 value 自动补子键，避免同类互相覆盖
        key = _ensure_appendable_entity(key, value)
        # NSFW 关闭时，丢弃高敏感事实（兜底，即便模型仍抽了出来）
        if not config.NSFW_ENABLED and profile_schema.is_sensitive_key(key):
            continue
        confidence = _clamp_float(f.get("confidence"), 0.0, 1.0, 0.6)
        await stores.upsert_fact(session, key, value, confidence, now, user_id, role_id)
    if skipped_facts:
        log.warning("facts 校验丢弃 %d 条（key/value 非法）session=%s", skipped_facts, session)

    # ---- episode：event 截断 + importance clamp + sensitive 标记 ----
    ep = data.get("episode") or {}
    event = _clean_str(ep.get("event"), 300)
    if event:
        emotion = _clean_str(ep.get("emotion"), 50) or "neutral"
        importance = _clamp_int(ep.get("importance"), 1, 10, 3)
        sensitive = bool(ep.get("sensitive"))
        if sensitive and not config.NSFW_ENABLED:
            pass  # NSFW 关闭：不存敏感事件
        else:
            await stores.add_episode(session, event, emotion, importance, now, sensitive,
                                     user_id, role_id)

    # ---- relationship：delta clamp + stage/mood 截断 ----
    rel = data.get("relationship") or {}
    stores.update_relationship(
        session,
        intimacy_delta=_clamp_float(rel.get("intimacy_delta"), -0.2, 0.3, 0.0),
        trust_delta=_clamp_float(rel.get("trust_delta"), -0.2, 0.3, 0.0),
        stage=_clean_str(rel.get("stage"), 30),
        mood=_clean_str(rel.get("mood"), 30),
        turn=now, user_id=user_id, role_id=role_id,
    )

    await _update_summary(session, dialogue, now, user_id, role_id, char_name, user_name)
    await _maybe_reflect(session, now, user_id, role_id, char_name, user_name)


async def _update_summary(session: str, dialogue: str, now: int,
                          user_id: str = "", role_id: str = "",
                          char_name: str = "", user_name: str = "") -> None:
    """滚动摘要：用 旧摘要 + 本批新对话 增量更新，存进 relationship.summary。"""
    user_tok = (user_name or "").strip() or "the user"
    char_tok = (char_name or "").strip() or "the character"
    try:
        prev = stores.get_relationship(session).get("summary") or "(none)"
        summary = await llm.chat(
            [{"role": "user", "content": _SUMMARY_PROMPT.format(
                prev=prev, dialogue=dialogue, user_tok=user_tok, char_tok=char_tok)}],
            model=config.EXTRACT_MODEL, temperature=0.3, max_tokens=300,
            use_extract_endpoint=True,
        )
        summary = (summary or "").strip()
        if summary:
            stores.update_relationship(session, summary=summary, turn=now,
                                       user_id=user_id, role_id=role_id)
    except Exception as e:
        log.warning("滚动摘要更新失败 session=%s err=%s", session, e)


async def _maybe_reflect(session: str, now: int, user_id: str = "", role_id: str = "",
                         char_name: str = "", user_name: str = "") -> None:
    """更长周期的反思：每 ~5 个加工周期归纳一条高层洞察。"""
    episodes = stores.all_episodes(session)
    if len(episodes) < config.PROCESS_EVERY * 2 or now % (config.PROCESS_EVERY * 3) != 0:
        return
    episodes.sort(key=lambda e: e["turn"], reverse=True)
    recent = episodes[:8]
    text = "\n".join(f"- {e['event']}" for e in recent)
    user_tok = (user_name or "").strip() or "the user"
    char_tok = (char_name or "").strip() or "the character"
    try:
        data = await llm.extract_json(_REFLECT_PROMPT.format(
            episodes=text, user_tok=user_tok, char_tok=char_tok))
        insight = (data or {}).get("insight")
        if insight:
            await stores.add_episode(session, f"[insight] {insight}", "understanding",
                                     int(data.get("importance", 8)), now,
                                     user_id=user_id, role_id=role_id)
    except Exception as e:
        log.warning("反思失败 session=%s err=%s", session, e)
