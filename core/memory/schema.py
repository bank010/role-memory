"""结构化用户画像 Schema —— 提取/存储的单一事实源（single source of truth）。

设计目标（面向全球 AI 社交陪伴产品）：
- 把"自由抽取"升级为分模块、可控类别的结构化画像。
- 每个字段带：模块(module)、更新频率(freq)、是否长期保存(long_term)、是否高敏感(sensitive)。
- 存储沿用扁平 facts 表，key 编码为 "<module>:<field>" 或 "<module>:<field>:<entity>"。
  例：identity:job、nsfw:orientation、preference:content:cyberpunk。

key 约定：
- 单值字段（会被新值覆盖）：module:field，如 identity:nickname、emotional:attachment_style。
- 多条目字段（同类可并存）：module:field:entity，如 preference:content:cyberpunk、gaming:title:rimworld。
  （沿用 stores.upsert_fact 的 category+entity 语义合并，避免互相覆盖。）

freq（更新频率）与 long_term（是否长期保存）目前作为元信息记录在 schema 中，
供 prompt 生成与未来的差异化淘汰策略使用；当前淘汰仍走 facts 覆盖 + episodes 上限。
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Field:
    module: str          # 模块名，如 Identity / Personality / NSFW
    field: str           # 字段标识（英文，进 key），如 job / orientation
    desc: str            # 提取内容说明
    example: str         # 示例
    freq: str            # 更新频率：low / mid / high
    long_term: bool      # 是否长期保存
    sensitive: bool = False   # 是否高敏感（NSFW/XP 等）
    multi: bool = False       # 是否允许同类多条目（用 entity 子键）

    @property
    def key_prefix(self) -> str:
        """该字段在 facts 表里的 key 前缀。"""
        return f"{self.module.lower()}:{self.field}"


# =============================================================================
# 14 模块画像 Schema（对应需求表）
# =============================================================================
SCHEMA: List[Field] = [
    # ---- Identity ----
    Field("identity", "nickname", "nickname / what to call them", "Mike", "low", True),
    Field("identity", "age", "age", "26", "low", True),
    Field("identity", "region", "region / location", "Tokyo", "low", True),
    Field("identity", "job", "occupation", "programmer", "low", True),
    Field("identity", "language", "languages used", "bilingual JP/EN", "low", True),
    Field("identity", "social_platform", "preferred social platforms", "uses Discord a lot", "mid", True, multi=True),

    # ---- Personality ----
    Field("personality", "trait", "personality trait", "tends to overthink", "mid", True, multi=True),
    Field("personality", "expression_style", "expression style", "dislikes small talk", "mid", True),
    Field("personality", "social_tendency", "social tendency (proactive/passive)", "passive", "mid", True),

    # ---- Preference (aesthetic / AI-interaction prefs that are NOT interest domains) ----
    Field("preference", "aesthetic", "aesthetic preference", "prefers cool tones", "mid", True, multi=True),
    Field("preference", "ai_interaction", "AI interaction preference (friend/lover)", "likes flirty conversation", "mid", True),

    # ---- Behavior ----
    Field("behavior", "routine", "daily routine", "active late at night", "high", True),
    Field("behavior", "messaging", "messaging behavior", "often disappears for days", "high", True),
    Field("behavior", "conflict", "conflict behavior", "goes silent when angry", "mid", True),

    # ---- Emotional ----
    Field("emotional", "pattern", "emotional pattern", "fears abandonment", "mid", True),
    Field("emotional", "stressor", "stressor", "fears losing their job", "mid", True, multi=True),
    Field("emotional", "security_source", "source of security", "likes being understood", "mid", True, multi=True),
    Field("emotional", "trigger", "emotional trigger", "hurt by coldness", "high", True, multi=True),

    # ---- Relationship ----
    Field("relationship", "family", "family relationship", "poor relationship with father", "mid", True, multi=True),
    Field("relationship", "romance", "romantic relationship", "just broke up", "high", True),
    Field("relationship", "social_circle", "social circle", "has a regular gaming group", "mid", True, multi=True),
    Field("relationship", "ai_relation", "how the user views the AI", "treats it as a lover", "high", True),

    # ---- Timeline ----
    Field("timeline", "life_event", "life event", "just started a company", "mid", True, multi=True),
    Field("timeline", "recent_event", "recent event", "had insomnia last week", "high", False, multi=True),

    # ---- Goal ----
    Field("goal", "long_term", "long-term goal", "wants to make an indie game", "mid", True, multi=True),
    Field("goal", "short_term", "short-term goal", "JLPT exam next week", "high", False, multi=True),

    # ---- Values ----
    Field("values", "value", "core value", "values respect highly", "low", True, multi=True),
    Field("values", "moral_boundary", "moral boundary", "does not accept dishonesty", "low", True, multi=True),

    # ---- Interest (single broad module, split by <domain> sub-key, fits any field) ----
    # key looks like interest:<domain>:<specific>; pick one domain below; specific item in the last segment.
    # e.g. interest:game:rimworld, interest:music:jazz, interest:anime:cyberpunk_edgerunners,
    #      interest:film:nolan, interest:book:scifi, interest:sport:climbing, interest:other:worldbuilding
    Field("interest", "game", "games (genre/specific title)", "RimWorld, CRPG", "mid", True, multi=True),
    Field("interest", "music", "music (style/artist)", "jazz, post-rock", "mid", True, multi=True),
    Field("interest", "film", "film/TV (movie/show/director)", "Nolan, cyberpunk", "mid", True, multi=True),
    Field("interest", "anime", "anime / otaku culture", "EVA, Edgerunners", "mid", True, multi=True),
    Field("interest", "book", "books / reading", "sci-fi, philosophy", "mid", True, multi=True),
    Field("interest", "sport", "sports / fitness", "climbing, gym", "mid", True, multi=True),
    Field("interest", "other", "other interests (none of the above)", "worldbuilding, painting", "mid", True, multi=True),

    # ---- NSFW (adult preferences consolidated here) ----
    Field("nsfw", "orientation", "sexual orientation", "bisexual", "low", True, sensitive=True),
    Field("nsfw", "xp", "kink / XP / fetish (e.g. SM, yandere, training)", "likes SM", "mid", True, sensitive=True, multi=True),
    Field("nsfw", "content_pref", "adult content preference (vanilla/dominance/plot-driven)", "prefers vanilla", "mid", True, sensitive=True, multi=True),
    Field("nsfw", "intimacy_pref", "intimacy preference", "likes a clingy vibe", "mid", True, sensitive=True, multi=True),
    Field("nsfw", "boundary", "hard limits / turn-offs", "dislikes humiliation play", "mid", True, sensitive=True, multi=True),
    Field("nsfw", "intensity", "adult content intensity", "ok with explicit chat", "mid", True, sensitive=True),

    # Note: the former Memory module (shared experiences / high-emotion / intimate events) are
    # ONE-TIME EVENTS, not stable profile attributes; moved to L2 episodes (with sensitive flag),
    # not part of this schema.

    # ---- Conversation ----
    Field("conversation", "catchphrase", "catchphrase / frequent words", "lmao", "high", False, multi=True),
    Field("conversation", "language_habit", "language habit (abbrev/tone)", "rarely uses periods", "high", False),
    Field("conversation", "meme", "meme culture", "likes internet memes", "high", False, multi=True),
]


# 合法模块集合（小写），供 key 校验用
MODULES: Dict[str, List[Field]] = {}
for f in SCHEMA:
    MODULES.setdefault(f.module.lower(), []).append(f)

# 合法字段前缀集合（module:field），供 _clean_key 白名单校验用
VALID_FIELD_PREFIXES = {f.key_prefix for f in SCHEMA}

# 高敏感字段前缀集合，用于打 sensitive 标记
SENSITIVE_PREFIXES = {f.key_prefix for f in SCHEMA if f.sensitive}

# 多条目字段前缀集合（允许 module:field:entity）
MULTI_PREFIXES = {f.key_prefix for f in SCHEMA if f.multi}

# 天然单值字段：同一时刻只可能有一个值，新值应覆盖旧值（不追加）。
# 设计原则：默认所有画像都可“追加并存”，只有这些本质单值的属性才覆盖。
# 注意：这是“覆盖”的白名单，白名单之外一律走追加（自动补 entity）。
SINGLE_VALUE_PREFIXES = {
    "identity:nickname", "identity:age", "identity:region", "identity:job",
    "identity:language",
    "personality:expression_style", "personality:social_tendency",
    "preference:ai_interaction",
    "behavior:routine", "behavior:messaging", "behavior:conflict",
    "emotional:pattern",
    "relationship:romance", "relationship:ai_relation",
    "nsfw:orientation", "nsfw:intensity",
    "conversation:language_habit",
}


def is_single_value_key(key: str) -> bool:
    """key 是否为天然单值字段（新值覆盖旧值）。其余一律追加并存。"""
    parts = key.split(":")
    if len(parts) < 2:
        return True  # 裸 module 名当作单值，安全兜底
    return f"{parts[0]}:{parts[1]}" in SINGLE_VALUE_PREFIXES


def is_sensitive_key(key: str) -> bool:
    """key 是否属于高敏感字段（按 module:field 前缀匹配）。"""
    parts = key.split(":")
    if len(parts) < 2:
        return False
    return f"{parts[0]}:{parts[1]}" in SENSITIVE_PREFIXES


def is_multi_key(key: str) -> bool:
    """key 是否属于多条目字段（同类可并存，需 module:field:entity 形式）。"""
    parts = key.split(":")
    if len(parts) < 2:
        return False
    return f"{parts[0]}:{parts[1]}" in MULTI_PREFIXES


def prompt_field_catalog(include_sensitive: bool = True) -> str:
    """生成给抽取 LLM 的字段目录（紧凑版），作为可选 key 的白名单提示。"""
    lines = []
    cur_mod = None
    for f in SCHEMA:
        if not include_sensitive and f.sensitive:
            continue
        if f.module != cur_mod:
            cur_mod = f.module
            lines.append(f"\n[{f.module}]")
        suffix = ":<entity>" if f.multi else ""
        flag = " (sensitive)" if f.sensitive else ""
        lines.append(f"- {f.key_prefix}{suffix} : {f.desc} (e.g. {f.example}){flag}")
    return "\n".join(lines).strip()
