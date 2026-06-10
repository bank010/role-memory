"""集中配置。对话端点与向量端点解耦：
- 对话(chat)：填了 CHAT_API_KEY 即走真实模型（同时驱动事实抽取、反思、语言归一化）
- 向量(embed)：填了 EMBED_API_KEY 才走真实 embedding，否则用本地哈希向量
没有 chat key → mock 模式。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get(*names, default=""):
    for n in names:
        v = os.getenv(n)
        if v:
            return v.strip()
    return default


# ---- 对话 / 抽取 / 归一化（同一个 chat 端点）----
CHAT_API_KEY = _get("CHAT_API_KEY", "OPENAI_API_KEY")
CHAT_BASE_URL = _get("CHAT_BASE_URL", "OPENAI_BASE_URL", default="https://api.openai.com/v1").rstrip("/")
CHAT_MODEL = _get("CHAT_MODEL", default="gpt-4o-mini")
EXTRACT_MODEL = _get("EXTRACT_MODEL", default=CHAT_MODEL)
# 抽取端点可与对话端点分开（对话放得开用 NSFW 版，抽取要稳用官方 DeepSeek）。
# 留空则回退复用对话端点。
EXTRACT_API_KEY = _get("EXTRACT_API_KEY", default=CHAT_API_KEY)
EXTRACT_BASE_URL = _get("EXTRACT_BASE_URL", default=CHAT_BASE_URL).rstrip("/")
# 原生 JSON 模式（response_format=json_object）。默认关：部分端点（BytePlus Ark 某些模型）不支持会 400。
# 官方 DeepSeek 支持，分开端点后可设 EXTRACT_JSON_MODE=1 提升 JSON 稳定性。
EXTRACT_JSON_MODE = os.getenv("EXTRACT_JSON_MODE", "0") == "1"

# ---- 向量（独立端点，可选）----
EMBED_API_KEY = _get("EMBED_API_KEY")
EMBED_BASE_URL = _get("EMBED_BASE_URL", default=CHAT_BASE_URL).rstrip("/")
EMBED_MODEL = _get("EMBED_MODEL", default="text-embedding-3-small")

# ---- Rerank 精排（独立端点，可选；Qwen3-Reranker via vLLM）----
# 两阶段检索的第二阶段：向量粗召回 top-N → reranker 精排 top-K。
# 注意：Qwen3-Reranker 强依赖官方 chat 模板，rerank.py 已封装；裸文本会近乎随机。
RERANK_BASE_URL = _get("RERANK_BASE_URL").rstrip("/")
RERANK_MODEL = _get("RERANK_MODEL")
RERANK_API_KEY = _get("RERANK_API_KEY", default="vllm")
RERANK_ENABLED = bool(RERANK_BASE_URL and RERANK_MODEL)
# 粗召回候选倍数：实际取 top_k * 此值 条送入精排
RERANK_CANDIDATE_MULT = int(os.getenv("RERANK_CANDIDATE_MULT", "4"))
# 精排延迟预算（毫秒）：超时立即降级回三维打分/向量排序，保证读路径不被精排拖垮。
# 0 = 不限制（不推荐生产使用）。
RERANK_TIMEOUT_MS = int(os.getenv("RERANK_TIMEOUT_MS", "300"))

# 没有 chat key 就进入 mock 模式：规则抽取 + 模板回复
MOCK_MODE = not bool(CHAT_API_KEY)
EMBED_REAL = bool(EMBED_API_KEY)

# 向量维度：Qwen3-Embedding-8B=4096，text-embedding-3-small=1536，本地哈希=256。
# pgvector 列需固定维度，这里据此建表。
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536" if EMBED_REAL else "256"))

# 语言归一化开关：
# - 用多语言 embedding（如 Qwen3-Embedding）时应【关闭】：直接 embed 原文，
#   省掉每条记忆/查询的一次翻译调用，也不丢失语义细节（已实测跨语言召回正常）。
# - 仅当使用单语 embedding 时才开启：先把文本翻译到基准语言再向量化。
NORMALIZE_ENABLED = os.getenv("NORMALIZE_ENABLED", "0") == "1"

# NSFW / 高敏感画像与事件的提取+注入总开关：
# - 开（默认）：照常提取高敏感字段/事件，仅打 sensitive 标记隔离。
# - 关：抽取 prompt 不暴露敏感字段目录，且拼上下文时过滤掉 sensitive 的事实与情节。
NSFW_ENABLED = os.getenv("NSFW_ENABLED", "1") == "1"

# ---- 记忆参数 ----
WORKING_WINDOW = int(os.getenv("WORKING_WINDOW", "6"))
PROCESS_EVERY = int(os.getenv("PROCESS_EVERY", "5"))
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "4"))
RECENCY_DECAY = float(os.getenv("RECENCY_DECAY", "0.02"))
# 逐字召回的 KNN 候选条数（postgres 预筛；候选内再做向量+词法混合重排）
VERBATIM_CANDIDATES = int(os.getenv("VERBATIM_CANDIDATES", "64"))

# 三维打分权重：relevance / recency / importance
SCORE_WEIGHTS = (0.55, 0.20, 0.25)

# ---- 记忆体量上限（遗忘/淘汰）----
# 超过上限时：情节按"重要度 × 新近度"打分淘汰最低分；chunk 按时间淘汰最旧；
# facts 按"置信度 × 新近度"淘汰最低分（单值身份类字段不淘汰）
MAX_EPISODES = int(os.getenv("MAX_EPISODES", "200"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "500"))
MAX_FACTS = int(os.getenv("MAX_FACTS", "150"))

# facts 注入上限：超过此数时按"与当前 query 相关性 + 置信度 + 新近度"选 top-K 注入，
# 防止重度用户的画像撑爆 system prompt（token 成本 + 注意力稀释）
FACTS_INJECT_TOP_K = int(os.getenv("FACTS_INJECT_TOP_K", "30"))

# 真实时间衰减（按天）：召回打分在 turn 衰减外再乘 exp(-此值 × 距今天数)，
# 用户离开两周回来，旧记忆不再"鲜活如昨"
RECENCY_TIME_DECAY = float(os.getenv("RECENCY_TIME_DECAY", "0.01"))

# 检索 query 增强：用户消息短于此字符数时（"那它呢？"这类指代），
# 拼上一轮对话一起做检索，提升指代场景的召回
QUERY_AUGMENT_MAX_LEN = int(os.getenv("QUERY_AUGMENT_MAX_LEN", "24"))

# 事实语义合并阈值：同类别下，实体向量相似度 >= 此值视为"同一条目"，更新而非新增。
# 偏高以避免误合并（如 cilantro 与 shrimp 不应合并）。
FACT_MERGE_THRESHOLD = float(os.getenv("FACT_MERGE_THRESHOLD", "0.86"))

# ---- 连接池（面向 5000+ 并发设计）----
# httpx 连接池：每个外部端点（LLM chat/extract、embedding、rerank）独立池
# max_connections = 单端点最大并发 TCP 连接数；keepalive = 长连接复用数
HTTPX_MAX_CONNECTIONS = int(os.getenv("HTTPX_MAX_CONNECTIONS", "500"))
HTTPX_MAX_KEEPALIVE = int(os.getenv("HTTPX_MAX_KEEPALIVE", "100"))
# 各外部 API 最大并发信号量：防止瞬间打爆下游，超出的排队等待
EMBED_CONCURRENCY = int(os.getenv("EMBED_CONCURRENCY", "50"))
LLM_CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "100"))
RERANK_CONCURRENCY = int(os.getenv("RERANK_CONCURRENCY", "30"))
# 外部调用失败重试次数
API_RETRIES = int(os.getenv("API_RETRIES", "2"))
# httpx 超时（秒）
HTTPX_CONNECT_TIMEOUT = float(os.getenv("HTTPX_CONNECT_TIMEOUT", "15"))
HTTPX_READ_TIMEOUT = float(os.getenv("HTTPX_READ_TIMEOUT", "60"))

# ---- 存储后端 ----
# sqlite  : 零依赖，开箱即跑（demo 默认）
# postgres: 生产级，pgvector 存向量 + 原生 KNN 预筛
STORE_BACKEND = _get("STORE_BACKEND", default="sqlite").lower()
DB_PATH = BASE_DIR / "memory.db"
PG_DSN = _get("PG_DSN", "DATABASE_URL", default="postgresql://localhost:5432/role_memory")
# Postgres 异步连接池：5000 并发建议 pool_max >= 100（配合 PgBouncer 可更小）
PG_POOL_MIN = int(os.getenv("PG_POOL_MIN", "10"))
PG_POOL_MAX = int(os.getenv("PG_POOL_MAX", "100"))

# ---- Redis 热缓存（留空则关闭，自动降级为直查后端）----
REDIS_URL = _get("REDIS_URL")
CACHE_ENABLED = bool(REDIS_URL)
CACHE_TTL = int(os.getenv("CACHE_TTL", "600"))
# 查询 embedding 缓存 TTL（读路径毫秒级的关键之一：重复/相近问法免一次 embed 调用）
EMBED_CACHE_TTL = int(os.getenv("EMBED_CACHE_TTL", "3600"))
# Redis 连接池
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "200"))
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "3"))

# ---- CORS（前后端分离）----
# 逗号分隔的前端域名白名单；留空则放行所有来源（"*"）。
# 例：CORS_ORIGINS=https://app.example.com,http://localhost:3000
CORS_ORIGINS = _get("CORS_ORIGINS", default="")
