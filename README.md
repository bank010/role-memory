# 角色扮演记忆系统 · Demo

面向**全球多语言 AI 社交陪伴产品**的角色扮演长期记忆系统。把记忆拆成分层结构，
写路径异步加工、读路径毫秒拼装，按 **用户 × 角色** 完全隔离存储，并用一个可视化界面
让你**看见记忆是怎么长出来、又怎么被召回**的。

## 核心特性

- **分层记忆模型**：人设 / 结构化画像（事实）/ 关系状态 / 情节 / 逐字原话，各司其职。
- **用户 × 角色隔离**：每个 `(user_id, role_id)` 组合是一段完全独立的记忆（独立画像、关系、事件）。同一用户对不同角色互不串扰，支持 1 对多 / 多对多。
- **多语言原生**：用 Qwen3-Embedding 直接对原文向量化，无需翻译，跨语言召回（中/英/日…）。
- **两阶段检索**：向量粗召回 → Qwen3-Reranker 精排，显著提升相关性。
- **结构化画像**：14 模块 schema 引导抽取，**默认可追加**（同类多值并存），仅天然单值属性（年龄/职业/性取向等）覆盖更新。
- **真名存储**：情节/画像/摘要直接用真实角色名/用户名写入，展示与检索都自然。
- **NSFW 分级**：敏感画像与事件打 `sensitive` 标记，`NSFW_ENABLED` 总开关控制提取与注入。
- **主动推进剧情**：注入引导让 AI 不被动应答，每轮推进情节并给出明确钩子。
- **上线加固**：session 级加工锁（防并发竞态）、加工失败不丢记忆、94 项回归测试。

## 记忆分层

| 层 | 内容 | 存储 |
|---|---|---|
| L0 人设 | 角色是谁（静态，含 `{{char}}`/`{{user}}` 占位符） | `app/personas.py` |
| L1 结构化画像 | 用户事实/偏好（14 模块 schema，带置信度、可追加） | `facts` 表 |
| L1 关系状态 | 亲密度/信任/情绪/滚动摘要 | `relationship` 表 |
| L2 情节记忆 | 发生过的事（事件+向量，三维打分召回，带 `sensitive`） | `episodes` 表 |
| L3 逐字记忆 | 每轮原话（向量+关键词混合检索，管精确细节） | `chunks` 表 |
| 真相源 | 原始对话日志（append-only，可重建一切） | `turns` 表 |

- **读路径**（在线、快）：`assembler.build_context` 查缓存并拼装上下文，不做 LLM 计算；注入时把记忆里的占位符/真名对齐到当前会话。
- **写路径**（离线、异步）：`pipeline.maybe_process` 每 `PROCESS_EVERY` 轮触发事实抽取 / 情节归纳 / 关系更新 / 滚动摘要 / 反思。**session 级锁保证同会话串行，加工失败不推进进度（下次重试不丢记忆）**。
- **检索打分**：情节按 `relevance × recency × importance` 三维打分，再经 reranker 精排；逐字按 `向量 + 关键词` 混合。

## 数据隔离：用户 × 角色

所有表都带 `user_id` / `role_id` 独立列 + 索引，内部以 `session = user_id␟role_id` 作主键。

```
用户 U1 ──┬── 角色 A：独立画像 + 关系 + 事件
          └── 角色 B：另一套，互不干扰
用户 U2 ──── 角色 A：又是独立的一套
```

好处：记忆隔离正确、检索候选集更小（更快）、支持运营查询（某用户的全部角色 / 某角色的全部用户）。

## 存储后端（可插拔）

业务层 `app/memory/stores.py` 只面向 `app/store/base.py` 接口编程，换后端只改环境变量 `STORE_BACKEND`：

| 后端 | 适用 | 向量 | 召回 |
|---|---|---|---|
| `sqlite`（默认） | 零依赖、开箱跑 | float32 blob | 内存内三维打分 |
| `postgres` | 生产级 | pgvector `vector` 列 + HNSW 索引 | 原生 KNN 预筛 + 三维重排 |

**Redis 热缓存**（`app/cache.py`）：缓存 `facts` / `relationship` 这两条对话主链路上的热读，
读穿透 + 写失效 + TTL 兜底。**Redis 未配置或不可用时自动降级为直查后端，业务无感**（旁路优化，绝不成为故障点）。

## 运行

```bash
cd role-memory-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 配置模型端点（不配 CHAT_API_KEY 则自动进 mock 模式，照样能跑）
cp .env.example .env   # 然后按需填写下方环境变量

uvicorn app.main:app --reload --port 8011
```

打开 http://localhost:8011

### 关键环境变量

```bash
# 对话端点（角色扮演回复）
CHAT_API_KEY=...        CHAT_BASE_URL=...        CHAT_MODEL=...
# 抽取/摘要端点（可与对话分开：对话放得开，抽取要稳）。留空则复用对话端点
EXTRACT_API_KEY=...     EXTRACT_BASE_URL=...     EXTRACT_MODEL=...   EXTRACT_JSON_MODE=1
# 多语言向量端点（Qwen3-Embedding via vLLM）
EMBED_API_KEY=vllm      EMBED_BASE_URL=.../v1    EMBED_MODEL=...     EMBED_DIM=4096
# 精排端点（Qwen3-Reranker via vLLM）。留空则关闭，检索退回纯向量召回
RERANK_BASE_URL=.../v1  RERANK_MODEL=...         RERANK_API_KEY=vllm
# 记忆参数
WORKING_WINDOW=6  PROCESS_EVERY=3  RETRIEVE_TOP_K=4  RECENCY_DECAY=0.02
# 开关
NSFW_ENABLED=1          # 敏感画像/事件的提取与注入总开关
NORMALIZE_ENABLED=0     # 用多语言 embedding 时关闭（直接 embed 原文，省一次翻译）
```

### 切到生产级存储（Postgres + Redis）

```bash
docker compose up -d        # 拉起 pgvector + redis
# 在 .env 写：
#   STORE_BACKEND=postgres
#   PG_DSN=postgresql://memory:memory@localhost:5432/role_memory
#   REDIS_URL=redis://localhost:6379/0
uvicorn app.main:app --port 8011
```

业务代码零改动，启动日志会打印 `store=postgres | cache=True`。

## 两种模式

- **真实模式**：配置了 `CHAT_API_KEY`（OpenAI / DeepSeek / 任意兼容网关）。抽取/摘要/回复都由模型完成。
- **Mock 模式**：不配 key。正则抽取 + 模板回复 + 本地哈希向量。回复很"傻"，但**记忆的抽取/存储/召回/关系演进机制完全真实可见**，适合先看架构跑通。

## API

| 端点 | 说明 |
|---|---|
| `POST /api/chat` | 聊天，参数含 `user_id` / `role_id` / `message` / `persona_id` / `char_name` / `user_name` |
| `GET /api/memory` | 查某 `(user_id, role_id)` 的画像 / 情节 / 关系 |
| `GET /api/history` | 恢复最近 N 轮对话（刷新页面用） |
| `POST /api/reprocess` | 重置加工进度，用新逻辑从头重抽（prompt 改动后补救旧 session） |
| `POST /api/reset` | 清空该会话记忆 |
| `GET /api/health` | 健康检查（含各后端/精排开关状态） |

## 测试

```bash
pytest -q        # 离线运行（强制本地哈希向量，不打真实网络），全套约 1 秒
```

覆盖：key 归一化 / entity 派生 / 追加 vs 覆盖 / 三维打分 / 情节去重 / (用户×角色) 隔离 / session 加工锁串行性 / 校验兜底。
需要 Postgres 集成测试时设 `RUN_PG_TESTS=1`。

## 目录

```
app/
  main.py            FastAPI：/api/chat /api/memory /api/history /api/reprocess /api/reset /api/health
  config.py          配置（环境变量，含后端/缓存/NSFW/归一化开关）
  schemas.py         请求体模型（含 user_id/role_id）
  session.py         会话标识：(user_id, role_id) <-> session 组装/拆解
  llm.py             LLM 客户端（对话 + 抽取双端点，OpenAI 兼容 + mock）
  embeddings.py      向量（Qwen3-Embedding + 本地哈希降级）
  rerank.py          Qwen3-Reranker 精排（封装官方 chat 模板）
  normalizer.py      多语言归一化（可选，默认关）
  personas.py        L0 人设（自动加载 RolePrompts）
  cache.py           Redis 热缓存（读穿透/写失效/优雅降级）
  store/             ★ 存储后端（仓储模式，可插拔，均带 user_id/role_id 列+索引）
    base.py          后端接口（ABC）
    sqlite_store.py  SQLite 实现（默认）
    postgres_store.py PostgreSQL + pgvector 实现（生产级）
    __init__.py      工厂：按 STORE_BACKEND 选实现
  memory/
    profile_schema.py 14 模块结构化画像 schema（多值/敏感/单值定义）
    stores.py        各类记忆【编排层】（向量化/去重/语义合并/缓存/淘汰）
    retrieval.py     情节三维打分 + 逐字混合检索 + reranker 精排
    assembler.py     上下文拼装 + 占位符填充 + 主动推进剧情引导 + 反幻觉护栏
    pipeline.py      异步记忆加工（抽取/归纳/关系/摘要/反思 + session 锁）
tests/               回归测试（离线、约 1 秒）
static/              可视化前端（用户ID + 角色切换，切换即换记忆）
docker-compose.yml   一键拉起 pgvector + redis
```
