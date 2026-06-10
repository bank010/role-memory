# Role Memory — AI 角色扮演长期记忆系统

面向**全球多语言 AI 社交陪伴产品**的角色扮演长期记忆引擎。

把记忆拆分为五层结构，写路径全程异步加工（不阻塞对话）、读路径毫秒级拼装，
按 **用户 × 角色** 完全隔离存储，配一个可视化 Demo 让你**看见记忆是怎么长出来、又怎么被召回**的。

## Demo 预览

![Demo 截图：左侧对话界面，右侧实时展示情节记忆面板（重要度/情绪/轮次标签）](image/image.png)

> 左侧为对话主界面，右侧**情节记忆（全部）**面板实时展示已抽取的情节记忆条目，每条附带重要度、情绪标签、所在轮次，以及 Insight（反思洞察）注释。底部状态栏显示「每 5 轮触发记忆压缩一次，左侧面板/关系/状态在加工后更新」。

---

> 📊 **想快速看懂每条链路？** 全部流程图见 [docs/FLOW.md](docs/FLOW.md)：
> 系统总览 / 一轮对话全链路 / 读路径拼装 / 两路召回 / 压缩管线 / facts·episode 写入 / 两级锁 / 遗忘治理。

## 目录

- [Demo 预览](#demo-预览)
- [核心特性](#核心特性)
- [架构概览](#架构概览)
- [记忆分层](#记忆分层)
- [一轮对话的完整链路](#一轮对话的完整链路)
- [记忆加工：调度与游标机制](#记忆加工调度与游标机制)
- [检索：三维打分 + 两阶段 Rerank](#检索三维打分--两阶段-rerank)
- [用户画像：14 模块 Schema](#用户画像14-模块-schema)
- [数据隔离：用户 × 角色](#数据隔离用户--角色)
- [存储后端（可插拔）](#存储后端可插拔)
- [快速开始](#快速开始)
- [环境变量完整说明](#环境变量完整说明)
- [生产级部署](#生产级部署)
- [API 接口](#api-接口)
- [目录结构](#目录结构)

---

## 核心特性

| 特性 | 说明 |
|---|---|
| **分层记忆模型** | L0 人设 / L1 画像+关系 / L2 情节 / L3 逐字原话，各层独立演进 |
| **用户 × 角色隔离** | `(user_id, role_id)` 完全隔离，1:N / N:N，独立画像、关系、事件 |
| **多语言原生** | 记忆直接用用户的语言存储（不翻译，省 token、小语种不失真）；Qwen3-Embedding 直接对原文向量化；词法检索语言无关（空格分词语言按词、中/日/泰按字符 bigram） |
| **毫秒级读路径** | query 单次 embed（带 Redis 缓存）+ 各路记忆并行 gather + KNN 候选预筛 + 精排延迟预算，debug 带分段计时 |
| **两阶段检索** | 向量粗召回 → Qwen3-Reranker 精排；精排有严格延迟预算（默认 300ms），超时降级绝不拖垮读路径 |
| **全异步存储** | Postgres 走 AsyncConnectionPool 异步连接池，episodes/chunks 双 HNSW 索引原生 KNN；turn 取号原子化（并发安全） |
| **14 模块结构化画像** | 覆盖身份/性格/兴趣/NSFW 等，默认可追加（同类多值并存）；entity slug 多语言安全（unicode 分词 + hash 兜底，非英文 value 不互相覆盖） |
| **画像体量治理** | 超 `MAX_FACTS` 按"置信度×新近度"淘汰（核心身份不淘汰）；注入超 `FACTS_INJECT_TOP_K` 时按"query 相关性+置信度+新近度"裁剪，重度用户不撑爆 prompt |
| **矛盾事实处理** | 抽取 prompt 强制同 key 更新 + 支持 `op:delete` 撤回（用户改口即遗忘），不留两条互相矛盾的画像 |
| **真实时间感知** | 召回打分叠加按天的时间衰减；注入标签用自然时间（"3 days ago" 而非 "5 turns ago"），离开两周回来角色能自然感知 |
| **指代检索增强** | 短消息（"后来呢？"）自动拼上一轮对话做检索，向量/词法两路都有语境信号 |
| **分层摘要** | 滚动摘要每 ~10 个加工周期归档为 `[chapter]` 情节（可向量召回），长期剧情不被反复压缩丢失 |
| **情节合并不丢信息** | 语义去重命中时保留更详细的事件文本（连同新向量），合并只增不减 |
| **真名存储** | 情节/画像/摘要直接用真实角色名/用户名写入，自然可读 |
| **NSFW 分级** | `sensitive` 标记隔离，`NSFW_ENABLED` 总开关控制提取与注入 |
| **主动推进剧情** | System prompt 内嵌叙事引导，AI 主动制造情节、不被动应答 |
| **并发安全** | 进程内 session 级加工锁 + Redis 分布式锁（多 worker/多实例互斥）；加工失败不推进游标，下次重试不丢记忆 |
| **连接池 + 信号量限流** | 所有外部 API（LLM/Embedding/Rerank）统一 httpx 连接池 + 并发信号量，防止瞬间打爆下游；失败自动重试 |
| **全链路容错** | Embedding/LLM/Rerank 任一环节超时或失败均返回结构化错误（503），不再裸 500；精排超时降级、缓存不可用降级 |
| **可插拔存储** | SQLite（零依赖，WAL 模式）/ PostgreSQL + pgvector + HNSW（生产级），一行配置切换 |
| **Redis 热缓存** | facts/relationship 热读 + 查询 embedding 缓存，未配置时自动降级，绝不成为故障点 |
| **真实时间线注入** | System prompt 注入首次对话时间和消息总数，模型不再编造虚假的相识历史 |
| **CORS 支持** | 内置跨域中间件，前后端分离开箱即用 |

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                         前端 / 调用方                         │
└───────────────────────────┬─────────────────────────────────┘
                            │  POST /api/chat
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI  app/main.py                    │
│                                                              │
│  ┌──────── 读路径（在线，毫秒级）────────────────────────┐   │
│  │  1. assembler.build_context                          │   │
│  │     ├─ L0 人设（前端直传 persona_text）               │   │
│  │     ├─ L1 facts（画像）+ relationship（关系/摘要）    │   │
│  │     ├─ L2 情节向量召回 → Reranker 精排               │   │
│  │     └─ L3 逐字向量+关键词混合召回                     │   │
│  │  2. llm.chat(messages) → reply                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────── 写路径（离线，异步不阻塞）──────────────────┐   │
│  │  3. turns 表 append_turn（原话落库）                 │   │
│  │  4. asyncio.create_task(_index_and_process)         │   │
│  │     ├─ chunks 逐字向量化（每轮）                      │   │
│  │     └─ pipeline.maybe_process（每 N 轮触发）         │   │
│  │         ├─ 事实抽取 → facts 表                       │   │
│  │         ├─ 情节归纳 → episodes 表                    │   │
│  │         ├─ 关系更新 → relationship 表                │   │
│  │         ├─ 滚动摘要 → relationship.summary           │   │
│  │         └─ 反思洞察 → episodes [insight]             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
          SQLite /      Redis 缓存     Embedding /
          Postgres      (可选)         Reranker
         (6 张表)                      (vLLM)
```

---

## 记忆分层

| 层 | 名称 | 内容 | 存储表 | 带向量 | 写入时机 |
|---|---|---|---|---|---|
| L0 | 人设 | 角色系统提示词（静态，含 `{{char}}`/`{{user}}` 占位符） | 前端管理 | — | 每轮注入 |
| L1 | 结构化画像 | 用户事实/偏好，14 模块，带置信度，默认可追加 | `facts` | ✅ | 每 N 轮抽取 |
| L1 | 关系状态 | 亲密度/信任度/情绪/阶段/滚动摘要 | `relationship` | — | 每 N 轮更新 |
| L2 | 情节记忆 | 发生过的事件，带重要度/情绪/敏感标记 | `episodes` | ✅ | 每 N 轮归纳 |
| L3 | 逐字记忆 | 每轮原话，管精确细节 | `chunks` | ✅ | 每轮写入 |
| — | 真相源 | 原始对话日志（append-only，可重建一切） | `turns` | — | 每轮写入 |

### 读路径（在线，每轮，毫秒级预算）

```
用户消息 → query 向量化【一次，带 Redis 缓存】
         → 并行 gather：
             L1 facts + relationship（Redis 读穿透）
             L2 情节 KNN 粗召回(TopK×4) → Reranker 精排(延迟预算内) → TopK
             L3 逐字 KNN 粗召回(top-64) → 向量+词法混合重排 → 精排 → TopK
             工作窗口 recent_turns
         → 拼装 system prompt（L0+L1+L2+L3+对话窗口）
         → LLM 生成回复
```

### 写路径（异步，不阻塞回复）

```
对话结束 → turns 落库（同步，立即）
         → create_task（异步，不等待）：
             逐字向量化 index_chunk（每轮）
             maybe_process（差值 ≥ PROCESS_EVERY=5 才触发压缩）：
               ┌ LLM 抽取 facts + episode + relationship_delta ┐
               └ LLM 更新滚动摘要                              ┘ ← 两路并行
               关系 delta + 新摘要 合并为一次写入
               LLM 反思洞察 / 章节归档（更长周期，按边界穿越触发）
               last_processed 游标推进
```

---

## 一轮对话的完整链路

```
用户发消息
    │
    ▼ [读路径，毫秒级]
① query 向量化（Qwen3-Embedding）
② 情节库粗召回 top-16 → Reranker 精排 → top-4
③ 逐字库粗召回 top-16 → Reranker 精排 → top-4
④ 从 Redis/DB 取 facts + relationship
⑤ 拼装 system prompt
⑥ LLM 生成回复
    │ 立即返回给用户
    │
    ▼ [写路径，后台异步]
⑦ turns 表写入本轮原话
⑧ user/assistant 各 embed 一条 → chunks 表
⑨ max_turn - last_processed ≥ 5 ？
      是 → 加工锁 → 并行两路 LLM：
              抽取 JSON ──→ facts（用户稳定偏好） → facts 表
              │             episode（本批事件）  → episodes 表
              滚动摘要 ──→ 与 relationship delta 合并一次写入
           last_processed = max_turn（成功才推进）
      否 → 结束
```

---

## 记忆加工：调度与游标机制

加工状态不在内存里，而是持久化在数据库 `meta` 表：

```sql
CREATE TABLE meta (
    session TEXT PRIMARY KEY,   -- user_id + role_id 组成的会话键
    last_processed_turn INTEGER DEFAULT 0  -- 已加工到第几轮
);
```

触发判断（两级锁 + 两级检查，防并发重复加工）：

```python
# 锁外预检（避免无谓抢锁）
if max_turn - last_processed < PROCESS_EVERY:
    return

async with session_lock:                  # 进程内锁：同进程并发请求串行
    token = await cache.acquire_lock(...)  # Redis 分布式锁：多 worker/多实例互斥
    if token is None:
        return  # 别的实例正在加工，本次放弃（下一轮触发会补上）
    # 锁内二次确认（可能已被其他协程/实例加工）
    if max_turn - last_processed < PROCESS_EVERY:
        return
    await _process(...)
    last_processed = max_turn  # 成功才推进，失败下次重试
```

> Redis 未配置时分布式锁自动退化为仅进程内锁（单实例部署不受影响）。
> 多 worker（`--workers 4`）或多实例部署务必配置 `REDIS_URL`。

**时间轴示例**（PROCESS_EVERY=5）：

| 轮次 | last_processed | 差值 | 是否加工 |
|---|---|---|---|
| 1 | 0 | 1 | ❌ |
| 4 | 0 | 4 | ❌ |
| **5** | 0 | 5 | ✅ → 推进到 5 |
| 6 | 5 | 1 | ❌ |
| **10** | 5 | 5 | ✅ → 推进到 10 |

---

## 检索：三维打分 + 两阶段 Rerank

### 情节三维打分

```
score = w_r × relevance + w_t × recency + w_i × importance
      = 0.55 × cos_sim(query, episode)
      + 0.20 × exp(-RECENCY_DECAY × turns_ago)
      + 0.25 × (importance / 10)
```

### 两阶段检索流程

```
query → embed（Redis 缓存命中则免远程调用） → 向量粗召回 top-(K×4)
              → RERANK_ENABLED ?
                  是 → Qwen3-Reranker 精排（RERANK_TIMEOUT_MS 预算内）→ top-K
                  否 / 超时 / 失败 → 三维打分排序 → top-K
              → 注入 system prompt
```

Reranker 使用 Qwen3 官方 chat 模板（`rerank.py` 已封装），裸文本调用会得到随机分数。
精排有严格延迟预算（默认 300ms）：在线读路径是毫秒级 SLO，超时立即降级回粗排顺序，
一次网络抖动不会把整条读路径拖到秒级。

### 多语言词法检索

逐字混合检索的词法部分语言无关：
- 空格分词语言（英/俄/韩/阿拉伯等）：按词匹配，整词权重 2.0（强信号：名字、专名）；
- 无空格语言（中文/日文假名/泰文）：按字符 bigram 匹配，bigram 权重 2.0，单字低权重；
- 中文虚词、日文助词等高频停用字权重压到最低，不产生噪声命中。

---

## 用户画像：14 模块 Schema

抽取 LLM 的输出由 `profile_schema.py` 中 44 个字段引导，key 格式：
- 单值字段（新值覆盖）：`module:field`，如 `identity:age`
- 多值字段（同类追加）：`module:field:entity`，如 `nsfw:xp:bondage`

| 模块 | 字段示例 | 特性 |
|---|---|---|
| `identity` | nickname / age / region / job / language | 基础身份 |
| `personality` | trait / expression_style / social_tendency | 性格特征 |
| `preference` | aesthetic / ai_interaction | 偏好 |
| `behavior` | routine / messaging / conflict | 行为习惯 |
| `emotional` | pattern / stressor / trigger | 情绪模式 |
| `relationship` | romance / family / ai_relation | 关系背景 |
| `timeline` | life_event / recent_event | 人生时间线 |
| `goal` | long_term / short_term | 目标 |
| `values` | value / moral_boundary | 价值观 |
| `interest` | game / music / film / anime / book / sport | 兴趣（统一大类） |
| `nsfw` | orientation / xp / content_pref / boundary | 成人偏好（sensitive=true） |
| `conversation` | catchphrase / language_habit / meme | 语言习惯 |

**默认可追加**：任何 2 段 key（`module:field`）不在单值白名单时，自动从 value 派生 entity 变成 3 段（如 `nsfw:xp` + "bondage" → `nsfw:xp:bondage`），多个偏好并存不覆盖。

**单值白名单**（`SINGLE_VALUE_PREFIXES`）：`identity:age`、`identity:job`、`nsfw:orientation` 等天然只有一个值的字段，新值直接覆盖。

---

## 数据隔离：用户 × 角色

所有表都带 `user_id` / `role_id` 独立列 + 组合索引，内部以 `session = user_id\x1frole_id` 为主键。

```
用户 U1 ──┬── 角色 A：独立 facts + relationship + episodes + chunks
          └── 角色 B：完全独立的一套，互不干扰
用户 U2 ────── 角色 A：又是独立的一套
```

**三重好处**：
1. 记忆隔离正确（不同角色不串记忆）
2. 检索候选集更小（只扫当前 session 的数据，毫秒级）
3. 支持运营查询：某用户的全部角色 / 某角色的全部用户

---

## 向量数据库选型

系统的向量检索层采用**仓储模式**（`app/store/base.py` 接口），可按需切换存储后端，无需改业务代码。

### 当前支持后端对比

| 后端 | 适用场景 | 向量存储 | ANN 检索 | 依赖 |
|---|---|---|---|---|
| `sqlite`（默认） | 本地开发 / 小规模 demo | float32 blob | 全量加载 + 内存三维打分 | 零依赖 |
| `postgres` | 生产级 / 多并发 | pgvector `vector(dim)` 列 | HNSW 索引 KNN 预筛 + 三维重排 | pgvector ≥ 0.5 |

```sql
-- pgvector HNSW 索引（启动时自动建）
CREATE INDEX idx_episodes_hnsw ON episodes
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### 其他向量数据库选型参考

| 数据库 | 推荐理由 | 适合规模 | 接入方式 |
|---|---|---|---|
| **pgvector**（当前集成）| 与 PostgreSQL 共存，无额外运维，支持 HNSW/IVFFlat，事务内一致 | 百万向量以内 | `store/postgres_store.py` 已实现 |
| **Qdrant** | 纯向量数据库，过滤检索强（支持 `user_id` payload 过滤），Rust 写性能好，有 Cloud 托管 | 千万级，多租户 | 扩展 `base.py` 实现 `QdrantStore` |
| **Weaviate** | 自带多租户隔离（对应 user×role 模型），支持混合检索（BM25+向量），有 GraphQL 查询 | 千万级，SaaS 场景 | 扩展 `base.py` 实现 `WeaviateStore` |
| **Milvus / Zilliz** | 高并发写入性能极强，分布式架构，适合全量用户画像向量化 | 亿级，高并发写 | 扩展 `base.py` 实现 `MilvusStore` |
| **Chroma** | 轻量本地优先，API 简洁，适合快速原型 | 百万级以内 | 可替换 SQLite 后端 |

> **本项目推荐**：中小规模（单机 < 500 万向量）直接用 **pgvector**，业务 DB 和向量 DB 合一，运维最简；千万级用户规模升级 **Qdrant**，其 payload filter 与本项目的 `user_id/role_id` 隔离模型天然契合。

**Redis 热缓存**（`app/cache.py`）：`facts` / `relationship` 两条主链路热读，读穿透 + 写失效 + TTL 兜底。未配置或连接失败时自动降级为直查后端，绝不成为故障点。

---

## Embedding 与 Reranker 模型选型

### Embedding 模型推荐

| 模型 | 维度 | 语言支持 | 推荐理由 | 部署方式 |
|---|---|---|---|---|
| **Qwen3-Embedding-8B**（当前集成）| 4096（可变） | 中/英/日/韩/多语言 | MTEB 多语言 SOTA，支持可变输出维度（节省存储），无需翻译直接跨语言召回 | vLLM |
| **Qwen3-Embedding-0.6B** | 1024 | 中/英/日/韩 | 轻量版，性能略低但资源消耗低 5-6 倍，适合低成本部署 | vLLM |
| **text-embedding-3-large** | 3072 | 多语言 | OpenAI 托管，无需自部署，多语言质量高 | API 直调 |
| **text-embedding-3-small** | 1536 | 多语言 | OpenAI 最低成本方案 | API 直调 |
| **bge-m3** | 1024 | 100+ 语言 | 开源，支持稠密/稀疏/多粒度三路检索，MTEB 表现优秀 | vLLM / SentenceTransformers |

> **部署命令（Qwen3-Embedding-8B）**：
> ```bash
> vllm serve Qwen/Qwen3-Embedding-8B \
>   --task embed \
>   --hf-overrides '{"is_causal": false}' \
>   --port 8001
> ```
> 设置 `EMBED_BASE_URL=http://host:8001/v1` `EMBED_MODEL=Qwen/Qwen3-Embedding-8B` `EMBED_DIM=4096`

### Reranker 模型推荐

Reranker 接收「query + passage」对，输出 0~1 相关性分数，用于对粗召回结果精排，显著提升最终 top-K 质量。

| 模型 | 参数量 | 推荐理由 | 部署方式 |
|---|---|---|---|
| **Qwen3-Reranker-8B**（当前集成）| 8B | 多语言 Rerank SOTA，中英日效果最佳，必须用官方 chat 模板（`rerank.py` 已封装） | vLLM |
| **Qwen3-Reranker-0.6B** | 0.6B | 速度快成本低，准确率略低于 8B，中低并发场景可用 | vLLM |
| **bge-reranker-v2-m3** | 568M | 开源，多语言，使用 cross-encoder 架构，SentenceTransformers 直接调用 | 本地加载 |
| **BAAI/bge-reranker-large** | 560M | 中英双语质量高，推理速度快 | 本地加载 |
| **Cohere Rerank 3** | 托管 | 无需自部署，多语言 API，延迟稳定 | API 直调 |

> **注意**：Qwen3-Reranker 系列必须通过官方 chat 模板调用（以 yes/no token 概率作为分数），裸文本调用会得到随机结果。本项目的 `app/rerank.py` 已正确封装此逻辑，Reranker 不可用时自动回退到纯向量三维打分，不影响功能。

> **部署命令（Qwen3-Reranker-8B）**：
> ```bash
> vllm serve Qwen/Qwen3-Reranker-8B \
>   --task classify \
>   --hf-overrides '{"is_causal": false, "classifier_from_token": ["yes","no"]}' \
>   --port 8002
> ```
> 设置 `RERANK_BASE_URL=http://host:8002/v1` `RERANK_MODEL=Qwen/Qwen3-Reranker-8B`

---

## 快速开始

### 1. 安装依赖

```bash
git clone https://github.com/bank010/role-memory.git
cd role-memory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 按需填写 .env（不填 CHAT_API_KEY 自动进 mock 模式，记忆机制照样可视化）
```

### 3. 启动服务

```bash
uvicorn app.main:app --reload --port 8011
```

打开 http://localhost:8011，点右上角 **「+ 新建角色」** 写一个角色提示词，就能开始聊了。

### Mock 模式 vs 真实模式

| | Mock 模式 | 真实模式 |
|---|---|---|
| **触发条件** | 不配 `CHAT_API_KEY` | 配置任意 OpenAI 兼容 key |
| **回复质量** | 模板回复，较"傻" | 真实 LLM 生成 |
| **记忆机制** | 正则抽取，完全可见 | LLM 抽取，质量更高 |
| **向量** | 本地哈希（dim=256） | Qwen3-Embedding（dim=4096） |
| **适用** | 看架构、跑通流程 | 体验真实效果 |

---

## 环境变量完整说明

```bash
# ── 对话端点（角色扮演回复）──────────────────────────────────
CHAT_API_KEY=          # 必填（留空=mock 模式）
CHAT_BASE_URL=https://api.openai.com/v1
CHAT_MODEL=gpt-4o-mini

# ── 抽取/摘要端点（建议与对话分开，用更稳定的模型）───────────
# 留空则复用对话端点
EXTRACT_API_KEY=
EXTRACT_BASE_URL=https://api.deepseek.com/v1
EXTRACT_MODEL=deepseek-chat
EXTRACT_JSON_MODE=1    # 官方 DeepSeek/OpenAI 支持，开启提升 JSON 稳定性

# ── 向量端点（Qwen3-Embedding via vLLM）─────────────────────
# 留空则降级为本地哈希向量（功能完整，语义质量低）
EMBED_API_KEY=vllm
EMBED_BASE_URL=https://your-host/v1
EMBED_MODEL=your_embedding_model_path
EMBED_DIM=4096         # Qwen3-Embedding-8B=4096, text-embedding-3-small=1536

# ── Rerank 精排（Qwen3-Reranker via vLLM）───────────────────
# 留空则关闭，检索退回纯向量+三维打分
RERANK_BASE_URL=https://your-host/v1
RERANK_MODEL=your_reranker_model_path
RERANK_API_KEY=vllm
RERANK_TIMEOUT_MS=300  # 精排延迟预算，超时降级回粗排顺序（0=不限制）

# ── 记忆参数 ────────────────────────────────────────────────
WORKING_WINDOW=6       # 对话窗口保留最近多少轮原文
PROCESS_EVERY=5        # 每累计多少轮触发一次记忆压缩（后台异步，抽取/摘要并行）
RETRIEVE_TOP_K=4       # 情节/逐字召回最终 top-K 条数
RECENCY_DECAY=0.02     # 新近度衰减系数（越大越偏向最近）
VERBATIM_CANDIDATES=64 # 逐字召回 KNN 候选条数（postgres 预筛）
MAX_EPISODES=200       # 情节库上限（超限按重要度×新近度淘汰）
MAX_CHUNKS=500         # 逐字库上限（超限淘汰最旧）
FACT_MERGE_THRESHOLD=0.86  # 事实语义合并阈值（同类实体相似度超此值则合并）

# ── 开关 ────────────────────────────────────────────────────
NSFW_ENABLED=1         # 敏感画像/事件提取与注入总开关
NORMALIZE_ENABLED=0    # 多语言归一化（用 Qwen3-Embedding 时关闭）

# ── 连接池与并发控制（面向高并发生产环境）─────────────────────
HTTPX_MAX_CONNECTIONS=500   # 每端点最大 TCP 连接数
HTTPX_MAX_KEEPALIVE=100     # 长连接复用数
HTTPX_CONNECT_TIMEOUT=15    # 建连超时（秒）
HTTPX_READ_TIMEOUT=60       # 读超时（秒）
EMBED_CONCURRENCY=50        # Embedding API 最大并发（信号量限流，防打爆 vLLM）
LLM_CONCURRENCY=100         # LLM API 最大并发
RERANK_CONCURRENCY=30       # Rerank API 最大并发
API_RETRIES=2               # 外部 API 失败重试次数

# ── 存储后端 ─────────────────────────────────────────────────
STORE_BACKEND=sqlite   # sqlite | postgres
PG_DSN=postgresql://memory:memory@localhost:5432/role_memory
PG_POOL_MIN=10         # Postgres 异步连接池下限（高并发建议 ≥10）
PG_POOL_MAX=100        # Postgres 异步连接池上限（5000 并发建议 ≥100）

# ── Redis 热缓存 ─────────────────────────────────────────────
# 启用后同时获得：facts/relationship 热读缓存、查询 embedding 缓存、
# 跨实例记忆加工分布式锁（多 worker 部署强烈建议开启）
# REDIS_URL=redis://localhost:6379/0
CACHE_TTL=600
EMBED_CACHE_TTL=3600        # 查询 embedding 缓存 TTL
REDIS_MAX_CONNECTIONS=200   # Redis 连接池大小
REDIS_SOCKET_TIMEOUT=5      # Redis 读写超时（秒）

# ── CORS（前后端分离）────────────────────────────────────────
# 留空=放行所有来源；生产建议填前端域名白名单
# CORS_ORIGINS=https://app.yourdomain.com,http://localhost:3000
```

---

## 生产级部署

### Step 1：存储层（Postgres + pgvector + Redis）

```bash
docker compose up -d   # 一键拉起 pgvector + redis
```

在 `.env` 中修改：

```bash
STORE_BACKEND=postgres
PG_DSN=postgresql://memory:memory@localhost:5432/role_memory
REDIS_URL=redis://localhost:6379/0
```

业务代码零改动，启动日志打印 `store=postgres | cache=True`。

### Step 2：向量模型（Qwen3-Embedding-8B，vLLM）

```bash
vllm serve Qwen/Qwen3-Embedding-8B \
  --task embed \
  --hf-overrides '{"is_causal": false}' \
  --port 8001
```

```bash
# .env
EMBED_BASE_URL=http://localhost:8001/v1
EMBED_MODEL=Qwen/Qwen3-Embedding-8B
EMBED_API_KEY=vllm
EMBED_DIM=4096
```

> 不想自部署？替换为 `EMBED_BASE_URL=https://api.openai.com/v1` + `EMBED_MODEL=text-embedding-3-large` 即可，无需改代码。

### Step 3：Reranker 精排（Qwen3-Reranker-8B，vLLM）

```bash
vllm serve Qwen/Qwen3-Reranker-8B \
  --task classify \
  --hf-overrides '{"is_causal": false, "classifier_from_token": ["yes","no"]}' \
  --port 8002
```

```bash
# .env
RERANK_BASE_URL=http://localhost:8002/v1
RERANK_MODEL=Qwen/Qwen3-Reranker-8B
RERANK_API_KEY=vllm
```

> 不部署 Reranker 时留空即可，系统自动回退到三维打分排序，功能完整。

### Step 4：启动服务

```bash
uvicorn app.main:app --workers 4 --port 8011
```

### 压力测试

项目自带压力测试脚本（`stress_test.py`），支持自定义并发数和轮数：

```bash
python stress_test.py -c 200           # 200 并发，每用户 1 轮
python stress_test.py -c 1000 -r 3     # 1000 并发，每用户连续 3 轮
```

**实测结果**（SQLite + 远程 vLLM Embedding + BytePlus Ark LLM，单 worker）：

| 并发数 | 成功率 | QPS | 平均延迟 | P95 |
|---|---|---|---|---|
| 200 | 100% | 9.5 | 9.3s | 14.8s |
| 500 | 100% | 8.8 | 10.4s | 17.0s |
| 1000 | 99.7% | 8.3 | 20.9s | 35.4s |

> 瓶颈在远程 LLM/Embedding API 的排队延迟，非应用层。切 Postgres + 多 worker + Embedding 多副本可进一步提升。

### 并发调优指南

| 目标并发 | 建议配置 |
|---|---|
| < 100 | 默认配置即可（SQLite + 单 worker） |
| 100 ~ 500 | 调高 `EMBED_CONCURRENCY=80`，开启 Redis |
| 500 ~ 2000 | 切 Postgres（`PG_POOL_MAX=100`），`--workers 4`，Embedding 多副本 |
| 2000+ | 多实例部署 + PgBouncer + Redis 集群 + LLM/Embedding 服务横向扩展 |

---

## API 接口

完整接口文档见 [`app/API.md`](app/API.md)，以下为速查表：

| 方法 | 端点 | 说明 |
|---|---|---|
| `POST` | `/api/chat` | 发送消息，获取 AI 回复（后台异步触发记忆加工） |
| `GET` | `/api/memory` | 查询用户画像 / 情节库 / 关系状态 |
| `GET` | `/api/history` | 获取最近 N 轮对话记录（刷新恢复用） |
| `POST` | `/api/reprocess` | 重置加工进度，从头重新抽取所有记忆 |
| `POST` | `/api/reset` | 清空该会话全部记忆（不可逆） |
| `GET` | `/api/health` | 健康检查，返回各后端/开关状态 |

---

## 目录结构

```
role-memory/
├── app/
│   ├── main.py              # FastAPI 入口，6 个 API 端点 + CORS + 后台任务管理
│   ├── config.py            # 集中配置（所有环境变量的读取与默认值）
│   ├── schemas.py           # Pydantic 请求体模型
│   ├── session.py           # (user_id, role_id) ↔ session 组装/拆解
│   ├── llm.py               # LLM 客户端（对话+抽取双端点，OpenAI 兼容 + mock）
│   ├── embeddings.py        # 向量化（Qwen3-Embedding + 本地哈希降级）
│   ├── rerank.py            # Qwen3-Reranker 精排（封装官方 chat 模板）
│   ├── normalizer.py        # 多语言归一化（可选，默认关闭）
│   ├── personas.py          # L0 人设（兜底，角色由调用方通过 persona_text 管理）
│   ├── cache.py             # Redis 热缓存（读穿透/写失效/优雅降级）
│   ├── API.md               # 完整服务端接口文档
│   ├── store/
│   │   ├── base.py          # 存储后端抽象接口（ABC）
│   │   ├── sqlite_store.py  # SQLite 实现（默认，零依赖）
│   │   ├── postgres_store.py# PostgreSQL + pgvector 实现（生产级）
│   │   └── __init__.py      # 工厂：按 STORE_BACKEND 选实现
│   └── memory/
│       ├── profile_schema.py# 14 模块画像 Schema（多值/敏感/单值白名单定义）
│       ├── stores.py        # 记忆编排层（向量化/去重/语义合并/缓存/淘汰）
│       ├── retrieval.py     # 三维打分 + 逐字混合检索 + Reranker 精排
│       ├── assembler.py     # 上下文拼装（L0~L3 + 占位符填充 + 剧情引导）
│       └── pipeline.py      # 异步记忆加工管线（抽取/归纳/关系/摘要/反思 + session 锁）
├── static/
│   ├── index.html           # 可视化 Demo 页面
│   ├── app.js               # 前端逻辑（角色管理 localStorage + 记忆面板）
│   └── style.css            # 深色主题样式
├── .env.example             # 环境变量模板（含所有字段说明）
├── docker-compose.yml       # 一键拉起 pgvector + Redis
├── requirements.txt         # Python 依赖
├── stress_test.py           # 压力测试脚本（200~5000 并发，含分段计时）
└── pytest.ini               # 测试配置
```

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bank010/role-memory&type=Date)](https://star-history.com/#bank010/role-memory&Date)
