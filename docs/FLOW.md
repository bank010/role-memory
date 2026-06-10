# 流程图全集 —— 记忆系统每条链路的可视化

> 配合 [README](../README.md) 阅读。所有图基于当前代码绘制，与实现一一对应。
>
> - 读路径 = 在线、毫秒级预算，用户每发一条消息走一次
> - 写路径 = 后台异步，永不阻塞对话

---

## 目录

1. [系统总览](#1-系统总览)
2. [一轮对话全链路](#2-一轮对话全链路)
3. [读路径：上下文拼装](#3-读路径上下文拼装)
4. [两路召回细节](#4-两路召回细节)
5. [写路径：记忆压缩管线](#5-写路径记忆压缩管线)
6. [facts 写入：清洗 → 派生 → 合并 → 淘汰](#6-facts-写入清洗--派生--合并--淘汰)
7. [episode 写入：语义去重合并](#7-episode-写入语义去重合并)
8. [并发控制：两级锁](#8-并发控制两级锁)
9. [遗忘与体量治理](#9-遗忘与体量治理)

---

## 1. 系统总览

```mermaid
flowchart TB
    subgraph Client["前端 / 调用方"]
        U["用户消息"]
    end

    subgraph Online["在线读路径（毫秒级）"]
        A["assembler.build_context<br/>上下文拼装"]
        R1["retrieval 两路召回"]
        LLM1["对话 LLM<br/>生成回复"]
    end

    subgraph Offline["后台写路径（异步）"]
        IX["index_chunk<br/>逐字向量化"]
        P["pipeline.maybe_process<br/>记忆压缩（每 5 轮）"]
        LLM2["抽取 LLM ∥ 摘要 LLM<br/>两路并行"]
    end

    subgraph Storage["存储层（可插拔）"]
        direction LR
        DB[("SQLite / PostgreSQL+pgvector<br/>turns·chunks·facts·episodes·relationship·meta")]
        RD[("Redis（可选）<br/>热读缓存·embedding 缓存·分布式锁")]
    end

    subgraph Models["模型服务"]
        direction LR
        EMB["Qwen3-Embedding<br/>多语言向量"]
        RRK["Qwen3-Reranker<br/>二阶段精排"]
    end

    U --> A
    A --> R1
    R1 --> LLM1
    LLM1 -->|"回复立即返回"| U
    LLM1 -.->|"create_task 不等待"| IX
    IX -.-> P
    P --> LLM2

    A <--> RD
    A <--> DB
    R1 <--> DB
    R1 --> EMB
    R1 --> RRK
    IX --> EMB
    LLM2 --> DB
```

**分层记忆模型**（每层独立演进，互为补充）：

```mermaid
flowchart LR
    subgraph Layers["记忆分层"]
        direction TB
        L0["L0 人设<br/>角色系统提示词（静态）"]
        L1a["L1 结构化画像 facts<br/>14 模块·带置信度·带向量"]
        L1b["L1 关系状态 relationship<br/>亲密度·信任·情绪·滚动摘要"]
        L2["L2 情节记忆 episodes<br/>事件·重要度·情绪·敏感标记·带向量"]
        L3["L3 逐字记忆 chunks<br/>每轮原话·带向量·管精确细节"]
        T["真相源 turns<br/>append-only 原始日志·可重建一切"]
    end
    L0 ~~~ L1a ~~~ L1b ~~~ L2 ~~~ L3 ~~~ T
```

---

## 2. 一轮对话全链路

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户
    participant API as /api/chat
    participant ASM as assembler（读路径）
    participant RET as retrieval
    participant L as 对话 LLM
    participant BG as 后台 task（写路径）
    participant PIPE as pipeline（压缩）

    U->>API: 发消息
    API->>ASM: build_context
    ASM->>ASM: 并行取 facts / relationship / 工作窗口 / max_turn
    ASM->>ASM: 短消息？→ 拼上一轮对话做检索增强
    ASM->>ASM: query embed 一次（Redis 缓存）
    par 两路召回 + 画像裁剪（并行）
        ASM->>RET: retrieve_episodes（三维打分+精排）
    and
        ASM->>RET: retrieve_verbatim（向量+词法混合+精排）
    and
        ASM->>RET: select_facts（超 30 条按相关性裁剪）
    end
    ASM-->>API: system prompt（人设+画像+关系+情节+逐字+窗口）
    API->>L: 生成回复
    L-->>U: 回复立即返回 ✅
    API->>API: turns 落库（同步，真相源）
    API-)BG: create_task（不等待）
    BG->>BG: index_chunk ×2（user/assistant 原话向量化）
    BG->>PIPE: maybe_process
    alt max_turn - last_processed ≥ 5
        PIPE->>PIPE: 压缩（见第 5 节）
    else 未到触发点
        PIPE->>PIPE: 直接返回
    end
```

---

## 3. 读路径：上下文拼装

`assembler.build_context` —— 毫秒级预算，debug 带分段计时（embed_ms / retrieve_ms / total_ms）。

```mermaid
flowchart TD
    S(["用户消息进入"]) --> P1

    subgraph P1["① 轻量数据并行读（asyncio.gather）"]
        direction LR
        F["all_facts<br/>（Redis 热读）"]
        REL["get_relationship<br/>（Redis 热读）"]
        W["recent_turns<br/>工作窗口"]
        MT["max_turn"]
    end

    P1 --> Q{"消息信息量<br/>< 24？<br/>（CJK 按 2 计）"}
    Q -->|"是（如「后来呢？」）"| AUG["拼上一轮 user_msg + ai_reply<br/>补全指代语境"]
    Q -->|否| RAW["原消息作为检索 query"]
    AUG --> E
    RAW --> E

    E["query embed 一次<br/>（Redis embedding 缓存，重复问法免远程调用）"]

    E --> P2
    subgraph P2["② 召回 + 裁剪并行（asyncio.gather，共享 qvec）"]
        direction LR
        EP["retrieve_episodes<br/>情节召回"]
        VB["retrieve_verbatim<br/>逐字召回"]
        SF["select_facts<br/>画像超 30 条按<br/>相关性+置信度+新近度裁剪<br/>（身份字段保底）"]
    end

    P2 --> NS{"NSFW_ENABLED?"}
    NS -->|关| FIL["过滤 sensitive 画像/情节"]
    NS -->|开| ASM2
    FIL --> ASM2

    ASM2["拼装 system prompt"] --> OUT(["messages + debug(timing_ms)"])

    ASM2 -.- NOTE["注入顺序：<br/>人设 → 关系状态 → 画像 → 滚动摘要<br/>→ 相关情节（自然时间标签）<br/>→ 逐字引用 → 行为准则"]
```

---

## 4. 两路召回细节

### 4a. 情节召回（retrieve_episodes）

```mermaid
flowchart TD
    Q(["qvec（调用方传入，不重复 embed）"]) --> C

    C{"存储后端？"}
    C -->|PostgreSQL| KNN["pgvector HNSW KNN<br/>库内预筛 top-N 候选"]
    C -->|SQLite| ALL["全量加载（demo 规模）"]
    KNN --> SCORE
    ALL --> SCORE

    SCORE["三维打分：<br/>score = 0.55×relevance + 0.25×recency + 0.20×importance<br/><br/>recency = exp(-0.02×轮次差) × exp(-0.01×距今天数)<br/>（轮次衰减 × 真实时间衰减）"]

    SCORE --> SORT["粗排取 top-16"]
    SORT --> RR{"Reranker 启用？"}
    RR -->|是| RERANK["Qwen3-Reranker 精排<br/>⏱️ 预算 300ms，超时降级回粗排"]
    RR -->|否| TOPK
    RERANK --> TOPK["取 top-4"]

    TOPK --> MARK["mark_recalled 丢后台 task<br/>（统计性写，不占读路径）"]
    TOPK --> OUT(["情节列表（含 ts，注入时显示自然时间）"])
```

### 4b. 逐字召回（retrieve_verbatim）—— 混合检索

```mermaid
flowchart TD
    Q(["qvec + 原 query 文本"]) --> C["candidate_chunks<br/>（pg: KNN 预筛 64 条；sqlite: 全量）"]

    C --> EX["排除工作窗口内的近期轮次<br/>（窗口里已有原文，避免重复注入）"]

    EX --> H["混合打分：score = 0.7×向量相似 + 0.3×词法匹配"]

    H --> LEX["词法分词（语言无关）：<br/>· 空格分词语言（英/俄/阿/韩）→ 按词<br/>· 连写语言（中/日/泰）→ 字符 bigram<br/>· 多语言停用词过滤"]
    LEX --> H

    H --> SORT["粗排"] --> RR["Reranker 精排（同预算）"] --> OUT(["top-6 逐字引用<br/>管名字/数字/专名等精确细节"])
```

---

## 5. 写路径：记忆压缩管线

`pipeline.maybe_process` → `_process`，整体在后台 task 中异步执行。

```mermaid
flowchart TD
    T(["后台 task 触发"]) --> PRE{"锁外预检：<br/>max_turn - last_processed ≥ 5？"}
    PRE -->|否| END1(["返回（未到触发点）"])
    PRE -->|是| LK1["进程内 session 锁<br/>（asyncio.Lock，同进程串行）"]

    LK1 --> LK2{"Redis 分布式锁<br/>SET NX，TTL 120s"}
    LK2 -->|"抢不到"| END2(["返回（别的实例在加工）"])
    LK2 -->|"抢到"| RECHECK{"锁内二次确认：<br/>差值仍 ≥ 5？"}
    RECHECK -->|否| REL1["释放锁"] --> END3(["返回（已被处理）"])
    RECHECK -->|是| PROC

    subgraph PROC["_process：单次压缩"]
        direction TB
        READ["并行读：turns_after（本批对话）<br/>+ all_facts + relationship（旧摘要）"]

        READ --> PAR
        subgraph PAR["两路 LLM 并行（asyncio.gather）<br/>墙钟 ≈ 1 次 LLM 调用"]
            direction LR
            EXT["结构化抽取<br/>facts + episode + rel_delta<br/>（JSON 模式）"]
            SUM["滚动摘要<br/>旧摘要 + 本批对话<br/>→ 120 词新摘要"]
        end

        PAR --> CH{"跨过 50 轮边界？<br/>（_crossed 边界穿越判断）"}
        CH -->|是| ARCH["旧摘要归档为 [chapter] 情节<br/>重要度 7，可被向量召回"]
        CH -->|否| FACTS
        ARCH --> FACTS

        FACTS["facts 落库<br/>（清洗/派生/合并/淘汰，见第 6 节）"]
        FACTS --> EPI["episode 落库<br/>（语义去重合并，见第 7 节）"]
        EPI --> RELW["relationship 一次读-改-写：<br/>亲密度/信任 delta + stage/mood + 新摘要<br/>（合并写入，防并行双写丢更新）"]
        RELW --> RFL{"跨过 15 轮边界<br/>且情节 ≥ 10 条？"}
        RFL -->|是| INS["反思 LLM → [insight] 情节<br/>高层洞察，重要度 8+"]
        RFL -->|否| DONE2["压缩完成"]
        INS --> DONE2
    end

    PROC -->|"全程成功"| ADV["last_processed = max_turn<br/>✅ 游标推进"]
    PROC -->|"任何异常"| NOADV["❌ 游标不动<br/>只记日志，下次重试不丢记忆"]
    ADV --> REL2["释放 Redis 锁"]
    NOADV --> REL2
    REL2 --> END4(["结束"])
```

---

## 6. facts 写入：清洗 → 派生 → 合并 → 淘汰

每条抽取出的 fact 走这条流水线（`pipeline` 校验 + `stores.upsert_fact`）。

```mermaid
flowchart TD
    F(["LLM 抽出的一条 fact<br/>{key, value, confidence, op?}"]) --> OP{"op == delete？"}

    OP -->|"是"| KNOWN{"key 在已知<br/>facts 里？"}
    KNOWN -->|是| DEL["删除该事实<br/>（用户改口撤回）"] --> ENDD(["完成"])
    KNOWN -->|"否（防 LLM 幻觉误删）"| SKIP1(["忽略"])

    OP -->|否| CLEAN["_clean_key：小写/截断/白名单校验<br/>category 别名归一（preference→interest 等）"]
    CLEAN --> VALID{"key/value 合法？"}
    VALID -->|否| SKIP2(["丢弃 + 记日志"])
    VALID -->|是| SV{"单值字段？<br/>（identity:job 等白名单）"}

    SV -->|"是 → 新值覆盖旧值"| UPSERT
    SV -->|"否 → 多值可并存"| ENT{"key 已带 entity？<br/>module:field:entity"}
    ENT -->|是| UPSERT
    ENT -->|否| SLUG["_slug_from_value 派生 entity：<br/>① unicode 分词去停用词取尾部<br/>　（中文 value → 中文 slug，多语言安全）<br/>② 完全提不出 → sha1 前 10 位兜底<br/>　（确定且唯一，绝不互相覆盖）"]
    SLUG --> UPSERT

    UPSERT["stores.upsert_fact"] --> MERGE{"同 category 下存在<br/>entity 向量相似度 ≥ 0.86<br/>的旧事实？"}
    MERGE -->|"是（cilantro≈coriander）"| SAME["写入旧 key（语义合并）<br/>值更新，置信度取较大"]
    MERGE -->|否| NEW["写入新 key"]

    SAME --> EVICT
    NEW --> EVICT
    EVICT{"总数 > MAX_FACTS<br/>（150）？"}
    EVICT -->|是| KICK["按 置信度×新近度 淘汰最低分<br/>（单值身份字段豁免）"]
    EVICT -->|否| ENDF(["完成 + 缓存失效"])
    KICK --> ENDF
```

---

## 7. episode 写入：语义去重合并

`stores.add_episode` —— 合并只增不减，不丢信息。

```mermaid
flowchart TD
    E(["新情节 event + importance"]) --> EMB["embed(event)"]
    EMB --> KNN["取最相似的 8 条候选<br/>（pg: KNN / sqlite: 全量）"]
    KNN --> SIM{"最高相似度<br/>≥ 去重阈值？"}

    SIM -->|"否"| INS["插入新情节"] --> EV{"总数 > MAX_EPISODES<br/>（200）？"}
    EV -->|是| KICK["按 重要度×新近度<br/>淘汰最低分"] --> DONE(["完成"])
    EV -->|否| DONE

    SIM -->|"是（重复事件）"| LEN{"新文本比旧文本<br/>更详细（更长）？"}
    LEN -->|是| RICH["更新为新文本 + 新向量<br/>importance 取较大，turn/ts 刷新"]
    LEN -->|否| KEEP["保留旧文本<br/>仅刷新 importance/turn/ts"]
    RICH --> DONE
    KEEP --> DONE
```

---

## 8. 并发控制：两级锁

多 worker / 多实例部署下，同一 session 的压缩绝不并发执行。

```mermaid
flowchart LR
    subgraph I1["实例 A"]
        W1["worker 协程 1"]
        W2["worker 协程 2"]
        LA["进程内 asyncio.Lock<br/>（per-session，WeakValueDict）"]
        W1 --> LA
        W2 --> LA
    end

    subgraph I2["实例 B"]
        W3["worker 协程 3"]
        LB["进程内 asyncio.Lock"]
        W3 --> LB
    end

    LA --> RL{{"Redis 分布式锁<br/>SET NX process:&lt;session&gt;<br/>TTL 120s + token 校验释放"}}
    LB --> RL
    RL -->|"唯一持有者"| GO["执行压缩"]
    RL -->|"未抢到"| GIVEUP["本次放弃<br/>（游标未动，下一轮触发补上）"]
```

> Redis 未配置时自动退化为仅进程内锁（单实例部署不受影响）；
> 加工失败不推进游标 → 天然的 at-least-once 重试语义。

---

## 9. 遗忘与体量治理

| 对象 | 上限 | 淘汰策略 | 豁免 |
|---|---|---|---|
| facts 画像 | `MAX_FACTS=150` | 置信度 × 轮次新近度，最低分先淘汰 | 单值身份字段（姓名/年龄/职业等） |
| episodes 情节 | `MAX_EPISODES=200` | 重要度 × 新近度，最低分先淘汰 | — |
| chunks 逐字 | `MAX_CHUNKS=500` | 按时间淘汰最旧 | — |
| 滚动摘要 | 120 词 | 每次压缩覆盖重写 | 每 ~50 轮归档为 `[chapter]` 情节 |

**注入端预算**（防止重度用户撑爆 prompt）：

| 注入内容 | 预算 | 超限策略 |
|---|---|---|
| facts | `FACTS_INJECT_TOP_K=30` | 按 query 相关性 + 置信度 + 新近度选 top-K，身份字段保底 |
| episodes | `RETRIEVE_TOP_K=4` | 三维打分 + 精排 |
| verbatim | `RETRIEVE_TOP_K+2=6` | 混合打分 + 精排 |
| 工作窗口 | `WORKING_WINDOW=6` 轮 | 滚动窗口 |
