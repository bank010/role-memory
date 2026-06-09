# Role Memory API 接口文档

> Base URL: `http://your-server:8011`
> 所有接口均支持跨域（CORS），可直接从任意前端域名调用。
> Content-Type: `application/json`

---

## 目录

- [POST /api/chat](#post-apichat) — 发送消息，获取 AI 回复
- [GET /api/memory](#get-apimemory) — 查询用户画像 / 情节 / 关系状态
- [GET /api/history](#get-apihistory) — 获取历史对话记录
- [POST /api/reprocess](#post-apireprocess) — 重新加工记忆
- [POST /api/reset](#post-apireset) — 清空会话记忆
- [GET /api/health](#get-apihealth) — 服务健康检查

---

## 核心概念

### 记忆隔离：user_id × role_id

每个 `(user_id, role_id)` 组合是一段**完全独立**的记忆（独立画像、关系、事件）。  
同一用户对不同角色、不同用户对同一角色，记忆互不干扰，支持 1:N / N:N。

```
用户 U1 ── 角色 A：独立画像 + 关系 + 事件
         └─ 角色 B：另一套，互不干扰
用户 U2 ── 角色 A：又是独立的一套
```

### 角色提示词（persona_text）

角色完全由调用方管理，每次请求通过 `persona_text` 字段直传，支持 `{{char}}` / `{{user}}` 占位符，后端自动替换为真实名字。

---

## POST /api/chat

发送一条用户消息，获取角色 AI 的回复。  
后台异步触发记忆加工（每累计 `PROCESS_EVERY` 轮），不阻塞本次响应。

### Request Body

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `user_id` | string | 否 | 用户唯一标识，留空则自动生成 |
| `role_id` | string | 否 | 角色唯一标识（建议必填，用于记忆隔离） |
| `message` | string | **是** | 用户本轮发送的消息内容 |
| `persona_text` | string | 否 | 角色提示词正文（系统设定），支持 `{{char}}` / `{{user}}` 占位符 |
| `char_name` | string | 否 | 角色真实名字，用于替换 `{{char}}`，并写入记忆 |
| `user_name` | string | 否 | 用户真实名字，用于替换 `{{user}}`，并写入记忆 |
| `session` | string | 否 | 兼容旧调用，直接传内部 session 串（优先使用 user_id+role_id） |

### 示例请求

```bash
curl -X POST http://localhost:8011/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_1001",
    "role_id": "role_vivica",
    "message": "Hi, I am Mike. I love climbing.",
    "persona_text": "You are {{char}}, a warm and patient companion who enjoys chatting with {{user}} about the outdoors.",
    "char_name": "Vivica",
    "user_name": "Mike"
  }'
```

### 成功响应 `200 OK`

```json
{
  "reply": "Hey Mike! Climbing sounds amazing...",
  "turn": 3,
  "debug": {
    "retrieved_episodes": [
      {
        "id": 1,
        "event": "Mike told Vivica he loves climbing.",
        "emotion": "happy",
        "importance": 5,
        "turn": 1,
        "sensitive": false,
        "score": 0.82
      }
    ],
    "retrieved_verbatim": [
      {
        "turn": 1,
        "role": "user",
        "text": "Hi, I am Mike. I love climbing.",
        "score": 0.91
      }
    ],
    "facts_injected": [
      {
        "key": "identity:job",
        "value": "Mike is a programmer.",
        "confidence": 0.9
      }
    ],
    "relationship": {
      "intimacy": 0.20,
      "trust": 0.15,
      "stage": "getting to know each other",
      "mood": "happy",
      "summary": "Mike and Vivica have been chatting about outdoor activities."
    },
    "window_turns": 2,
    "system_prompt": "...完整注入 LLM 的 system prompt..."
  }
}
```

| 字段 | 说明 |
|---|---|
| `reply` | AI 角色本轮回复内容 |
| `turn` | 本轮对话轮次序号（从 1 递增） |
| `debug.retrieved_episodes` | 本轮从情节库召回的相关事件（含打分） |
| `debug.retrieved_verbatim` | 本轮从逐字库召回的原话片段（含打分） |
| `debug.facts_injected` | 本轮注入 LLM 的用户画像事实 |
| `debug.relationship` | 当前关系状态（亲密度/信任度/阶段/情绪） |
| `debug.window_turns` | 对话窗口保留的最近轮数 |
| `debug.system_prompt` | 本轮完整的 system prompt（调试用） |

### 错误响应 `503 Service Unavailable`

模型超时或网络波动时返回（JSON 格式，不会返回裸文本 500）：

```json
{
  "error": "llm_unavailable",
  "message": "模型暂时不可用（超时或网络波动），请稍后重试。",
  "detail": "httpx.ConnectTimeout: ..."
}
```

---

## GET /api/memory

查询指定 `(user_id, role_id)` 的完整记忆状态：用户画像、情节库、关系状态。

### Query Parameters

| 参数 | 类型 | 说明 |
|---|---|---|
| `user_id` | string | 用户标识 |
| `role_id` | string | 角色标识 |

### 示例请求

```bash
curl "http://localhost:8011/api/memory?user_id=u_1001&role_id=role_vivica"
```

### 响应 `200 OK`

```json
{
  "facts": [
    {
      "key": "identity:job",
      "value": "Mike is a programmer.",
      "confidence": 0.9,
      "updated_turn": 3
    },
    {
      "key": "interest:sport:climbing",
      "value": "Mike likes mountain climbing.",
      "confidence": 0.95,
      "updated_turn": 3
    },
    {
      "key": "nsfw:xp:bondage",
      "value": "Mike likes bondage play.",
      "confidence": 0.85,
      "updated_turn": 9,
      "sensitive": true
    }
  ],
  "episodes": [
    {
      "id": 2,
      "event": "Mike and Vivica had their first intimate conversation.",
      "emotion": "excited",
      "importance": 8,
      "turn": 6,
      "sensitive": true,
      "recall_count": 3,
      "last_recalled_turn": 9
    }
  ],
  "relationship": {
    "intimacy": 0.45,
    "trust": 0.38,
    "stage": "becoming close",
    "mood": "playful",
    "summary": "Mike and Vivica have been getting closer through shared activities and intimate conversations."
  },
  "max_turn": 12,
  "last_processed": 12
}
```

| 字段 | 说明 |
|---|---|
| `facts` | 用户结构化画像（14 模块 schema，key 格式 `module:field` 或 `module:field:entity`） |
| `facts[].sensitive` | `true` 表示 NSFW/高敏感字段 |
| `episodes` | 情节记忆库（发生过的事件），按轮次倒序 |
| `episodes[].importance` | 重要度 1~10，影响召回优先级和淘汰顺序 |
| `relationship.intimacy` | 亲密度 0.0~1.0，随对话自动演进 |
| `relationship.trust` | 信任度 0.0~1.0，随对话自动演进 |
| `max_turn` | 该会话当前最大轮次 |
| `last_processed` | 记忆加工已处理到第几轮（`max_turn - last_processed < PROCESS_EVERY` 时会触发下次加工） |

---

## GET /api/history

获取最近 n 轮对话原文，用于页面刷新后恢复聊天记录。

### Query Parameters

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `user_id` | string | — | 用户标识 |
| `role_id` | string | — | 角色标识 |
| `n` | integer | 40 | 获取最近多少轮 |

### 示例请求

```bash
curl "http://localhost:8011/api/history?user_id=u_1001&role_id=role_vivica&n=20"
```

### 响应 `200 OK`

```json
{
  "turns": [
    {
      "turn": 1,
      "user_msg": "Hi, I am Mike. I love climbing.",
      "ai_reply": "Hey Mike! Climbing sounds amazing...",
      "ts": 1749369600.123
    },
    {
      "turn": 2,
      "user_msg": "What do you think about outdoor adventures?",
      "ai_reply": "I think they are wonderful...",
      "ts": 1749369660.456
    }
  ]
}
```

---

## POST /api/reprocess

将记忆加工进度重置为 0，然后从头重新抽取所有历史对话的事实/情节/关系/摘要。  
**适用场景**：角色提示词调整后补救旧会话，或手动触发补全漏掉的记忆。  
此接口是**同步**的，等待加工完成后才返回。

### Request Body

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | string | 用户标识 |
| `role_id` | string | 角色标识 |
| `char_name` | string | 角色名（用于记忆写入，可选） |
| `user_name` | string | 用户名（用于记忆写入，可选） |

### 示例请求

```bash
curl -X POST http://localhost:8011/api/reprocess \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "u_1001",
    "role_id": "role_vivica",
    "char_name": "Vivica",
    "user_name": "Mike"
  }'
```

### 响应 `200 OK`

```json
{
  "ok": true,
  "max_turn": 12,
  "last_processed": 12
}
```

---

## POST /api/reset

清空指定会话的**全部记忆**（画像、情节、关系、对话历史、加工进度）。  
⚠️ 不可逆，谨慎调用。

### Request Body

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | string | 用户标识 |
| `role_id` | string | 角色标识 |

### 示例请求

```bash
curl -X POST http://localhost:8011/api/reset \
  -H "Content-Type: application/json" \
  -d '{ "user_id": "u_1001", "role_id": "role_vivica" }'
```

### 响应 `200 OK`

```json
{ "ok": true }
```

---

## GET /api/health

服务健康检查，返回当前运行模式和各后端状态。

### 示例请求

```bash
curl http://localhost:8011/api/health
```

### 响应 `200 OK`

```json
{
  "mock_mode": false,
  "chat_model": "ep-20260601104633-s7hxl",
  "window": 6,
  "process_every": 3,
  "store_backend": "sqlite",
  "cache_enabled": false,
  "rerank_enabled": true,
  "personas": []
}
```

| 字段 | 说明 |
|---|---|
| `mock_mode` | `true` = 未配置 API Key，使用规则回复 + 本地哈希向量 |
| `chat_model` | 当前对话模型名 |
| `window` | 工作窗口轮数（WORKING_WINDOW） |
| `process_every` | 记忆加工触发间隔轮数（PROCESS_EVERY） |
| `store_backend` | 存储后端：`sqlite` \| `postgres` |
| `cache_enabled` | Redis 热缓存是否启用 |
| `rerank_enabled` | Reranker 精排是否启用 |
| `personas` | 内置角色列表（当前为空，角色由调用方通过 persona_text 管理） |

---

## 服务端集成建议

### 典型调用流程

```
1. 用户发消息
   └─ POST /api/chat  →  获取 reply，存 turn
      └─ 后台自动：每 3 轮触发记忆加工（异步，不影响响应速度）

2. 页面加载 / 切换会话
   ├─ GET /api/history   →  恢复聊天记录
   └─ GET /api/memory    →  展示用户画像 / 情节 / 关系状态

3. 运营/管理
   ├─ POST /api/reprocess  →  重新提取记忆（调整 prompt 后补救）
   └─ POST /api/reset      →  清空会话
```

### 记忆加工触发规则

记忆加工（事实抽取 + 情节归纳 + 关系更新 + 滚动摘要）在后台**异步**执行，触发条件：

```
当前轮次(max_turn) - 上次加工轮次(last_processed) >= PROCESS_EVERY(默认 3)
```

加工期间对话正常进行，加工结果下一轮起生效。可通过 `GET /api/memory` 的 `last_processed` 字段判断加工进度。

### 错误处理建议

| HTTP 状态码 | 含义 | 建议处理 |
|---|---|---|
| `200` | 成功 | 正常解析 JSON |
| `503` + `error: llm_unavailable` | 模型超时/网络波动 | 展示提示，引导用户重试 |
| `422` | 请求参数校验失败 | 检查必填字段（`message` 为必填） |
| `500` | 服务端未知异常 | 记录日志，重试或联系运维 |

### 关系状态参考值

| 阶段(stage) | 亲密度参考范围 | 对角色行为的影响 |
|---|---|---|
| 初识 / getting to know | 0.0 ~ 0.2 | 礼貌、有距离感 |
| 熟悉 / becoming familiar | 0.2 ~ 0.4 | 轻松自然，开始有玩笑 |
| 暧昧 / becoming close | 0.4 ~ 0.6 | 主动靠近，情感流露 |
| 亲密 / intimate | 0.6 ~ 0.8 | 亲昵、敞开内心 |
| 深度依赖 / deeply bonded | 0.8 ~ 1.0 | 高度默契，极度亲密 |
