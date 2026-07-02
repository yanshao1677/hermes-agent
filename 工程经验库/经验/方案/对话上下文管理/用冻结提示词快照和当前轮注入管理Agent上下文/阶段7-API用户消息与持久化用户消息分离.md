---
title: 阶段7-API用户消息与持久化用户消息分离
level: atomic
parent: 用冻结提示词快照和当前轮注入管理Agent上下文
status: draft
tags:
  - ai-agent
  - persistence
  - gateway
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段9-只持久化真实历史并用刷新索引避免重复写入.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段7：API 用户消息与持久化用户消息分离

## 触发条件

调用 `run_conversation()` 时，调用方传入可选参数 `persist_user_message`（已清洗、不含仅 API 使用的前缀，例如技能斜杠命令展开前的原文）。

## 输入字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_message` | `str` | 写入真实历史并送模型的 API 用户内容（可含技能注入） |
| `persist_user_message` | `str \| None` | 持久化/预取查询用的清洗版；缺省则与 `user_message` 相同 |
| `current_turn_user_idx` | `int` | 本轮 user 在 `messages` 中的下标 |

## 判定规则

### R1：回合初始化

```text
_persist_user_message_override = persist_user_message  # 可为 None
_persist_user_message_idx = None  # 稍后设为 current_turn_user_idx
```

### R2：真实历史写入

```text
messages.append({"role": "user", "content": user_message})
current_turn_user_idx = len(messages) - 1
_persist_user_message_idx = current_turn_user_idx
```

API 请求始终使用 `user_message`（及阶段 1 在其副本上的注入），**不**在构造请求前替换为 override。

### R3：预取/插件查询词

```text
original_user_message = persist_user_message if persist_user_message is not None else user_message
```

外部记忆 prefetch 与 `pre_llm_call` 的 `user_message` 参数使用 `original_user_message`，避免技能前缀污染召回查询。

### R4：持久化前覆盖

在 `_persist_session()` / `_flush_messages_to_session_db()` 调用 `_apply_persist_user_message_override(messages)`：

```text
if override is not None and idx is not None and 0 <= idx < len(messages):
    if messages[idx].role == "user":
        messages[idx]["content"] = override
```

覆盖**就地**修改真实历史，随后写入 JSON 日志与 SQLite。

## 状态读写位置

- 进程内：`_persist_user_message_override`、`_persist_user_message_idx`（回合级，每轮 `run_conversation` 重置）。
- 持久化：Session DB 与 JSON 日志存 override 版 user 内容。
- API 副本：仍基于 `user_message` + 临时注入。

## 正常路径

1. 接收 `user_message` 与可选 `persist_user_message`。
2. 将 `user_message` 追加到真实历史。
3. 用 `original_user_message` 驱动阶段 3 的 prefetch/插件。
4. 模型看到的是 API 版 user（含技能等）。
5. 退出路径 persist 前把 DB/日志中的 user 行替换为 override。

## 分支路径

- `persist_user_message is None` → 不覆盖；持久化内容与 API user 相同。
- override 与 API 内容相同 → R4 为无害赋值。
- idx 越界或目标非 user → R4 跳过。

## 失败处理

覆盖失败不单独处理；依赖 persist 前 idx 校验。代理字符在入口对两个字符串均执行 `_sanitize_surrogates()`。

## 幂等性 / 一致性约束

- override 只影响**持久化视图**，不影响当前轮已发出的 API 请求内容。
- 同一轮多次 `_persist_session` 重复调用 R4 结果一致（幂等赋值）。
- 不能把 override 当作“回滚 API 历史”机制；内存中 `messages` 在 persist 前可能被改为 override，返回给调用方的 `messages` 也是清洗版。

## 代码骨架

```python
def begin_turn(user_message: str, persist_user_message: str | None, messages: list):
    state.override = persist_user_message
    messages.append({"role": "user", "content": user_message})
    state.persist_idx = len(messages) - 1
    state.original_query = persist_user_message if persist_user_message is not None else user_message
    return state.persist_idx, state.original_query


def apply_persist_override(messages: list, idx: int | None, override: str | None):
    if override is None or idx is None:
        return
    if 0 <= idx < len(messages) and messages[idx].get("role") == "user":
        messages[idx]["content"] = override
```

## 最小验证清单

1. gateway 传入 skill 展开后的 `user_message` 与原始 `persist_user_message` 时，API 看到前者，DB 存后者。
2. `prefetch_all` 的 query 等于 `persist_user_message`，不等于含技能块的 API user。
3. 未传 `persist_user_message` 时 DB 与 API user 一致。
4. `_persist_user_message_idx` 指向本轮唯一 user 消息。

## 来源证据（仅供溯源核实）

- `run_agent.py:7400-7414`：`persist_user_message` 参数说明。
- `run_agent.py:7444-7445`、`7520`：override 与 idx 初始化。
- `run_agent.py:7504`：`original_user_message` 选择逻辑。
- `run_agent.py:7517-7518`：真实历史追加 `user_message`。
- `run_agent.py:2164-2178`：persist 前覆盖实现。
- `run_agent.py:2188`、`2201`：persist 与 flush 均调用覆盖。
