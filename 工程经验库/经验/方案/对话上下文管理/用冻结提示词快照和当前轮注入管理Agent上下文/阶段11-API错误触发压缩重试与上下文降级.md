---
title: 阶段11-API错误触发压缩重试与上下文降级
level: atomic
parent: 用冻结提示词快照和当前轮注入管理Agent上下文
status: draft
tags:
  - ai-agent
  - context-compression
  - error-recovery
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段8-主循环前预压缩上下文.md
  - 阶段10-上下文压缩后重建系统提示词并创建父子会话链路.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段11：API 错误触发压缩重试与上下文降级

## 触发条件

主循环内 API 调用失败，错误经分类器判定为可压缩类，且 `compression_attempts <= max_compression_attempts`（**3**）。与阶段 8 预压缩不同：本阶段在**收到 provider 错误后**触发。

## 输入字段

| 字段 | 说明 |
|------|------|
| `classified.reason` | 失败原因枚举 |
| `error_msg` | 原始错误文本 |
| `compression_attempts` | 当前 API 调用重试循环内的压缩计数 |
| `max_compression_attempts` | 固定 **3** |
| `approx_tokens` | 当前请求估算 token |
| `messages` / `active_system_prompt` | 可变异状态 |

## 判定规则

### 分支 A：`long_context_tier`（Anthropic 长上下文订阅受限）

1. 将 `context_length` 降为 **200_000**（若原值更大）。
2. `_context_probe_persistable = False`（套餐限制，不持久化探测值）。
3. 调用阶段 10 `_compress_context()`。
4. `conversation_history = None`。
5. 若消息数减少或 context 已降级 → `restart_with_compressed_messages = True`，`sleep(2)`，break 重试循环。

### 分支 B：`payload_too_large`（HTTP 413）

1. `compression_attempts += 1`；若 > 3 → 返回 partial 错误，提示 `/new` 或 `/compress`。
2. 调用 `_compress_context()`；清空 `conversation_history`。
3. 若 `len(messages) < original_len` → 重试；否则返回“无法进一步压缩”。

### 分支 C：`context_overflow`（上下文超长）

**C1 — 输出上限问题**（`parse_available_output_tokens_from_error` 有值）：

- 不压缩；设置 `_ephemeral_max_output_tokens = available_out - 64`。
- 计入 `compression_attempts`；超限则失败返回。
- `restart_with_compressed_messages = True`（仅调 max_tokens 重试）。

**C2 — 输入过大**：

1. `parsed_limit = parse_context_limit_from_error(error_msg)`；若有效且 `< old_ctx` 则 `new_ctx = parsed_limit`，否则 `new_ctx = get_next_probe_tier(old_ctx)`。
2. 若 `new_ctx < old_ctx`：`compressor.update_model(context_length=new_ctx)`；可持久化探测值当且仅当 `parsed_limit == new_ctx`。
3. `compression_attempts += 1`；> 3 则失败返回。
4. 调用 `_compress_context()`；`conversation_history = None`。
5. 若消息数减少 **或** context 已降级 → 重试；否则失败返回。

### 分支 D：主循环 token 压力（非 API 错误）

工具循环结束后，若 `compression_enabled` 且 `should_compress(last_prompt_tokens)` → 调用 `_compress_context()`，清空 `conversation_history`（与阶段 8 相同清空语义）。

## 状态读写位置

- 写：`messages`、`active_system_prompt`、`session_id`（经阶段 10）、`conversation_history` 引用、`compression_attempts`、可选 `_ephemeral_max_output_tokens`、compressor.context_length。
- 读：错误分类结果、approx_tokens。

## 正常路径（以 context_overflow 输入过大为例）

1. API 返回 context overflow。
2. 解析或 tier 降级 context_length。
3. compress + 清空 history 引用。
4. sleep(2) 后 `continue` 重试 API（重新走阶段 1–6 构造请求）。

## 分支路径

- 压缩次数用尽 → persist partial 并返回 error 字符串含 `max_compression_attempts`。
- 仅 max_tokens 过大 → 不调用 compress，只缩 output cap。
- 压缩后消息数未减且已达最小 tier → 硬失败。

## 失败处理

所有失败路径在返回前尽量 `_persist_session(messages, conversation_history)`，避免丢历史。用户面向提示建议 `/new` 或 `/compress`。

## 幂等性 / 一致性约束

- `compression_attempts` 在单次 API 调用重试循环内累计；切换 fallback provider 时可能重置（rate limit 分支 `compression_attempts = 0`）。
- 每次 compress 成功必须 `conversation_history = None`，与阶段 8 一致。
- context_length 降级与 compress 可组合：重试条件为 `len(messages) < original_len OR new_ctx < old_ctx`。

## 代码骨架

```python
MAX_COMPRESSION_ATTEMPTS = 3

def handle_context_error(reason, error_msg, state, messages, attempts):
    if attempts > MAX_COMPRESSION_ATTEMPTS:
        return "fail", messages
    if reason == "payload_too_large":
        compressed, prompt = compress_context(messages, state)
        if len(compressed) < len(messages):
            return "retry", compressed
        return "fail", compressed
    if reason == "context_overflow":
        avail_out = parse_available_output_tokens(error_msg)
        if avail_out is not None:
            state.ephemeral_max_output = max(1, avail_out - 64)
            return "retry", messages
        maybe_downgrade_context_length(error_msg, state.compressor)
        compressed, prompt = compress_context(messages, state)
        return "retry", compressed
    ...
```

## 最小验证清单

1. 413 第 4 次压缩尝试被拒绝，返回 partial。
2. context overflow 且错误为 max_tokens → 只改 output cap，不 compress。
3. 每次 compress 后 `conversation_history is None`。
4. long_context_tier 将 context 上限设为 200k。
5. 主循环 `should_compress` 在工具轮次后触发 compress（非仅 API 错误）。

## 来源证据（仅供溯源核实）

- `run_agent.py:7867`：`max_compression_attempts = 3`。
- `run_agent.py:8628-8670`：`long_context_tier` → 200k + compress。
- `run_agent.py:8694-8740`：413 payload too large 分支。
- `run_agent.py:8742-8854`：`context_overflow` 与 max_tokens / input 过大分支。
- `run_agent.py:9497-9507`：工具循环后 `should_compress` 触发 compress。
