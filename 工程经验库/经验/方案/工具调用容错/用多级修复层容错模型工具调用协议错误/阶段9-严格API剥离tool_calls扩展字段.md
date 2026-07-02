---
title: 阶段9-严格API剥离tool_calls扩展字段
level: atomic
parent: 用多级修复层容错模型工具调用协议错误
status: draft
tags:
  - strict-api
  - tool-calling
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段1-请求前清理孤立工具结果并补齐缺失结果.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段9：严格 API 剥离 tool_calls 扩展字段

## 触发条件

构造 `api_messages` 复制 assistant 消息时，且 `_should_sanitize_tool_calls()` 为 True。

## 输入字段

| 条件 | 行为 |
|------|------|
| `api_mode == "codex_responses"` | **不**剥离（保留扩展字段） |
| 其它 api_mode | 剥离 |

剥离键：`_STRIP_KEYS = {"call_id", "response_item_id"}`（Codex Responses 扩展）。

## 判定规则

1. `_should_sanitize_tool_calls()` ≡ `(api_mode != "codex_responses")`。
2. 对每个 `api_msg` 若含 `tool_calls` 列表：
   - 复制每个 tc dict，去掉 strip 键。
   - 内部 `messages` 列表保留完整 tc（供 Codex 切回）。
3. 仅修改 **api 副本**；`messages` 原对象不变（7773 注释）。

## 状态读写位置

- 写：api_messages 中 assistant 条目。
- 不写：持久化 messages、session DB。

## 正常路径

Mistral/Fireworks 等 strict OpenAI 兼容 API 收到无扩展字段的 tool_calls → 避免 400/422。

## 分支路径

- codex_responses → 跳过，保留 call_id 等。
- 无 tool_calls → no-op。

## 失败处理

无；纯字段过滤。

## 幂等性 / 一致性约束

对同一 api_msg 重复 strip 无副作用。内部 messages 与 API 副本刻意分叉，切换 api_mode 时依赖内部完整结构。

## 代码骨架

```python
STRIP = {"call_id", "response_item_id"}

def sanitize_tool_calls_for_strict_api(api_msg, api_mode):
    if api_mode == "codex_responses":
        return api_msg
    tcs = api_msg.get("tool_calls")
    if not isinstance(tcs, list):
        return api_msg
    api_msg["tool_calls"] = [
        {k: v for k, v in tc.items() if k not in STRIP} if isinstance(tc, dict) else tc
        for tc in tcs
    ]
    return api_msg
```

## 最小验证清单

- chat_completions + Mistral 路径 strip 两字段。
- codex_responses 不 strip。
- 内部 messages 仍含扩展字段。

## 来源证据（附录，不进正文）

- `run_agent.py:6285-6311`：`_sanitize_tool_calls_for_strict_api`、`_should_sanitize_tool_calls`。
- `run_agent.py:7773-7775`：仅 API 副本 strip。
