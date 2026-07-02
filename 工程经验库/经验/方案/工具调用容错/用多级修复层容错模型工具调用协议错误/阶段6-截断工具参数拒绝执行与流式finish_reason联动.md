---
title: 阶段6-截断工具参数拒绝执行与流式finish_reason联动
level: atomic
parent: 用多级修复层容错模型工具调用协议错误
status: draft
tags:
  - truncation
  - tool-calling
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段5-工具参数JSON校验与空参归一化.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段6：截断工具参数拒绝执行与流式 finish_reason 联动

## 触发条件

阶段 5 发现 `invalid_json_args` 非空；或 chat_completions 路径检测到 tool_calls 存在但 arguments JSON 无效。

## 输入字段

| 字段 | 值 |
|------|-----|
| 截断启发式 | `not arguments.rstrip().endswith(("}", "]"))` |
| `_invalid_json_retries` | 截断路径强制置 0 |
| `truncated_tool_call_retries` | 每 API 迭代局部变量；上限 **1** |
| `finish_reason` | 可能被流式路径改为 `"length"` |

## 判定规则

**A. 阶段 5 内截断检测（invalid_json 且像截断）**

对 invalid 工具名集合内的 tc：

- 若 `(arguments or "").rstrip()` 不以 `}` 或 `]` 结尾 → 视为截断。
- 动作：
  - `_invalid_json_retries = 0`
  - `_cleanup_task_resources(task_id)`
  - `_persist_session(...)`
  - 返回 `{partial: True, error: "响应因输出长度限制被截断", final_response: None}`

**B. 流式聚合路径（mock tool_calls 构建时）**

- 非空 arguments 若 `json.loads` 失败 → `has_truncated_tool_args = True`
- 最终 `effective_finish_reason = "length"` if truncated else original

**C. chat_completions 响应后 tool_calls 存在**

- 若 `truncated_tool_call_retries < 1`：计数+1，`continue` 重试 API（不 append 损坏回复）
- 否则：cleanup + persist + partial，`error: "工具回复因长度限制被截断"`

**非截断 invalid JSON** → 交回阶段 5 的 3 次重试逻辑。

## 状态读写位置

- 写：重试计数、partial 返回。
- 不执行工具。

## 正常路径

完整 JSON 结尾 → 不进入本阶段终止分支。

## 分支路径

- 路由器把 finish_reason 改为 tool_calls 但 JSON 仍截断 → A 仍生效。
- 流式 B 使后续 A 更容易触发。

## 失败处理

一律 partial 终止当前回合工具执行，不注入 stub 执行。

## 幂等性 / 一致性约束

截断拒绝与「格式错误可重试」严格分离；截断时不计数 json 重试 3 次路径。

## 代码骨架

```python
def looks_truncated(args_str):
    return not (args_str or "").rstrip().endswith(("}", "]"))

if invalid_json and any(looks_truncated(tc.function.arguments) for tc in bad_tcs):
    return partial(error="响应因输出长度限制被截断")

if response_has_tool_calls and truncated_tool_call_retries < 1:
    truncated_tool_call_retries += 1
    continue  # retry API once
return partial(error="工具回复因长度限制被截断")
```

## 最小验证清单

- 以 `{` 开头不以 `}` 结尾的 invalid JSON → partial，不执行工具。
- truncated_tool_call_retries 只重试 1 次。
- 截断时 `_invalid_json_retries` 重置为 0。

## 来源证据（附录，不进正文）

- `run_agent.py:9300-9325`：截断检测与 partial 返回。
- `run_agent.py:4995-5018`：流式 finish_reason 联动。
- `run_agent.py:8248-8272`：`truncated_tool_call_retries`。
- `run_agent.py:9435`：工具执行成功后重置 truncated 计数。
