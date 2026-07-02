---
title: 阶段4-无效工具名注入tool错误并限制重试
level: atomic
parent: 用多级修复层容错模型工具调用协议错误
status: draft
tags:
  - tool-calling
  - invalid-tool
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段3-工具名模糊修复后再校验.md
  - 阶段5-工具参数JSON校验与空参归一化.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段4：无效工具名注入 tool 错误并限制重试

## 触发条件

阶段 3 之后仍存在 `tc.function.name ∉ valid_tool_names`。

## 输入字段

| 字段 | 默认 |
|------|------|
| `_invalid_tool_retries` | 0；每 user turn 开头重置 |
| 上限 | 3 |
| `available` | `", ".join(sorted(valid_tool_names))` |

## 判定规则

1. 收集 `invalid_tool_calls` 名称列表。
2. 若非空：`_invalid_tool_retries += 1`。
3. **若 retries >= 3**：
   - 重置计数为 0
   - `_persist_session(...)`
   - 返回 `{partial: True, completed: False, final_response: None, error: "模型生成了无效的工具调用: ..."}`
4. **若 retries < 3**（含第 1、2 次）：
   - append `_build_assistant_message(...)`（含全部 tool_calls）
   - 对每个 tc append **tool** 消息：
     - 无效名：`content = f"工具 '{name}' 不存在。可用工具有: {available}"`
     - 同批合法名：`content = "跳过：本轮另一工具调用名无效，请重试此工具调用。"`
   - `continue` 主循环（**不是** user 消息）

5. 全部 valid → `_invalid_tool_retries = 0`，进入阶段 5。

## 状态读写位置

- 读写：`_invalid_tool_retries`。
- 写：messages（assistant + tool）；达上限时 persist。

## 正常路径

第 1 次幻觉工具名 → tool 错误 → 模型下轮修正 → 计数重置。

## 分支路径

- 同批混合 valid/invalid → valid 的 tc 也收到「跳过」tool 结果。
- 第 3 次仍 invalid → partial 终止，不执行任何工具。

## 失败处理

partial 返回；用户可见回合结束，error 字段说明原因。

## 幂等性 / 一致性约束

计数不跨 user turn。错误必须用 **tool role** 保持 assistant→tool 配对，禁止用 user 消息（旧文档错误已修正）。

## 代码骨架

```python
invalid = [tc.function.name for tc in tool_calls if tc.function.name not in valid_names]
if invalid:
    state.invalid_tool_retries += 1
    if state.invalid_tool_retries >= 3:
        state.invalid_tool_retries = 0
        return partial_result(error=f"无效工具: {invalid[0]}")
    messages.append(build_assistant_message(assistant_message))
    for tc in tool_calls:
        if tc.function.name in invalid:
            msg = f"工具 '{tc.function.name}' 不存在。可用工具有: {available}"
        else:
            msg = "跳过：本轮另一工具调用名无效，请重试此工具调用。"
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": msg})
    continue
state.invalid_tool_retries = 0
```

## 最小验证清单

- 错误消息 role 为 tool，非 user。
- 第 3 次 invalid 返回 partial=True。
- 新 user turn 计数从 0 开始。
- 同批合法工具收到「跳过」tool 内容。

## 来源证据（附录，不进正文）

- `run_agent.py:7451-7452`：回合重置计数。
- `run_agent.py:9237-9278`：完整分支。
- `run_agent.py:9251-9262`：第 3 次 partial 返回。
