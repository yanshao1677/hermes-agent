---
title: 阶段5-工具参数JSON校验与空参归一化
level: atomic
parent: 用多级修复层容错模型工具调用协议错误
status: draft
tags:
  - json-validation
  - tool-calling
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段6-截断工具参数拒绝执行与流式finish_reason联动.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段5：工具参数 JSON 校验与空参归一化

## 触发条件

阶段 4 工具名全部 valid 后，执行参数预处理与 JSON 解析校验。

## 输入字段

- `tc.function.arguments`：str / dict / list / 其它。
- `_invalid_json_retries`：每 user turn 重置；上限 3（与阶段 6 截断路径分离）。

## 判定规则

对每个 tool_call 的 `args`：

1. **已是 dict 或 list** → `tc.function.arguments = json.dumps(args)`，视为 valid。
2. **非 str 且非 None** → `str(args)` 再当字符串处理。
3. **空或仅空白** → `tc.function.arguments = "{}"`，视为 valid。
4. **非空 str** → `json.loads(args)`；成功则 valid。
5. **JSONDecodeError** → 加入 `invalid_json_args` 列表 `(tool_name, error_str)`。

**若 `invalid_json_args` 非空**

- 先交阶段 6 判断是否截断；非截断路径：
  - `_invalid_json_retries += 1`
  - **< 3**：`continue` 重试 API，**不** append 坏 assistant/tool 到 messages
  - **>= 3**：append assistant + 对每个 tc  append tool 错误；无效名列表内：
    ```
    错误：参数不是有效的JSON。{err}。如工具无必填参数请用空对象: {}。请重试并确保JSON有效。
    ```
    其它 tc：`跳过：本次回复中有其他工具调用参数无效。`；重置 json 计数，`continue`

**全部 valid** → `_invalid_json_retries = 0`，进入阶段 7。

## 状态读写位置

- 写：`tc.function.arguments` 字符串形式。
- 写 messages：仅 json 重试达上限时。

## 正常路径

空参数 → `{}` → 通过 → 执行阶段。

## 分支路径

- dict 参数 → dumps 后通过。
- 第 1–2 次坏 JSON → 无 history 污染重试。

## 失败处理

见阶段 6 截断 vs 本阶段 3 次 tool 错误注入。

## 幂等性 / 一致性约束

`< 3` 次重试不写入 messages，避免坏 tool_calls 进入持久化历史。执行阶段（阶段 8）若 parse 仍失败兜底 `{}` 是**另一层**（校验后不应常触发）。

## 代码骨架

```python
for tc in tool_calls:
    args = tc.function.arguments
    if isinstance(args, (dict, list)):
        tc.function.arguments = json.dumps(args)
    elif args is not None and not isinstance(args, str):
        tc.function.arguments = str(args)
    elif not args or not str(args).strip():
        tc.function.arguments = "{}"
    else:
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            invalid.append((tc.function.name, str(e)))
```

## 最小验证清单

- `""` 和 whitespace → `{}`。
- dict 参数被 dumps。
- 第 1 次坏 JSON 不 append messages。
- 第 3 次坏 JSON 产生 tool role 错误。

## 来源证据（附录，不进正文）

- `run_agent.py:9280-9298`：归一化与 parse。
- `run_agent.py:9327-9364`：重试与注入。
- `run_agent.py:7452`：回合重置 `_invalid_json_retries`。
