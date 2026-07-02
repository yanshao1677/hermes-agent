---
title: 阶段3-按实际可用工具动态裁剪schema描述
level: atomic
parent: 用导入时自注册和可用性检查实现工具注册表
status: draft
tags:
  - schema-postprocess
  - hallucination-control
created_at: 2026-06-09
updated_at: 2026-06-09
confidence: high
related:
  - ../用导入时自注册和可用性检查实现工具注册表.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - model_tools.py
---

# 阶段3：按实际可用工具动态裁剪schema描述

## 触发条件

工具 schema 已经过工具集解析和可用性过滤，即将返回给模型。

## 输入字段

- `filtered_tools`：已通过可用性检查的工具定义列表。
- `available_tool_names`：从 `filtered_tools` 中提取出的实际可用工具名集合。
- 静态 schema 中可能包含跨工具引用的工具定义。

## 判定规则

1. 动态后处理只能基于 `available_tool_names`，不能基于目标工具集合；目标集合中可能包含检查失败的工具。
2. 如果某工具 schema 的描述会列出可在沙箱中调用的工具，则必须用实际可用集合与允许列表求交后重建描述。
3. 如果某工具描述建议优先使用其它工具，而这些被引用工具均不可用，则必须删除该建议句。
4. 后处理只修改返回给模型的 schema，不修改注册表中的原始 schema。
5. 后处理完成后，更新最近解析工具名列表。

## 状态读写位置

- 读：过滤后的 schema、实际可用工具名集合。
- 写：本次返回的 schema 列表、最近解析工具名列表。
- 不写：注册表原始 schema。

## 正常路径

1. 从过滤后 schema 提取工具名集合。
2. 对需要列出子工具能力的 schema，使用实际可用集合重建动态 schema。
3. 对需要提及替代工具的描述，若替代工具不可用，删除对应描述片段。
4. 返回修改后的 schema 列表。

## 分支路径

- 被后处理的工具本身不可用 → 不处理它的 schema。
- 被引用工具部分可用 → 只保留可用工具引用。
- 被引用工具全部不可用 → 删除跨工具建议，避免模型幻想调用不存在工具。

## 失败处理

证据不足：动态 schema 构建函数自身异常时是否有专门兜底；当前证据只覆盖调用这些后处理的条件和目的。可复用实现应将后处理异常降级为保留原 schema 并记录警告。

## 幂等性 / 一致性约束

后处理必须是纯函数式地修改本次返回值；不能把裁剪后的描述写回注册表，否则下一次在工具可用性变化后无法恢复完整描述。

## 代码骨架

```python
def postprocess_schemas(filtered_tools):
    names = {t["function"]["name"] for t in filtered_tools}
    result = list(filtered_tools)
    if "sandbox" in names:
        allowed = SANDBOX_ALLOWED_TOOLS & names
        replace_schema(result, "sandbox", build_sandbox_schema(allowed))
    if "browser" in names and not ({"web_search", "web_extract"} & names):
        strip_description_sentence(result, "browser", "prefer web_search or web_extract")
    return result
```

## 最小验证清单

- 目标工具集合包含某工具但该工具检查失败时，schema 描述不能引用它。
- 引用工具可用时，描述保留该引用。
- 引用工具不可用时，描述删除该引用。
- 多次获取 schema 后，注册表原始 schema 未被永久改写。

## 来源证据（附录，不进正文）

- `model_tools.py:301-308`：实际可用工具名从过滤后 schema 计算，而不是从目标集合计算。
- `model_tools.py:310-321`：执行类 schema 用实际可用工具集合重建。
- `model_tools.py:323-344`：浏览类 schema 在 web 工具不可用时删除跨工具建议。
- `model_tools.py:350-351`：保存最近解析出的可用工具名列表。
