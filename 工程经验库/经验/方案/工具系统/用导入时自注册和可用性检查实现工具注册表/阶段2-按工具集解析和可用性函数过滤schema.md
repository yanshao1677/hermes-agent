---
title: 阶段2-按工具集解析和可用性函数过滤schema
level: atomic
parent: 用导入时自注册和可用性检查实现工具注册表
status: draft
tags:
  - toolset
  - schema-filter
  - availability-check
created_at: 2026-06-09
updated_at: 2026-06-09
confidence: high
related:
  - ../用导入时自注册和可用性检查实现工具注册表.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - model_tools.py
  - tools/registry.py
  - toolsets.py
---

# 阶段2：按工具集解析和可用性函数过滤schema

## 触发条件

准备向模型发起请求前，需要生成本次可见工具 schema。

## 输入字段

- `enabled_toolsets`：可选工具集名列表；非空时只启用这些工具集。
- `disabled_toolsets`：可选工具集名列表；仅在未指定启用列表时生效。
- `quiet_mode`：布尔值，控制是否输出状态提示。
- `TOOLSETS`：工具集定义表。
- 注册表中的工具条目和 `check_fn`。

## 判定规则

1. 若 `enabled_toolsets` 非空：逐个校验并解析这些工具集，目标集合从空集累加。
2. 若未指定启用列表但有 `disabled_toolsets`：先加载全部工具集，再从目标集合中删除被禁用工具集解析出的工具。
3. 若两者都未指定：加载全部工具集。
4. 未知工具集只产生警告，不抛异常。
5. 目标工具名集合交给注册表获取 schema。
6. 注册表对每个工具执行 `check_fn`；同一个函数在一次获取中只执行一次并缓存结果。
7. `check_fn` 返回 False 或抛异常的工具不进入 schema 列表。
8. 返回的 schema 必须补齐 `name` 字段，优先使用注册条目的工具名。

## 状态读写位置

- 读：工具集定义、注册表工具条目、可用性函数。
- 写：本次调用局部目标工具集合；进程全局的“最近解析工具名列表”可供沙箱类工具参考。

## 正常路径

1. 根据启用/禁用参数计算目标工具名集合。
2. 调用注册表获取过滤后的 OpenAI 格式工具定义。
3. 从过滤结果计算实际可用工具名集合。
4. 保存最近解析工具名列表。
5. 返回过滤后的工具定义。

## 分支路径

- 启用列表包含未知工具集 → 跳过该项。
- 禁用列表包含未知工具集 → 跳过该项。
- 工具名不在注册表 → 跳过该工具。
- 可用性函数异常 → 该工具视为不可用。

## 失败处理

证据显示可用性函数异常会被吞掉并将工具标为不可用；工具集解析自身对未知名称只警告。证据不足：组合工具集循环引用时的防护未在本卡深读。

## 幂等性 / 一致性约束

本阶段每次调用都重新计算可见工具集合。若外部环境变量或凭据在运行中变化，下一次 schema 获取可能暴露不同工具；长会话若要求提示词缓存稳定，不应在会话中途随意改变工具集。

## 代码骨架

```python
def get_tool_definitions(enabled=None, disabled=None):
    target = set()
    if enabled is not None:
        for name in enabled:
            if validate_toolset(name):
                target.update(resolve_toolset(name))
    elif disabled:
        for name in all_toolsets():
            target.update(resolve_toolset(name))
        for name in disabled:
            if validate_toolset(name):
                target.difference_update(resolve_toolset(name))
    else:
        for name in all_toolsets():
            target.update(resolve_toolset(name))
    return registry.definitions(target)
```

## 最小验证清单

- 指定启用工具集时，只返回该集合内且可用的工具。
- 指定禁用工具集时，全集中不包含禁用集合工具。
- 工具检查函数返回 False 时，该工具 schema 不返回。
- 两个工具共享同一检查函数时，该函数一次获取中只执行一次。
- 未知工具集不会使 schema 获取失败。

## 来源证据（附录，不进正文）

- `model_tools.py:234-302`：启用/禁用/默认三种路径解析工具集合。
- `tools/registry.py:116-143`：按工具名和 `check_fn` 过滤 schema，并缓存检查结果。
- `toolsets.py:68-243`：工具集定义。
- `model_tools.py:350-351`：记录最近解析工具名列表。
