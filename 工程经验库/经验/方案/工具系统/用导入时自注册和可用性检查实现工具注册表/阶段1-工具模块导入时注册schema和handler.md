---
title: 阶段1-工具模块导入时注册schema和handler
level: atomic
parent: 用导入时自注册和可用性检查实现工具注册表
status: draft
tags:
  - tool-registry
  - import-registration
created_at: 2026-06-09
updated_at: 2026-06-09
confidence: high
related:
  - ../用导入时自注册和可用性检查实现工具注册表.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - tools/registry.py
  - model_tools.py
---

# 阶段1：工具模块导入时注册schema和handler

## 触发条件

工具发现阶段导入某个工具模块，且该模块在顶层调用注册表注册方法。

## 输入字段

- `name`：工具唯一名，字符串。
- `toolset`：所属工具集名，字符串。
- `schema`：模型可见函数 schema。
- `handler`：执行处理器，接收参数字典和运行时上下文。
- `check_fn`：可选可用性检查函数。
- `requires_env`：可选环境变量名列表。
- `is_async`：布尔值，标记处理器是否异步。
- `max_result_size_chars`：可选结果大小限制。

## 判定规则

1. 注册时以 `name` 作为主键写入注册表。
2. 如果已有同名工具且工具集不同，记录名称冲突警告，然后以后注册者覆盖旧条目。
3. `requires_env` 缺省时保存为空列表。
4. `description` 缺省时从 `schema.description` 读取。
5. 如果提供 `check_fn` 且所属工具集尚未绑定检查函数，把该函数记录为工具集检查函数。
6. 工具模块导入失败不能阻断其它工具模块导入。

## 状态读写位置

- 写：进程内注册表的 `tools[name]` 和 `toolset_checks[toolset]`。
- 不写：模型请求消息、会话历史、用户配置。

## 正常路径

1. 工具发现器遍历模块名列表。
2. 导入工具模块。
3. 工具模块顶层调用注册表注册方法。
4. 注册表构造工具条目并保存。
5. 工具发现器继续导入下一个模块。

## 分支路径

- 同名同工具集重复注册 → 后者覆盖前者，不触发“工具集不同”警告。
- 同名不同工具集 → 记录冲突警告并覆盖。
- 模块导入异常 → 记录调试信息或跳过，继续其它模块。

## 失败处理

注册函数本身未见对 schema 结构做深校验；证据不足：schema 缺少 `parameters` 或 `description` 时是否应拒绝注册。可复用实现应在注册时加 schema 校验，避免运行时才暴露协议错误。

## 幂等性 / 一致性约束

导入式注册在单进程内是最终覆盖语义，不是严格幂等。重复导入同一模块通常由语言导入缓存保证只执行一次；如果显式重载模块，可能覆盖已有条目。

## 代码骨架

```python
class Registry:
    def __init__(self):
        self.tools = {}
        self.toolset_checks = {}

    def register(self, name, toolset, schema, handler, check_fn=None, requires_env=None):
        existing = self.tools.get(name)
        if existing and existing["toolset"] != toolset:
            log_warning(f"tool name collision: {name}")
        self.tools[name] = {
            "name": name,
            "toolset": toolset,
            "schema": schema,
            "handler": handler,
            "check_fn": check_fn,
            "requires_env": requires_env or [],
        }
        if check_fn and toolset not in self.toolset_checks:
            self.toolset_checks[toolset] = check_fn
```

## 最小验证清单

- 导入一个工具模块后，注册表包含该工具名。
- 注册同名不同工具集时，最终条目为后注册者。
- `requires_env=None` 时保存为空列表。
- 首个有 `check_fn` 的同工具集工具会设置工具集检查函数。
- 某个工具模块导入失败后，其它模块仍被导入。

## 来源证据（附录，不进正文）

- `tools/registry.py:24-45`：工具条目字段。
- `tools/registry.py:59-93`：注册方法覆盖同名工具、保存元数据、记录工具集检查函数。
- `model_tools.py:132-183`：工具发现器逐个导入模块，导入触发注册。
