---
title: 阶段1-状态敏感工具schema可见但执行在主循环拦截
level: atomic
parent: 用Agent级拦截和注册表分发分离状态敏感工具
status: draft
tags:
  - agent-loop
  - stateful-tool
created_at: 2026-06-09
updated_at: 2026-06-09
confidence: high
related:
  - ../用Agent级拦截和注册表分发分离状态敏感工具.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
  - model_tools.py
---

# 阶段1：状态敏感工具schema可见但执行在主循环拦截

## 触发条件

模型返回的工具调用名属于状态敏感工具集合，例如待办、记忆、会话搜索或子任务委托。

## 输入字段

- `function_name`：工具名。
- `function_args`：解析后的参数字典。
- 当前智能体的待办存储、记忆存储、会话数据库、会话标识、父智能体对象。

## 判定规则

1. 工具名是待办工具时，调用待办处理器并传入当前会话待办存储。
2. 工具名是会话搜索工具时，必须存在会话数据库；存在时传入数据库和当前会话标识，不存在时返回失败 JSON。
3. 工具名是记忆工具时，调用记忆处理器并传入当前记忆存储。
4. 工具名是子任务委托工具时，调用委托处理器并传入当前智能体作为父对象。
5. 这些工具不得落入普通注册表分发路径执行。

## 状态读写位置

- 待办工具读写当前智能体内存待办存储。
- 记忆工具读写当前记忆存储及其持久文件。
- 会话搜索只读会话数据库。
- 委托工具读取父智能体运行时配置并创建子智能体。

## 正常路径

1. 主循环解析工具名和 JSON 参数。
2. 命中状态敏感工具分支。
3. 从当前智能体取出必要状态对象。
4. 调用对应工具处理器。
5. 把工具结果作为 `role=tool` 消息追加到真实历史。

## 分支路径

- 会话搜索无数据库 → 返回 `success=false`。
- 记忆工具执行写入后，如存在外部记忆管理器，还可通知外部提供者。
- 委托工具可根据单任务或批任务创建一个或多个子智能体。

## 失败处理

证据不足：每个状态敏感工具内部所有异常分支未在本卡完整展开。本卡确认主循环对会话数据库缺失有明确失败返回，普通分发层对误入状态敏感工具有防漏错误。

## 幂等性 / 一致性约束

状态敏感工具必须绑定当前智能体实例；不能用全局单例替代当前会话状态。子智能体和并发执行路径也必须复用同一拦截规则，否则状态语义会分裂。

## 代码骨架

```python
def invoke_agent_tool(agent, name, args):
    if name == "todo":
        return todo_tool(todos=args.get("todos"), merge=args.get("merge", False), store=agent.todo_store)
    if name == "session_search":
        if not agent.session_db:
            return json.dumps({"success": False, "error": "session db unavailable"})
        return session_search(args.get("query", ""), db=agent.session_db, current_session_id=agent.session_id)
    if name == "memory":
        return memory_tool(action=args.get("action"), target=args.get("target", "memory"), store=agent.memory_store)
    if name == "delegate_task":
        return delegate_task(goal=args.get("goal"), parent_agent=agent)
    raise KeyError(name)
```

## 最小验证清单

- 待办工具调用后，当前智能体待办存储变化。
- 会话搜索工具调用时参数包含当前会话标识。
- 无会话数据库时，会话搜索返回失败而不是抛异常。
- 委托工具调用时能读取父智能体配置。
- 状态敏感工具不调用注册表 dispatch。

## 来源证据（附录，不进正文）

- `model_tools.py:360-364`：状态敏感工具集合。
- `run_agent.py:6953-7024`：串行执行分支注入当前 store、db、parent_agent。
- `run_agent.py:6592-6624`：统一 `_invoke_tool()` 分支处理 Agent 级工具。
