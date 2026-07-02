---
title: 用Agent级拦截和注册表分发分离状态敏感工具
level: pattern
parent: 
status: draft
tags:
  - agent-loop
  - tool-dispatch
  - stateful-tool
created_at: 2026-06-09
updated_at: 2026-06-09
confidence: high
related:
  - 用Agent级拦截和注册表分发分离状态敏感工具/阶段1-状态敏感工具schema可见但执行在主循环拦截.md
  - 用Agent级拦截和注册表分发分离状态敏感工具/阶段2-普通工具统一走注册表分发并注入运行时上下文.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
  - model_tools.py
  - tools/registry.py
---

# 用Agent级拦截和注册表分发分离状态敏感工具

## 1. 问题

模型可见的工具不都适合由普通注册表直接执行。有些工具需要访问当前智能体实例的内部状态，例如待办列表、长期记忆存储、会话数据库、子任务创建所需的父智能体上下文。若这些工具直接走通用分发层，处理器拿不到正确状态，或会误操作全局状态。

## 2. 适用约束

- 工具 schema 仍必须对模型可见。
- 工具执行时有些工具需要当前会话、当前智能体、当前数据库或当前内存对象。
- 普通工具仍应由统一注册表分发，以保留插件钩子、错误包装和异步桥接。

## 3. 核心思路

把“schema 可见性”和“执行归属”分开：状态敏感工具仍注册 schema，但主循环在执行阶段按工具名拦截并注入当前智能体状态；通用分发层遇到这些工具只返回防漏错误，普通工具继续走注册表。

## 4. 通用结构

- **AgentLoopTools**：一组必须由主循环处理的工具名。
- **AgentStateAdapters**：把当前待办存储、记忆存储、会话数据库、父智能体对象注入具体工具处理器。
- **RegistryDispatcher**：处理无状态或只需通用运行时上下文的工具。
- **FallbackStub**：防止状态敏感工具绕过主循环时静默误执行。

## 5. 处理流程

1. 工具注册表照常持有所有工具 schema。
2. 模型返回工具调用后，主循环解析工具名和参数。
3. 若工具名属于状态敏感集合，则在主循环内执行专门分支。
4. 专门分支把当前实例状态传入工具处理器。
5. 若工具名不属于状态敏感集合，则调用通用分发函数。
6. 通用分发函数执行插件前后钩子和注册表分发。
7. 如果状态敏感工具误入通用分发，返回明确错误提示。

## 6. 异常处理

状态敏感工具各自返回结构化结果。普通工具由注册表捕获未知工具和处理器异常。通用分发对状态敏感工具返回“必须由主循环处理”的 JSON 错误，避免错误地无状态执行。

## 7. 具体语言实现

```python
AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}

class Agent:
    def invoke_tool(self, name, args):
        if name == "todo":
            return todo_tool(args, store=self.todo_store)
        if name == "memory":
            return memory_tool(args, store=self.memory_store)
        if name == "session_search":
            return session_search(args, db=self.session_db, current_session_id=self.session_id)
        if name == "delegate_task":
            return delegate_task(args, parent_agent=self)
        return dispatch_registered_tool(name, args, session_id=self.session_id)

def dispatch_registered_tool(name, args, **context):
    if name in AGENT_LOOP_TOOLS:
        return json.dumps({"error": f"{name} must be handled by the agent loop"})
    return registry.dispatch(name, args, **context)
```

## 8. 测试点

- 状态敏感工具 schema 可出现在工具定义中。
- 主循环执行待办工具时使用当前会话待办存储。
- 主循环执行会话搜索时传入当前会话标识用于排除当前链路。
- 状态敏感工具绕过主循环进入通用分发时返回错误。
- 普通工具继续走注册表分发。

## 9. 适用场景 / 不适用场景

适用于工具处理器既包含无状态工具又包含强会话状态工具的智能体运行时。不适用于所有工具都通过外部服务执行且状态显式放在请求参数中的架构。

## 10. 风险与反模式

- 新增状态敏感工具时，只注册 schema 但忘记加入主循环拦截，会导致模型可见但执行失败。
- 主循环分支过多会膨胀，需要定期抽象状态适配器。
- 状态敏感工具不应在子智能体中默认开放，否则会写共享记忆或跨平台发消息。

## 11. 标签

Agent 级工具、状态注入、工具分发、会话状态、工具安全边界。

## 12. 来源证据（附录，不进正文，仅供溯源）

- `model_tools.py:360-364`：定义必须由 agent loop 处理的工具集合。
- `model_tools.py:496-499`：通用分发遇到这些工具返回错误 JSON。
- `run_agent.py:6953-7024`：串行工具执行中对 `todo`、`session_search`、`memory`、`delegate_task` 分别注入当前状态。
- `run_agent.py:6584-6654`：并发工具路径也通过 `_invoke_tool()` 兼容 Agent 级工具和注册表工具。
- `tools/registry.py:149-166`：普通工具统一由注册表分发并捕获异常。
