---
title: 阶段4-Agent级拦截执行会话搜索
level: atomic
parent: 用全文检索和摘要实现历史会话情景召回
status: draft
tags:
  - session-search
  - agent-intercept
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段2-用FTS检索并按父链路聚合历史会话.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
  - tools/session_search_tool.py
---

# 阶段4：Agent 级拦截执行会话搜索

## 触发条件

主循环 `handle_function_call` / `_invoke_tool` 解析到 `function_name == "session_search"`。

## 输入字段

| 参数 | 来源 | 默认 |
|------|------|------|
| `query` | tool args | `""` |
| `role_filter` | tool args | None |
| `limit` | tool args | 3 |
| `db` | `self._session_db` | 须非空 |
| `current_session_id` | `self.session_id` | 自动注入 |

## 判定规则

1. **不走 registry 分发**：与 todo/memory 一样在主循环 elif 链拦截。
2. 若 `not self._session_db` → 返回 `{"success": false, "error": "会话数据库不可用。"}` JSON 字符串（中英文实现略有差异，语义相同）。
3. 否则调用 `session_search(query, role_filter, limit, db=_session_db, current_session_id=self.session_id)`。
4. `current_session_id` **必须**注入——否则阶段 2 lineage 排除失效，召回当前对话重复内容。
5. 并发工具路径 `_invoke_tool` 与串行路径逻辑一致。
6. 返回 JSON 字符串直接作为 tool result message content。

## 状态读写位置

- 读：SessionDB（经 tool 只读）。
- 不写 session 状态。

## 正常路径

1. 模型调用 session_search。
2. Agent 注入 db + current_session_id。
3. tool 模块执行阶段 1–3。
4. JSON 返回模型。

## 分支路径

- CLI 无 session 持久化 → db 空 → 明确 error，模型可降级回答。
- gateway 每消息新 agent 但共享 session_id → lineage 仍正确。

## 失败处理

tool 内部 error JSON；Agent 不额外 wrap except 标准 tool 消息格式。

## 幂等性 / 一致性约束

只读工具；可并发（与 todo 不同，无 write 冲突）。多次搜索同 query 结果依赖 DB 状态。

## 代码骨架

```python
elif function_name == "session_search":
    if not agent.session_db:
        return json.dumps({"success": False, "error": "Session database not available."})
    return session_search(
        query=args.get("query", ""),
        role_filter=args.get("role_filter"),
        limit=args.get("limit", 3),
        db=agent.session_db,
        current_session_id=agent.session_id,
    )
```

## 最小验证清单

- 无 session_db 时不调用 session_search 模块。
- 有 db 时 current_session_id 等于 agent.session_id。
- 拦截路径不经过 handle_function_call registry。

## 来源证据（附录，不进正文）

- `run_agent.py:6963-6974`：串行工具路径。
- `run_agent.py:6597-6607`：`_invoke_tool` 并发路径。
- `tools/session_search_tool.py:247-253`：`session_search()` 签名。
