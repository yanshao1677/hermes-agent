---
title: 阶段2-用FTS检索并按父链路聚合历史会话
level: atomic
parent: 用全文检索和摘要实现历史会话情景召回
status: draft
tags:
  - fts
  - session-lineage
created_at: 2026-06-09
updated_at: 2026-07-02
confidence: high
related:
  - 阶段1-空查询浏览最近会话.md
  - 阶段3-命中会话并发摘要失败时降级预览.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - hermes_state.py
  - tools/session_search_tool.py
---

# 阶段2：用 FTS 检索并按父链路聚合历史会话

## 触发条件

`session_search` 收到非空 `query.strip()`。

## 输入字段

| 字段 | 说明 |
|------|------|
| `query` | 非空检索串 |
| `role_filter` | 可选，逗号分隔角色 |
| `limit` | 结果会话数，入口 `min(limit, 5)` |
| `current_session_id` | 当前会话，用于 lineage 排除 |

## 判定规则

1. `role_list = [r.strip() for r in role_filter.split(",") if strip]` 或 None。
2. `db.search_messages(query, role_filter=role_list, exclude_sources=["tool"], limit=50, offset=0)`。
3. FTS 语法错误 → DB 层返回 `[]`（不 raise）。
4. 无命中 → JSON `success=true, results=[], count=0`。
5. `_resolve_to_parent(session_id)`：while parent 存在且未 visited，向上；异常 break。
6. `current_lineage_root = _resolve_to_parent(current_session_id)` 若 id 非空。
7. 遍历 raw_results：
   - `resolved_sid = _resolve_to_parent(raw_sid)`。
   - skip：`resolved_sid == current_lineage_root` 或 `raw_sid == current_session_id`。
   - `seen_sessions[resolved_sid] = result` 首次保留。
   - `len(seen_sessions) >= limit` → break。

## 状态读写位置

- 只读 FTS 索引与会话元数据。
- 输出：候选 `{root_session_id: hit_metadata}` 供阶段 3。

## 正常路径

1. strip query。
2. FTS 50 条。
3. 聚合唯一根会话至 limit。

## 分支路径

- current_session_id 为空 → 不做 lineage 排除（仅 skip raw_sid==current 分支仍无效）。
- parent 链断 → 停在可得 session id。

## 失败处理

DB None → `tool_error("Session database not available.")`。外层 except → tool_error 包装。

## 幂等性 / 一致性约束

只读。parent 链不完整时 exclusion 退化，可能召回同对话其它 fragment。

## 代码骨架

```python
def collect_sessions(db, query, limit, current_session_id):
    hits = db.search_messages(query=query.strip(), exclude_sources=["tool"], limit=50)
    current_root = resolve_root(db, current_session_id) if current_session_id else None
    seen = {}
    for hit in hits:
        root = resolve_root(db, hit["session_id"])
        if current_root and root == current_root:
            continue
        if hit["session_id"] == current_session_id:
            continue
        seen.setdefault(root, hit)
        if len(seen) >= min(limit, 5):
            break
    return seen
```

## 最小验证清单

- limit=10 最多聚合 5 个根会话。
- 同 root 多消息只保留一条 hit。
- FTS 非法语法返回空而非异常。

## 来源证据（附录，不进正文）

- `tools/session_search_tool.py:263-345`：非空 query 主路径。
- `hermes_state.py:990-1091`：`search_messages()` 与语法错误处理。
