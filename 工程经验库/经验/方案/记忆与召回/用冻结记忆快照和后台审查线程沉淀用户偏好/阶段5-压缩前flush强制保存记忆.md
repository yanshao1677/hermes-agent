---
title: 阶段5-压缩前flush强制保存记忆
level: atomic
parent: 用冻结记忆快照和后台审查线程沉淀用户偏好
status: draft
tags:
  - memory
  - flush
  - compression
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段6-压缩后失效快照并重新加载.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段5：压缩前 flush 强制保存记忆

## 触发条件

`_compress_context()` 开始时第一行调用 `flush_memories(messages, min_turns=0)`；或 CLI 退出等路径调用 `flush_memories()`（默认 min_turns 配置）。

## 输入字段

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `messages` | list | `_session_messages` | 将被 append flush 消息 |
| `min_turns` | int | `flush_min_turns`（6） | 压缩时显式传 0=总是 flush |
| `memory_flush_min_turns` | int | 6 | 配置；若为 0 且 min_turns 为 None 则直接 return |
| `_user_turn_count` | int | — | 须 >= effective_min |
| `valid_tool_names` | set | — | 须含 `"memory"` |

## 判定规则

**早退条件（任一满足则跳过）**

1. `_memory_flush_min_turns == 0` 且 `min_turns is None`。
2. `"memory" not in valid_tool_names` 或 `not _memory_store`。
3. `_user_turn_count < effective_min`（压缩时 effective_min=0 永不触发此条）。
4. `messages` 为空或 `len(messages) < 3`。
5. 找不到 memory tool schema → pop 已 append 的 flush 消息并 return。

**flush 消息**

- content：`"[System: 当前对话将被压缩，请保存任何值得长期记忆的内容——优先保存用户偏好、纠正与反复操作的模式，其次是具体任务细节。]"`
- 带 `_flush_sentinel = f"__flush_{id(self)}_{time.monotonic()}"` 标记。

**API 调用**

1. 构造 `api_messages`：复制 messages，剥离 `reasoning`/`finish_reason`/`_flush_sentinel`/`_thinking_prefill`；assistant 加 `reasoning_content`。
2. 前置 `[{"role":"system","content": _cached_system_prompt}]`（若存在）。
3. **仅** memory 一个 tool；temperature=0.3；max_tokens=5120。
4. 优先 auxiliary client `task="flush_memories"`；失败则按 api_mode 回退 primary。

**执行工具**

- 解析响应中 `memory` tool_calls → 直接调用 `memory_tool(..., store=_memory_store)`，不经过完整 agent 循环。

**清理（finally）**

- 从 `messages` 尾部 pop 直到遇到带 `_flush_sentinel == sentinel` 的消息并移除；中间所有消息一并移除（含 assistant/tool 若产生）。

## 状态读写位置

- 读：messages、cached system prompt、memory schema。
- 写：MEMORY/USER 文件（通过 memory_tool）、不保留 flush 相关消息。

## 正常路径

1. append flush user 消息。
2. 单次 LLM 调用（仅 memory tool）。
3. 执行返回的 memory 写操作。
4. finally  strip flush 段，messages 回到压缩前用户可见历史。

## 分支路径

- auxiliary 不可用 → 回退 codex / anthropic / openai primary。
- 无 tool_calls → finally 仍清理 sentinel。
- tool 执行异常 → `logger.debug`，继续清理。
- 整个 API 异常 → debug 日志，finally 清理。

## 失败处理

flush 全程不 raise 到压缩主路径；压缩继续。未保存的记忆依赖阶段 6 reload 前磁盘状态（flush 失败则仅保留此前已写入条目）。

## 幂等性 / 一致性约束

flush 消息**永不**进入持久化 session DB（sentinel 清理 + 不写入真实历史的 flush 段）。同一次压缩只调用一次 flush（min_turns=0）。

## 代码骨架

```python
def flush_memories(agent, messages, min_turns=None):
    if not agent.can_flush(min_turns):
        return
    sentinel = new_sentinel()
    messages.append({"role": "user", "content": FLUSH_PROMPT, "_flush_sentinel": sentinel})
    try:
        tool_calls = call_llm_once(messages, tools=[memory_tool_only], temp=0.3, max_tokens=5120)
        for tc in tool_calls:
            if tc.name == "memory":
                memory_tool(**parse(tc), store=agent.memory_store)
    finally:
        strip_from_sentinel(messages, sentinel)
```

## 最小验证清单

- `_compress_context` 调用时 `_user_turn_count=0` 仍执行 flush（min_turns=0）。
- flush 后 messages 无 `_flush_sentinel` 残留。
- 常规退出 `_user_turn_count < flush_min_turns` 时不 flush。
- flush 产生的 memory 写入可在阶段 6 reload 后进新 snapshot。

## 来源证据（附录，不进正文）

- `run_agent.py:6313-6462`：`flush_memories()` 完整实现。
- `run_agent.py:6480-6481`：压缩前调用 `min_turns=0`。
- `run_agent.py:1087`：`flush_min_turns` 默认 6。
