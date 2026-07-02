---
title: 阶段5-Anthropic提示词缓存断点策略
level: atomic
parent: 用冻结提示词快照和当前轮注入管理Agent上下文
status: draft
tags:
  - ai-agent
  - prompt-cache
  - anthropic
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段4-系统消息拼接与预填充消息插入请求副本.md
  - 阶段6-请求副本归一化与协议清理.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - agent/prompt_caching.py
  - run_agent.py
---

# 阶段5：Anthropic 提示词缓存断点策略

## 触发条件

阶段 4 完成后，且 `_use_prompt_caching == True`。通常在 Claude 模型经 OpenRouter 或 native Anthropic API 时启用。

## 输入字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `api_messages` | `List[Dict]` | 含 system、prefill、历史的完整请求副本 |
| `cache_ttl` | `str` | `"5m"`（默认）或 `"1h"` |
| `native_anthropic` | `bool` | `api_mode == "anthropic_messages"` 时为真 |

## 判定规则

策略名 **`system_and_3`**：最多 **4** 个 `cache_control` 断点。

### R1：断点标记

```text
marker = {"type": "ephemeral"}
if cache_ttl == "1h":
    marker["ttl"] = "1h"
```

### R2：断点 1 — system

若 `api_messages[0].role == "system"`：对第一条 system 消息应用 marker，`breakpoints_used = 1`。

### R3：断点 2–4 — 最后 3 条非 system

```text
remaining = 4 - breakpoints_used
non_sys_indices = [i for i in range(len(messages)) if role != "system"]
for idx in non_sys_indices[-remaining:]:
    apply marker to messages[idx]
```

### R4：单消息 marker 放置（`_apply_cache_marker`）

- `role == "tool"` 且 `native_anthropic == False` → **不**加 marker（直接 return）。
- `role == "tool"` 且 `native_anthropic == True` → marker 加在消息级 `cache_control`。
- `content` 为空或 `None` → 消息级 `cache_control`。
- `content` 为字符串 → 转为 `[{"type":"text","text": content, "cache_control": marker}]`。
- `content` 为 list → marker 加在**最后一个** content block。

### R5：深拷贝

函数返回 `copy.deepcopy(api_messages)`，不修改调用方传入的列表。

## 状态读写位置

- 读：完整 `api_messages`。
- 写：带 `cache_control` 的新副本。
- 不修改真实历史。

## 正常路径

1. 深拷贝 `api_messages`。
2. 对 system 应用 R4。
3. 对最后 3 条非 system 应用 R4。
4. 返回标记后的副本，交给阶段 6。

## 分支路径

- `api_messages` 为空 → 返回空列表。
- 非 system 消息不足 3 条 → 对所有非 system 消息打 marker（在 remaining 限额内）。
- 无 system 消息 → 4 个断点全部用于非 system 尾部窗口。
- `_use_prompt_caching == False` → 跳过本阶段。

## 失败处理

纯函数，无 IO。空消息列表安全返回。

## 幂等性 / 一致性约束

- 必须在阶段 4 之后、阶段 6 **之前**调用：归一化（strip/sort_keys）会改变 content 字节，影响 cache hit；源码顺序为 cache → sanitize → normalize（见阶段 6 说明）。
- 同一轮内 stable system 前缀不变，ephemeral system 变化会改变 system 断点内容，属预期。

## 代码骨架

```python
def apply_system_and_3_cache(
    api_messages: list[dict],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
) -> list[dict]:
    messages = copy.deepcopy(api_messages)
    marker = {"type": "ephemeral"}
    if cache_ttl == "1h":
        marker["ttl"] = "1h"
    used = 0
    if messages and messages[0].get("role") == "system":
        apply_marker(messages[0], marker, native_anthropic)
        used = 1
    non_sys = [i for i, m in enumerate(messages) if m.get("role") != "system"]
    for idx in non_sys[-(4 - used):]:
        apply_marker(messages[idx], marker, native_anthropic)
    return messages
```

## 最小验证清单

1. 有 system 时至少 1 个断点在 system 上。
2. 非 system 尾部最多 3 个断点；总断点 ≤ 4。
3. `cache_ttl="1h"` 时 marker 含 `"ttl": "1h"`。
4. OpenRouter 模式下 `role=tool` 消息不加 cache_control。
5. 返回列表与输入不是同一对象（deepcopy）。

## 来源证据（仅供溯源核实）

- `agent/prompt_caching.py:41-72`：`apply_anthropic_cache_control()` 与 `system_and_3` 策略。
- `agent/prompt_caching.py:15-38`：`_apply_cache_marker()` 各 content 形态与 tool 分支。
- `run_agent.py:7799-7800`：主循环调用点与 `cache_ttl`、`native_anthropic` 参数。
