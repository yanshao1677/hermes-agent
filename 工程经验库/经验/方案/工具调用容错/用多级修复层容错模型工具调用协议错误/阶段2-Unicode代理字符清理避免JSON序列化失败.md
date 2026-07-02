---
title: 阶段2-Unicode代理字符清理避免JSON序列化失败
level: atomic
parent: 用多级修复层容错模型工具调用协议错误
status: draft
tags:
  - unicode
  - json-serialization
created_at: 2026-07-02
updated_at: 2026-07-02
confidence: high
related:
  - 阶段1-请求前清理孤立工具结果并补齐缺失结果.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
---

# 阶段2：Unicode 代理字符清理避免 JSON 序列化失败

## 触发条件

1. **回合入口**：`user_message` / `persist_user_message` 为 str 时（粘贴富文本常见）。
2. **API 失败恢复**：捕获 `UnicodeEncodeError` 且 `_unicode_sanitization_passes < 2` 时，对 `messages` 就地清理后 `continue` 重试。

## 输入字段

| 字段 | 说明 |
|------|------|
| `_SURROGATE_RE` | `[\ud800-\udfff]` |
| 替换字符 | `U+FFFD`（`\ufffd`） |
| `_unicode_sanitization_passes` | 每 user turn 初始 0，最多 2 次 sanitize 重试 |

## 判定规则

**单字符串 `_sanitize_surrogates(text)`**

- 若匹配 surrogate → 全部替换为 `\ufffd`；否则原样返回。

**消息列表 `_sanitize_messages_surrogates(messages)` 就地修改**

扫描每条 dict 消息：

- `content` 字符串
- `content` 列表中 part 的 `text`
- `name`
- `tool_calls[].id`
- `tool_calls[].function.name`
- `tool_calls[].function.arguments`

任一命中 surrogate → 替换并设 `found=True`。返回是否发现代理字符。

**API 错误恢复分支**

1. 先 `_sanitize_messages_surrogates(messages)`；若 found → passes+=1，重试。
2. 若错误信息含 `'ascii'` codec 且仍失败 → 可选 `_sanitize_messages_non_ascii`（本卡不展开）。

## 状态读写位置

- 写：user 输入副本、API 重试时的 messages 就地字段。
- 不写 DB（除非后续 persist 已发生的 messages）。

## 正常路径

回合开始 sanitize 用户输入 → 正常 API 无 UnicodeEncodeError。

## 分支路径

- 无 surrogate → 空操作。
- API 抛 UnicodeEncodeError + 清理成功 → 重试同一请求。

## 失败处理

清理后仍失败 → 落入常规 API 错误处理，不再无限 sanitize。

## 幂等性 / 一致性约束

已替换的 `\ufffd` 不再匹配 surrogate regex；二次清理为 no-op。替换可能改变用户粘贴内容的语义（罕见可接受 vs 崩溃）。

## 代码骨架

```python
SURROGATE_RE = re.compile(r'[\ud800-\udfff]')

def sanitize_surrogates(text: str) -> str:
    if SURROGATE_RE.search(text):
        return SURROGATE_RE.sub('\ufffd', text)
    return text
```

## 最小验证清单

- 含 `\ud800` 的用户消息在回合入口被替换。
- tool_calls.arguments 中 surrogate 在 API 重试路径被清理。
- `_unicode_sanitization_passes` 每 turn 从 0 开始。

## 来源证据（附录，不进正文）

- `run_agent.py:337-350`：`_sanitize_surrogates`。
- `run_agent.py:353-398`：`_sanitize_messages_surrogates`。
- `run_agent.py:7437-7440`：回合入口用户消息。
- `run_agent.py:8437-8456`：UnicodeEncodeError 恢复。
