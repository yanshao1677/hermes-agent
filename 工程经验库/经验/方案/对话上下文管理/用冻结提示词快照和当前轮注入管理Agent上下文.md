---
title: 用冻结提示词快照和当前轮注入管理Agent上下文
level: pattern
parent: 
status: draft
tags:
  - ai-agent
  - context-management
  - session-persistence
  - memory-prefetch
  - prompt-cache
created_at: 2026-06-09
updated_at: 2026-07-02
confidence: high
related:
  - ../../架构/用稳定系统提示词与临时上下文分层保持长会话智能体稳定.md
  - 用冻结提示词快照和当前轮注入管理Agent上下文/方案目录.md
source_repo: hermes-agent
source_commit: ce1dc9f77865c2c92c9e39f45fcf9c3ca46ecfdc
source_paths:
  - run_agent.py
  - agent/memory_manager.py
  - agent/prompt_caching.py
  - hermes_state.py
---

# 用冻结提示词快照和当前轮注入管理Agent上下文

> **架构关系**：ADR《用稳定系统提示词与临时上下文分层》定义 WHY（稳定/临时/真实历史三层）；本方案定义 HOW（11 阶段流水线）。外部记忆与 FTS 召回分别在记忆域方案中展开。

## 1. 问题

长会话智能体既要“记住稳定规则”，又要“理解当前任务相关的临时信息”。如果所有信息都实时拼进系统提示词，会破坏缓存并导致规则漂移；如果只用固定提示词，又无法把当前轮相关召回提供给模型。

本方案解决的问题是：如何在同一个会话中同时支持稳定系统提示词、当前轮临时召回、真实历史持久化、会话恢复和上下文压缩，而不让这些信息相互污染。

## 2. 适用约束

适用前提：

- 系统存在会话级稳定提示词，且构建成本或 token 数较高。
- 每轮可能出现临时上下文，例如外部记忆 prefetch、插件 hook 返回内容、少样本预填充。
- 需要把真实用户、助手、工具消息持久化到数据库或日志。
- 有跨进程、跨入口恢复会话的需求。
- 上下文可能超过模型窗口，需要压缩旧历史。

不要求必须使用某个特定模型或数据库，但需要具备以下能力：会话元数据存储、消息列表复制、请求前注入、压缩后重建快照。

## 3. 核心思路

把“会话事实”和“模型请求”拆成两份数据：真实历史只保存用户、助手、工具消息；每次模型调用都从真实历史复制出请求副本，并只在副本中追加本轮临时上下文和预填充内容。

## 4. 通用结构

### 4.1 数据对象

- **Session**：会话元数据，至少包含 `id`、`system_prompt_snapshot`、`parent_session_id`、开始/结束时间。
- **Messages**：真实消息序列，只包含可回放、可审计、可搜索的 user/assistant/tool 消息。
- **PromptSnapshot**：会话级系统提示词快照，包含身份、规则、工具指导、长期记忆快照、技能索引、平台规则等稳定内容。
- **RequestMessages**：一次模型调用的副本，由 `PromptSnapshot + Prefill + copied Messages + current-turn injections` 组成。
- **EphemeralContext**：调用级上下文，例如外部记忆召回、插件结果、临时系统补充，不作为真实历史写入。
- **ContextCompressor**：当 token 超预算时，把中间历史压缩成摘要，保护头部和尾部上下文，并产生新的会话片段。

### 4.2 状态字段

- `cached_system_prompt`：进程内当前会话稳定快照。
- `current_turn_user_idx`：当前轮 user 消息在真实历史中的下标，用于只向本轮 user 注入临时上下文。
- `persist_user_message_override` / `persist_user_message_idx`：API user 与持久化 user 分离（见阶段 7）。
- `last_flushed_db_idx`：记录已写入数据库的消息位置，防止重复写入。
- `parent_session_id`：压缩或委托产生的新会话指向父会话。

## 5. 处理流程

完整 11 阶段见 [方案目录](用冻结提示词快照和当前轮注入管理Agent上下文/方案目录.md)。主路径摘要：

| 顺序 | 阶段 | 要点 |
|------|------|------|
| 1 | 准备真实历史 + user 分离 | 追加 user；记录 idx；可选 persist override |
| 2 | 恢复/构建 system 快照 | cached → DB → rebuild |
| 3 | 预压缩（可选） | 主循环前；含 tools token 估算 |
| 4 | 收集临时上下文 | prefetch + 插件；一次收集多次复用 |
| 5 | 构造 API 副本 | user 注入 → system/prefill → cache → sanitize |
| 6 | 模型与工具循环 | 每轮重新走阶段 5 |
| 7 | 持久化 | 真实 history + flush 游标 |
| 8 | 压缩 | flush memory → compress → 新 session + 新 snapshot |
| 9 | API 错误压缩 | 413 / overflow / tier 降级；最多 3 次 |

### 请求副本构造子流水线（每轮 API 调用）

1. **阶段 1**：复制历史，当前 user 注入 prefetch/插件。
2. **阶段 4**：插入 system（stable + ephemeral）与 prefill。
3. **阶段 5**：Anthropic `system_and_3` 缓存断点。
4. **阶段 6**：sanitize、strip、tool args sort_keys。

## 6. 异常处理

- 系统提示词从会话元数据恢复失败：退回重新构建；该退回会影响缓存一致性，但比中断会话更可用。
- 插件临时上下文失败：记录警告后忽略，不阻断当前轮。
- 外部记忆 prefetch 失败：跳过失败 provider，不影响其他 provider。
- 消息持久化失败：记录警告；真实历史仍在内存返回值中，但跨进程恢复能力下降。
- 压缩会话拆分失败：继续使用压缩后的内存消息，但新会话可能无法被索引。
- API context/413 错误：阶段 11 触发 compress 重试，最多 3 次；失败返回 partial 并建议 `/new`。
- 摘要模型不可用：证据不足：已核实上下文压缩器具备失败冷却常量和摘要流程，但本方案未深读完整压缩失败分支，不能确认所有失败时的最终返回策略。

## 7. 具体语言实现

下面是一个 Python 骨架，表达该方案的数据分层和注入时机。它不绑定具体模型 SDK，可作为实现时的结构参考。

```python
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Session:
    id: str
    system_prompt_snapshot: str = ""
    parent_session_id: Optional[str] = None


@dataclass
class TurnRuntime:
    cached_system_prompt: Optional[str] = None
    last_flushed_idx: int = 0
    persist_override: Optional[str] = None
    persist_idx: Optional[int] = None


class SessionStore:
    def __init__(self):
        self.sessions: dict[str, Session] = {}
        self.messages: dict[str, list[dict]] = {}

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def ensure_session(self, session_id: str) -> Session:
        return self.sessions.setdefault(session_id, Session(id=session_id))

    def update_system_prompt(self, session_id: str, prompt: str) -> None:
        session = self.ensure_session(session_id)
        session.system_prompt_snapshot = prompt

    def append_messages(self, session_id: str, messages: list[dict], start: int) -> int:
        bucket = self.messages.setdefault(session_id, [])
        for message in messages[start:]:
            bucket.append(dict(message))
        return len(messages)


class ContextRunner:
    def __init__(
        self,
        session_id: str,
        store: SessionStore,
        build_system_prompt: Callable[[], str],
        call_model: Callable[[list[dict]], dict],
        apply_cache: Callable[[list[dict]], list[dict]] | None = None,
        finalize_request: Callable[[list[dict]], list[dict]] | None = None,
    ):
        self.session_id = session_id
        self.store = store
        self.build_system_prompt = build_system_prompt
        self.call_model = call_model
        self.apply_cache = apply_cache
        self.finalize_request = finalize_request
        self.runtime = TurnRuntime()

    def get_or_build_prompt(self, has_history: bool) -> str:
        if self.runtime.cached_system_prompt:
            return self.runtime.cached_system_prompt
        stored = self.store.get_session(self.session_id)
        if has_history and stored and stored.system_prompt_snapshot:
            self.runtime.cached_system_prompt = stored.system_prompt_snapshot
        else:
            self.runtime.cached_system_prompt = self.build_system_prompt()
            self.store.update_system_prompt(self.session_id, self.runtime.cached_system_prompt)
        return self.runtime.cached_system_prompt

    def build_request_messages(
        self,
        messages: list[dict],
        current_turn_user_idx: int,
        ephemeral_context: str = "",
        ephemeral_system_prompt: str = "",
        prefill_messages: Optional[list[dict]] = None,
    ) -> list[dict]:
        api_messages: list[dict] = []
        for idx, message in enumerate(messages):
            request_message = dict(message)
            if idx == current_turn_user_idx and request_message.get("role") == "user":
                if ephemeral_context:
                    request_message["content"] = (
                        str(request_message.get("content", ""))
                        + "\n\n"
                        + ephemeral_context
                    )
            api_messages.append(request_message)

        system_content = self.runtime.cached_system_prompt or ""
        if ephemeral_system_prompt:
            system_content = (system_content + "\n\n" + ephemeral_system_prompt).strip()
        if system_content:
            api_messages.insert(0, {"role": "system", "content": system_content})

        if prefill_messages:
            offset = 1 if system_content else 0
            for index, message in enumerate(prefill_messages):
                api_messages.insert(offset + index, dict(message))

        if self.apply_cache:
            api_messages = self.apply_cache(api_messages)
        if self.finalize_request:
            api_messages = self.finalize_request(api_messages)
        return api_messages

    def run_turn(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
        persist_user_message: Optional[str] = None,
        ephemeral_context: str = "",
        ephemeral_system_prompt: str = "",
        prefill_messages: Optional[list[dict]] = None,
    ) -> dict:
        messages = list(history or [])
        messages.append({"role": "user", "content": user_message})
        current_turn_user_idx = len(messages) - 1
        self.runtime.persist_idx = current_turn_user_idx
        self.runtime.persist_override = persist_user_message

        self.get_or_build_prompt(has_history=bool(history))
        request_messages = self.build_request_messages(
            messages,
            current_turn_user_idx,
            ephemeral_context=ephemeral_context,
            ephemeral_system_prompt=ephemeral_system_prompt,
            prefill_messages=prefill_messages,
        )
        response = self.call_model(request_messages)
        messages.append({"role": "assistant", "content": response.get("content", "")})
        if self.runtime.persist_override is not None:
            messages[current_turn_user_idx]["content"] = self.runtime.persist_override
        self.runtime.last_flushed_idx = self.store.append_messages(
            self.session_id,
            messages,
            self.runtime.last_flushed_idx,
        )
        return {"messages": messages, "response": response}
```

## 8. 测试点

- 续会话已有 `system_prompt_snapshot` 时，不调用系统提示词构建函数。
- 当前轮临时上下文只追加到当前用户消息副本，不改变真实历史中的用户消息（persist 前）。
- 预填充消息只出现在请求副本中，真实历史不包含预填充消息。
- 插件或外部记忆召回失败时，模型调用仍使用无该上下文的请求副本继续执行。
- 多次工具循环中，真实历史会追加助手与工具消息；每次请求都重新复制历史并重新注入当前轮上下文。
- 压缩后，新会话拥有新的系统提示词快照，并保留父会话引用。
- `persist_user_message` 与 `user_message` 不同时，DB 存前者、API 用后者。
- 主循环前 token 超阈值时预压缩，且压缩后 `conversation_history` 引用清空。

## 9. 适用场景 / 不适用场景

适用：

- 多轮智能体、工具调用、长期记忆、会话恢复、上下文压缩并存的运行时。
- 系统提示词中包含用户偏好、工具指导、技能索引等高价值稳定内容。
- 外部召回内容与当前任务相关，但不应成为长期规则。

不适用：

- 一次性脚本式调用。
- 无会话恢复、无外部召回、无压缩需求的简单聊天。
- 临时上下文需要严格审计和回放的合规场景；此时应把临时上下文单独审计，而不是只存在请求副本。

## 10. 风险与反模式

- **反模式：把 prefetch 结果放入系统 prompt。** 这会把当前任务相关背景误提升为长期规则。
- **反模式：把请求副本持久化。** 这会让临时上下文、预填充示例和清理后的 API 字段污染真实历史。
- **反模式：续会话时重新构建快照。** 这会让跨进程恢复得到不同的系统提示词，破坏前缀一致性。
- **风险：调试不可见。** 临时注入不进持久层，线上问题排查时需要额外记录注入摘要或请求审计。
- **风险：快照过旧。** 会话中途写入长期记忆不会立即进入稳定快照，必须通过新会话或压缩后重建生效。

## 11. 标签

context-management, session-persistence, prompt-cache, memory-prefetch, request-copy, long-session

## 附录：来源证据（仅供溯源核实，阅读正文无需依赖此节）

- `run_agent.py:7490-7520`：`run_conversation()` 复制历史、追加当前 user 消息，并记录 `current_turn_user_idx`。
- `run_agent.py:7444-7450`、`7504`、`2164-2178`：`persist_user_message` 分离与 persist 前覆盖。
- `run_agent.py:7531-7564`：优先恢复会话 DB 中的 `system_prompt`；否则构建新系统提示词并更新 DB。
- `run_agent.py:7569-7615`：主循环前预压缩（含 tools token、最多 3 pass、`conversation_history=None`）。
- `run_agent.py:7617-7673`：插件与外部记忆 prefetch；注释说明临时 context 不进 session DB。
- `run_agent.py:7738-7830`：API 副本构造流水线（user 注入 → system/prefill → cache → sanitize → normalize）。
- `run_agent.py:6464-6521`：`_compress_context()` 压缩前后状态迁移。
- `run_agent.py:7867`、`8628-8854`、`9497-9507`：API 错误与 token 压力触发的 compress 重试。
- `run_agent.py:2180-2236`：持久化与 `_last_flushed_db_idx` 防重复写入。
- `agent/prompt_caching.py:41-72`：system_and_3 缓存断点。
- `hermes_state.py:41-68`：sessions 表 `system_prompt` 与 `parent_session_id`。
- 证据不足：压缩摘要失败后的完整恢复策略未在本次任务中深读到足够细节。
