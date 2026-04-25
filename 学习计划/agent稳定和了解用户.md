我看下来，Hermes 稳定和“越来越懂用户”的核心不是模型更强，而是它自己实现了一套 **runtime + 信息分层**。

**让 Agent 稳定的点**

1. **System prompt 稳定，不在每轮乱变**  
   `_build_system_prompt()` 每个会话只构建一次，缓存到 `_cached_system_prompt`，继续会话时还会从 SQLite 复用旧的 system prompt 快照。这样模型的身份、规则、工具说明不会每轮漂移，也能吃到 Anthropic prefix cache。  
   参考：[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:3001)、[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:7621)

2. **临时上下文和长期提示分开**  
   `ephemeral_system_prompt`、memory prefetch、plugin context 都只在 API 调用时注入，不写入会话历史和 system prompt。这样当前任务可以得到额外信息，但不会污染长期上下文。  
   参考：[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:7869)

3. **工具不是随便暴露，而是 registry + toolset 管控**  
   工具统一注册到 `tools/registry.py`，再由 `toolsets.py` 决定哪些工具能给模型。`check_fn` 失败的工具不会暴露，避免模型调用不可用工具。  
   参考：[tools/registry.py](/d:/git/geren/hermes-agent/tools/registry.py:48)、[model_tools.py](/d:/git/geren/hermes-agent/model_tools.py:234)、[toolsets.py](/d:/git/geren/hermes-agent/toolsets.py:68)

4. **agent-level tools 单独拦截**  
   `memory`、`todo`、`session_search`、`delegate_task` 这类需要 Agent 内部状态的工具不走普通 registry dispatch，而是在 `run_agent.py` 里特殊处理。这样不会丢状态，也不会误用。  
   参考：[model_tools.py](/d:/git/geren/hermes-agent/model_tools.py:364)

5. **上下文压缩避免长会话崩溃**  
   `ContextCompressor` 会保护开头、保护最近尾部，把中间历史总结掉；还会在请求前 preflight，遇到 413/context overflow 时尝试压缩后重试。  
   参考：[agent/context_compressor.py](/d:/git/geren/hermes-agent/agent/context_compressor.py:60)、[agent/context_compressor.py](/d:/git/geren/hermes-agent/agent/context_compressor.py:666)、[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:7674)

6. **工具调用协议有修复层**  
   它会清理孤立 tool result、补 stub、修复/拒绝无效工具名、处理无效 JSON 参数，让模型犯小错时能自我纠正，而不是整个会话炸掉。  
   参考：[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:3179)、[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:9500)

7. **子智能体隔离噪音**  
   `delegate_task` 给子 agent 新上下文、独立 terminal session、受限 toolset，并且 `skip_memory=True`、`skip_context_files=True`，父 agent 只拿 summary。复杂任务不会把主上下文塞满。  
   参考：[tools/delegate_tool.py](/d:/git/geren/hermes-agent/tools/delegate_tool.py:1)、[tools/delegate_tool.py](/d:/git/geren/hermes-agent/tools/delegate_tool.py:348)

8. **会话、审批、checkpoint 提供恢复能力**  
   SQLite 保存消息、system prompt、FTS 搜索；危险命令有 approval；文件修改前可 checkpoint。这些是产品级 agent 稳定性的底座。  
   参考：[hermes_state.py](/d:/git/geren/hermes-agent/hermes_state.py:5)、[tools/approval.py](/d:/git/geren/hermes-agent/tools/approval.py:690)、[tools/checkpoint_manager.py](/d:/git/geren/hermes-agent/tools/checkpoint_manager.py:1)

**让它越来越懂用户的点**

1. **Memory 只存“长期有用”的用户信息**  
   `MEMORY.md` 存环境、项目习惯、工具经验；`USER.md` 存用户偏好、沟通风格、工作习惯。并且有字符上限，避免记忆无限膨胀。  
   参考：[tools/memory_tool.py](/d:/git/geren/hermes-agent/tools/memory_tool.py:1)、[tools/memory_tool.py](/d:/git/geren/hermes-agent/tools/memory_tool.py:100)

2. **Memory 是冻结快照，不会中途改 system prompt**  
   中途写 memory 会立刻落盘，但本轮 system prompt 不变，下一会话再刷新。这保证“越来越懂用户”和“当前会话稳定”不冲突。  
   参考：[tools/memory_tool.py](/d:/git/geren/hermes-agent/tools/memory_tool.py:335)

3. **MemoryManager 支持外部记忆提供商**  
   每轮前 `prefetch_all()` 拉相关记忆，注入到当前用户消息；每轮结束 `sync_all()` 同步用户输入和最终回答。  
   参考：[agent/memory_manager.py](/d:/git/geren/hermes-agent/agent/memory_manager.py:167)、[agent/memory_manager.py](/d:/git/geren/hermes-agent/agent/memory_manager.py:199)

4. **session_search 负责“情景记忆”**  
   它不是把所有历史塞 prompt，而是用 FTS5 搜索历史会话，再把命中的会话总结成当前任务需要的 recall。适合“上次我们怎么做的”。  
   参考：[tools/session_search_tool.py](/d:/git/geren/hermes-agent/tools/session_search_tool.py:247)、[hermes_state.py](/d:/git/geren/hermes-agent/hermes_state.py:990)

5. **Skills 让 Agent 记住可复用工作流**  
   memory 记事实和偏好，skills 记“以后怎么做”。系统 prompt 只放技能索引，真正需要时再 `skill_view` 加载全文，避免 prompt 噪音。  
   参考：[agent/prompt_builder.py](/d:/git/geren/hermes-agent/agent/prompt_builder.py:563)

6. **后台 review 自动提炼记忆和技能**  
   主任务完成后，它会后台审查对话，看是否应该保存 memory 或更新 skill，不和用户当前任务争上下文注意力。  
   参考：[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:2066)、[run_agent.py](/d:/git/geren/hermes-agent/run_agent.py:10254)

所以可以概括成一句话：

Hermes 的稳定来自 **固定核心 prompt、受控工具、上下文压缩、错误恢复、子任务隔离**；它越来越懂用户来自 **USER.md/MEMORY.md、session_search、skills、后台提炼、按用户/平台隔离会话**。真正关键是“什么信息放在哪一层、什么时候注入、什么时候持久化”。