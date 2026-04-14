# Hermes Agent 学习计划

## 学习目标

这份计划的目标不是只看懂几个源码文件，而是建立下面 5 个能力：

1. 理解一个产品级 AI Agent 是怎么运行的
2. 理解为什么 Hermes 很多能力不完全交给框架，而是自己实现
3. 理解 prompt、memory、skills、session search、context compression 的分工
4. 理解主智能体和子智能体的职责边界
5. 形成自己的技术判断：什么时候用框架，什么时候更适合自研 runtime

---

## 总体路线

建议采用这条主线：

1. 先用 Hermes 建立“产品级 Agent 系统”整体感
2. 再拆主智能体、工具系统、记忆与压缩、子智能体、平台交付
3. 最后再回看 LangChain/LangGraph，做对照理解

一句话概括：

- LangChain 更适合学习“框架抽象”
- Hermes 更适合学习“系统落地”

---

## 阶段一：建立全局认知

预计时间：2-3 天

目标：

- 理解 Hermes 的核心卖点和整体架构
- 建立入口层、核心层、工具层、支撑层的全局视图

建议阅读顺序：

1. [README.md](/d:/git/geren/hermes-agent/README.md)
2. [AGENTS.md](/d:/git/geren/hermes-agent/AGENTS.md)
3. [pyproject.toml](/d:/git/geren/hermes-agent/pyproject.toml)

重点问题：

1. 这个项目到底解决什么问题
2. 它有哪些入口：CLI、Gateway、Batch、ACP
3. Agent 主循环在哪
4. 工具系统如何组织
5. 哪些能力已经超出“普通聊天封装”

阶段产出：

- 一页纸架构草图
- 一段 3 分钟项目概述

---

## 阶段二：主智能体与主提示词

预计时间：4-6 天

目标：

- 理解主智能体如何创建、如何运行、如何调用模型
- 理解 system prompt 如何组装、缓存、失效和重建

建议阅读顺序：

1. [run_agent.py](/d:/git/geren/hermes-agent/run_agent.py)
2. [agent/prompt_builder.py](/d:/git/geren/hermes-agent/agent/prompt_builder.py)
3. [agent/prompt_caching.py](/d:/git/geren/hermes-agent/agent/prompt_caching.py)

重点问题：

1. `AIAgent` 是怎么创建的
2. `run_conversation()` 的主循环怎么跑
3. `_build_system_prompt()` 组装了哪些层
4. `_cached_system_prompt` 为什么不能每轮都重建
5. `ephemeral_system_prompt` 是什么，为什么不进持久 system prompt
6. 用户通过 `/personality` 改的是哪一层提示词

阶段产出：

- 一张“主智能体一次请求前的时序图”
- 一份“主提示词分层说明”

---

## 阶段三：工具系统与工具调度

预计时间：4-5 天

目标：

- 理解 Hermes 的工具不是散落在各处，而是 registry 驱动
- 理解 toolset 为什么是这个项目的关键设计

建议阅读顺序：

1. [model_tools.py](/d:/git/geren/hermes-agent/model_tools.py)
2. [tools/registry.py](/d:/git/geren/hermes-agent/tools/registry.py)
3. [toolsets.py](/d:/git/geren/hermes-agent/toolsets.py)
4. [tools/file_tools.py](/d:/git/geren/hermes-agent/tools/file_tools.py)
5. [tools/web_tools.py](/d:/git/geren/hermes-agent/tools/web_tools.py)

重点问题：

1. 工具如何注册
2. 工具 schema 如何暴露给模型
3. toolset 如何控制能力范围
4. 为什么不是把所有工具始终开放给模型
5. agent-level tools 为什么要单独拦截

阶段产出：

- 一张“registry -> toolset -> dispatch”的流程图
- 一段“新增一个工具需要改哪些层”的说明

---

## 阶段四：记忆、技能、会话检索、上下文压缩

预计时间：5-7 天

目标：

- 理解 Hermes 为什么不靠“大 prompt”硬扛复杂度
- 学会从信息分层角度理解 Agent 系统

建议阅读顺序：

1. [agent/context_compressor.py](/d:/git/geren/hermes-agent/agent/context_compressor.py)
2. [agent/memory_manager.py](/d:/git/geren/hermes-agent/agent/memory_manager.py)
3. [agent/skill_commands.py](/d:/git/geren/hermes-agent/agent/skill_commands.py)
4. [tools/skills_tool.py](/d:/git/geren/hermes-agent/tools/skills_tool.py)
5. [tools/skill_manager_tool.py](/d:/git/geren/hermes-agent/tools/skill_manager_tool.py)
6. [tools/session_search_tool.py](/d:/git/geren/hermes-agent/tools/session_search_tool.py)

重点问题：

1. memory 负责什么，不负责什么
2. session search 与 memory 的区别是什么
3. skills 为什么不直接全文塞进 system prompt
4. context overflow 出现时 Hermes 如何恢复
5. 为什么这几个能力要分层，而不是混成一种“上下文”

阶段产出：

- 一张“system prompt / memory / skills / session search / compressed history”信息分层图
- 一份“Hermes 如何控制 prompt 噪音”的总结

---

## 阶段五：子智能体与任务隔离

预计时间：3-4 天

目标：

- 理解 Hermes 为什么引入子智能体
- 理解“子智能体不是更聪明，而是更隔离”

建议阅读顺序：

1. [tools/delegate_tool.py](/d:/git/geren/hermes-agent/tools/delegate_tool.py)
2. [run_agent.py](/d:/git/geren/hermes-agent/run_agent.py) 中与 `delegate_task` 相关部分

重点问题：

1. 子智能体是如何创建的
2. 为什么 `skip_memory=True`
3. 为什么 `skip_context_files=True`
4. 为什么要缩小 toolset
5. 为什么 parent 只看 summary，不看子过程
6. 哪些任务适合委托，哪些任务不适合

阶段产出：

- 一张“主智能体 vs 子智能体”的职责对比表
- 一段“为什么子智能体能降低噪音”的总结

---

## 阶段六：平台交付、返回协议、用户交互

预计时间：4-6 天

目标：

- 理解 Agent 产品最后一公里如何落地
- 理解“模型输出”如何转成用户真正看到的交互

建议阅读顺序：

1. [gateway/run.py](/d:/git/geren/hermes-agent/gateway/run.py)
2. [gateway/platforms/base.py](/d:/git/geren/hermes-agent/gateway/platforms/base.py)
3. 任选一个平台深入：
   - [gateway/platforms/telegram.py](/d:/git/geren/hermes-agent/gateway/platforms/telegram.py)
   - [gateway/platforms/slack.py](/d:/git/geren/hermes-agent/gateway/platforms/slack.py)

重点问题：

1. 用户消息是如何进入 agent 的
2. 返回文本是如何发回平台的
3. `MEDIA:`、`[[audio_as_voice]]` 协议如何告诉模型、如何解析
4. 为什么按钮、审批卡、model picker 不完全依赖模型协议
5. 平台断连、重试、用户中断如何处理

阶段产出：

- 一张“用户输入 -> agent -> 返回结果”的链路图
- 一张“返回协议与平台组件”总结表

---

## 阶段七：回看 LangChain / LangGraph

预计时间：2-3 天

目标：

- 不是只学 API，而是带着 Hermes 的理解去看框架抽象
- 建立自己的技术判断

建议关注的问题：

1. LangChain 把 Hermes 的哪些能力抽象掉了
2. 哪些地方用框架确实更高效
3. 哪些地方 Hermes 自己写更合理
4. 如果未来做自己的项目，哪些场景用框架，哪些场景倾向自研

阶段产出：

- 一张“LangChain vs Hermes”对照表
- 一份你自己的技术判断

建议判断模板：

- PoC / Demo / 简单业务：框架优先
- 多平台 / 长会话 / 强容错 / 多 provider / 复杂交付：更偏向自研 runtime

---

## 推荐节奏

如果按业余时间学习：

- 第 1 周：阶段一 + 阶段二
- 第 2 周：阶段三 + 阶段四
- 第 3 周：阶段五 + 阶段六
- 第 4 周：阶段七 + 总结

如果每天可投入较多时间：

- 10-14 天可以完成第一轮

---

## 每个阶段的固定输出方法

每完成一个阶段，都做下面 3 件事：

1. 写出这一层解决了什么问题
2. 写出如果没有这一层，会出什么问题
3. 写出如果用 LangChain/通用框架来做，哪些部分方便，哪些部分不够

这样你学到的不是“这个文件里写了什么”，而是“为什么要这样设计”。

---

## 最终要掌握的 5 个问题

学完后，你至少要能独立讲清楚：

1. 主智能体是怎么创建和运行的
2. 工具系统为什么要 registry + toolset
3. 为什么不能把所有能力都塞进主 prompt
4. 子智能体为什么要隔离 context、memory 和 toolset
5. 为什么生产级 agent 的关键是 runtime，而不只是模型和框架

---

## 一句话总结

这份学习计划的核心不是“读完 Hermes 源码”，而是：

通过 Hermes 学会如何从产品级、运行时、信息分层、容错和交付链路的角度理解 AI Agent 系统。
