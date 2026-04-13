<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/文档-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/许可证-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/由%20Nous%20Research%20构建-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**由 [Nous Research](https://nousresearch.com) 构建的自我改进 AI 代理。** 它是唯一具有内置学习循环的代理——从经验中创建技能，在使用过程中改进技能，主动提醒自己持久化知识，搜索自己的历史对话，并跨会话构建对用户的深层理解模型。可以在 $5 的 VPS 上运行，也可以在 GPU 集群上运行，或者在空闲时几乎零成本的 serverless 基础设施上运行。它不局限于你的笔记本电脑——可以从 Telegram 与它交流，同时它在云 VM 上工作。

使用你想要的任何模型——[Nous Portal](https://portal.nousresearch.com)、[OpenRouter](https://openrouter.ai)（200+ 模型）、[z.ai/GLM](https://z.ai)、[Kimi/Moonshot](https://platform.moonshot.ai)、[MiniMax](https://www.minimax.io)、OpenAI，或你自己的端点。使用 `hermes model` 切换——无需代码更改，无锁定。

<table>
<tr><td><b>真正的终端界面</b></td><td>完整的 TUI，支持多行编辑、斜杠命令自动补全、对话历史、中断重定向和流式工具输出。</td></tr>
<tr><td><b>在你所在的地方运行</b></td><td>Telegram、Discord、Slack、WhatsApp、Signal 和 CLI——全部来自单一的 gateway 进程。语音备忘录转录、跨平台对话连续性。</td></tr>
<tr><td><b>闭环学习</b></td><td>代理管理的记忆，带有周期性提醒。复杂任务后自主创建技能。技能在使用过程中自我改进。FTS5 会话搜索配合 LLM 摘要用于跨会话回忆。<a href="https://github.com/plastic-labs/honcho">Honcho</a> 辩证式用户建模。兼容 <a href="https://agentskills.io">agentskills.io</a> 开放标准。</td></tr>
<tr><td><b>计划自动化</b></td><td>内置 cron 调度器，可分发到任何平台。每日报告、夜间备份、每周审计——全部用自然语言描述，无人值守运行。</td></tr>
<tr><td><b>委托与并行化</b></td><td>为并行工作流生成隔离的子代理。编写通过 RPC 调用工具的 Python 脚本，将多步骤管道压缩为零上下文成本的轮次。</td></tr>
<tr><td><b>随处运行，不限于笔记本电脑</b></td><td>六种终端后端——本地、Docker、SSH、Daytona、Singularity 和 Modal。Daytona 和 Modal 提供 serverless 持久化——你的代理环境在空闲时休眠，按需唤醒，会话间隔成本几乎为零。在 $5 的 VPS 或 GPU 集群上运行。</td></tr>
<tr><td><b>研究就绪</b></td><td>批量轨迹生成、Atropos RL 环境、轨迹压缩，用于训练下一代工具调用模型。</td></tr>
</table>

---

## 快速安装

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

适用于 Linux、macOS、WSL2 和通过 Termux 的 Android。安装程序会为你处理平台特定的设置。

> **Android / Termux：** 经过测试的手动路径已记录在 [Termux 指南](https://hermes-agent.nousresearch.com/docs/getting-started/termux) 中。在 Termux 上，Hermes 安装精选的 `.[termux]` extra，因为完整的 `.[all]` extra 目前会拉取 Android 不兼容的语音依赖。
>
> **Windows：** 原生 Windows 不受支持。请安装 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 并运行上述命令。

安装后：

```bash
source ~/.bashrc    # 重载 shell（或：source ~/.zshrc）
hermes              # 开始对话！
```

---

## 入门指南

```bash
hermes              # 交互式 CLI——开始对话
hermes model        # 选择你的 LLM 提供商和模型
hermes tools        # 配置启用的工具
hermes config set   # 设置单个配置值
hermes gateway      # 启动消息 gateway（Telegram、Discord 等）
hermes setup        # 运行完整设置向导（一次性配置所有内容）
hermes claw migrate # 从 OpenClaw 迁移（如果你来自 OpenClaw）
hermes update       # 更新到最新版本
hermes doctor       # 诊断任何问题
```

📖 **[完整文档 →](https://hermes-agent.nousresearch.com/docs/)**

## CLI 与消息平台快速参考

Hermes 有两个入口点：使用 `hermes` 启动终端 UI，或运行 gateway 并从 Telegram、Discord、Slack、WhatsApp、Signal 或 Email 与它交流。一旦进入对话，许多斜杠命令在两个界面间共享。

| 操作 | CLI | 消息平台 |
|---------|-----|---------------------|
| 开始聊天 | `hermes` | 运行 `hermes gateway setup` + `hermes gateway start`，然后给机器人发消息 |
| 开始新对话 | `/new` 或 `/reset` | `/new` 或 `/reset` |
| 更换模型 | `/model [provider:model]` | `/model [provider:model]` |
| 设置个性 | `/personality [name]` | `/personality [name]` |
| 重试或撤销上一轮 | `/retry`、`/undo` | `/retry`、`/undo` |
| 压缩上下文 / 查看用量 | `/compress`、`/usage`、`/insights [--days N]` | `/compress`、`/usage`、`/insights [days]` |
| 浏览技能 | `/skills` 或 `/<skill-name>` | `/skills` 或 `/<skill-name>` |
| 中断当前工作 | `Ctrl+C` 或发送新消息 | `/stop` 或发送新消息 |
| 平台特定状态 | `/platforms` | `/status`、`/sethome` |

完整的命令列表，请参阅 [CLI 指南](https://hermes-agent.nousresearch.com/docs/user-guide/cli) 和 [消息 Gateway 指南](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)。

---

## 文档

所有文档位于 **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**：

| 章节 | 涵盖内容 |
|---------|---------------|
| [快速入门](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart) | 安装 → 设置 → 2分钟内开始首次对话 |
| [CLI 使用](https://hermes-agent.nousresearch.com/docs/user-guide/cli) | 命令、快捷键、个性、会话 |
| [配置](https://hermes-agent.nousresearch.com/docs/user-guide/configuration) | 配置文件、提供商、模型、所有选项 |
| [消息 Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) | Telegram、Discord、Slack、WhatsApp、Signal、Home Assistant |
| [安全](https://hermes-agent.nousresearch.com/docs/user-guide/security) | 命令审批、DM 配对、容器隔离 |
| [工具与工具集](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) | 40+ 工具、工具集系统、终端后端 |
| [技能系统](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | 程序性记忆、技能中心、创建技能 |
| [记忆](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) | 持久记忆、用户档案、最佳实践 |
| [MCP 集成](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) | 连接任何 MCP 服务器以扩展能力 |
| [Cron 调度](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) | 带平台分发的计划任务 |
| [上下文文件](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | 塑造每次对话的项目上下文 |
| [架构](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) | 项目结构、代理循环、关键类 |
| [贡献](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) | 开发设置、PR 流程、代码风格 |
| [CLI 参考](https://hermes-agent.nousresearch.com/docs/reference/cli-commands) | 所有命令和标志 |
| [环境变量](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | 完整的环境变量参考 |

---

## 从 OpenClaw 迁移

如果你来自 OpenClaw，Hermes 可以自动导入你的设置、记忆、技能和 API 密钥。

**首次设置期间：** 设置向导（`hermes setup`）会自动检测 `~/.openclaw` 并在配置开始前提供迁移选项。

**安装后任何时候：**

```bash
hermes claw migrate              # 交互式迁移（完整预设）
hermes claw migrate --dry-run    # 预览将要迁移的内容
hermes claw migrate --preset user-data   # 不迁移敏感信息
hermes claw migrate --overwrite  # 覆盖现有冲突
```

导入内容：
- **SOUL.md** — 人设文件
- **记忆** — MEMORY.md 和 USER.md 条目
- **技能** — 用户创建的技能 → `~/.hermes/skills/openclaw-imports/`
- **命令白名单** — 审批模式
- **消息设置** — 平台配置、允许用户、工作目录
- **API 密钥** — 白名单密钥（Telegram、OpenRouter、OpenAI、Anthropic、ElevenLabs）
- **TTS 资源** — 工作空间音频文件
- **工作空间指令** — AGENTS.md（带 `--workspace-target`）

查看 `hermes claw migrate --help` 了解所有选项，或使用 `openclaw-migration` 技能进行交互式代理引导迁移，带 dry-run 预览。

---

## 贡献

我们欢迎贡献！请参阅 [贡献指南](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) 了解开发设置、代码风格和 PR 流程。

贡献者快速入门：

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

> **RL 训练（可选）：** 要参与 RL/Tinker-Atropos 集成开发：
> ```bash
> git submodule update --init tinker-atropos
> uv pip install -e "./tinker-atropos"
> ```

---

## 社区

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [技能中心](https://agentskills.io)
- 🐛 [问题](https://github.com/NousResearch/hermes-agent/issues)
- 💡 [讨论](https://github.com/NousResearch/hermes-agent/discussions)
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — 社区微信桥接：在同一微信账号上运行 Hermes Agent 和 OpenClaw。

---

## 许可证

MIT — 请参阅 [LICENSE](LICENSE)。

由 [Nous Research](https://nousresearch.com) 构建。