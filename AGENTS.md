# Hermes Agent - 开发指南

AI 编码助手和在 hermes-agent 代码库上工作的开发者的指令。

## 开发环境

```bash
source venv/bin/activate  # 运行 Python 前必须激活
```

## 项目结构

```
hermes-agent/
├── run_agent.py          # AIAgent 类——核心对话循环
├── model_tools.py        # 工具编排、_discover_tools()、handle_function_call()
├── toolsets.py           # 工具集定义、_HERMES_CORE_TOOLS 列表
├── cli.py                # HermesCLI 类——交互式 CLI 协调器
├── hermes_state.py       # SessionDB——SQLite 会话存储（FTS5 搜索）
├── agent/                # 代理内部模块
│   ├── prompt_builder.py     # 系统提示词组装
│   ├── context_compressor.py # 自动上下文压缩
│   ├── prompt_caching.py     # Anthropic 提示词缓存
│   ├── auxiliary_client.py   # 辅助 LLM 客户端（视觉、摘要）
│   ├── model_metadata.py     # 模型上下文长度、token 估算
│   ├── models_dev.py         # models.dev 注册表集成（提供商感知上下文）
│   ├── display.py            # KawaiiSpinner、工具预览格式化
│   ├── skill_commands.py     # 技能斜杠命令（CLI/gateway 共享）
│   └── trajectory.py         # 轨迹保存辅助
├── hermes_cli/           # CLI 子命令和设置
│   ├── main.py           # 入口点——所有 `hermes` 子命令
│   ├── config.py         # DEFAULT_CONFIG、OPTIONAL_ENV_VARS、迁移
│   ├── commands.py       # 斜杠命令定义 + SlashCommandCompleter
│   ├── callbacks.py      # 终端回调（clarify、sudo、approval）
│   ├── setup.py          # 交互式设置向导
│   ├── skin_engine.py    # Skin/主题引擎——CLI 视觉定制
│   ├── skills_config.py  # `hermes skills`——按平台启用/禁用技能
│   ├── tools_config.py   # `hermes tools`——按平台启用/禁用工具
│   ├── skills_hub.py     # `/skills` 斜杠命令（搜索、浏览、安装）
│   ├── models.py         # 模型目录、提供商模型列表
│   ├── model_switch.py   # 共享 /model 切换管道（CLI + gateway）
│   └── auth.py           # 提供商凭证解析
├── tools/                # 工具实现（每个工具一个文件）
│   ├── registry.py       # 中央工具注册表（schema、处理器、分发）
│   ├── approval.py       # 危险命令检测
│   ├── terminal_tool.py  # 终端编排
│   ├── process_registry.py # 后台进程管理
│   ├── file_tools.py     # 文件读/写/搜索/补丁
│   ├── web_tools.py      # Web 搜索/提取（Parallel + Firecrawl）
│   ├── browser_tool.py   # Browserbase 浏览器自动化
│   ├── code_execution_tool.py # execute_code 沙箱
│   ├── delegate_tool.py  # 子代理委托
│   ├── mcp_tool.py       # MCP 客户端（~1050 行）
│   └── environments/     # 终端后端（local、docker、ssh、modal、daytona、singularity）
├── gateway/              # 消息平台 gateway
│   ├── run.py            # 主循环、斜杠命令、消息分发
│   ├── session.py        # SessionStore——对话持久化
│   └── platforms/        # 适配器：telegram、discord、slack、whatsapp、homeassistant、signal
├── acp_adapter/          # ACP 服务器（VS Code / Zed / JetBrains 集成）
├── cron/                 # 调度器（jobs.py、scheduler.py）
├── environments/         # RL 训练环境（Atropos）
├── tests/                # Pytest 测试套件（~3000 测试）
└── batch_runner.py       # 并行批处理
```

**用户配置：** `~/.hermes/config.yaml`（设置）、`~/.hermes/.env`（API 密钥）

## 文件依赖链

```
tools/registry.py （无依赖——被所有工具文件导入）
       ↑
tools/*.py  （每个在导入时调用 registry.register()）
       ↑
model_tools.py  （导入 tools/registry + 触发工具发现）
       ↑
run_agent.py, cli.py, batch_runner.py, environments/
```

---

## AIAgent 类 (run_agent.py)

```python
class AIAgent:
    def __init__(self,
        model: str = "anthropic/claude-opus-4.6",
        max_iterations: int = 90,
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        quiet_mode: bool = False,
        save_trajectories: bool = False,
        platform: str = None,           # "cli"、"telegram" 等
        session_id: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        # ... 以及 provider、api_mode、callbacks、routing 参数
    ): ...

    def chat(self, message: str) -> str:
        """简单接口——返回最终响应字符串。"""

    def run_conversation(self, user_message: str, system_message: str = None,
                         conversation_history: list = None, task_id: str = None) -> dict:
        """完整接口——返回包含 final_response + messages 的字典。"""
```

### 代理循环

核心循环位于 `run_conversation()` 内——完全同步：

```python
while api_call_count < self.max_iterations and self.iteration_budget.remaining > 0:
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else:
        return response.content
```

消息遵循 OpenAI 格式：`{"role": "system/user/assistant/tool", ...}`。推理内容存储在 `assistant_msg["reasoning"]`。

---

## CLI 架构 (cli.py)

- **Rich** 用于横幅/面板，**prompt_toolkit** 用于带自动补全的输入
- **KawaiiSpinner**（`agent/display.py`）——API 调用期间动画表情、`┊` 活动日志用于工具结果
- cli.py 中的 `load_cli_config()` 合并硬编码默认值 + 用户配置 YAML
- **Skin 引擎**（`hermes_cli/skin_engine.py`）——数据驱动的 CLI 主题；启动时从 `display.skin` 配置键初始化；skin 定制横幅颜色、spinner 表情/动词/翅膀、工具前缀、响应框、品牌文本
- `process_command()` 是 `HermesCLI` 上的方法——通过中央注册表的 `resolve_command()` 解析的规范命令名进行分发
- 技能斜杠命令：`agent/skill_commands.py` 扫描 `~/.hermes/skills/`，作为**用户消息**注入（非系统提示词）以保留提示词缓存

### 斜杠命令注册表 (`hermes_cli/commands.py`)

所有斜杠命令在中央 `COMMAND_REGISTRY` 中定义为 `CommandDef` 对象列表。每个下游消费者自动从该注册表派生：

- **CLI** —— `process_command()` 通过 `resolve_command()` 解析别名，按规范名分发
- **Gateway** —— `GATEWAY_KNOWN_COMMANDS` frozenset 用于 hook 发射，`resolve_command()` 用于分发
- **Gateway 帮助** —— `gateway_help_lines()` 生成 `/help` 输出
- **Telegram** —— `telegram_bot_commands()` 生成 BotCommand 菜单
- **Slack** —— `slack_subcommand_map()` 生成 `/hermes` 子命令路由
- **自动补全** —— `COMMANDS` 平坦字典供 `SlashCommandCompleter` 使用
- **CLI 帮助** —— `COMMANDS_BY_CATEGORY` 字典供 `show_help()` 使用

### 添加斜杠命令

1. 在 `hermes_cli/commands.py` 的 `COMMAND_REGISTRY` 中添加 `CommandDef` 条目：
```python
CommandDef("mycommand", "Description of what it does", "Session",
           aliases=("mc",), args_hint="[arg]"),
```
2. 在 `cli.py` 的 `HermesCLI.process_command()` 中添加处理器：
```python
elif canonical == "mycommand":
    self._handle_mycommand(cmd_original)
```
3. 如果命令在 gateway 中可用，在 `gateway/run.py` 中添加处理器：
```python
if canonical == "mycommand":
    return await self._handle_mycommand(event)
```
4. 对于持久设置，在 `cli.py` 中使用 `save_config_value()`

**CommandDef 字段：**
- `name` —— 无斜杠的规范名（例如 `"background"`）
- `description` —— 人类可读描述
- `category` —— `"Session"`、`"Configuration"`、`"Tools & Skills"`、`"Info"`、`"Exit"` 之一
- `aliases` —— 别名元组（例如 `("bg",)`）
- `args_hint` —— 帮助中显示的参数占位符（例如 `"<prompt>"`、`"[name]"`)
- `cli_only` —— 仅在交互式 CLI 中可用
- `gateway_only` —— 仅在消息平台中可用
- `gateway_config_gate` —— 配置路径（例如 `"display.tool_progress_command"`）；当在 `cli_only` 命令上设置时，如果配置值为真，命令在 gateway 中可用。`GATEWAY_KNOWN_COMMANDS` 始终包含配置门控命令以便 gateway 可以分发；帮助/菜单仅在门打开时显示它们。

**添加别名**只需要在现有 `CommandDef` 的 `aliases` 元组中添加。无需其他文件更改——分发、帮助文本、Telegram 菜单、Slack 映射和自动补全全部自动更新。

---

## 添加新工具

需要在**3 个文件**中更改：

**1. 创建 `tools/your_tool.py`：**
```python
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(param=args.get("param", ""), task_id=kw.get("task_id")),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

**2. 在 `model_tools.py` `_discover_tools()` 列表中添加导入。**

**3. 添加到 `toolsets.py`**——要么 `_HERMES_CORE_TOOLS`（所有平台），要么新建工具集。

注册表处理 schema 收集、分发、可用性检查和错误包装。所有处理器必须返回 JSON 字符串。

**工具 schema 中的路径引用**：如果 schema 描述提及文件路径（例如默认输出目录），使用 `display_hermes_home()` 使其 profile 感知。schema 在导入时生成，这在 `_apply_profile_override()` 设置 `HERMES_HOME` 之后。

**状态文件**：如果工具存储持久状态（缓存、日志、检查点），使用 `get_hermes_home()` 作为基础目录——不要用 `Path.home() / ".hermes"`。这确保每个 profile 有自己的状态。

**代理级工具**（todo、memory）：由 `run_agent.py` 在 `handle_function_call()` 之前拦截。参见 `todo_tool.py` 的模式。

---

## 添加配置

### config.yaml 选项：
1. 在 `hermes_cli/config.py` 中添加到 `DEFAULT_CONFIG`
2. 增加 `_config_version`（当前为 5）以触发现有用户迁移

### .env 变量：
1. 在 `hermes_cli/config.py` 中添加到 `OPTIONAL_ENV_VARS` 并附带元数据：
```python
"NEW_API_KEY": {
    "description": "What it's for",
    "prompt": "Display name",
    "url": "https://...",
    "password": True,
    "category": "tool",  # provider、tool、messaging、setting
},
```

### 配置加载器（两个独立系统）：

| 加载器 | 使用者 | 位置 |
|--------|---------|----------|
| `load_cli_config()` | CLI 模式 | `cli.py` |
| `load_config()` | `hermes tools`、`hermes setup` | `hermes_cli/config.py` |
| 直接 YAML 加载 | Gateway | `gateway/run.py` |

---

## Skin/主题系统

Skin 引擎（`hermes_cli/skin_engine.py`）提供数据驱动的 CLI 视觉定制。Skin 是**纯数据**——添加新 skin 无需代码更改。

### 架构

```
hermes_cli/skin_engine.py    # SkinConfig dataclass、内置 skin、YAML 加载器
~/.hermes/skins/*.yaml       # 用户安装的自定义 skin（直接放入）
```

- `init_skin_from_config()` —— CLI 启动时调用，从配置读取 `display.skin`
- `get_active_skin()` —— 返回当前 skin 的缓存 `SkinConfig`
- `set_active_skin(name)` —— 运行时切换 skin（由 `/skin` 命令使用）
- `load_skin(name)` —— 先从用户 skin 加载，再内置，最后回退到默认
- 缺失的 skin 值自动从 `default` skin 继承

### Skin 可定制内容

| 元素 | Skin 键 | 使用者 |
|---------|----------|---------|
| 横幅面板边框 | `colors.banner_border` | `banner.py` |
| 横幅面板标题 | `colors.banner_title` | `banner.py` |
| 横幅章节标题 | `colors.banner_accent` | `banner.py` |
| 横幅暗淡文本 | `colors.banner_dim` | `banner.py` |
| 横幅正文文本 | `colors.banner_text` | `banner.py` |
| 响应框边框 | `colors.response_border` | `cli.py` |
| Spinner 表情（等待） | `spinner.waiting_faces` | `display.py` |
| Spinner 表情（思考） | `spinner.thinking_faces` | `display.py` |
| Spinner 动词 | `spinner.thinking_verbs` | `display.py` |
| Spinner 翅膀（可选） | `spinner.wings` | `display.py` |
| 工具输出前缀 | `tool_prefix` | `display.py` |
| 每工具表情 | `tool_emojis` | `display.py` → `get_tool_emoji()` |
| 代理名称 | `branding.agent_name` | `banner.py`、`cli.py` |
| 欢迎消息 | `branding.welcome` | `cli.py` |
| 响应框标签 | `branding.response_label` | `cli.py` |
| 提示符 | `branding.prompt_symbol` | `cli.py` |

### 内置 skin

- `default` —— 经典 Hermes 金色/kawaii（当前外观）
- `ares` —— 深红/青铜战神主题，带自定义 spinner 翅膀
- `mono` —— 清爽灰度单色
- `slate` —— 冷蓝色开发者主题

### 添加内置 skin

在 `hermes_cli/skin_engine.py` 的 `_BUILTIN_SKINS` 字典中添加：

```python
"mytheme": {
    "name": "mytheme",
    "description": "Short description",
    "colors": { ... },
    "spinner": { ... },
    "branding": { ... },
    "tool_prefix": "┊",
},
```

### 用户 skin (YAML)

用户创建 `~/.hermes/skins/<name>.yaml`：

```yaml
name: cyberpunk
description: Neon-soaked terminal theme

colors:
  banner_border: "#FF00FF"
  banner_title: "#00FFFF"
  banner_accent: "#FF1493"

spinner:
  thinking_verbs: ["jacking in", "decrypting", "uploading"]
  wings:
    - ["⟨⚡", "⚡⟩"]

branding:
  agent_name: "Cyber Agent"
  response_label: " ⚡ Cyber "

tool_prefix: "▏"
```

用 `/skin cyberpunk` 或在 config.yaml 中设置 `display.skin: cyberpunk` 激活。

---

## 重要策略
### 提示词缓存不可破坏

Hermes-Agent 确保缓存在整个对话中保持有效。**不要实现会：**
- 在对话中途更改过去上下文
- 在对话中途更改工具集
- 在对话中途重载记忆或重建系统提示词

的更改。

破坏缓存会大幅提高成本。唯一更改上下文的时间是上下文压缩期间。

### 工作目录行为
- **CLI**：使用当前目录（`.` → `os.getcwd()`）
- **消息平台**：使用 `MESSAGING_CWD` 环境变量（默认：主目录）

### 后台进程通知 (Gateway)

当使用 `terminal(background=true, notify_on_complete=true)` 时，gateway 运行监视器检测进程完成并触发新代理轮次。用 config.yaml 中的 `display.background_process_notifications`（或 `HERMES_BACKGROUND_NOTIFICATIONS` 环境变量）控制后台进程消息的详细程度：

- `all` —— 运行输出更新 + 最终消息（默认）
- `result` —— 仅最终完成消息
- `error` —— 仅退出码 != 0 时的最终消息
- `off` —— 无监视器消息

---

## Profile：多实例支持

Hermes 支持 **profile**——多个完全隔离的实例，每个有自己的 `HERMES_HOME` 目录（配置、API 密钥、记忆、会话、技能、gateway 等）。

核心机制：`hermes_cli/main.py` 中的 `_apply_profile_override()` 在任何模块导入之前设置 `HERMES_HOME`。所有 119+ 个 `get_hermes_home()` 引用自动作用到活跃 profile。

### Profile 安全代码规则

1. **所有 HERMES_HOME 路径使用 `get_hermes_home()`。** 从 `hermes_constants` 导入。
   绝不要在读写状态的代码中硬编码 `~/.hermes` 或 `Path.home() / ".hermes"`。
   ```python
   # 正确
   from hermes_constants import get_hermes_home
   config_path = get_hermes_home() / "config.yaml"

   # 错误——破坏 profile
   config_path = Path.home() / ".hermes" / "config.yaml"
   ```

2. **用户面向消息使用 `display_hermes_home()`。** 从 `hermes_constants` 导入。
   默认返回 `~/.hermes`，profile 返回 `~/.hermes/profiles/<name>`。
   ```python
   # 正确
   from hermes_constants import display_hermes_home
   print(f"Config saved to {display_hermes_home()}/config.yaml")

   # 错误——profile 显示错误路径
   print("Config saved to ~/.hermes/config.yaml")
   ```

3. **模块级常量可以**——它们在导入时缓存 `get_hermes_home()`，
   这在 `_apply_profile_override()` 设置环境变量之后。只需用 `get_hermes_home()`，
   不要用 `Path.home() / ".hermes"`。

4. **模拟 `Path.home()` 的测试必须同时设置 `HERMES_HOME`**——因为代码现在用
   `get_hermes_home()`（读取环境变量），不用 `Path.home() / ".hermes"`：
   ```python
   with patch.object(Path, "home", return_value=tmp_path), \
        patch.dict(os.environ, {"HERMES_HOME": str(tmp_path / ".hermes")}):
       ...
   ```

5. **Gateway 平台适配器应使用令牌锁**——如果适配器用唯一凭证（bot 令牌、API 密钥）连接，在 `connect()`/`start()` 方法中从 `gateway.status` 调用 `acquire_scoped_lock()`，在 `disconnect()`/`stop()` 中调用 `release_scoped_lock()`。这防止两个 profile 使用相同凭证。
   参见 `gateway/platforms/telegram.py` 的规范模式。

6. **Profile 操作锚定 HOME，不是 ERMES_HOME**——`_get_profiles_root()`
   返回 `Path.home() / ".hermes" / "profiles"`，不是 `get_hermes_home() / "profiles"`。
   这是故意——让 `hermes -p coder profile list` 看到所有 profile，无论哪个活跃。

## 已知陷阱

### 不要硬编码 `~/.hermes` 路径
代码路径用 `hermes_constants` 的 `get_hermes_home()`。用户面向打印/日志消息用 `display_hermes_home()`。
硬编码 `~/.hermes` 破坏 profile——每个 profile 有自己的 `HERMES_HOME` 目录。这是 PR #3575 修复的 5 个 bug 的根源。

### 不要用 `simple_term_menu` 做交互菜单
tmux/iTerm2 中渲染 bug——滚动重影。用 `curses`（stdlib）代替。参见 `hermes_cli/tools_config.py` 的模式。

### 不要在 spinner/display 代码中用 `\033[K`（ANSI 擦除到行尾）
在 `prompt_toolkit` 的 `patch_stdout` 下会泄漏为字面 `?[K` 文本。用空格填充：`f"\r{line}{' ' * pad}"`。

### `_last_resolved_tool_names` 是 `model_tools.py` 的进程全局变量
`delegate_tool.py` 的 `_run_single_child()` 在子代理执行前后保存和恢复此全局变量。如果你添加读取此全局的新代码，注意它在子代理运行期间可能暂时过期。

### 不要在 schema 描述中硬编码跨工具引用
工具 schema 描述不得提及其他工具集的工具名（例如 `browser_navigate` 说 "prefer web_search"）。那些工具可能不可用（缺少 API 密钥、禁用工具集），导致模型幻觉调用不存在的工具。如果需要跨引用，在 `model_tools.py` 的 `get_tool_definitions()` 中动态添加——参见 `browser_navigate` / `execute_code` 后处理块的模式。

### 测试不得写入 `~/.hermes/`
`tests/conftest.py` 的 `_isolate_hermes_home` autouse fixture 将 `HERMES_HOME` 重定向到临时目录。测试中绝不要硬编码 `~/.hermes/` 路径。

**Profile 测试**：测试 profile 功能时，也要模拟 `Path.home()` 以使 `_get_profiles_root()` 和 `_get_default_hermes_home()` 在临时目录内解析。
使用 `tests/hermes_cli/test_profiles.py` 的模式：
```python
@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home
```

---

## 测试

```bash
source venv/bin/activate
python -m pytest tests/ -q          # 完整套件（~3000 测试，~3 分钟）
python -m pytest tests/test_model_tools.py -q   # 工具集解析
python -m pytest tests/test_cli_init.py -q       # CLI 配置加载
python -m pytest tests/gateway/ -q               # Gateway 测试
python -m pytest tests/tools/ -q                 # 工具级测试
```

推送更改前始终运行完整套件。