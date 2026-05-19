#!/usr/bin/env python3
"""
AI Agent运行器 - 支持工具调用

本模块提供了一个简洁、独立的Agent，可以执行具有工具调用能力的AI模型。
它处理对话循环、工具执行和响应管理。

功能特性:
- 自动工具调用循环直至完成
- 可配置的模型参数
- 错误处理和恢复
- 消息历史管理
- 支持多种模型提供商

使用方法:
    from run_agent import AIAgent

    agent = AIAgent(base_url="http://localhost:30000/v1", model="claude-opus-4-20250514")
    response = agent.run_conversation("Tell me about the latest Python updates")
"""

import asyncio
import base64
import concurrent.futures
import copy
import hashlib
import json
import logging
logger = logging.getLogger(__name__)
import os
import random
import re
import sys
import tempfile
import time
import threading
from types import SimpleNamespace
import uuid
from typing import List, Dict, Any, Optional
from openai import OpenAI
import fire
from datetime import datetime
from pathlib import Path

from hermes_constants import get_hermes_home

# 首先从 ~/.hermes/.env 加载环境变量，然后从项目根目录加载作为开发环境的后备。
# 用户管理的环境文件应该在重启时覆盖过期的shell导出变量。
from hermes_cli.env_loader import load_hermes_dotenv

_hermes_home = get_hermes_home()
_project_env = Path(__file__).parent / '.env'
_loaded_env_paths = load_hermes_dotenv(hermes_home=_hermes_home, project_env=_project_env)
if _loaded_env_paths:
    for _env_path in _loaded_env_paths:
        logger.info("Loaded environment variables from %s", _env_path)
else:
    logger.info("No .env file found. Using system environment variables.")


# 导入我们的工具系统
from model_tools import (
    get_tool_definitions,
    get_toolset_for_tool,
    handle_function_call,
    check_toolset_requirements,
)
from tools.terminal_tool import cleanup_vm, get_active_env, is_persistent_env
from tools.tool_result_storage import maybe_persist_tool_result, enforce_turn_budget
from tools.interrupt import set_interrupt as _set_interrupt
from tools.browser_tool import cleanup_browser


from hermes_constants import OPENROUTER_BASE_URL

# Agent内部组件已提取到agent/包中以实现模块化
from agent.memory_manager import build_memory_context_block
from agent.retry_utils import jittered_backoff
from agent.error_classifier import classify_api_error, FailoverReason
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, PLATFORM_HINTS,
    MEMORY_GUIDANCE, SESSION_SEARCH_GUIDANCE, SKILLS_GUIDANCE,
    build_nous_subscription_prompt,
)
from agent.model_metadata import (
    fetch_model_metadata,
    estimate_tokens_rough, estimate_messages_tokens_rough, estimate_request_tokens_rough,
    get_next_probe_tier, parse_context_limit_from_error,
    parse_available_output_tokens_from_error,
    save_context_length, is_local_endpoint,
    query_ollama_num_ctx,
)
from agent.context_compressor import ContextCompressor
from agent.subdirectory_hints import SubdirectoryHintTracker
from agent.prompt_caching import apply_anthropic_cache_control
from agent.prompt_builder import build_skills_system_prompt, build_context_files_prompt, build_environment_hints, load_soul_md, TOOL_USE_ENFORCEMENT_GUIDANCE, TOOL_USE_ENFORCEMENT_MODELS, DEVELOPER_ROLE_MODELS, GOOGLE_MODEL_OPERATIONAL_GUIDANCE, OPENAI_MODEL_EXECUTION_GUIDANCE
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from agent.display import (
    KawaiiSpinner, build_tool_preview as _build_tool_preview,
    get_cute_tool_message as _get_cute_tool_message_impl,
    _detect_tool_failure,
    get_tool_emoji as _get_tool_emoji,
)
from agent.trajectory import (
    convert_scratchpad_to_think, has_incomplete_scratchpad,
    save_trajectory as _save_trajectory_to_file,
)
from utils import atomic_json_write, env_var_enabled



class _SafeWriter:
    """透明的stdio包装器，捕获断开管道产生的OSError/ValueError异常。

    当hermes-agent作为systemd服务、Docker容器或无头守护进程运行时，
    stdout/stderr管道可能会变得不可用（空闲超时、缓冲区耗尽、套接字重置）。
    任何print()调用都会引发``OSError: [Errno 5] Input/output error``，
    这可能导致agent设置或run_conversation()崩溃——特别是当except处理程序
    也尝试打印时会产生双重故障。

    此外，当子agent在ThreadPoolExecutor线程中运行时，共享的stdout句柄
    可能在线程拆卸和清理之间关闭，从而引发``ValueError: I/O operation on closed file``
    而不是OSError。

    此包装器将所有写入委托给底层流，并静默捕获OSError和ValueError。
    当被包装的流健康时，它是透明的。
    """

    __slots__ = ("_inner",)

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def write(self, data):
        try:
            return self._inner.write(data)
        except (OSError, ValueError):
            return len(data) if isinstance(data, str) else 0

    def flush(self):
        try:
            self._inner.flush()
        except (OSError, ValueError):
            pass

    def fileno(self):
        return self._inner.fileno()

    def isatty(self):
        try:
            return self._inner.isatty()
        except (OSError, ValueError):
            return False

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _install_safe_stdio() -> None:
    """包装stdout/stderr，确保尽力而为的控制台输出不会导致agent崩溃。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and not isinstance(stream, _SafeWriter):
            setattr(sys, stream_name, _SafeWriter(stream))


class IterationBudget:
    """Agent的线程安全迭代计数器。

    每个agent（父agent或子agent）都有自己的``IterationBudget``。
    父agent的预算上限为``max_iterations``（默认90）。
    每个子agent获得独立的预算上限，上限为
    ``delegation.max_iterations``（默认50）——这意味着
    父agent+子agent的总迭代次数可以超过父agent的上限。
    用户可以通过config.yaml中的``delegation.max_iterations``
    控制每个子agent的限制。

    ``execute_code``（程序化工具调用）迭代通过:meth:`refund`退还，
    因此它们不会消耗预算。
    """

    def __init__(self, max_total: int):
        self.max_total = max_total
        self._used = 0
        self._lock = threading.Lock()

    def consume(self) -> bool:
        """尝试消耗一次迭代。如果允许则返回True。"""
        with self._lock:
            if self._used >= self.max_total:
                return False
            self._used += 1
            return True

    def refund(self) -> None:
        """退还一次迭代（例如用于execute_code轮次）。"""
        with self._lock:
            if self._used > 0:
                self._used -= 1

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_total - self._used)


# 绝不能并发运行的工具（交互式/面向用户）。
# 当这些工具出现在批次中时，我们回退到顺序执行。
_NEVER_PARALLEL_TOOLS = frozenset({"clarify"})

# 没有共享可变会话状态的只读工具。
_PARALLEL_SAFE_TOOLS = frozenset({
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "read_file",
    "search_files",
    "session_search",
    "skill_view",
    "skills_list",
    "vision_analyze",
    "web_extract",
    "web_search",
})

# 文件工具可以在目标路径独立时并发运行。
_PATH_SCOPED_TOOLS = frozenset({"read_file", "write_file", "patch"})

# 并行工具执行的最大工作线程数。
_MAX_TOOL_WORKERS = 8

# 表示终端命令可能修改/删除文件的模式。
_DESTRUCTIVE_PATTERNS = re.compile(
    r"""(?:^|\s|&&|\|\||;|`)(?:
        rm\s|rmdir\s|
        mv\s|
        sed\s+-i|
        truncate\s|
        dd\s|
        shred\s|
        git\s+(?:reset|clean|checkout)\s
    )""",
    re.VERBOSE,
)
# 覆盖文件的输出重定向（> 但不包括 >>）
_REDIRECT_OVERWRITE = re.compile(r'[^>]>[^>]|^>[^>]')


def _is_destructive_command(cmd: str) -> bool:
    """启发式判断：此终端命令是否看起来像是修改/删除文件？"""
    if not cmd:
        return False
    if _DESTRUCTIVE_PATTERNS.search(cmd):
        return True
    if _REDIRECT_OVERWRITE.search(cmd):
        return True
    return False


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """当工具调用批次可以安全并发运行时返回True。"""
    if len(tool_calls) <= 1:
        return False

    tool_names = [tc.function.name for tc in tool_calls]
    if any(name in _NEVER_PARALLEL_TOOLS for name in tool_names):
        return False

    reserved_paths: list[Path] = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        try:
            function_args = json.loads(tool_call.function.arguments)
        except Exception:
            logging.debug(
                "Could not parse args for %s — defaulting to sequential; raw=%s",
                tool_name,
                tool_call.function.arguments[:200],
            )
            return False
        if not isinstance(function_args, dict):
            logging.debug(
                "Non-dict args for %s (%s) — defaulting to sequential",
                tool_name,
                type(function_args).__name__,
            )
            return False

        if tool_name in _PATH_SCOPED_TOOLS:
            scoped_path = _extract_parallel_scope_path(tool_name, function_args)
            if scoped_path is None:
                return False
            if any(_paths_overlap(scoped_path, existing) for existing in reserved_paths):
                return False
            reserved_paths.append(scoped_path)
            continue

        if tool_name not in _PARALLEL_SAFE_TOOLS:
            return False

    return True


def _extract_parallel_scope_path(tool_name: str, function_args: dict) -> Path | None:
    """返回路径范围工具的规范化文件目标。"""
    if tool_name not in _PATH_SCOPED_TOOLS:
        return None

    raw_path = function_args.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    expanded = Path(raw_path).expanduser()
    if expanded.is_absolute():
        return Path(os.path.abspath(str(expanded)))

    # 避免使用resolve()；文件可能尚不存在。
    return Path(os.path.abspath(str(Path.cwd() / expanded)))


def _paths_overlap(left: Path, right: Path) -> bool:
    """当两个路径可能指向同一子树时返回True。"""
    left_parts = left.parts
    right_parts = right.parts
    if not left_parts or not right_parts:
        # 空路径不应该到达这里（上游已做防护），但要安全处理。
        return bool(left_parts) == bool(right_parts) and bool(left_parts)
    common_len = min(len(left_parts), len(right_parts))
    return left_parts[:common_len] == right_parts[:common_len]



_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')




def _sanitize_surrogates(text: str) -> str:
    """将孤立的代理码点替换为U+FFFD（替换字符）。

    代理字符在UTF-8中无效，会导致OpenAI SDK内部的``json.dumps()``崩溃。
    当文本不含代理字符时，这是一个快速的空操作。
    """
    if _SURROGATE_RE.search(text):
        return _SURROGATE_RE.sub('\ufffd', text)
    return text


def _sanitize_messages_surrogates(messages: list) -> bool:
    """从消息列表中的所有字符串内容清理代理字符。

    就地处理消息字典。如果发现并替换了任何代理字符则返回True，
    否则返回False。覆盖content/text、name和工具调用metadata/arguments，
    以便重试不会因非内容字段而失败。
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str) and _SURROGATE_RE.search(content):
            msg["content"] = _SURROGATE_RE.sub('\ufffd', content)
            found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and _SURROGATE_RE.search(text):
                        part["text"] = _SURROGATE_RE.sub('\ufffd', text)
                        found = True
        name = msg.get("name")
        if isinstance(name, str) and _SURROGATE_RE.search(name):
            msg["name"] = _SURROGATE_RE.sub('\ufffd', name)
            found = True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id")
                if isinstance(tc_id, str) and _SURROGATE_RE.search(tc_id):
                    tc["id"] = _SURROGATE_RE.sub('\ufffd', tc_id)
                    found = True
                fn = tc.get("function")
                if isinstance(fn, dict):
                    fn_name = fn.get("name")
                    if isinstance(fn_name, str) and _SURROGATE_RE.search(fn_name):
                        fn["name"] = _SURROGATE_RE.sub('\ufffd', fn_name)
                        found = True
                    fn_args = fn.get("arguments")
                    if isinstance(fn_args, str) and _SURROGATE_RE.search(fn_args):
                        fn["arguments"] = _SURROGATE_RE.sub('\ufffd', fn_args)
                        found = True
    return found


def _strip_non_ascii(text: str) -> str:
    """移除非ASCII字符，替换为最接近的ASCII等效字符或直接删除。

    作为最后的手段，用于系统编码为ASCII且无法处理任何非ASCII字符时
    （例如Chromebook上的LANG=C）。
    """
    return text.encode('ascii', errors='ignore').decode('ascii')


def _sanitize_messages_non_ascii(messages: list) -> bool:
    """从消息列表中的所有字符串内容剥离非ASCII字符。

    这是针对仅支持ASCII编码的系统（LANG=C、Chromebook、最小容器）的
    最后手段恢复。如果发现并清理了任何非ASCII内容则返回True。
    """
    found = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        # 清理content字段（字符串）
        content = msg.get("content")
        if isinstance(content, str):
            sanitized = _strip_non_ascii(content)
            if sanitized != content:
                msg["content"] = sanitized
                found = True
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        sanitized = _strip_non_ascii(text)
                        if sanitized != text:
                            part["text"] = sanitized
                            found = True
        # 清理name字段（可能包含工具结果中的非ASCII字符）
        name = msg.get("name")
        if isinstance(name, str):
            sanitized = _strip_non_ascii(name)
            if sanitized != name:
                msg["name"] = sanitized
                found = True
        # 清理tool_calls
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    if isinstance(fn, dict):
                        fn_args = fn.get("arguments")
                        if isinstance(fn_args, str):
                            sanitized = _strip_non_ascii(fn_args)
                            if sanitized != fn_args:
                                fn["arguments"] = sanitized
                                found = True
    return found





# =========================================================================
# 大型工具结果处理器 — 将超大输出保存到临时文件
# =========================================================================


# =========================================================================
# Qwen Portal请求头 — 模拟QwenCode CLI以兼容portal.qwen.ai。
# 提取为模块级辅助函数，以便__init__和
# _apply_client_headers_for_base_url可以共用。
# =========================================================================
_QWEN_CODE_VERSION = "0.14.1"


def _qwen_portal_headers() -> dict:
    """返回Qwen Portal API所需的默认HTTP请求头。"""
    import platform as _plat

    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }


class AIAgent:
    """
    支持工具调用能力的AI Agent。

    此类管理对话流程、工具执行和响应处理，
    用于支持函数调用的AI模型。
    """

    # ── Class-level context pressure dedup (survives across instances) ──
    # The gateway creates a new AIAgent per message, so instance-level flags
    # reset every time.  This dict tracks {session_id: (warn_level, timestamp)}
    # to suppress duplicate warnings within a cooldown window.
    _context_pressure_last_warned: dict = {}
    _CONTEXT_PRESSURE_COOLDOWN = 300  # seconds between re-warning same session

    @property
    def base_url(self) -> str:
        return self._base_url

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._base_url = value
        self._base_url_lower = value.lower() if value else ""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,
        acp_command: str = None,
        acp_args: list[str] | None = None,
        command: str = None,
        args: list[str] | None = None,
        model: str = "",
        max_iterations: int = 90,  # 默认工具调用迭代次数（与子agent共享）
        tool_delay: float = 1.0,
        enabled_toolsets: List[str] = None,
        disabled_toolsets: List[str] = None,
        save_trajectories: bool = False,
        verbose_logging: bool = False,
        quiet_mode: bool = False,
        ephemeral_system_prompt: str = None,
        log_prefix_chars: int = 100,
        log_prefix: str = "",
        providers_allowed: List[str] = None,
        providers_ignored: List[str] = None,
        providers_order: List[str] = None,
        provider_sort: str = None,
        provider_require_parameters: bool = False,
        provider_data_collection: str = None,
        session_id: str = None,
        tool_progress_callback: callable = None,
        tool_start_callback: callable = None,
        tool_complete_callback: callable = None,
        thinking_callback: callable = None,
        reasoning_callback: callable = None,
        clarify_callback: callable = None,
        step_callback: callable = None,
        stream_delta_callback: callable = None,
        interim_assistant_callback: callable = None,
        tool_gen_callback: callable = None,
        status_callback: callable = None,
        max_tokens: int = None,
        reasoning_config: Dict[str, Any] = None,
        service_tier: str = None,
        request_overrides: Dict[str, Any] = None,
        prefill_messages: List[Dict[str, Any]] = None,
        platform: str = None,
        user_id: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        session_db=None,
        parent_session_id: str = None,
        iteration_budget: "IterationBudget" = None,
        fallback_model: Dict[str, Any] = None,
        credential_pool=None,
        checkpoints_enabled: bool = False,
        checkpoint_max_snapshots: int = 50,
        pass_session_id: bool = False,
        persist_session: bool = True,
    ):
        """
        初始化AI Agent。

        参数:
            base_url (str): 模型API的基础URL（可选）
            api_key (str): 用于认证的API密钥（可选，如未提供则使用环境变量）
            provider (str): 提供商标识符（可选；用于遥测/路由提示）
            api_mode (str): API模式覆盖："chat_completions"或"codex_responses"
            model (str): 要使用的模型名称（默认："anthropic/claude-opus-4.6"）
            max_iterations (int): 工具调用迭代的最大次数（默认：90）
            tool_delay (float): 工具调用之间的延迟秒数（默认：1.0）
            enabled_toolsets (List[str]): 仅启用这些工具集的工具（可选）
            disabled_toolsets (List[str]): 禁用这些工具集的工具（可选）
            save_trajectories (bool): 是否将对话轨迹保存到JSONL文件（默认：False）
            verbose_logging (bool): 启用详细日志记录用于调试（默认：False）
            quiet_mode (bool): 抑制进度输出以获得干净的CLI体验（默认：False）
            ephemeral_system_prompt (str): agent执行期间使用但不保存到轨迹的系统提示（可选）
            log_prefix_chars (int): 工具调用/响应日志预览中显示的字符数（默认：100）
            log_prefix (str): 添加到所有日志消息的前缀，用于并行处理中的识别（默认：""）
            providers_allowed (List[str]): 允许的OpenRouter提供商（可选）
            providers_ignored (List[str]): 忽略的OpenRouter提供商（可选）
            providers_order (List[str]): 按顺序尝试的OpenRouter提供商（可选）
            provider_sort (str): 按价格/吞吐量/延迟排序提供商（可选）
            session_id (str): 用于日志记录的预生成会话ID（可选，如未提供则自动生成）
            tool_progress_callback (callable): 进度通知的回调函数(tool_name, args_preview)
            clarify_callback (callable): 交互式用户问题的回调函数(question, choices) -> str。
                由平台层（CLI或gateway）提供。如果为None，clarify工具返回错误。
            max_tokens (int): 模型响应的最大token数（可选，如未设置则使用模型默认值）
            reasoning_config (Dict): OpenRouter推理配置覆盖（例如{"effort": "none"}禁用思考）。
                如果为None，对OpenRouter默认为{"enabled": True, "effort": "medium"}。
                设置以禁用/自定义推理。
            prefill_messages (List[Dict]): 作为预填充上下文 prepend 到对话历史的消息。
                用于注入少样本示例或预设模型响应风格。
                示例：[{"role": "user", "content": "Hi!"}, {"role": "assistant", "content": "Hello!"}]
            platform (str): 用户所在的界面平台（如"cli"、"telegram"、"discord"、"whatsapp"）。
                用于将平台特定的格式提示注入系统提示。
            skip_context_files (bool): 如果为True，跳过自动注入SOUL.md、AGENTS.md和.cursorrules
                到系统提示。用于批处理和数据生成，避免用用户特定的角色或项目指令
                污染轨迹。
        """
        _install_safe_stdio()

        self.model = model
        self.max_iterations = max_iterations
        # 共享迭代预算 — 父agent创建，子agent继承。
        # 由父agent和所有子agent的每个LLM轮次消耗。
        self.iteration_budget = iteration_budget or IterationBudget(max_iterations)
        self.tool_delay = tool_delay
        self.save_trajectories = save_trajectories
        self.verbose_logging = verbose_logging
        self.quiet_mode = quiet_mode
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.platform = platform  # "cli"、"telegram"、"discord"、"whatsapp"等
        self._user_id = user_id  # 平台用户标识符（gateway会话）
        # 可插拔的打印函数 — CLI将其替换为_cprint，以便
        # 原始ANSI状态行通过prompt_toolkit的渲染器路由，
        # 而不是直接发送到stdout，因为patch_stdout的StdoutProxy
        # 会破坏转义序列。None = 使用builtins.print。
        self._print_fn = None
        self.background_review_callback = None  # 用于gateway交付的可选同步回调
        self.skip_context_files = skip_context_files
        self.pass_session_id = pass_session_id
        self.persist_session = persist_session
        self._credential_pool = credential_pool
        self.log_prefix_chars = log_prefix_chars
        self.log_prefix = f"{log_prefix} " if log_prefix else ""
        # 存储有效基础URL用于功能检测（提示缓存、推理等）
        self.base_url = base_url or ""
        provider_name = provider.strip().lower() if isinstance(provider, str) and provider.strip() else None
        self.provider = provider_name or ""
        self.acp_command = acp_command or command
        self.acp_args = list(acp_args or args or [])
        if api_mode in {"chat_completions", "codex_responses", "anthropic_messages"}:
            self.api_mode = api_mode
        elif self.provider == "openai-codex":
            self.api_mode = "codex_responses"
        elif (provider_name is None) and "chatgpt.com/backend-api/codex" in self._base_url_lower:
            self.api_mode = "codex_responses"
            self.provider = "openai-codex"
        elif self.provider == "anthropic" or (provider_name is None and "api.anthropic.com" in self._base_url_lower):
            self.api_mode = "anthropic_messages"
            self.provider = "anthropic"
        elif self._base_url_lower.rstrip("/").endswith("/anthropic"):
            # 第三方Anthropic兼容端点（如MiniMax、DashScope）
            # 使用以/anthropic结尾的URL约定。自动检测这些，以便
            # 使用Anthropic Messages API适配器而不是chat completions。
            self.api_mode = "anthropic_messages"
        else:
            self.api_mode = "chat_completions"

        try:
            from hermes_cli.model_normalize import (
                _AGGREGATOR_PROVIDERS,
                normalize_model_for_provider,
            )

            if self.provider not in _AGGREGATOR_PROVIDERS:
                self.model = normalize_model_for_provider(self.model, self.provider)
        except Exception:
            pass

        # GPT-5.x模型需要Responses API路径 — 在/v1/chat/completions上
        # OpenAI和OpenRouter都会拒绝它们。同时
        # 对直接OpenAI URL（api.openai.com）自动升级，因为所有
        # 新的工具调用模型在那里都优先使用Responses。
        if self.api_mode == "chat_completions" and (
            self._is_direct_openai_url()
            or self._model_requires_responses_api(self.model)
        ):
            self.api_mode = "codex_responses"

        # 在后台线程中预热OpenRouter模型元数据缓存。
        # fetch_model_metadata()缓存1小时；这避免了
        # 在第一个API响应时因估算定价而产生阻塞HTTP请求。
        if self.provider == "openrouter" or self._is_openrouter_url():
            threading.Thread(
                target=lambda: fetch_model_metadata(),
                daemon=True,
            ).start()

        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.suppress_status_output = False
        self.thinking_callback = thinking_callback
        self.reasoning_callback = reasoning_callback
        self.clarify_callback = clarify_callback
        self.step_callback = step_callback
        self.stream_delta_callback = stream_delta_callback
        self.interim_assistant_callback = interim_assistant_callback
        self.status_callback = status_callback
        self.tool_gen_callback = tool_gen_callback

        
        # 工具执行状态 — 即使注册了流消费者，也允许工具执行期间
        # 进行_vprint（那时没有token流式传输）
        self._executing_tools = False

        # 中断机制用于跳出工具循环
        self._interrupt_requested = False
        self._interrupt_message = None  # 触发中断的可选消息
        self._execution_thread_id: int | None = None  # 在run_conversation()开始时设置
        self._client_lock = threading.RLock()
        
        # 子agent委托状态
        self._delegate_depth = 0        # 0 = 顶层agent，子agent时递增
        self._active_children = []      # 运行中的子AIAgent（用于中断传播）
        self._active_children_lock = threading.Lock()
        
        # 存储OpenRouter提供商偏好
        self.providers_allowed = providers_allowed
        self.providers_ignored = providers_ignored
        self.providers_order = providers_order
        self.provider_sort = provider_sort
        self.provider_require_parameters = provider_require_parameters
        self.provider_data_collection = provider_data_collection

        # 存储工具集过滤选项
        self.enabled_toolsets = enabled_toolsets
        self.disabled_toolsets = disabled_toolsets
        
        # 模型响应配置
        self.max_tokens = max_tokens  # None = 使用模型默认值
        self.reasoning_config = reasoning_config  # None = 使用默认值（OpenRouter为medium）
        self.service_tier = service_tier
        self.request_overrides = dict(request_overrides or {})
        self.prefill_messages = prefill_messages or []  # 预填充的对话轮次
        
        # Anthropic提示缓存：对通过OpenRouter的Claude模型自动启用。
        # 通过缓存对话前缀，在多轮对话中减少约75%的输入成本。
        # 使用system_and_3策略（4个断点）。
        is_openrouter = self._is_openrouter_url()
        is_claude = "claude" in self.model.lower()
        is_native_anthropic = self.api_mode == "anthropic_messages" and self.provider == "anthropic"
        self._use_prompt_caching = (is_openrouter and is_claude) or is_native_anthropic
        self._cache_ttl = "5m"  # 默认5分钟TTL（1.25倍写入成本）
        
        # 迭代预算：LLM仅在实际耗尽迭代预算时才会被通知
        # （api_call_count >= max_iterations）。此时我们注入
        # 一条消息，允许最后一次API调用，如果模型没有产生
        # 文本响应，则强制发送一条用户消息要求它总结。
        # 无中间压力警告 — 它们曾导致模型在复杂任务上过早"放弃"（#7915）。
        self._budget_exhausted_injected = False
        self._budget_grace_call = False

        # 上下文压力警告：当上下文填满时通知用户（而非LLM）。
        # 纯信息性 — 显示在CLI输出中并通过status_callback发送给
        # gateway平台。不注入到消息中。
        # 分层：在压缩阈值的85%和95%时触发。
        self._context_pressure_warned_at = 0.0  # 已显示的最高层级

        # 活动追踪 — 在每次API调用、工具执行和流chunk时更新。
        # 用于gateway超时处理程序报告agent被终止时正在做什么，
        # 以及"仍在工作"通知显示进度。
        self._last_activity_ts: float = time.time()
        self._last_activity_desc: str = "初始化"
        self._current_tool: str | None = None
        self._api_call_count: int = 0

        # 速率限制追踪 — 从每次API调用后的x-ratelimit-*响应头更新。
        # 由/usage斜杠命令访问。
        self._rate_limit_state: Optional["RateLimitState"] = None

        # 集中式日志 — agent.log（INFO+）和errors.log（WARNING+）
        # 都位于~/.hermes/logs/下。幂等的，所以gateway模式
        # （每条消息创建新AIAgent）不会重复添加处理程序。
        from hermes_logging import setup_logging, setup_verbose_logging
        setup_logging(hermes_home=_hermes_home)

        if self.verbose_logging:
            setup_verbose_logging()
            logger.info("详细日志已启用（第三方库日志已抑制）")
        else:
            if self.quiet_mode:
                # 在安静模式（CLI默认）下，抑制*控制台*上的所有工具/基础设施
                # 日志噪音。TUI有自己的丰富状态显示；
                # logger的INFO/WARNING消息只会让它变得杂乱。
                # 文件处理程序（agent.log、errors.log）仍然捕获所有内容。
                for quiet_logger in [
                    'tools',               # 所有tools.*（terminal、browser、web、file等）
                    'run_agent',            # agent runner内部
                    'trajectory_compressor',
                    'cron',                 # 调度器（仅在守护进程模式下相关）
                    'hermes_cli',           # CLI助手
                ]:
                    logging.getLogger(quiet_logger).setLevel(logging.ERROR)
        
        # 内部流回调（在流式TTS期间设置）。
        # 在此处初始化以便run_conversation之前_vprint可以引用它。
        self._stream_callback = None
        # 延迟段落换行标志 — 在工具迭代后设置，
        # 以便在下一个真实文本delta前添加单个"\n\n"。
        self._stream_needs_break = False
        # 当前模型响应期间通过实时token回调已发送的可见助手文本。
        # 用于避免当提供商稍后将其作为完成的中间助手消息返回时，
        # 重新发送相同的评论。
        self._current_streamed_assistant_text = ""

        # 可选的当前轮次用户消息覆盖，当面向API的用户消息
        # 与持久化记录有意不同时使用
        # （例如CLI语音模式为实时调用添加临时前缀）。
        self._persist_user_message_idx = None
        self._persist_user_message_override = None

        # 缓存Anthropic图像转文本回退，针对每个图像载荷/URL，
        # 以便单个工具循环不会在相同的图像历史上
        # 重复运行辅助视觉处理。
        self._anthropic_image_fallback_cache: Dict[str, str] = {}

        # 通过集中式提供商路由器初始化LLM客户端。
        # 路由器处理认证解析、基础URL、请求头和
        # 所有已知提供商的Codex/Anthropic包装。
        # raw_codex=True因为主agent需要直接responses.stream()
        # 访问用于Codex Responses API流式传输。
        self._anthropic_client = None
        self._is_anthropic_oauth = False

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token
            # 仅当提供商实际为Anthropic时才回退到ANTHROPIC_TOKEN。
            # 其他anthropic_messages提供商（MiniMax、Alibaba等）必须使用自己的API密钥。
            # 回退会将Anthropic凭证发送到第三方端点（修复#1739、#minimax-401）。
            _is_native_anthropic = self.provider == "anthropic"
            effective_key = (api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url
            from agent.anthropic_adapter import _is_oauth_token as _is_oat
            self._is_anthropic_oauth = _is_oat(effective_key)
            self._anthropic_client = build_anthropic_client(effective_key, base_url)
            # Anthropic模式不需要OpenAI客户端
            self.client = None
            self._client_kwargs = {}
            if not self.quiet_mode:
                print(f"🤖 AI Agent initialized with model: {self.model} (Anthropic native)")
                if effective_key and len(effective_key) > 12:
                    print(f"🔑 Using token: {effective_key[:8]}...{effective_key[-4:]}")
        else:
            if api_key and base_url:
                # 来自CLI/gateway的显式凭证 — 直接构造。
                # 运行时提供商解析器已经为我们处理了认证。
                client_kwargs = {"api_key": api_key, "base_url": base_url}
                if self.provider == "copilot-acp":
                    client_kwargs["command"] = self.acp_command
                    client_kwargs["args"] = self.acp_args
                effective_base = base_url
                if "openrouter" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                        "X-OpenRouter-Title": "Hermes Agent",
                        "X-OpenRouter-Categories": "productivity,cli-agent",
                    }
                elif "api.githubcopilot.com" in effective_base.lower():
                    from hermes_cli.models import copilot_default_headers

                    client_kwargs["default_headers"] = copilot_default_headers()
                elif "api.kimi.com" in effective_base.lower():
                    client_kwargs["default_headers"] = {
                        "User-Agent": "KimiCLI/1.30.0",
                    }
                elif "portal.qwen.ai" in effective_base.lower():
                    client_kwargs["default_headers"] = _qwen_portal_headers()
            else:
                # 无显式凭证 — 使用集中式提供商路由器
                from agent.auxiliary_client import resolve_provider_client
                _routed_client, _ = resolve_provider_client(
                    self.provider or "auto", model=self.model, raw_codex=True)
                if _routed_client is not None:
                    client_kwargs = {
                        "api_key": _routed_client.api_key,
                        "base_url": str(_routed_client.base_url),
                    }
                    # 保留路由器设置的任何default_headers
                    if hasattr(_routed_client, '_default_headers') and _routed_client._default_headers:
                        client_kwargs["default_headers"] = dict(_routed_client._default_headers)
                else:
                    # 当用户显式选择了非OpenRouter提供商
                    # 但未找到凭证时，快速失败并给出清晰
                    # 消息，而不是静默路由通过OpenRouter。
                    _explicit = (self.provider or "").strip().lower()
                    if _explicit and _explicit not in ("auto", "openrouter", "custom"):
                        raise RuntimeError(
                            f"Provider '{_explicit}' is set in config.yaml but no API key "
                            f"was found. Set the {_explicit.upper()}_API_KEY environment "
                            f"variable, or switch to a different provider with `hermes model`."
                        )
                    # 最终回退：尝试原始OpenRouter密钥
                    client_kwargs = {
                        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
                        "base_url": OPENROUTER_BASE_URL,
                        "default_headers": {
                            "HTTP-Referer": "https://hermes-agent.nousresearch.com",
                            "X-OpenRouter-Title": "Hermes Agent",
                            "X-OpenRouter-Categories": "productivity,cli-agent",
                        },
                    }
            
            self._client_kwargs = client_kwargs  # 存储用于中断后重建

            # 为OpenRouter上的Claude启用细粒度工具流式传输。
            # 没有这个，Anthropic会缓冲整个工具调用，在思考时
            # 静默数分钟 — OpenRouter的上游代理会在静默期间超时。
            # beta请求头让Anthropic逐token流式传输工具调用参数，
            # 保持连接存活。
            _effective_base = str(client_kwargs.get("base_url", "")).lower()
            if "openrouter" in _effective_base and "claude" in (self.model or "").lower():
                headers = client_kwargs.get("default_headers") or {}
                existing_beta = headers.get("x-anthropic-beta", "")
                _FINE_GRAINED = "fine-grained-tool-streaming-2025-05-14"
                if _FINE_GRAINED not in existing_beta:
                    if existing_beta:
                        headers["x-anthropic-beta"] = f"{existing_beta},{_FINE_GRAINED}"
                    else:
                        headers["x-anthropic-beta"] = _FINE_GRAINED
                    client_kwargs["default_headers"] = headers

            self.api_key = client_kwargs.get("api_key", "")
            self.base_url = client_kwargs.get("base_url", self.base_url)
            try:
                self.client = self._create_openai_client(client_kwargs, reason="agent_init", shared=True)
                if not self.quiet_mode:
                    print(f"🤖 AI Agent initialized with model: {self.model}")
                    if base_url:
                        print(f"🔗 Using custom base URL: {base_url}")
                    # 始终显示API密钥信息（掩码）用于调试认证问题
                    key_used = client_kwargs.get("api_key", "none")
                    if key_used and key_used != "dummy-key" and len(key_used) > 12:
                        print(f"🔑 Using API key: {key_used[:8]}...{key_used[-4:]}")
                    else:
                        print(f"⚠️  警告：API密钥似乎无效或缺失（得到：'{key_used[:20] if key_used else 'none'}...')")
            except Exception as e:
                raise RuntimeError(f"初始化OpenAI客户端失败：{e}")
        
        # 提供商回退链 — 当主要提供商耗尽（速率限制、过载、
        # 连接失败）时尝试的有序备份提供商列表。
        # 支持旧的单一字典``fallback_model``格式和
        # 新的列表``fallback_providers``格式。
        if isinstance(fallback_model, list):
            self._fallback_chain = [
                f for f in fallback_model
                if isinstance(f, dict) and f.get("provider") and f.get("model")
            ]
        elif isinstance(fallback_model, dict) and fallback_model.get("provider") and fallback_model.get("model"):
            self._fallback_chain = [fallback_model]
        else:
            self._fallback_chain = []
        self._fallback_index = 0
        self._fallback_activated = False
        # 保留旧属性用于向后兼容（测试、外部调用者）
        self._fallback_model = self._fallback_chain[0] if self._fallback_chain else None
        if self._fallback_chain and not self.quiet_mode:
            if len(self._fallback_chain) == 1:
                fb = self._fallback_chain[0]
                print(f"🔄 回退模型：{fb['model']} ({fb['provider']})")
            else:
                print(f"🔄 回退链（{len(self._fallback_chain)}个提供商）：" +
                      " → ".join(f"{f['model']} ({f['provider']})" for f in self._fallback_chain))

        # 获取可用工具（带过滤）
        self.tools = get_tool_definitions(
            enabled_toolsets=enabled_toolsets,
            disabled_toolsets=disabled_toolsets,
            quiet_mode=self.quiet_mode,
        )
        
        # 显示工具配置并存储有效工具名称用于验证
        self.valid_tool_names = set()
        if self.tools:
            self.valid_tool_names = {tool["function"]["name"] for tool in self.tools}
            tool_names = sorted(self.valid_tool_names)
            if not self.quiet_mode:
                print(f"🛠️  已加载{len(self.tools)}个工具：{', '.join(tool_names)}")
                
                # 如已应用过滤则显示过滤信息
                if enabled_toolsets:
                    print(f"   ✅ 已启用工具集：{', '.join(enabled_toolsets)}")
                if disabled_toolsets:
                    print(f"   ❌ 已禁用工具集：{', '.join(disabled_toolsets)}")
        elif not self.quiet_mode:
            print("🛠️  未加载工具（所有工具被过滤或不可用）")
        
        # 检查工具需求
        if self.tools and not self.quiet_mode:
            requirements = check_toolset_requirements()
            missing_reqs = [name for name, available in requirements.items() if not available]
            if missing_reqs:
                print(f"⚠️  某些工具可能因缺失需求而无法工作：{missing_reqs}")
        
        # 显示轨迹保存状态
        if self.save_trajectories and not self.quiet_mode:
            print("📝 轨迹保存已启用")
        
        # 显示临时系统提示状态
        if self.ephemeral_system_prompt and not self.quiet_mode:
            prompt_preview = self.ephemeral_system_prompt[:60] + "..." if len(self.ephemeral_system_prompt) > 60 else self.ephemeral_system_prompt
            print(f"🔒 临时系统提示：'{prompt_preview}'（不保存到轨迹）")
        
        # 显示提示缓存状态
        if self._use_prompt_caching and not self.quiet_mode:
            source = "原生Anthropic" if is_native_anthropic else "通过OpenRouter的Claude"
            print(f"💾 提示缓存：已启用（{source}，{self._cache_ttl} TTL）")
        
        # 会话日志设置 - 自动保存对话轨迹用于调试
        self.session_start = datetime.now()
        if session_id:
            # 使用提供的会话ID（例如来自CLI）
            self.session_id = session_id
        else:
            # 生成新的会话ID
            timestamp_str = self.session_start.strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:6]
            self.session_id = f"{timestamp_str}_{short_uuid}"
        
        # 会话日志存放在~/.hermes/sessions/，与gateway会话并列
        hermes_home = get_hermes_home()
        self.logs_dir = hermes_home / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
        
        # 追踪会话日志的对话消息
        self._session_messages: List[Dict[str, Any]] = []
        
        # 缓存的系统提示 -- 每个会话构建一次，仅在压缩时重建
        self._cached_system_prompt: Optional[str] = None
        
        # 文件系统检查点管理器（透明 — 不是工具）
        from tools.checkpoint_manager import CheckpointManager
        self._checkpoint_mgr = CheckpointManager(
            enabled=checkpoints_enabled,
            max_snapshots=checkpoint_max_snapshots,
        )
        
        # SQLite会话存储（可选 — 由CLI或gateway提供）
        self._session_db = session_db
        self._parent_session_id = parent_session_id
        self._last_flushed_db_idx = 0  # 追踪DB写入游标以防止重复写入
        if self._session_db:
            try:
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    model_config={
                        "max_iterations": self.max_iterations,
                        "reasoning_config": reasoning_config,
                        "max_tokens": max_tokens,
                    },
                    user_id=None,
                    parent_session_id=self._parent_session_id,
                )
            except Exception as e:
                # 短暂的SQLite锁争用（例如CLI和gateway并发写入）
                # 绝不能永久禁用此agent的session_search。
                # 保持_session_db存活 — 一旦锁清除，
                # 后续消息刷新和session_search调用仍然有效。
                # 会话行可能缺失于此次运行的索引中，
                # 但这是可恢复的（刷新会upsert行）。
                logger.warning(
                    "会话DB create_session失败（session_search仍可用）： %s", e
                )
        
        # 用于任务规划的内建待办列表（每个agent/会话一个）
        from tools.todo_tool import TodoStore
        self._todo_store = TodoStore()
        
        # 加载配置一次用于memory、skills和compression部分
        try:
            from hermes_cli.config import load_config as _load_agent_config
            _agent_cfg = _load_agent_config()
        except Exception:
            _agent_cfg = {}

        # 持久化记忆（MEMORY.md + USER.md） -- 从磁盘加载
        self._memory_store = None
        self._memory_enabled = False
        self._user_profile_enabled = False
        self._memory_nudge_interval = 10
        self._memory_flush_min_turns = 6
        self._turns_since_memory = 0
        self._iters_since_skill = 0
        if not skip_memory:
            try:
                mem_config = _agent_cfg.get("memory", {})
                self._memory_enabled = mem_config.get("memory_enabled", False)
                self._user_profile_enabled = mem_config.get("user_profile_enabled", False)
                self._memory_nudge_interval = int(mem_config.get("nudge_interval", 10))
                self._memory_flush_min_turns = int(mem_config.get("flush_min_turns", 6))
                if self._memory_enabled or self._user_profile_enabled:
                    from tools.memory_tool import MemoryStore
                    self._memory_store = MemoryStore(
                        memory_char_limit=mem_config.get("memory_char_limit", 2200),
                        user_char_limit=mem_config.get("user_char_limit", 1375),
                    )
                    self._memory_store.load_from_disk()
            except Exception:
                pass  # 记忆是可选的 -- 不要破坏agent初始化
        


        # 记忆提供商插件（外部 — 一次一个，与内建并存）
        # 从config读取memory.provider以选择要激活的插件。
        self._memory_manager = None
        if not skip_memory:
            try:
                _mem_provider_name = mem_config.get("provider", "") if mem_config else ""

                # 自动迁移：如果Honcho曾被主动配置（启用+凭证）
                # 但memory.provider未设置，则自动激活honcho插件。
                # 仅存在配置文件不够 — 用户可能已禁用Honcho，
                # 或文件可能来自其他工具。
                if not _mem_provider_name:
                    try:
                        from plugins.memory.honcho.client import HonchoClientConfig as _HCC
                        _hcfg = _HCC.from_global_config()
                        if _hcfg.enabled and (_hcfg.api_key or _hcfg.base_url):
                            _mem_provider_name = "honcho"
                            # 持久化以便此自动迁移仅执行一次
                            try:
                                from hermes_cli.config import load_config as _lc, save_config as _sc
                                _cfg = _lc()
                                _cfg.setdefault("memory", {})["provider"] = "honcho"
                                _sc(_cfg)
                            except Exception:
                                pass
                            if not self.quiet_mode:
                                print("  ✓ 已自动迁移Honcho到记忆提供商插件。")
                                print("    您的配置和数据已保留。\n")
                    except Exception:
                        pass

                if _mem_provider_name:
                    from agent.memory_manager import MemoryManager as _MemoryManager
                    from plugins.memory import load_memory_provider as _load_mem
                    self._memory_manager = _MemoryManager()
                    _mp = _load_mem(_mem_provider_name)
                    if _mp and _mp.is_available():
                        self._memory_manager.add_provider(_mp)
                    if self._memory_manager.providers:
                        from hermes_constants import get_hermes_home as _ghh
                        _init_kwargs = {
                            "session_id": self.session_id,
                            "platform": platform or "cli",
                            "hermes_home": str(_ghh()),
                            "agent_context": "primary",
                        }
                        # 为gateway用户身份线程传递用于按用户记忆范围
                        if self._user_id:
                            _init_kwargs["user_id"] = self._user_id
                        # 身份配置用于按配置文件提供商范围
                        try:
                            from hermes_cli.profiles import get_active_profile_name
                            _profile = get_active_profile_name()
                            _init_kwargs["agent_identity"] = _profile
                            _init_kwargs["agent_workspace"] = "hermes"
                        except Exception:
                            pass
                        self._memory_manager.initialize_all(**_init_kwargs)
                        logger.info("记忆提供商'%s'已激活", _mem_provider_name)
                    else:
                        logger.debug("记忆提供商'%s'未找到或不可用", _mem_provider_name)
                        self._memory_manager = None
            except Exception as _mpe:
                logger.warning("记忆提供商插件初始化失败： %s", _mpe)
                self._memory_manager = None

        # 将记忆提供商工具schema注入工具表面
        if self._memory_manager and self.tools is not None:
            for _schema in self._memory_manager.get_all_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)

        # 技能配置：技能创建提醒的提示间隔
        self._skill_nudge_interval = 10
        try:
            skills_config = _agent_cfg.get("skills", {})
            self._skill_nudge_interval = int(skills_config.get("creation_nudge_interval", 10))
        except Exception:
            pass

        # 工具使用强制配置："auto"（默认 — 匹配硬编码
        # 模型列表）、true（始终）、false（从不）或子字符串列表。
        _agent_section = _agent_cfg.get("agent", {})
        if not isinstance(_agent_section, dict):
            _agent_section = {}
        self._tool_use_enforcement = _agent_section.get("tool_use_enforcement", "auto")

        # 初始化上下文压缩器用于自动上下文管理
        # 当接近模型上下文限制时压缩对话
        # 通过config.yaml配置（compression部分）
        _compression_cfg = _agent_cfg.get("compression", {})
        if not isinstance(_compression_cfg, dict):
            _compression_cfg = {}
        compression_threshold = float(_compression_cfg.get("threshold", 0.50))
        compression_enabled = str(_compression_cfg.get("enabled", True)).lower() in ("true", "1", "yes")
        compression_summary_model = _compression_cfg.get("summary_model") or None
        compression_target_ratio = float(_compression_cfg.get("target_ratio", 0.20))
        compression_protect_last = int(_compression_cfg.get("protect_last_n", 20))

        # 从模型配置读取显式的context_length覆盖值
        _model_cfg = _agent_cfg.get("model", {})
        if isinstance(_model_cfg, dict):
            _config_context_length = _model_cfg.get("context_length")
        else:
            _config_context_length = None
        if _config_context_length is not None:
            try:
                _config_context_length = int(_config_context_length)
            except (TypeError, ValueError):
                _config_context_length = None

        # 存储以便switch_model复用（使配置覆盖在模型切换时保持）
        self._config_context_length = _config_context_length

        # 检查custom_providers中每个模型的context_length
        if _config_context_length is None:
            _custom_providers = _agent_cfg.get("custom_providers")
            if isinstance(_custom_providers, list):
                for _cp_entry in _custom_providers:
                    if not isinstance(_cp_entry, dict):
                        continue
                    _cp_url = (_cp_entry.get("base_url") or "").rstrip("/")
                    if _cp_url and _cp_url == self.base_url.rstrip("/"):
                        _cp_models = _cp_entry.get("models", {})
                        if isinstance(_cp_models, dict):
                            _cp_model_cfg = _cp_models.get(self.model, {})
                            if isinstance(_cp_model_cfg, dict):
                                _cp_ctx = _cp_model_cfg.get("context_length")
                                if _cp_ctx is not None:
                                    try:
                                        _config_context_length = int(_cp_ctx)
                                    except (TypeError, ValueError):
                                        pass
                        break
        
        # 选择上下文引擎：配置驱动（类似记忆提供商）。
        # 1. 检查config.yaml中的context.engine设置
        # 2. 检查plugins/context_engine/<name>/目录（仓库内置）
        # 3. 检查通用插件系统（用户安装的插件）
        # 4. 回退到内建的ContextCompressor
        _selected_engine = None
        _engine_name = "compressor"  # 默认值
        try:
            _ctx_cfg = _agent_cfg.get("context", {}) if isinstance(_agent_cfg, dict) else {}
            _engine_name = _ctx_cfg.get("engine", "compressor") or "compressor"
        except Exception:
            pass

        if _engine_name != "compressor":
            # 尝试从plugins/context_engine/<name>/加载
            try:
                from plugins.context_engine import load_context_engine
                _selected_engine = load_context_engine(_engine_name)
            except Exception as _ce_load_err:
                logger.debug("从plugins/context_engine/加载上下文引擎： %s", _ce_load_err)

            # 尝试通用插件系统作为回退
            if _selected_engine is None:
                try:
                    from hermes_cli.plugins import get_plugin_context_engine
                    _candidate = get_plugin_context_engine()
                    if _candidate and _candidate.name == _engine_name:
                        _selected_engine = _candidate
                except Exception:
                    pass

            if _selected_engine is None:
                logger.warning(
                    "上下文引擎'%s'未找到 — 回退到内建压缩器",
                    _engine_name,
                )
        # else: 配置指定"compressor" — 使用内建，不自动激活插件

        if _selected_engine is not None:
            self.context_compressor = _selected_engine
            if not self.quiet_mode:
                logger.info("使用上下文引擎： %s", _selected_engine.name)
        else:
            self.context_compressor = ContextCompressor(
                model=self.model,
                threshold_percent=compression_threshold,
                protect_first_n=3,
                protect_last_n=compression_protect_last,
                summary_target_ratio=compression_target_ratio,
                summary_model_override=compression_summary_model,
                quiet_mode=self.quiet_mode,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                config_context_length=_config_context_length,
                provider=self.provider,
                api_mode=self.api_mode,
            )
        self.compression_enabled = compression_enabled

        # 拒绝上下文窗口低于可靠工具调用工作流所需最小值
        # （64K tokens）的模型。
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH
        _ctx = getattr(self.context_compressor, "context_length", 0)
        if _ctx and _ctx < MINIMUM_CONTEXT_LENGTH:
            raise ValueError(
                f"模型{self.model}的上下文窗口为{_ctx:,} tokens，"
                f"低于Hermes Agent所需的最小值{MINIMUM_CONTEXT_LENGTH:,}。"
                f"请选择至少{MINIMUM_CONTEXT_LENGTH // 1000}K上下文的模型，"
                f"或在config.yaml中设置model.context_length以覆盖。"
            )

        # 注入上下文引擎工具schema（如lcm_grep、lcm_describe、lcm_expand）
        self._context_engine_tool_names: set = set()
        if hasattr(self, "context_compressor") and self.context_compressor and self.tools is not None:
            for _schema in self.context_compressor.get_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                self.tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    self.valid_tool_names.add(_tname)
                    self._context_engine_tool_names.add(_tname)

        # 通知上下文引擎会话开始
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_start(
                    self.session_id,
                    hermes_home=str(get_hermes_home()),
                    platform=self.platform or "cli",
                    model=self.model,
                    context_length=getattr(self.context_compressor, "context_length", 0),
                )
            except Exception as _ce_err:
                logger.debug("上下文引擎on_session_start： %s", _ce_err)

        self._subdirectory_hints = SubdirectoryHintTracker(
            working_dir=os.getenv("TERMINAL_CWD") or None,
        )
        self._user_turn_count = 0

        # 会话的累计token使用量
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0
        self.session_api_calls = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # ── Ollama num_ctx注入 ──
        # Ollama默认使用2048上下文，无论模型实际能力如何。
        # 当连接Ollama服务器时，检测模型的最大上下文
        # 并在每次chat请求时传递num_ctx以使用完整窗口。
        # 用户覆盖：在config.yaml中设置model.ollama_num_ctx以限制VRAM使用。
        self._ollama_num_ctx: int | None = None
        _ollama_num_ctx_override = None
        if isinstance(_model_cfg, dict):
            _ollama_num_ctx_override = _model_cfg.get("ollama_num_ctx")
        if _ollama_num_ctx_override is not None:
            try:
                self._ollama_num_ctx = int(_ollama_num_ctx_override)
            except (TypeError, ValueError):
                logger.debug("无效的ollama_num_ctx配置值： %r", _ollama_num_ctx_override)
        if self._ollama_num_ctx is None and self.base_url and is_local_endpoint(self.base_url):
            try:
                _detected = query_ollama_num_ctx(self.model, self.base_url)
                if _detected and _detected > 0:
                    self._ollama_num_ctx = _detected
            except Exception as exc:
                logger.debug("Ollama num_ctx检测失败： %s", exc)
        if self._ollama_num_ctx and not self.quiet_mode:
            logger.info(
                "Ollama num_ctx：将请求%d tokens（模型最大值来自/api/show）",
                self._ollama_num_ctx,
            )

        if not self.quiet_mode:
            if compression_enabled:
                print(f"📊 上下文限制：{self.context_compressor.context_length:,} tokens（在{int(compression_threshold*100)}%时压缩 = {self.context_compressor.threshold_tokens:,}）")
            else:
                print(f"📊 上下文限制：{self.context_compressor.context_length:,} tokens（自动压缩已禁用）")

        # 立即检查以便CLI用户在启动时看到警告。
        # Gateway status_callback尚未连接，所以任何警告存储在
        # _compression_warning中并在第一次run_conversation()时重放。
        self._compression_warning = None
        self._check_compression_model_feasibility()

        # 快照主要运行时用于每轮恢复。当回退在一轮中激活时，
        # 下一轮恢复这些值以便首选模型每次获得新的尝试机会。
        # 使用单个字典以便新状态字段易于添加，无需N个单独属性。
        _cc = self.context_compressor
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            # _try_activate_fallback()覆盖的上下文引擎状态。
            # 使用getattr获取model/base_url/api_key/provider，因为插件
            # 引擎可能没有这些（它们是ContextCompressor特有的）。
            "compressor_model": getattr(_cc, "model", self.model),
            "compressor_base_url": getattr(_cc, "base_url", self.base_url),
            "compressor_api_key": getattr(_cc, "api_key", ""),
            "compressor_provider": getattr(_cc, "provider", self.provider),
            "compressor_context_length": _cc.context_length,
            "compressor_threshold_tokens": _cc.threshold_tokens,
        }
        if self.api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

    def reset_session_state(self):
        """Reset all session-scoped token counters to 0 for a fresh session.
        
        This method encapsulates the reset logic for all session-level metrics
        including:
        - Token usage counters (input, output, total, prompt, completion)
        - Cache read/write tokens
        - API call count
        - Reasoning tokens
        - Estimated cost tracking
        - Context compressor internal counters
        
        The method safely handles optional attributes (e.g., context compressor)
        using ``hasattr`` checks.
        
        This keeps the counter reset logic DRY and maintainable in one place
        rather than scattering it across multiple methods.
        """
        # Token usage counters
        self.session_total_tokens = 0
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_cache_read_tokens = 0
        self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.session_api_calls = 0
        self.session_estimated_cost_usd = 0.0
        self.session_cost_status = "unknown"
        self.session_cost_source = "none"
        
        # Turn counter (added after reset_session_state was first written — #2635)
        self._user_turn_count = 0

        # Context engine reset (works for both built-in compressor and plugins)
        if hasattr(self, "context_compressor") and self.context_compressor:
            self.context_compressor.on_session_reset()
    
    def switch_model(self, new_model, new_provider, api_key='', base_url='', api_mode=''):
        """为运行中的代理就地切换模型/提供商。

        在 ``model_switch.switch_model()`` 解析了凭证并验证模型后，
        由 /model 命令处理器（CLI 和 gateway）调用。此方法执行实际的运行时替换：
        重建客户端、更新缓存标志并刷新上下文压缩器。

        实现镜像了 ``_try_activate_fallback()`` 的客户端替换逻辑，但也更新了
        ``_primary_runtime``，以便更改在多个轮次间持续存在（不同于仅在单个轮次内生效的 fallback）。
        """
        import logging
        from hermes_cli.providers import determine_api_mode

        # ── 如果未提供则确定 api_mode ──
        if not api_mode:
            api_mode = determine_api_mode(new_provider, base_url)

        old_model = self.model
        old_provider = self.provider

        # ── 替换核心运行时字段 ──
        self.model = new_model
        self.provider = new_provider
        self.base_url = base_url or self.base_url
        self.api_mode = api_mode
        if api_key:
            self.api_key = api_key

        # ── 构建新客户端 ──
        if api_mode == "anthropic_messages":
            from agent.anthropic_adapter import (
                build_anthropic_client,
                resolve_anthropic_token,
                _is_oauth_token,
            )
            # 仅当提供商确实是 Anthropic 时才回退到 ANTHROPIC_TOKEN。
            # 其他 anthropic_messages 提供商（MiniMax、阿里巴巴等）必须使用自己的
            # API 密钥——回退会将 Anthropic 凭证发送到第三方端点。
            _is_native_anthropic = new_provider == "anthropic"
            effective_key = (api_key or self.api_key or resolve_anthropic_token() or "") if _is_native_anthropic else (api_key or self.api_key or "")
            self.api_key = effective_key
            self._anthropic_api_key = effective_key
            self._anthropic_base_url = base_url or getattr(self, "_anthropic_base_url", None)
            self._anthropic_client = build_anthropic_client(
                effective_key, self._anthropic_base_url,
            )
            self._is_anthropic_oauth = _is_oauth_token(effective_key)
            self.client = None
            self._client_kwargs = {}
        else:
            effective_key = api_key or self.api_key
            effective_base = base_url or self.base_url
            self._client_kwargs = {
                "api_key": effective_key,
                "base_url": effective_base,
            }
            self.client = self._create_openai_client(
                dict(self._client_kwargs),
                reason="switch_model",
                shared=True,
            )

        # ── 重新评估提示词缓存 ──
        is_native_anthropic = api_mode == "anthropic_messages" and new_provider == "anthropic"
        self._use_prompt_caching = (
            ("openrouter" in (self.base_url or "").lower() and "claude" in new_model.lower())
            or is_native_anthropic
        )

        # ── 更新上下文压缩器 ──
        if hasattr(self, "context_compressor") and self.context_compressor:
            from agent.model_metadata import get_model_context_length
            new_context_length = get_model_context_length(
                self.model,
                base_url=self.base_url,
                api_key=self.api_key,
                provider=self.provider,
                config_context_length=getattr(self, "_config_context_length", None),
            )
            self.context_compressor.update_model(
                model=self.model,
                context_length=new_context_length,
                base_url=self.base_url,
                api_key=getattr(self, "api_key", ""),
                provider=self.provider,
                api_mode=self.api_mode,
            )

        # ── 使缓存的系统提示词无效，以便下次轮次重建 ──
        self._cached_system_prompt = None

        # ── 更新 _primary_runtime，以便更改在多个轮次间持续存在 ──
        _cc = self.context_compressor if hasattr(self, "context_compressor") and self.context_compressor else None
        self._primary_runtime = {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "api_mode": self.api_mode,
            "api_key": getattr(self, "api_key", ""),
            "client_kwargs": dict(self._client_kwargs),
            "use_prompt_caching": self._use_prompt_caching,
            "compressor_model": getattr(_cc, "model", self.model) if _cc else self.model,
            "compressor_base_url": getattr(_cc, "base_url", self.base_url) if _cc else self.base_url,
            "compressor_api_key": getattr(_cc, "api_key", "") if _cc else "",
            "compressor_provider": getattr(_cc, "provider", self.provider) if _cc else self.provider,
            "compressor_context_length": _cc.context_length if _cc else 0,
            "compressor_threshold_tokens": _cc.threshold_tokens if _cc else 0,
        }
        if api_mode == "anthropic_messages":
            self._primary_runtime.update({
                "anthropic_api_key": self._anthropic_api_key,
                "anthropic_base_url": self._anthropic_base_url,
                "is_anthropic_oauth": self._is_anthropic_oauth,
            })

        # ── 重置 fallback 状态 ──
        self._fallback_activated = False
        self._fallback_index = 0

        logging.info(
            "Model switched in-place: %s (%s) -> %s (%s)",
            old_model, old_provider, new_model, new_provider,
        )

    def _safe_print(self, *args, **kwargs):
        """静默处理管道中断/标准输出关闭的打印函数。

        在无头环境（systemd、Docker、nohup）中，标准输出可能在会话中途变得不可用。
        原始的 ``print()`` 会引发 ``OSError``，这可能导致 cron 任务崩溃并丢失已完成的工作。

        内部通过 ``self._print_fn`` 路由（默认：内置 ``print``），
        以便像 CLI 这样的调用者可以注入一个正确处理 ANSI 转义序列的渲染器
        （例如 prompt_toolkit 的 ``print_formatted_text(ANSI(...))``）而无需修改此方法。
        """
        try:
            fn = self._print_fn or print
            fn(*args, **kwargs)
        except (OSError, ValueError):
            pass

    def _vprint(self, *args, force: bool = False, **kwargs):
        """详细打印——在主动流式传输 token 时被抑制。

        对于应始终显示的错误/警告消息，即使在流式播放期间（TTS 或显示），
        也传递 ``force=True``。

        在工具执行期间（``_executing_tools`` 为 True），即使注册了流消费者也允许打印，
        因为此时没有正在流式传输的 token。

        在主响应已交付且剩余的工具调用是响应后的清理工作（``_mute_post_response``）后，
        所有非强制输出都被抑制。

        ``suppress_status_output`` 是更严格的 CLI 自动化模式，
        用于可解析的单查询流程，如 ``hermes chat -q``。在该模式下，
        通过 ``_vprint`` 路由的所有状态/诊断打印都被抑制，以便标准输出保持机器可读。
        """
        if getattr(self, "suppress_status_output", False):
            return
        if not force and getattr(self, "_mute_post_response", False):
            return
        if not force and self._has_stream_consumers() and not self._executing_tools:
            return
        self._safe_print(*args, **kwargs)

    def _should_start_quiet_spinner(self) -> bool:
        """当静默模式的旋转器输出有安全接收器时返回 True。

        在无头/stdio 协议环境中，没有自定义 ``_print_fn`` 的原始旋转器会回退到
        ``sys.stdout`` 并可能破坏协议流，如 ACP JSON-RPC。仅在以下情况允许静默旋转器：
        - 输出通过 ``_print_fn`` 显式重路由；或者
        - 标准输出是真实的 TTY。
        """
        if self._print_fn is not None:
            return True
        stream = getattr(sys, "stdout", None)
        if stream is None:
            return False
        try:
            return bool(stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    def _should_emit_quiet_tool_messages(self) -> bool:
        """Return True when quiet-mode tool summaries should print directly.

        When the caller provides ``tool_progress_callback`` (for example the CLI
        TUI or a gateway progress renderer), that callback owns progress display.
        Emitting quiet-mode summary lines here duplicates progress and leaks tool
        previews into flows that are expected to stay silent, such as
        ``hermes chat -q``.
        """
        return self.quiet_mode and not self.tool_progress_callback

    def _emit_status(self, message: str) -> None:
        """向 CLI 和 gateway 两个通道发送生命周期状态消息。

        CLI 用户通过 ``_vprint(force=True)`` 看到消息，因此无论详细/静默模式如何，
        它始终可见。Gateway 消费者通过 ``status_callback("lifecycle", ...)`` 接收它。

        此辅助函数从不引发异常——异常被吞掉，因此它不会中断重试/fallback 逻辑。
        """
        try:
            self._vprint(f"{self.log_prefix}{message}", force=True)
        except Exception:
            pass
        if self.status_callback:
            try:
                self.status_callback("lifecycle", message)
            except Exception:
                logger.debug("status_callback error in _emit_status", exc_info=True)

    def _current_main_runtime(self) -> Dict[str, str]:
        """返回会话范围的辅助路由的实时主运行时。"""
        return {
            "model": getattr(self, "model", "") or "",
            "provider": getattr(self, "provider", "") or "",
            "base_url": getattr(self, "base_url", "") or "",
            "api_key": getattr(self, "api_key", "") or "",
            "api_mode": getattr(self, "api_mode", "") or "",
        }

    def _check_compression_model_feasibility(self) -> None:
        """如果辅助压缩模型的上下文窗口小于主模型的压缩阈值，则在会话开始时发出警告。

        当辅助模型无法容纳需要摘要的内容时，压缩要么会直接失败（LLM 调用错误），
        要么会生成严重截断的摘要。

        在 ``__init__`` 期间调用，以便 CLI 用户立即看到警告（通过 ``_vprint``）。
        Gateway 在构造之后设置 ``status_callback``，因此 ``_replay_compression_warning()``
        在第一次 ``run_conversation()`` 调用时通过回调重新发送存储的警告。
        """
        if not self.compression_enabled:
            return
        try:
            from agent.auxiliary_client import get_text_auxiliary_client
            from agent.model_metadata import get_model_context_length

            client, aux_model = get_text_auxiliary_client(
                "compression",
                main_runtime=self._current_main_runtime(),
            )
            if client is None or not aux_model:
                msg = (
                    "⚠ No auxiliary LLM provider configured — context "
                    "compression will drop middle turns without a summary. "
                    "Run `hermes setup` or set OPENROUTER_API_KEY."
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "No auxiliary LLM provider for compression — "
                    "summaries will be unavailable."
                )
                return

            aux_base_url = str(getattr(client, "base_url", ""))
            aux_api_key = str(getattr(client, "api_key", ""))

            # 读取用户配置的压缩模型的 context_length。
            # 自定义端点通常不支持 /models API 查询，因此 get_model_context_length()
            # 会回退到 128K 默认值，忽略显式配置值。将其作为最高优先级提示传递，
            # 以便始终尊重配置值。
            _aux_cfg = (self.config or {}).get("auxiliary", {}).get("compression", {})
            _aux_context_config = _aux_cfg.get("context_length") if isinstance(_aux_cfg, dict) else None
            if _aux_context_config is not None:
                try:
                    _aux_context_config = int(_aux_context_config)
                except (TypeError, ValueError):
                    _aux_context_config = None

            aux_context = get_model_context_length(
                aux_model,
                base_url=aux_base_url,
                api_key=aux_api_key,
                config_context_length=_aux_context_config,
            )

            threshold = self.context_compressor.threshold_tokens
            if aux_context < threshold:
                # 建议一个适合辅助模型的阈值，向下舍入到整洁的百分比。
                safe_pct = int((aux_context / self.context_compressor.context_length) * 100)
                msg = (
                    f"⚠ Compression model ({aux_model}) context "
                    f"is {aux_context:,} tokens, but the main model's "
                    f"compression threshold is {threshold:,} tokens. "
                    f"Context compression will not be possible — the "
                    f"content to summarise will exceed the auxiliary "
                    f"model's context window.\n"
                    f"  Fix options (config.yaml):\n"
                    f"  1. Use a larger compression model:\n"
                    f"       auxiliary:\n"
                    f"         compression:\n"
                    f"           model: <model-with-{threshold:,}+-context>\n"
                    f"  2. Lower the compression threshold to fit "
                    f"the current model:\n"
                    f"       compression:\n"
                    f"         threshold: 0.{safe_pct:02d}"
                )
                self._compression_warning = msg
                self._emit_status(msg)
                logger.warning(
                    "Auxiliary compression model %s has %d token context, "
                    "below the main model's compression threshold of %d "
                    "tokens — compression summaries will fail or be "
                    "severely truncated.",
                    aux_model,
                    aux_context,
                    threshold,
                )
        except Exception as exc:
            logger.debug(
                "Compression feasibility check failed (non-fatal): %s", exc
            )

    def _replay_compression_warning(self) -> None:
        """通过 ``status_callback`` 重新发送压缩警告。

        在 ``__init__`` 期间，gateway 的 ``status_callback`` 尚未连接，
        因此 ``_emit_status`` 仅到达 ``_vprint``（CLI）。此方法在第一次
        ``run_conversation()`` 开始时调用一次——到那时 gateway 已设置回调，
        因此每个平台（Telegram、Discord、Slack 等）都能收到警告。
        """
        msg = getattr(self, "_compression_warning", None)
        if msg and self.status_callback:
            try:
                self.status_callback("lifecycle", msg)
            except Exception:
                pass

    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """当基础 URL 针对 OpenAI 的原生 API 时返回 True。"""
        url = (base_url or self._base_url_lower).lower()
        return "api.openai.com" in url and "openrouter" not in url

    def _is_openrouter_url(self) -> bool:
        """当基础 URL 针对 OpenRouter 时返回 True。"""
        return "openrouter" in self._base_url_lower

    @staticmethod
    def _model_requires_responses_api(model: str) -> bool:
        """对于需要 Responses API 路径的模型返回 True。

        GPT-5.x 模型在 /v1/chat/completions 上被 OpenAI 和 OpenRouter 都拒绝
        （错误：``unsupported_api_for_model``）。检测这些模型，以便无论哪个提供商
        为模型提供服务，都设置正确的 api_mode。
        """
        m = model.lower()
        # 去除供应商前缀（例如 "openai/gpt-5.4" → "gpt-5.4"）
        if "/" in m:
            m = m.rsplit("/", 1)[-1]
        return m.startswith("gpt-5")

    def _max_tokens_param(self, value: int) -> dict:
        """为当前提供商返回正确的最大 token 关键字参数。
        
        OpenAI 的较新模型（gpt-4o、o 系列、gpt-5+）需要 'max_completion_tokens'。
        OpenRouter、本地模型和较旧的 OpenAI 模型使用 'max_tokens'。
        """
        if self._is_direct_openai_url():
            return {"max_completion_tokens": value}
        return {"max_tokens": value}

    def _has_content_after_think_block(self, content: str) -> bool:
        """
        检查内容在任何推理/思考块之后是否有实际文本。

        这检测模型仅输出推理但没有实际响应的情况，这表示应该重试的不完整生成。
        必须与 _strip_think_blocks() 的标签变体保持同步。

        参数:
            content: 要检查的助手消息内容

        返回:
            如果在思考块后有有意义的内容则返回 True，否则返回 False
        """
        if not content:
            return False

        # 移除所有推理标签变体（必须与 _strip_think_blocks 匹配）
        cleaned = self._strip_think_blocks(content)

        # Check if there's any non-whitespace content remaining
        return bool(cleaned.strip())
    
    def _strip_think_blocks(self, content: str) -> str:
        """Remove reasoning/thinking blocks from content, returning only visible text."""
        if not content:
            return ""
        # Strip all reasoning tag variants: <think>, <thinking>, <THINKING>,
        # <reasoning>, <REASONING_SCRATCHPAD>, <thought> (Gemma 4)
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
        content = re.sub(r'<REASONING_SCRATCHPAD>.*?</REASONING_SCRATCHPAD>', '', content, flags=re.DOTALL)
        content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'</?(?:think|thinking|reasoning|thought|REASONING_SCRATCHPAD)>\s*', '', content, flags=re.IGNORECASE)
        return content

    def _looks_like_codex_intermediate_ack(
        self,
        user_message: str,
        assistant_content: str,
        messages: List[Dict[str, Any]],
    ) -> bool:
        """检测应该继续而不是结束轮次的计划/确认消息。"""
        if any(isinstance(msg, dict) and msg.get("role") == "tool" for msg in messages):
            return False

        assistant_text = self._strip_think_blocks(assistant_content or "").strip().lower()
        if not assistant_text:
            return False
        if len(assistant_text) > 1200:
            return False

        has_future_ack = bool(
            re.search(r"\b(i['’]ll|i will|let me|i can do that|i can help with that)\b", assistant_text)
        )
        if not has_future_ack:
            return False

        action_markers = (
            "look into",
            "look at",
            "inspect",
            "scan",
            "check",
            "analyz",
            "review",
            "explore",
            "read",
            "open",
            "run",
            "test",
            "fix",
            "debug",
            "search",
            "find",
            "walkthrough",
            "report back",
            "summarize",
        )
        workspace_markers = (
            "directory",
            "current directory",
            "current dir",
            "cwd",
            "repo",
            "repository",
            "codebase",
            "project",
            "folder",
            "filesystem",
            "file tree",
            "files",
            "path",
        )

        user_text = (user_message or "").strip().lower()
        user_targets_workspace = (
            any(marker in user_text for marker in workspace_markers)
            or "~/" in user_text
            or "/" in user_text
        )
        assistant_mentions_action = any(marker in assistant_text for marker in action_markers)
        assistant_targets_workspace = any(
            marker in assistant_text for marker in workspace_markers
        )
        return (user_targets_workspace or assistant_targets_workspace) and assistant_mentions_action
    
    
    def _extract_reasoning(self, assistant_message) -> Optional[str]:
        """
        从助手消息中提取推理/思考内容。
        
        OpenRouter 和各种提供商可以以多种格式返回推理：
        1. message.reasoning - 直接推理字段（DeepSeek、Qwen 等）
        2. message.reasoning_content - 替代字段（Moonshot AI、Novita 等）
        3. message.reasoning_details - {type, summary, ...} 对象数组（OpenRouter 统一格式）
        
        参数:
            assistant_message: 来自 API 响应的助手消息对象
            
        返回:
            组合的推理解析，或如果未找到推理则返回 None
        """
        reasoning_parts = []
        
        # 检查直接推理字段
        if hasattr(assistant_message, 'reasoning') and assistant_message.reasoning:
            reasoning_parts.append(assistant_message.reasoning)
        
        # 检查 reasoning_content 字段（一些提供商使用的替代名称）
        if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
            # 如果与推理相同则不要重复
            if assistant_message.reasoning_content not in reasoning_parts:
                reasoning_parts.append(assistant_message.reasoning_content)
        
        # 检查 reasoning_details 数组（OpenRouter 统一格式）
        # 格式：[{"type": "reasoning.summary", "summary": "...", ...}, ...]
        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            for detail in assistant_message.reasoning_details:
                if isinstance(detail, dict):
                    # 从推理详情对象中提取摘要
                    summary = (
                        detail.get('summary')
                        or detail.get('thinking')
                        or detail.get('content')
                        or detail.get('text')
                    )
                    if summary and summary not in reasoning_parts:
                        reasoning_parts.append(summary)

        # 一些提供商将推理直接嵌入在助手内容中，
        # 而不是返回结构化的推理字段。仅在未找到结构化推理时才回退到内联提取。
        content = getattr(assistant_message, "content", None)
        if not reasoning_parts and isinstance(content, str) and content:
            inline_patterns = (
                r"<think>(.*?)</think>",
                r"<thinking>(.*?)</thinking>",
                r"<reasoning>(.*?)</reasoning>",
                r"<REASONING_SCRATCHPAD>(.*?)</REASONING_SCRATCHPAD>",
            )
            for pattern in inline_patterns:
                flags = re.DOTALL | re.IGNORECASE
                for block in re.findall(pattern, content, flags=flags):
                    cleaned = block.strip()
                    if cleaned and cleaned not in reasoning_parts:
                        reasoning_parts.append(cleaned)
        
        # 组合所有推理部分
        if reasoning_parts:
            return "\n\n".join(reasoning_parts)
        
        return None

    def _cleanup_task_resources(self, task_id: str) -> None:
        """清理给定任务的 VM 和浏览器资源。

        当活动终端环境被标记为持久化（``persistent_filesystem=True``）时，
        跳过 ``cleanup_vm``，以便长生命周期的沙箱容器在轮次间存活。
        ``terminal_tool._cleanup_inactive_envs`` 中的空闲回收器在超过
        ``terminal.lifetime_seconds`` 后仍会将它们拆除。非持久化后端
        像以前一样在每轮次后拆除，以防止资源泄漏（这是 Morph 后端此钩子的原始意图，
        参见提交 fbd3a2fd）。
        """
        try:
            if is_persistent_env(task_id):
                if self.verbose_logging:
                    logging.debug(
                        f"Skipping per-turn cleanup_vm for persistent env {task_id}; "
                        f"idle reaper will handle it."
                    )
            else:
                cleanup_vm(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup VM for task {task_id}: {e}")
        try:
            cleanup_browser(task_id)
        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to cleanup browser for task {task_id}: {e}")

    # ------------------------------------------------------------------
    # 后台记忆/技能审查
    # ------------------------------------------------------------------

    _MEMORY_REVIEW_PROMPT = (
        "审查上面的对话并考虑在适当的情况下保存到记忆中。\n\n"
        "重点关注：\n"
        "1. 用户是否透露了关于自己的事情——他们的角色、愿望、偏好或值得记住的个人细节？\n"
        "2. 用户是否表达了关于你应该如何表现、他们的工作风格或他们希望你如何运作的期望？\n\n"
        "如果有突出的内容，使用记忆工具保存它。"
        "如果没有值得保存的内容，只需说'没有可保存的内容。'然后停止。"
    )

    _SKILL_REVIEW_PROMPT = (
        "审查上面的对话并考虑在适当的情况下保存或更新技能。\n\n"
        "重点关注：是否使用了一种非平凡的方法来完成需要反复试验的任务，"
        "或者由于途中的经验发现而改变了方向，或者用户是否期望或想要不同的方法或结果？\n\n"
        "如果相关技能已存在，用你学到的内容更新它。"
        "否则，如果该方法可重用，则创建一个新技能。\n"
        "如果没有值得保存的内容，只需说'没有可保存的内容。'然后停止。"
    )

    _COMBINED_REVIEW_PROMPT = (
        "Review the conversation above and consider two things:\n\n"
        "**Memory**: Has the user revealed things about themselves — their persona, "
        "desires, preferences, or personal details? Has the user expressed expectations "
        "about how you should behave, their work style, or ways they want you to operate? "
        "如果是这样，使用记忆工具保存。\n\n"
        "**技能**：是否使用了一种非平凡的方法来完成需要反复试验的任务，"
        "或者由于途中的经验发现而改变了方向，或者用户是否期望或想要不同的方法或结果？"
        "如果相关技能已存在，则更新它。否则，如果该方法可重用，则创建一个新技能。\n\n"
        "仅在确实有值得保存的内容时才采取行动。"
        "如果没有突出的内容，只需说'没有可保存的内容。'然后停止。"
    )

    def _spawn_background_review(
        self,
        messages_snapshot: List[Dict],
        review_memory: bool = False,
        review_skills: bool = False,
    ) -> None:
        """生成后台线程以审查对话以进行记忆/技能保存。

        创建一个完整的 AIAgent 分支，具有与主会话相同的模型、工具和上下文。
        审查提示词作为下一个用户轮次附加到分叉的对话中。直接写入共享的记忆/技能存储。
        从不修改主对话历史或产生用户可见的输出。
        """
        import threading

        # 根据触发的内容选择正确的提示词
        if review_memory and review_skills:
            prompt = self._COMBINED_REVIEW_PROMPT
        elif review_memory:
            prompt = self._MEMORY_REVIEW_PROMPT
        else:
            prompt = self._SKILL_REVIEW_PROMPT

        def _run_review():
            import contextlib, os as _os
            review_agent = None
            try:
                with open(_os.devnull, "w") as _devnull, \
                     contextlib.redirect_stdout(_devnull), \
                     contextlib.redirect_stderr(_devnull):
                    review_agent = AIAgent(
                        model=self.model,
                        max_iterations=8,
                        quiet_mode=True,
                        platform=self.platform,
                        provider=self.provider,
                    )
                    review_agent._memory_store = self._memory_store
                    review_agent._memory_enabled = self._memory_enabled
                    review_agent._user_profile_enabled = self._user_profile_enabled
                    review_agent._memory_nudge_interval = 0
                    review_agent._skill_nudge_interval = 0

                    review_agent.run_conversation(
                        user_message=prompt,
                        conversation_history=messages_snapshot,
                    )

                # 扫描审查代理的消息以查找成功的工具操作，并向用户显示紧凑摘要。
                actions = []
                for msg in getattr(review_agent, "_session_messages", []):
                    if not isinstance(msg, dict) or msg.get("role") != "tool":
                        continue
                    try:
                        data = json.loads(msg.get("content", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if not data.get("success"):
                        continue
                    message = data.get("message", "")
                    target = data.get("target", "")
                    if "created" in message.lower():
                        actions.append(message)
                    elif "updated" in message.lower():
                        actions.append(message)
                    elif "added" in message.lower() or (target and "add" in message.lower()):
                        label = "记忆" if target == "memory" else "用户资料" if target == "user" else target
                        actions.append(f"{label}已更新")
                    elif "Entry added" in message:
                        label = "记忆" if target == "memory" else "用户资料" if target == "user" else target
                        actions.append(f"{label}已更新")
                    elif "removed" in message.lower() or "replaced" in message.lower():
                        label = "记忆" if target == "memory" else "用户资料" if target == "user" else target
                        actions.append(f"{label}已更新")

                if actions:
                    summary = " · ".join(dict.fromkeys(actions))
                    self._safe_print(f"  💾 {summary}")
                    _bg_cb = self.background_review_callback
                    if _bg_cb:
                        try:
                            _bg_cb(f"💾 {summary}")
                        except Exception:
                            pass

            except Exception as e:
                logger.debug("后台记忆/技能审查失败：%s", e)
            finally:
                # 关闭所有资源（httpx 客户端、子进程等），以便 GC 不会尝试在
                # 已死亡的 asyncio 事件循环上清理它们（这会产生"事件循环已关闭"错误）。
                if review_agent is not None:
                    try:
                        review_agent.close()
                    except Exception:
                        pass

        t = threading.Thread(target=_run_review, daemon=True, name="bg-review")
        t.start()

    def _apply_persist_user_message_override(self, messages: List[Dict]) -> None:
        """在持久化/返回之前重写当前轮次的用户消息。

        某些调用路径需要仅用于 API 的用户消息变体，而不让该合成文本泄漏到
        持久化的转录本或恢复的会话历史中。当为活动轮次配置了覆盖时，就地修改
        内存中的消息列表，以便持久化和返回的历史都保持干净。
        """
        idx = getattr(self, "_persist_user_message_idx", None)
        override = getattr(self, "_persist_user_message_override", None)
        if override is None or idx is None:
            return
        if 0 <= idx < len(messages):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") == "user":
                msg["content"] = override

    def _persist_session(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """在任何退出路径上将会话状态保存到 JSON 日志和 SQLite 中。

        确保对话永远不会丢失，即使在错误或提前返回时也是如此。
        当 ``persist_session=False`` 时跳过（临时辅助流程）。
        """
        if not self.persist_session:
            return
        self._apply_persist_user_message_override(messages)
        self._session_messages = messages
        self._save_session_log(messages)
        self._flush_messages_to_session_db(messages, conversation_history)

    def _flush_messages_to_session_db(self, messages: List[Dict], conversation_history: List[Dict] = None):
        """将任何未刷新的消息持久化到 SQLite 会话存储中。

        使用 _last_flushed_db_idx 来跟踪哪些消息已经被写入，
        因此重复调用（来自多个退出路径）仅写入真正的新消息——防止重复写入错误（#860）。
        """
        if not self._session_db:
            return
        self._apply_persist_user_message_override(messages)
        try:
            # 如果 create_session() 在启动时失败（例如临时锁定），
            # 会话行可能还不存在。ensure_session() 使用 INSERT OR
            # IGNORE，因此当行已经存在时它是一个无操作。
            self._session_db.ensure_session(
                self.session_id,
                source=self.platform or "cli",
                model=self.model,
            )
            start_idx = len(conversation_history) if conversation_history else 0
            flush_from = max(start_idx, self._last_flushed_db_idx)
            for msg in messages[flush_from:]:
                role = msg.get("role", "unknown")
                content = msg.get("content")
                tool_calls_data = None
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_calls_data = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        for tc in msg.tool_calls
                    ]
                elif isinstance(msg.get("tool_calls"), list):
                    tool_calls_data = msg["tool_calls"]
                self._session_db.append_message(
                    session_id=self.session_id,
                    role=role,
                    content=content,
                    tool_name=msg.get("tool_name"),
                    tool_calls=tool_calls_data,
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning") if role == "assistant" else None,
                    reasoning_details=msg.get("reasoning_details") if role == "assistant" else None,
                    codex_reasoning_items=msg.get("codex_reasoning_items") if role == "assistant" else None,
                )
            self._last_flushed_db_idx = len(messages)
        except Exception as e:
            logger.warning("Session DB append_message failed: %s", e)

    def _get_messages_up_to_last_assistant(self, messages: List[Dict]) -> List[Dict]:
        """
        获取直到（但不包括）最后一个助手轮次的消息。
        
        当我们需要"回滚"到对话中最后一个成功点时使用，
        通常是在最终助手消息不完整或格式不正确时。
        
        参数:
            messages: 完整消息列表
            
        Returns:
            Messages up to the last complete assistant turn (ending with user/tool message)
        """
        if not messages:
            return []
        
        # Find the index of the last assistant message
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        if last_assistant_idx is None:
            # 未找到助手消息，返回所有消息
            return messages.copy()
        
        # 返回直到（不包括）最后一个助手消息的所有内容
        return messages[:last_assistant_idx]
    
    def _format_tools_for_system_message(self) -> str:
        """
        以轨迹格式为系统消息格式化工具定义。
        
        返回:
            str: 工具定义的 JSON 字符串表示
        """
        if not self.tools:
            return "[]"
        
        # 将工具定义转换为轨迹中期望的格式
        formatted_tools = []
        for tool in self.tools:
            func = tool["function"]
            formatted_tool = {
                "name": func["name"],
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {}),
                "required": None  # 匹配示例中的格式
            }
            formatted_tools.append(formatted_tool)
        
        return json.dumps(formatted_tools, ensure_ascii=False)
    
    def _convert_to_trajectory_format(self, messages: List[Dict[str, Any]], user_query: str, completed: bool) -> List[Dict[str, Any]]:
        """
        将内部消息格式转换为用于保存的轨迹格式。
        
        参数:
            messages (List[Dict]): 内部消息历史
            user_query (str): 原始用户查询
            completed (bool): 对话是否成功完成
            
        返回:
            List[Dict]: 轨迹格式的消息
        """
        trajectory = []
        
        # 添加带有工具定义的系统消息
        system_msg = (
            "你是一个函数调用 AI 模型。你会获得 <tools> </tools> XML 标签内的函数签名。"
            "你可以调用一个或多个函数来协助处理用户查询。如果可用工具与协助用户查询无关，"
            "只需以自然的对话语言回应。不要假设要代入函数的值。调用并执行函数后，"
            "你将获得 <tool_response> </tool_response> XML 标签内的函数结果。以下是可用工具：\n"
            f"<tools>\n{self._format_tools_for_system_message()}\n</tools>\n"
            "对于每个函数调用，返回一个 JSON 对象，每个都具有以下 pydantic 模型 json 模式：\n"
            "{'title': 'FunctionCall', 'type': 'object', 'properties': {'name': {'title': 'Name', 'type': 'string'}, "
            "'arguments': {'title': 'Arguments', 'type': 'object'}}, 'required': ['name', 'arguments']}\n"
            "每个函数调用都应包含在 <tool_call> </tool_call> XML 标签内。\n"
            "示例：\n<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>"
        )
        
        trajectory.append({
            "from": "system",
            "value": system_msg
        })
        
        # 添加实际的用户提示词（来自数据集）作为第一条人类消息
        trajectory.append({
            "from": "human",
            "value": user_query
        })
        
        # 跳过第一条消息（用户查询），因为我们已经在上面添加了它。
        # 预填充消息仅在 API 调用时注入（不在消息列表中），
        # 因此这里不需要偏移量调整。
        i = 1
        
        while i < len(messages):
            msg = messages[i]
            
            if msg["role"] == "assistant":
                # 检查此消息是否有工具调用
                if "tool_calls" in msg and msg["tool_calls"]:
                    # 格式化带有工具调用的助手消息
                    # 为轨迹存储在推理周围添加 <think> 标签
                    content = ""
                    
                    # 如果可用，在 <think> 标签中预先添加推理（原生思考 token）
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    if msg.get("content") and msg["content"].strip():
                        # 将任何 <REASONING_SCRATCHPAD> 标签转换为 <think> 标签
                        #（当禁用原生思考且模型通过 XML 进行推理时使用）
                        content += convert_scratchpad_to_think(msg["content"]) + "\n"
                    
                    # 添加用 XML 标签包装的工具调用
                    for tool_call in msg["tool_calls"]:
                        if not tool_call or not isinstance(tool_call, dict): continue
                        # 解析参数——应该总是成功，因为我们在对话期间验证了
                        # 但保留 try-except 作为安全网
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"]) if isinstance(tool_call["function"]["arguments"], str) else tool_call["function"]["arguments"]
                        except json.JSONDecodeError:
                            # 这不应该发生，因为我们在对话期间验证并重试，
                            # 但如果发生，记录警告并使用空字典
                            logging.warning(f"轨迹转换中意外的无效 JSON：{tool_call['function']['arguments'][:100]}")
                            arguments = {}
                        
                        tool_call_json = {
                            "name": tool_call["function"]["name"],
                            "arguments": arguments
                        }
                        content += f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>\n"
                    
                    # 确保每个 gpt 轮次都有一个 <think> 块（如果没有推理则为空）
                    # 以便训练数据的格式一致
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.rstrip()
                    })
                    
                    # 收集所有后续工具响应
                    tool_responses = []
                    j = i + 1
                    while j < len(messages) and messages[j]["role"] == "tool":
                        tool_msg = messages[j]
                        # 用 XML 标签格式化工具响应
                        tool_response = "<tool_response>\n"
                        
                        # 如果工具内容看起来像 JSON，尝试将其解析为 JSON
                        tool_content = tool_msg["content"]
                        try:
                            if tool_content.strip().startswith(("{", "[")):
                                tool_content = json.loads(tool_content)
                        except (json.JSONDecodeError, AttributeError):
                            pass  # 如果不是有效的 JSON，则保持为字符串
                        
                        tool_index = len(tool_responses)
                        tool_name = (
                            msg["tool_calls"][tool_index]["function"]["name"]
                            if tool_index < len(msg["tool_calls"])
                            else "unknown"
                        )
                        tool_response += json.dumps({
                            "tool_call_id": tool_msg.get("tool_call_id", ""),
                            "name": tool_name,
                            "content": tool_content
                        }, ensure_ascii=False)
                        tool_response += "\n</tool_response>"
                        tool_responses.append(tool_response)
                        j += 1
                    
                    # 将所有工具响应添加为单个消息
                    if tool_responses:
                        trajectory.append({
                            "from": "tool",
                            "value": "\n".join(tool_responses)
                        })
                        i = j - 1  # 跳过我们刚刚处理的工具消息
                
                else:
                    # 没有工具调用的常规助手消息
                    # 为轨迹存储在推理周围添加 <think> 标签
                    content = ""
                    
                    # 如果可用，在 <think> 标签中预先添加推理（原生思考 token）
                    if msg.get("reasoning") and msg["reasoning"].strip():
                        content = f"<think>\n{msg['reasoning']}\n</think>\n"
                    
                    # 将任何 <REASONING_SCRATCHPAD> 标签转换为 <think> 标签
                    #（当禁用原生思考且模型通过 XML 进行推理时使用）
                    raw_content = msg["content"] or ""
                    content += convert_scratchpad_to_think(raw_content)
                    
                    # 确保每个 gpt 轮次都有一个 <think> 块（如果没有推理则为空）
                    if "<think>" not in content:
                        content = "<think>\n</think>\n" + content
                    
                    trajectory.append({
                        "from": "gpt",
                        "value": content.strip()
                    })
            
            elif msg["role"] == "user":
                trajectory.append({
                    "from": "human",
                    "value": msg["content"]
                })
            
            i += 1
        
        return trajectory
    
    def _save_trajectory(self, messages: List[Dict[str, Any]], user_query: str, completed: bool):
        """
        将对话轨迹保存到 JSONL 文件。
        
        参数:
            messages (List[Dict]): 完整消息历史
            user_query (str): 原始用户查询
            completed (bool): 对话是否成功完成
        """
        if not self.save_trajectories:
            return
        
        trajectory = self._convert_to_trajectory_format(messages, user_query, completed)
        _save_trajectory_to_file(trajectory, self.model, completed)
    
    @staticmethod
    def _summarize_api_error(error: Exception) -> str:
        """从 API 错误中提取人类可读的一行摘要。

        通过提取 <title> 标签来处理 Cloudflare HTML 错误页面（502、503 等），
        而不是转储原始 HTML。对于其他所有情况，回退到截断的 str(error)。
        """
        import re as _re
        raw = str(error)

        # Cloudflare / 代理 HTML 页面：获取 <title> 以获得清晰的摘要
        if "<!DOCTYPE" in raw or "<html" in raw:
            m = _re.search(r"<title[^>]*>([^<]+)</title>", raw, _re.IGNORECASE)
            title = m.group(1).strip() if m else "HTML error page (title not found)"
            # 如果存在，也获取 Cloudflare Ray ID
            ray = _re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", raw)
            ray_id = ray.group(1).strip() if ray else None
            status_code = getattr(error, "status_code", None)
            parts = []
            if status_code:
                parts.append(f"HTTP {status_code}")
            parts.append(title)
            if ray_id:
                parts.append(f"Ray {ray_id}")
            return " — ".join(parts)

        # 来自 OpenAI/Anthropic SDK 的 JSON 正文错误
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            msg = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else body.get("message")
            if msg:
                status_code = getattr(error, "status_code", None)
                prefix = f"HTTP {status_code}: " if status_code else ""
                return f"{prefix}{msg[:300]}"

        # 回退：截断原始字符串，但比 200 个字符给更多空间
        status_code = getattr(error, "status_code", None)
        prefix = f"HTTP {status_code}: " if status_code else ""
        return f"{prefix}{raw[:500]}"

    def _mask_api_key_for_logs(self, key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if len(key) <= 12:
            return "***"
        return f"{key[:8]}...{key[-4:]}"

    def _clean_error_message(self, error_msg: str) -> str:
        """
        清理错误消息以供用户显示，删除 HTML 内容并截断。
        
        参数:
            error_msg: 来自 API 或异常的原始错误消息
            
        返回:
            干净、用户友好的错误消息
        """
        if not error_msg:
            return "未知错误"
            
        # 删除 HTML 内容（CloudFlare 和 gateway 错误页面常见）
        if error_msg.strip().startswith('<!DOCTYPE html') or '<html' in error_msg:
            return "服务暂时不可用（返回了 HTML 错误页面）"
            
        # 删除换行符和过多的空白
        cleaned = ' '.join(error_msg.split())
        
        # 如果太长则截断
        if len(cleaned) > 150:
            cleaned = cleaned[:150] + "..."
            
        return cleaned

    @staticmethod
    def _extract_api_error_context(error: Exception) -> Dict[str, Any]:
        """从提供商错误中提取结构化的速率限制详情。"""
        context: Dict[str, Any] = {}

        body = getattr(error, "body", None)
        payload = None
        if isinstance(body, dict):
            payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            reason = payload.get("code") or payload.get("error")
            if isinstance(reason, str) and reason.strip():
                context["reason"] = reason.strip()
            message = payload.get("message") or payload.get("error_description")
            if isinstance(message, str) and message.strip():
                context["message"] = message.strip()
            for key in ("resets_at", "reset_at"):
                value = payload.get(key)
                if value not in (None, ""):
                    context["reset_at"] = value
                    break
            retry_after = payload.get("retry_after")
            if retry_after not in (None, "") and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass

        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
            if retry_after and "reset_at" not in context:
                try:
                    context["reset_at"] = time.time() + float(retry_after)
                except (TypeError, ValueError):
                    pass
            ratelimit_reset = headers.get("x-ratelimit-reset")
            if ratelimit_reset and "reset_at" not in context:
                context["reset_at"] = ratelimit_reset

        if "message" not in context:
            raw_message = str(error).strip()
            if raw_message:
                context["message"] = raw_message[:500]

        if "reset_at" not in context:
            message = context.get("message") or ""
            if isinstance(message, str):
                delay_match = re.search(r"quotaResetDelay[:\s\"]+(\\d+(?:\\.\\d+)?)(ms|s)", message, re.IGNORECASE)
                if delay_match:
                    value = float(delay_match.group(1))
                    seconds = value / 1000.0 if delay_match.group(2).lower() == "ms" else value
                    context["reset_at"] = time.time() + seconds
                else:
                    sec_match = re.search(
                        r"retry\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(?:sec|secs|seconds|s\b)",
                        message,
                        re.IGNORECASE,
                    )
                    if sec_match:
                        context["reset_at"] = time.time() + float(sec_match.group(1))

        return context

    def _usage_summary_for_api_request_hook(self, response: Any) -> Optional[Dict[str, Any]]:
        """用于 ``post_api_request`` 插件的 token 桶（无原始 ``response`` 对象）。"""
        if response is None:
            return None
        raw_usage = getattr(response, "usage", None)
        if not raw_usage:
            return None
        from dataclasses import asdict

        cu = normalize_usage(raw_usage, provider=self.provider, api_mode=self.api_mode)
        summary = asdict(cu)
        summary.pop("raw_usage", None)
        summary["prompt_tokens"] = cu.prompt_tokens
        summary["total_tokens"] = cu.total_tokens
        return summary

    def _dump_api_request_debug(
        self,
        api_kwargs: Dict[str, Any],
        *,
        reason: str,
        error: Optional[Exception] = None,
    ) -> Optional[Path]:
        """
        为活动推理 API 转储便于调试的 HTTP 请求记录。

        从 api_kwargs 捕获请求正文（排除仅用于传输的键，如 timeout）。
        旨在调试提供商端的 4xx 失败，其中重试没有用处。
        """
        try:
            body = copy.deepcopy(api_kwargs)
            body.pop("timeout", None)
            body = {k: v for k, v in body.items() if v is not None}

            api_key = None
            try:
                api_key = getattr(self.client, "api_key", None)
            except Exception as e:
                logger.debug("Could not extract API key for debug dump: %s", e)

            dump_payload: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "reason": reason,
                "request": {
                    "method": "POST",
                    "url": f"{self.base_url.rstrip('/')}{'/responses' if self.api_mode == 'codex_responses' else '/chat/completions'}",
                    "headers": {
                        "Authorization": f"Bearer {self._mask_api_key_for_logs(api_key)}",
                        "Content-Type": "application/json",
                    },
                    "body": body,
                },
            }

            if error is not None:
                error_info: Dict[str, Any] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                for attr_name in ("status_code", "request_id", "code", "param", "type"):
                    attr_value = getattr(error, attr_name, None)
                    if attr_value is not None:
                        error_info[attr_name] = attr_value

                body_attr = getattr(error, "body", None)
                if body_attr is not None:
                    error_info["body"] = body_attr

                response_obj = getattr(error, "response", None)
                if response_obj is not None:
                    try:
                        error_info["response_status"] = getattr(response_obj, "status_code", None)
                        error_info["response_text"] = response_obj.text
                    except Exception as e:
                        logger.debug("Could not extract error response details: %s", e)

                dump_payload["error"] = error_info

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dump_file = self.logs_dir / f"request_dump_{self.session_id}_{timestamp}.json"
            dump_file.write_text(
                json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            self._vprint(f"{self.log_prefix}🧾 Request debug dump written to: {dump_file}")

            if env_var_enabled("HERMES_DUMP_REQUEST_STDOUT"):
                print(json.dumps(dump_payload, ensure_ascii=False, indent=2, default=str))

            return dump_file
        except Exception as dump_error:
            if self.verbose_logging:
                logging.warning(f"Failed to dump API request debug payload: {dump_error}")
            return None

    @staticmethod
    def _clean_session_content(content: str) -> str:
        """将 REASONING_SCRATCHPAD 转换为 think 标签并清理空白。"""
        if not content:
            return content
        content = convert_scratchpad_to_think(content)
        content = re.sub(r'\n+(<think>)', r'\n\1', content)
        content = re.sub(r'(</think>)\n+', r'\1\n', content)
        return content.strip()

    def _save_session_log(self, messages: List[Dict[str, Any]] = None):
        """
        将完整的原始会话保存到 JSON 文件。

        存储代理看到的每条消息：用户消息、助手消息（带有推理、finish_reason、tool_calls）、
        工具响应（带有 tool_call_id、tool_name）和注入的系统消息（压缩摘要、待办事项快照等）。

        REASONING_SCRATCHPAD 标签被转换为 <think> 块以保持一致性。
        每轮后覆盖，因此它始终反映最新状态。
        """
        messages = messages or self._session_messages
        if not messages:
            return

        try:
            # 为会话日志清理助手内容
            cleaned = []
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    msg = dict(msg)
                    msg["content"] = self._clean_session_content(msg["content"])
                cleaned.append(msg)

            # 保护：永远不要用更少的消息覆盖更大的会话日志。
            # 这可以防止当 --resume 加载一个其消息未完全写入 SQLite 的会话时的数据丢失——
            # 恢复的代理以部分历史开始，否则会覆盖完整的 JSON 日志。
            if self.session_log_file.exists():
                try:
                    existing = json.loads(self.session_log_file.read_text(encoding="utf-8"))
                    existing_count = existing.get("message_count", len(existing.get("messages", [])))
                    if existing_count > len(cleaned):
                        logging.debug(
                            "跳过会话日志覆盖：现有有 %d 条消息，当前有 %d 条",
                            existing_count, len(cleaned),
                        )
                        return
                except Exception:
                    pass  # 现有文件已损坏——允许覆盖

            entry = {
                "session_id": self.session_id,
                "model": self.model,
                "base_url": self.base_url,
                "platform": self.platform,
                "session_start": self.session_start.isoformat(),
                "last_updated": datetime.now().isoformat(),
                "system_prompt": self._cached_system_prompt or "",
                "tools": self.tools or [],
                "message_count": len(cleaned),
                "messages": cleaned,
            }

            atomic_json_write(
                self.session_log_file,
                entry,
                indent=2,
                default=str,
            )

        except Exception as e:
            if self.verbose_logging:
                logging.warning(f"Failed to save session log: {e}")
    
    def interrupt(self, message: str = None) -> None:
        """
        请求代理中断其当前的工具调用循环。
        
        从另一个线程调用此方法（例如，输入处理器、消息接收器）以优雅地
        停止代理并处理新消息。
        
        还会向长时间运行的工具执行（例如终端命令）发送信号以提前终止，
        以便代理可以立即响应。
        
        参数:
            message: 触发中断的可选新消息。如果提供，代理将在其响应上下文中包含此消息。
        
        示例（CLI）：
            # 在单独的输入线程中：
            if user_typed_something:
                agent.interrupt(user_input)
        
        示例（消息传递）：
            # 当活动会话有新消息到达时：
            if session_has_running_agent:
                running_agent.interrupt(new_message.text)
        """
        self._interrupt_requested = True
        self._interrupt_message = message
        # 向所有工具发送信号以立即中止任何进行中的操作。
        # 将中断范围限定到此代理的执行线程，以便在同一进程（gateway）中运行的其他代理不受影响。
        _set_interrupt(True, self._execution_thread_id)
        # 将中断传播到任何正在运行的子代理（子代理委托）
        with self._active_children_lock:
            children_copy = list(self._active_children)
        for child in children_copy:
            try:
                child.interrupt(message)
            except Exception as e:
                logger.debug("无法将中断传播到子代理：%s", e)
        if not self.quiet_mode:
            print("\n⚡ 已请求中断" + (f": '{message[:40]}...'" if message and len(message) > 40 else f": '{message}'" if message else ""))
    
    def clear_interrupt(self) -> None:
        """清除任何待处理的中断请求和每线程工具中断信号。"""
        self._interrupt_requested = False
        self._interrupt_message = None
        _set_interrupt(False, self._execution_thread_id)

    def _touch_activity(self, desc: str) -> None:
        """更新最后活动时间戳和描述（线程安全）。"""
        self._last_activity_ts = time.time()
        self._last_activity_desc = desc

    def _capture_rate_limits(self, http_response: Any) -> None:
        """从 HTTP 响应中解析 x-ratelimit-* 头并缓存状态。

        在每个流式 API 调用后调用。httpx Response 对象可通过
        ``stream.response`` 在 OpenAI SDK Stream 上获得。
        """
        if http_response is None:
            return
        headers = getattr(http_response, "headers", None)
        if not headers:
            return
        try:
            from agent.rate_limit_tracker import parse_rate_limit_headers
            state = parse_rate_limit_headers(headers, provider=self.provider)
            if state is not None:
                self._rate_limit_state = state
        except Exception:
            pass  # Never let header parsing break the agent loop

    def get_rate_limit_state(self):
        """Return the last captured RateLimitState, or None."""
        return self._rate_limit_state

    def get_activity_summary(self) -> dict:
        """返回代理当前活动的快照以进行诊断。

        由 gateway 超时处理器调用，以报告代理被终止时正在做什么，
        以及用于周期性的"仍在工作"通知。
        """
        elapsed = time.time() - self._last_activity_ts
        return {
            "last_activity_ts": self._last_activity_ts,
            "last_activity_desc": self._last_activity_desc,
            "seconds_since_activity": round(elapsed, 1),
            "current_tool": self._current_tool,
            "api_call_count": self._api_call_count,
            "max_iterations": self.max_iterations,
            "budget_used": self.iteration_budget.used,
            "budget_max": self.iteration_budget.max_total,
        }

    def shutdown_memory_provider(self, messages: list = None) -> None:
        """关闭记忆提供程序和上下文引擎——在实际会话边界调用。

        这会在记忆管理器上调用 on_session_end() 然后调用 shutdown_all()，
        并在上下文引擎上调用 on_session_end()。
        不在每轮调用——仅在 CLI 退出、/reset、gateway 会话到期等时调用。
        """
        if self._memory_manager:
            try:
                self._memory_manager.on_session_end(messages or [])
            except Exception:
                pass
            try:
                self._memory_manager.shutdown_all()
            except Exception:
                pass
        # 通知上下文引擎会话结束（刷新 DAG、关闭数据库等）
        if hasattr(self, "context_compressor") and self.context_compressor:
            try:
                self.context_compressor.on_session_end(
                    self.session_id or "",
                    messages or [],
                )
            except Exception:
                pass
    
    def close(self) -> None:
        """释放此代理实例持有的所有资源。

        清理否则会成为孤儿的子进程资源：
        - 在 ProcessRegistry 中跟踪的后台进程
        - 终端沙箱环境
        - 浏览器守护程序会话
        - 活动子代理（子代理委托）
        - OpenAI/httpx 客户端连接

        多次调用是安全的（幂等）。每个清理步骤都是独立保护的，
        因此一个步骤的失败不会阻止其他步骤。
        """
        task_id = getattr(self, "session_id", None) or ""

        # 1. 终止此任务的后台进程
        try:
            from tools.process_registry import process_registry
            process_registry.kill_all(task_id=task_id)
        except Exception:
            pass

        # 2. 清理终端沙箱环境
        try:
            from tools.terminal_tool import cleanup_vm
            cleanup_vm(task_id)
        except Exception:
            pass

        # 3. 清理浏览器守护程序会话
        try:
            from tools.browser_tool import cleanup_browser
            cleanup_browser(task_id)
        except Exception:
            pass

        # 4. 关闭活动子代理
        try:
            with self._active_children_lock:
                children = list(self._active_children)
                self._active_children.clear()
            for child in children:
                try:
                    child.close()
                except Exception:
                    pass
        except Exception:
            pass

        # 5. 关闭 OpenAI/httpx 客户端
        try:
            client = getattr(self, "client", None)
            if client is not None:
                self._close_openai_client(client, reason="agent_close", shared=True)
                self.client = None
        except Exception:
            pass

    def _hydrate_todo_store(self, history: List[Dict[str, Any]]) -> None:
        """
        从对话历史恢复待办事项状态。
        
        Gateway 为每条消息创建一个新的 AIAgent，因此内存中的 TodoStore 是空的。
        我们扫描历史以查找最近的待办事项工具响应并重放它以重建状态。
        """
        # 向后遍历历史以查找最近的待办事项工具响应
        last_todo_response = None
        for msg in reversed(history):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            # 快速检查：待办事项响应包含 "todos" 键
            if '"todos"' not in content:
                continue
            try:
                data = json.loads(content)
                if "todos" in data and isinstance(data["todos"], list):
                    last_todo_response = data["todos"]
                    break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if last_todo_response:
            # 将项目重放到存储中（替换模式）
            self._todo_store.write(last_todo_response, merge=False)
            if not self.quiet_mode:
                self._vprint(f"{self.log_prefix}📋 从历史恢复了 {len(last_todo_response)} 个待办事项")
        _set_interrupt(False)
    
    @property
    def is_interrupted(self) -> bool:
        """检查是否已请求中断。"""
        return self._interrupt_requested










    def _build_system_prompt(self, system_message: str = None) -> str:
        """
        从所有层组装完整的系统提示词。
        
        每个会话调用一次（缓存在 self._cached_system_prompt 上），
        仅在上下文压缩事件后重建。这确保系统提示词在会话中的所有轮次间保持稳定，
        最大化前缀缓存命中。
        """
        # 层（按顺序）：
        #   1. 代理身份——SOUL.md（如果可用），否则为 DEFAULT_AGENT_IDENTITY
        #   2. 用户/gateway 系统提示词（如果提供）
        #   3. 持久记忆（冻结快照）
        #   4. 技能指导（如果加载了技能工具）
        #   5. 上下文文件（AGENTS.md、.cursorrules——SOUL.md 在此处排除，当用作身份时）
        #   6. 当前日期和时间（构建时冻结）
        #   7. 平台特定的格式提示

        # 尝试 SOUL.md 作为主要身份（除非跳过上下文文件）
        _soul_loaded = False
        if not self.skip_context_files:
            _soul_content = load_soul_md()
            if _soul_content:
                prompt_parts = [_soul_content]
                _soul_loaded = True

        if not _soul_loaded:
            # 回退到硬编码的身份
            prompt_parts = [DEFAULT_AGENT_IDENTITY]

        # 工具感知的行为指导：仅在加载工具时注入
        tool_guidance = []
        if "memory" in self.valid_tool_names:
            tool_guidance.append(MEMORY_GUIDANCE)
        if "session_search" in self.valid_tool_names:
            tool_guidance.append(SESSION_SEARCH_GUIDANCE)
        if "skill_manage" in self.valid_tool_names:
            tool_guidance.append(SKILLS_GUIDANCE)
        if tool_guidance:
            prompt_parts.append(" ".join(tool_guidance))

        nous_subscription_prompt = build_nous_subscription_prompt(self.valid_tool_names)
        if nous_subscription_prompt:
            prompt_parts.append(nous_subscription_prompt)
        # 工具使用强制：告诉模型实际调用工具而不是描述预期操作。由 config.yaml 控制
        # agent.tool_use_enforcement：
        #   "auto"（默认）——匹配 TOOL_USE_ENFORCEMENT_MODELS
        #   true ——始终注入（所有模型）
        #   false ——从不注入
        #   list  — custom model-name substrings to match
        if self.valid_tool_names:
            _enforce = self._tool_use_enforcement
            _inject = False
            if _enforce is True or (isinstance(_enforce, str) and _enforce.lower() in ("true", "always", "yes", "on")):
                _inject = True
            elif _enforce is False or (isinstance(_enforce, str) and _enforce.lower() in ("false", "never", "no", "off")):
                _inject = False
            elif isinstance(_enforce, list):
                model_lower = (self.model or "").lower()
                _inject = any(p.lower() in model_lower for p in _enforce if isinstance(p, str))
            else:
                # "auto" or any unrecognised value — use hardcoded defaults
                model_lower = (self.model or "").lower()
                _inject = any(p in model_lower for p in TOOL_USE_ENFORCEMENT_MODELS)
            if _inject:
                prompt_parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
                _model_lower = (self.model or "").lower()
                # Google model operational guidance (conciseness, absolute
                # paths, parallel tool calls, verify-before-edit, etc.)
                if "gemini" in _model_lower or "gemma" in _model_lower:
                    prompt_parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
                # OpenAI GPT/Codex execution discipline (tool persistence,
                # prerequisite checks, verification, anti-hallucination).
                if "gpt" in _model_lower or "codex" in _model_lower:
                    prompt_parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)

        # 以便它可以将用户引向它们，而不是重新发明答案。

        # 注意：ephemeral_system_prompt 不包含在这里。它仅在 API 调用时注入，
        # 因此它不会出现在缓存/存储的系统提示词中。
        if system_message is not None:
            prompt_parts.append(system_message)

        if self._memory_store:
            if self._memory_enabled:
                mem_block = self._memory_store.format_for_system_prompt("memory")
                if mem_block:
                    prompt_parts.append(mem_block)
            # USER.md 在启用时始终包含在内。
            if self._user_profile_enabled:
                user_block = self._memory_store.format_for_system_prompt("user")
                if user_block:
                    prompt_parts.append(user_block)

        # 外部记忆提供程序系统提示词块（对内置的补充）
        if self._memory_manager:
            try:
                _ext_mem_block = self._memory_manager.build_system_prompt()
                if _ext_mem_block:
                    prompt_parts.append(_ext_mem_block)
            except Exception:
                pass

        has_skills_tools = any(name in self.valid_tool_names for name in ['skills_list', 'skill_view', 'skill_manage'])
        if has_skills_tools:
            avail_toolsets = {
                toolset
                for toolset in (
                    get_toolset_for_tool(tool_name) for tool_name in self.valid_tool_names
                )
                if toolset
            }
            skills_prompt = build_skills_system_prompt(
                available_tools=self.valid_tool_names,
                available_toolsets=avail_toolsets,
            )
        else:
            skills_prompt = ""
        if skills_prompt:
            prompt_parts.append(skills_prompt)

        if not self.skip_context_files:
            # 设置时使用 TERMINAL_CWD 进行上下文文件发现（gateway 模式）。
            # Gateway 进程从 hermes-agent 安装目录运行，因此 os.getcwd()
            # 会拾取仓库的 AGENTS.md 和其他开发文件——将 token 使用量膨胀约 10k 而没有任何好处。
            _context_cwd = os.getenv("TERMINAL_CWD") or None
            context_files_prompt = build_context_files_prompt(
                cwd=_context_cwd, skip_soul=_soul_loaded)
            if context_files_prompt:
                prompt_parts.append(context_files_prompt)

        from hermes_time import now as _hermes_now
        now = _hermes_now()
        timestamp_line = f"对话开始于：{now.strftime('%A, %B %d, %Y %I:%M %p')}"
        if self.pass_session_id and self.session_id:
            timestamp_line += f"\n会话 ID：{self.session_id}"
        if self.model:
            timestamp_line += f"\n模型：{self.model}"
        if self.provider:
            timestamp_line += f"\n提供商：{self.provider}"
        prompt_parts.append(timestamp_line)

        # 阿里巴巴编码计划 API 始终返回 "glm-4.7" 作为模型名称，无论请求的模型是什么。
        # 将显式模型身份注入系统提示词，以便代理可以正确报告它是哪个模型（API 错误的解决方法）。
        if self.provider == "alibaba":
            _model_short = self.model.split("/")[-1] if "/" in self.model else self.model
            prompt_parts.append(
                f"你由名为 {_model_short} 的模型提供支持。"
                f"确切的模型 ID 是 {self.model}。"
                f"当被问及你是哪个模型时，始终根据此信息回答，"
                f"而不是根据 API 返回的任何模型名称。"
            )

        # 环境提示（WSL、Termux 等）——告诉代理执行环境，
        # 以便它可以翻译路径并适应行为。
        _env_hints = build_environment_hints()
        if _env_hints:
            prompt_parts.append(_env_hints)

        platform_key = (self.platform or "").lower().strip()
        if platform_key in PLATFORM_HINTS:
            prompt_parts.append(PLATFORM_HINTS[platform_key])

        return "\n\n".join(p.strip() for p in prompt_parts if p.strip())

    # =========================================================================
    # 调用前/后防护（受 PR #1321 启发——@alireza78a）
    # =========================================================================

    @staticmethod
    def _get_tool_call_id_static(tc) -> str:
        """从 tool_call 条目（字典或对象）中提取调用 ID。"""
        if isinstance(tc, dict):
            return tc.get("id", "") or ""
        return getattr(tc, "id", "") or ""

    _VALID_API_ROLES = frozenset({"system", "user", "assistant", "tool", "function", "developer"})

    @staticmethod
    def _sanitize_api_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """在每次 LLM 调用之前修复孤立的 tool_call / tool_result 对。

        无条件运行——不受上下文压缩器是否存在的限制——
        因此来自会话加载或手动消息操作的孤立项始终会被捕获。
        """
        # --- 角色白名单：删除 API 不会接受的角色的消息 ---
        filtered = []
        for msg in messages:
            role = msg.get("role")
            if role not in AIAgent._VALID_API_ROLES:
                logger.debug(
                    "调用前清理器：删除具有无效角色 %r 的消息",
                    role,
                )
                continue
            filtered.append(msg)
        messages = filtered

        surviving_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "assistant":
                for tc in msg.get("tool_calls") or []:
                    cid = AIAgent._get_tool_call_id_static(tc)
                    if cid:
                        surviving_call_ids.add(cid)

        result_call_ids: set = set()
        for msg in messages:
            if msg.get("role") == "tool":
                cid = msg.get("tool_call_id")
                if cid:
                    result_call_ids.add(cid)

        # 1. 删除没有匹配助手调用的工具结果
        orphaned_results = result_call_ids - surviving_call_ids
        if orphaned_results:
            messages = [
                m for m in messages
                if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
            ]
            logger.debug(
                "调用前清理器：删除了 %d 个孤立的工具结果",
                len(orphaned_results),
            )

        # 2. 为结果被删除的调用注入存根结果
        missing_results = surviving_call_ids - result_call_ids
        if missing_results:
            patched: List[Dict[str, Any]] = []
            for msg in messages:
                patched.append(msg)
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        cid = AIAgent._get_tool_call_id_static(tc)
                        if cid in missing_results:
                            patched.append({
                                "role": "tool",
                                "content": "[结果不可用——请参阅上面的上下文摘要]",
                                "tool_call_id": cid,
                            })
            messages = patched
            logger.debug(
                "调用前清理器：添加了 %d 个存根工具结果",
                len(missing_results),
            )
        return messages

    @staticmethod
    def _cap_delegate_task_calls(tool_calls: list) -> list:
        """截断多余的 delegate_task 调用到 max_concurrent_children。

        delegate_tool 在单个调用中限制任务列表，但模型可以在一个轮次中发出多个单独的
        delegate_task tool_calls。这会截断多余的，保留所有非委托调用。

        如果不需要截断，则返回原始列表。
        """
        from tools.delegate_tool import _get_max_concurrent_children
        max_children = _get_max_concurrent_children()
        delegate_count = sum(1 for tc in tool_calls if tc.function.name == "delegate_task")
        if delegate_count <= max_children:
            return tool_calls
        kept_delegates = 0
        truncated = []
        for tc in tool_calls:
            if tc.function.name == "delegate_task":
                if kept_delegates < max_children:
                    truncated.append(tc)
                    kept_delegates += 1
            else:
                truncated.append(tc)
        logger.warning(
            "截断了 %d 个多余的 delegate_task 调用以强制执行 max_concurrent_children=%d 限制",
            delegate_count - max_children, max_children,
        )
        return truncated

    @staticmethod
    def _deduplicate_tool_calls(tool_calls: list) -> list:
        """在单个轮次中删除重复的 (tool_name, arguments) 对。

        仅保留每个唯一对的第一次出现。
        如果未找到重复项，则返回原始列表。
        """
        seen: set = set()
        unique: list = []
        for tc in tool_calls:
            key = (tc.function.name, tc.function.arguments)
            if key not in seen:
                seen.add(key)
                unique.append(tc)
            else:
                logger.warning("删除了重复的工具调用：%s", tc.function.name)
        return unique if len(unique) < len(tool_calls) else tool_calls

    def _repair_tool_call(self, tool_name: str) -> str | None:
        """在中止之前尝试修复不匹配的工具名称。

        1. 尝试小写
        2. 尝试规范化（小写 + 连字符/空格 -> 下划线）
        3. 尝试模糊匹配（difflib，cutoff=0.7）

        如果在 valid_tool_names 中找到修复的名称，则返回，否则返回 None。
        """
        from difflib import get_close_matches

        # 1. 小写
        lowered = tool_name.lower()
        if lowered in self.valid_tool_names:
            return lowered

        # 2. 规范化
        normalized = lowered.replace("-", "_").replace(" ", "_")
        if normalized in self.valid_tool_names:
            return normalized

        # 3. 模糊匹配
        matches = get_close_matches(lowered, self.valid_tool_names, n=1, cutoff=0.7)
        if matches:
            return matches[0]

        return None

    def _invalidate_system_prompt(self):
        """
        使缓存的系统提示词无效，强制在下一轮次重建。
        
        在上下文压缩事件后调用。还会从磁盘重新加载记忆，
        以便重建的提示词捕获此会话中的任何写入。
        """
        self._cached_system_prompt = None
        if self._memory_store:
            self._memory_store.load_from_disk()

    def _responses_tools(self, tools: Optional[List[Dict[str, Any]]] = None) -> Optional[List[Dict[str, Any]]]:
        """将 chat-completions 工具模式转换为 Responses 函数工具模式。"""
        source_tools = tools if tools is not None else self.tools
        if not source_tools:
            return None

        converted: List[Dict[str, Any]] = []
        for item in source_tools:
            fn = item.get("function", {}) if isinstance(item, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            converted.append({
                "type": "function",
                "name": name,
                "description": fn.get("description", ""),
                "strict": False,
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return converted or None

    @staticmethod
    def _deterministic_call_id(fn_name: str, arguments: str, index: int = 0) -> str:
        """从工具调用内容生成确定性的 call_id。

        当 API 不提供 call_id 时用作回退。
        确定性 ID 防止缓存失效——随机 UUID 会使每个 API 调用的前缀唯一，
        破坏 OpenAI 的提示词缓存。
        """
        import hashlib
        seed = f"{fn_name}:{arguments}:{index}"
        digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"call_{digest}"

    @staticmethod
    def _split_responses_tool_id(raw_id: Any) -> tuple[Optional[str], Optional[str]]:
        """将存储的工具 id 拆分为 (call_id, response_item_id)。"""
        if not isinstance(raw_id, str):
            return None, None
        value = raw_id.strip()
        if not value:
            return None, None
        if "|" in value:
            call_id, response_item_id = value.split("|", 1)
            call_id = call_id.strip() or None
            response_item_id = response_item_id.strip() or None
            return call_id, response_item_id
        if value.startswith("fc_"):
            return None, value
        return value, None

    def _derive_responses_function_call_id(
        self,
        call_id: str,
        response_item_id: Optional[str] = None,
    ) -> str:
        """构建有效的 Responses `function_call.id`（必须以 `fc_` 开头）。"""
        if isinstance(response_item_id, str):
            candidate = response_item_id.strip()
            if candidate.startswith("fc_"):
                return candidate

        source = (call_id or "").strip()
        if source.startswith("fc_"):
            return source
        if source.startswith("call_") and len(source) > len("call_"):
            return f"fc_{source[len('call_'):]}"

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "", source)
        if sanitized.startswith("fc_"):
            return sanitized
        if sanitized.startswith("call_") and len(sanitized) > len("call_"):
            return f"fc_{sanitized[len('call_'):]}"
        if sanitized:
            return f"fc_{sanitized[:48]}"

        seed = source or str(response_item_id or "") or uuid.uuid4().hex
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]
        return f"fc_{digest}"

    def _chat_messages_to_responses_input(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将内部聊天样式的消息转换为 Responses 输入项。"""
        items: List[Dict[str, Any]] = []
        seen_item_ids: set = set()

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "system":
                continue

            if role in {"user", "assistant"}:
                content = msg.get("content", "")
                content_text = str(content) if content is not None else ""

                if role == "assistant":
                    # 重放来自前几轮的加密推理项，以便 API 可以保持连贯的推理链。
                    codex_reasoning = msg.get("codex_reasoning_items")
                    has_codex_reasoning = False
                    if isinstance(codex_reasoning, list):
                        for ri in codex_reasoning:
                            if isinstance(ri, dict) and ri.get("encrypted_content"):
                                item_id = ri.get("id")
                                if item_id and item_id in seen_item_ids:
                                    continue
                                items.append(ri)
                                if item_id:
                                    seen_item_ids.add(item_id)
                                has_codex_reasoning = True

                    if content_text.strip():
                        items.append({"role": "assistant", "content": content_text})
                    elif has_codex_reasoning:
                        # Responses API 在每个推理项后需要一个后续项（否则：missing_following_item 错误）。
                        # 当助手只产生推理而没有可见内容时，发出一个空的助手消息作为所需的后续项。
                        items.append({"role": "assistant", "content": ""})

                    tool_calls = msg.get("tool_calls")
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get("function", {})
                            fn_name = fn.get("name")
                            if not isinstance(fn_name, str) or not fn_name.strip():
                                continue

                            embedded_call_id, embedded_response_item_id = self._split_responses_tool_id(
                                tc.get("id")
                            )
                            call_id = tc.get("call_id")
                            if not isinstance(call_id, str) or not call_id.strip():
                                call_id = embedded_call_id
                            if not isinstance(call_id, str) or not call_id.strip():
                                if (
                                    isinstance(embedded_response_item_id, str)
                                    and embedded_response_item_id.startswith("fc_")
                                    and len(embedded_response_item_id) > len("fc_")
                                ):
                                    call_id = f"call_{embedded_response_item_id[len('fc_'):]}"
                                else:
                                    _raw_args = str(fn.get("arguments", "{}"))
                                    call_id = self._deterministic_call_id(fn_name, _raw_args, len(items))
                            call_id = call_id.strip()

                            arguments = fn.get("arguments", "{}")
                            if isinstance(arguments, dict):
                                arguments = json.dumps(arguments, ensure_ascii=False)
                            elif not isinstance(arguments, str):
                                arguments = str(arguments)
                            arguments = arguments.strip() or "{}"

                            items.append({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": fn_name,
                                "arguments": arguments,
                            })
                    continue

                items.append({"role": role, "content": content_text})
                continue

            if role == "tool":
                raw_tool_call_id = msg.get("tool_call_id")
                call_id, _ = self._split_responses_tool_id(raw_tool_call_id)
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_tool_call_id, str) and raw_tool_call_id.strip():
                        call_id = raw_tool_call_id.strip()
                if not isinstance(call_id, str) or not call_id.strip():
                    continue
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(msg.get("content", "") or ""),
                })

        return items

    def _preflight_codex_input_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_items, list):
            raise ValueError("Codex Responses input must be a list of input items.")

        normalized: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise ValueError(f"Codex Responses input[{idx}] must be an object.")

            item_type = item.get("type")
            if item_type == "function_call":
                call_id = item.get("call_id")
                name = item.get("name")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing call_id.")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call is missing name.")

                arguments = item.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                elif not isinstance(arguments, str):
                    arguments = str(arguments)
                arguments = arguments.strip() or "{}"

                normalized.append(
                    {
                        "type": "function_call",
                        "call_id": call_id.strip(),
                        "name": name.strip(),
                        "arguments": arguments,
                    }
                )
                continue

            if item_type == "function_call_output":
                call_id = item.get("call_id")
                if not isinstance(call_id, str) or not call_id.strip():
                    raise ValueError(f"Codex Responses input[{idx}] function_call_output is missing call_id.")
                output = item.get("output", "")
                if output is None:
                    output = ""
                if not isinstance(output, str):
                    output = str(output)

                normalized.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id.strip(),
                        "output": output,
                    }
                )
                continue

            if item_type == "reasoning":
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    item_id = item.get("id")
                    if isinstance(item_id, str) and item_id:
                        if item_id in seen_ids:
                            continue
                        seen_ids.add(item_id)
                    reasoning_item = {"type": "reasoning", "encrypted_content": encrypted}
                    if isinstance(item_id, str) and item_id:
                        reasoning_item["id"] = item_id
                    summary = item.get("summary")
                    if isinstance(summary, list):
                        reasoning_item["summary"] = summary
                    else:
                        reasoning_item["summary"] = []
                    normalized.append(reasoning_item)
                continue

            role = item.get("role")
            if role in {"user", "assistant"}:
                content = item.get("content", "")
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    content = str(content)

                normalized.append({"role": role, "content": content})
                continue

            raise ValueError(
                f"Codex Responses input[{idx}] has unsupported item shape (type={item_type!r}, role={role!r})."
            )

        return normalized

    def _preflight_codex_api_kwargs(
        self,
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(api_kwargs, dict):
            raise ValueError("Codex Responses request must be a dict.")

        required = {"model", "instructions", "input"}
        missing = [key for key in required if key not in api_kwargs]
        if missing:
            raise ValueError(f"Codex Responses request missing required field(s): {', '.join(sorted(missing))}.")

        model = api_kwargs.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Codex Responses request 'model' must be a non-empty string.")
        model = model.strip()

        instructions = api_kwargs.get("instructions")
        if instructions is None:
            instructions = ""
        if not isinstance(instructions, str):
            instructions = str(instructions)
        instructions = instructions.strip() or DEFAULT_AGENT_IDENTITY

        normalized_input = self._preflight_codex_input_items(api_kwargs.get("input"))

        tools = api_kwargs.get("tools")
        normalized_tools = None
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("Codex Responses request 'tools' must be a list when provided.")
            normalized_tools = []
            for idx, tool in enumerate(tools):
                if not isinstance(tool, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] must be an object.")
                if tool.get("type") != "function":
                    raise ValueError(f"Codex Responses tools[{idx}] has unsupported type {tool.get('type')!r}.")

                name = tool.get("name")
                parameters = tool.get("parameters")
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"Codex Responses tools[{idx}] is missing a valid name.")
                if not isinstance(parameters, dict):
                    raise ValueError(f"Codex Responses tools[{idx}] is missing valid parameters.")

                description = tool.get("description", "")
                if description is None:
                    description = ""
                if not isinstance(description, str):
                    description = str(description)

                strict = tool.get("strict", False)
                if not isinstance(strict, bool):
                    strict = bool(strict)

                normalized_tools.append(
                    {
                        "type": "function",
                        "name": name.strip(),
                        "description": description,
                        "strict": strict,
                        "parameters": parameters,
                    }
                )

        store = api_kwargs.get("store", False)
        if store is not False:
            raise ValueError("Codex Responses contract requires 'store' to be false.")

        allowed_keys = {
            "model", "instructions", "input", "tools", "store",
            "reasoning", "include", "max_output_tokens", "temperature",
            "tool_choice", "parallel_tool_calls", "prompt_cache_key", "service_tier",
        }
        normalized: Dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": normalized_input,
            "store": False,
        }
        if normalized_tools is not None:
            normalized["tools"] = normalized_tools

        # 透传推理配置
        reasoning = api_kwargs.get("reasoning")
        if isinstance(reasoning, dict):
            normalized["reasoning"] = reasoning
        include = api_kwargs.get("include")
        if isinstance(include, list):
            normalized["include"] = include
        service_tier = api_kwargs.get("service_tier")
        if isinstance(service_tier, str) and service_tier.strip():
            normalized["service_tier"] = service_tier.strip()

        # 透传 max_output_tokens 和 temperature
        max_output_tokens = api_kwargs.get("max_output_tokens")
        if isinstance(max_output_tokens, (int, float)) and max_output_tokens > 0:
            normalized["max_output_tokens"] = int(max_output_tokens)
        temperature = api_kwargs.get("temperature")
        if isinstance(temperature, (int, float)):
            normalized["temperature"] = float(temperature)

        # 透传 tool_choice、parallel_tool_calls、prompt_cache_key
        for passthrough_key in ("tool_choice", "parallel_tool_calls", "prompt_cache_key"):
            val = api_kwargs.get(passthrough_key)
            if val is not None:
                normalized[passthrough_key] = val

        if allow_stream:
            stream = api_kwargs.get("stream")
            if stream is not None and stream is not True:
                raise ValueError("Codex Responses 'stream' must be true when set.")
            if stream is True:
                normalized["stream"] = True
            allowed_keys.add("stream")
        elif "stream" in api_kwargs:
            raise ValueError("Codex Responses stream flag is only allowed in fallback streaming requests.")

        unexpected = sorted(key for key in api_kwargs if key not in allowed_keys)
        if unexpected:
            raise ValueError(
                f"Codex Responses request has unsupported field(s): {', '.join(unexpected)}."
            )

        return normalized

    def _extract_responses_message_text(self, item: Any) -> str:
        """从 Responses 消息输出项中提取助手文本。"""
        content = getattr(item, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: List[str] = []
        for part in content:
            ptype = getattr(part, "type", None)
            if ptype not in {"output_text", "text"}:
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks).strip()

    def _extract_responses_reasoning_text(self, item: Any) -> str:
        """从 Responses 推理项中提取紧凑的推理文本。"""
        summary = getattr(item, "summary", None)
        if isinstance(summary, list):
            chunks: List[str] = []
            for part in summary:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    chunks.append(text)
            if chunks:
                return "\n".join(chunks).strip()
        text = getattr(item, "text", None)
        if isinstance(text, str) and text:
            return text.strip()
        return ""

    def _normalize_codex_response(self, response: Any) -> tuple[Any, str]:
        """将 Responses API 对象规范化为类似 assistant_message 的对象。"""
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            # Codex 后端可能在答案完全通过流事件发送时返回空输出。
            # 在抛出异常前，检查 output_text 作为最后手段的回退。
            out_text = getattr(response, "output_text", None)
            if isinstance(out_text, str) and out_text.strip():
                logger.debug(
                    "Codex response has empty output but output_text is present (%d chars); "
                    "synthesizing output item.", len(out_text.strip()),
                )
                output = [SimpleNamespace(
                    type="message", role="assistant", status="completed",
                    content=[SimpleNamespace(type="output_text", text=out_text.strip())],
                )]
                response.output = output
            else:
                raise RuntimeError("Responses API returned no output items")

        response_status = getattr(response, "status", None)
        if isinstance(response_status, str):
            response_status = response_status.strip().lower()
        else:
            response_status = None

        if response_status in {"failed", "cancelled"}:
            error_obj = getattr(response, "error", None)
            if isinstance(error_obj, dict):
                error_msg = error_obj.get("message") or str(error_obj)
            else:
                error_msg = str(error_obj) if error_obj else f"Responses API returned status '{response_status}'"
            raise RuntimeError(error_msg)

        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        reasoning_items_raw: List[Dict[str, Any]] = []
        tool_calls: List[Any] = []
        has_incomplete_items = response_status in {"queued", "in_progress", "incomplete"}
        saw_commentary_phase = False
        saw_final_answer_phase = False

        for item in output:
            item_type = getattr(item, "type", None)
            item_status = getattr(item, "status", None)
            if isinstance(item_status, str):
                item_status = item_status.strip().lower()
            else:
                item_status = None

            if item_status in {"queued", "in_progress", "incomplete"}:
                has_incomplete_items = True

            if item_type == "message":
                item_phase = getattr(item, "phase", None)
                if isinstance(item_phase, str):
                    normalized_phase = item_phase.strip().lower()
                    if normalized_phase in {"commentary", "analysis"}:
                        saw_commentary_phase = True
                    elif normalized_phase in {"final_answer", "final"}:
                        saw_final_answer_phase = True
                message_text = self._extract_responses_message_text(item)
                if message_text:
                    content_parts.append(message_text)
            elif item_type == "reasoning":
                reasoning_text = self._extract_responses_reasoning_text(item)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                # 捕获完整推理项用于多轮连续性。
                # encrypted_content 是 API 在后续轮次需要的
                # 不透明数据块，用于维持连贯的推理链。
                encrypted = getattr(item, "encrypted_content", None)
                if isinstance(encrypted, str) and encrypted:
                    raw_item = {"type": "reasoning", "encrypted_content": encrypted}
                    item_id = getattr(item, "id", None)
                    if isinstance(item_id, str) and item_id:
                        raw_item["id"] = item_id
                    # 捕获摘要 — API 在回放推理项时需要
                    summary = getattr(item, "summary", None)
                    if isinstance(summary, list):
                        raw_summary = []
                        for part in summary:
                            text = getattr(part, "text", None)
                            if isinstance(text, str):
                                raw_summary.append({"type": "summary_text", "text": text})
                        raw_item["summary"] = raw_summary
                    reasoning_items_raw.append(raw_item)
            elif item_type == "function_call":
                if item_status in {"queued", "in_progress", "incomplete"}:
                    continue
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))
            elif item_type == "custom_tool_call":
                fn_name = getattr(item, "name", "") or ""
                arguments = getattr(item, "input", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                raw_call_id = getattr(item, "call_id", None)
                raw_item_id = getattr(item, "id", None)
                embedded_call_id, _ = self._split_responses_tool_id(raw_item_id)
                call_id = raw_call_id if isinstance(raw_call_id, str) and raw_call_id.strip() else embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    call_id = self._deterministic_call_id(fn_name, arguments, len(tool_calls))
                call_id = call_id.strip()
                response_item_id = raw_item_id if isinstance(raw_item_id, str) else None
                response_item_id = self._derive_responses_function_call_id(call_id, response_item_id)
                tool_calls.append(SimpleNamespace(
                    id=call_id,
                    call_id=call_id,
                    response_item_id=response_item_id,
                    type="function",
                    function=SimpleNamespace(name=fn_name, arguments=arguments),
                ))

        final_text = "\n".join([p for p in content_parts if p]).strip()
        if not final_text and hasattr(response, "output_text"):
            out_text = getattr(response, "output_text", "")
            if isinstance(out_text, str):
                final_text = out_text.strip()

        assistant_message = SimpleNamespace(
            content=final_text,
            tool_calls=tool_calls,
            reasoning="\n\n".join(reasoning_parts).strip() if reasoning_parts else None,
            reasoning_content=None,
            reasoning_details=None,
            codex_reasoning_items=reasoning_items_raw or None,
        )

        if tool_calls:
            finish_reason = "tool_calls"
        elif has_incomplete_items or (saw_commentary_phase and not saw_final_answer_phase):
            finish_reason = "incomplete"
        elif reasoning_items_raw and not final_text:
            # 响应仅包含推理（加密思考状态），没有可见内容或工具调用。
            # 模型仍在思考，需要再一轮才能产生实际答案。
            # 将其标记为 "stop" 会进入空内容重试循环（消耗3次重试后失败），
            # 改为标记为 incomplete，让 Codex 续接路径正确处理。
            finish_reason = "incomplete"
        else:
            finish_reason = "stop"
        return assistant_message, finish_reason

    def _thread_identity(self) -> str:
        thread = threading.current_thread()
        return f"{thread.name}:{thread.ident}"

    def _client_log_context(self) -> str:
        provider = getattr(self, "provider", "unknown")
        base_url = getattr(self, "base_url", "unknown")
        model = getattr(self, "model", "unknown")
        return (
            f"thread={self._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    def _openai_client_lock(self) -> threading.RLock:
        lock = getattr(self, "_client_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._client_lock = lock
        return lock

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """检查 OpenAI 客户端是否已关闭。

        处理 is_closed 的属性和方法两种形式：
        - httpx.Client.is_closed 是布尔属性
        - openai.OpenAI.is_closed 是返回布尔值的方法

        之前的 bug：getattr(client, "is_closed", False) 返回绑定的方法，
        方法对象始终为真值，导致每次调用时都不必要地重建客户端。
        """
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            # 处理方法（openai SDK）vs 属性（httpx）
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    def _create_openai_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        if self.provider == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient

            client = CopilotACPClient(**client_kwargs)
            logger.info(
                "Copilot ACP client created (%s, shared=%s) %s",
                reason,
                shared,
                self._client_log_context(),
            )
            return client
        client = OpenAI(**client_kwargs)
        logger.info(
            "OpenAI client created (%s, shared=%s) %s",
            reason,
            shared,
            self._client_log_context(),
        )
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """强制关闭底层 TCP 套接字以防止 CLOSE-WAIT 累积。

        当提供商在流传输中途断开连接时，httpx 的 ``client.close()``
        执行优雅关闭，这会使套接字保持在 CLOSE-WAIT 状态直到
        操作系统超时（通常数分钟）。此方法遍历 httpx 传输池，
        发出 ``socket.shutdown(SHUT_RDWR)`` + ``socket.close()``
        强制立即 TCP RST，释放文件描述符。

        返回强制关闭的套接字数量。
        """
        import socket as _socket

        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return 0
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return 0
            # httpx 使用 httpcore 连接池；连接存储在
            # _connections（列表）或 _pool（列表）中，取决于版本。
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            for conn in list(connections):
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                try:
                    sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def _close_openai_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None:
            return
        # 先强制关闭 TCP 套接字以防止 CLOSE-WAIT 累积，
        # 然后执行 SDK 级别的优雅关闭。
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info(
                "OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s",
                reason,
                shared,
                force_closed,
                self._client_log_context(),
            )
        except Exception as exc:
            logger.debug(
                "OpenAI client close failed (%s, shared=%s) %s error=%s",
                reason,
                shared,
                self._client_log_context(),
                exc,
            )

    def _replace_primary_openai_client(self, *, reason: str) -> bool:
        with self._openai_client_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self._create_openai_client(self._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning(
                    "Failed to rebuild shared OpenAI client (%s) %s error=%s",
                    reason,
                    self._client_log_context(),
                    exc,
                )
                return False
            self.client = new_client
        self._close_openai_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def _ensure_primary_openai_client(self, *, reason: str) -> Any:
        with self._openai_client_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning(
            "Detected closed shared OpenAI client; recreating before use (%s) %s",
            reason,
            self._client_log_context(),
        )
        if not self._replace_primary_openai_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self._openai_client_lock():
            return self.client

    def _cleanup_dead_connections(self) -> bool:
        """检测并清理主客户端上的死亡 TCP 连接。

        检查 httpx 连接池中处于不健康状态的套接字
        （CLOSE-WAIT、错误）。如果发现任何此类连接，
        强制关闭所有套接字并从头重建主客户端。

        如果发现并清理了死亡连接则返回 True。
        """
        client = getattr(self, "client", None)
        if client is None:
            return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None:
                return False
            transport = getattr(http_client, "_transport", None)
            if transport is None:
                return False
            pool = getattr(transport, "_pool", None)
            if pool is None:
                return False
            connections = (
                getattr(pool, "_connections", None)
                or getattr(pool, "_pool", None)
                or []
            )
            dead_count = 0
            for conn in list(connections):
                # 检查空闲但套接字已关闭的连接
                stream = (
                    getattr(conn, "_network_stream", None)
                    or getattr(conn, "_stream", None)
                )
                if stream is None:
                    continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None:
                    continue
                # 通过非阻塞 recv 探测套接字健康状况
                import socket as _socket
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, _socket.MSG_PEEK | _socket.MSG_DONTWAIT)
                    if data == b"":
                        dead_count += 1
                except BlockingIOError:
                    pass  # 无数据可用 — 套接字健康
                except OSError:
                    dead_count += 1
                finally:
                    try:
                        sock.setblocking(True)
                    except OSError:
                        pass
            if dead_count > 0:
                logger.warning(
                    "Found %d dead connection(s) in client pool — rebuilding client",
                    dead_count,
                )
                self._replace_primary_openai_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def _create_request_openai_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock

        primary_client = self._ensure_primary_openai_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self._openai_client_lock():
            request_kwargs = dict(self._client_kwargs)
        return self._create_openai_client(request_kwargs, reason=reason, shared=False)

    def _close_request_openai_client(self, client: Any, *, reason: str) -> None:
        self._close_openai_client(client, reason=reason, shared=False)

    def _run_codex_stream(self, api_kwargs: dict, client: Any = None, on_first_delta: callable = None):
        """执行一个流式 Responses API 请求并返回最终响应。"""
        import httpx as _httpx

        active_client = client or self._ensure_primary_openai_client(reason="codex_stream_direct")
        max_stream_retries = 1
        has_tool_calls = False
        first_delta_fired = False
        # 累积流式文本，以便在 get_final_response() 返回空输出时
        # 进行恢复（例如 chatgpt.com backend-api 发送
        # response.incomplete 而非 response.completed）。
        self._codex_streamed_text_parts: list = []
        for attempt in range(max_stream_retries + 1):
            collected_output_items: list = []
            try:
                with active_client.responses.stream(**api_kwargs) as stream:
                    for event in stream:
                        if self._interrupt_requested:
                            break
                        event_type = getattr(event, "type", "")
                        # 在文本内容增量时触发回调（工具调用期间抑制）
                        if "output_text.delta" in event_type or event_type == "response.output_text.delta":
                            delta_text = getattr(event, "delta", "")
                            if delta_text:
                                self._codex_streamed_text_parts.append(delta_text)
                            if delta_text and not has_tool_calls:
                                if not first_delta_fired:
                                    first_delta_fired = True
                                    if on_first_delta:
                                        try:
                                            on_first_delta()
                                        except Exception:
                                            pass
                                self._fire_stream_delta(delta_text)
                        # 追踪工具调用以抑制文本流式传输
                        elif "function_call" in event_type:
                            has_tool_calls = True
                        # 触发推理回调
                        elif "reasoning" in event_type and "delta" in event_type:
                            reasoning_text = getattr(event, "delta", "")
                            if reasoning_text:
                                self._fire_reasoning_delta(reasoning_text)
                        # 收集已完成的输出项 — 某些后端
                        # (chatgpt.com/backend-api/codex) 通过
                        # response.output_item.done 流式传输有效项，
                        # 但 SDK 的 get_final_response() 返回空输出列表。
                        elif event_type == "response.output_item.done":
                            done_item = getattr(event, "item", None)
                            if done_item is not None:
                                collected_output_items.append(done_item)
                        # 记录未完成的终止事件用于诊断
                        elif event_type in ("response.incomplete", "response.failed"):
                            resp_obj = getattr(event, "response", None)
                            status = getattr(resp_obj, "status", None) if resp_obj else None
                            incomplete_details = getattr(resp_obj, "incomplete_details", None) if resp_obj else None
                            logger.warning(
                                "Codex Responses stream received terminal event %s "
                                "(status=%s, incomplete_details=%s, streamed_chars=%d). %s",
                                event_type, status, incomplete_details,
                                sum(len(p) for p in self._codex_streamed_text_parts),
                                self._client_log_context(),
                            )
                    final_response = stream.get_final_response()
                    # 补丁：ChatGPT Codex 后端流式传输有效的输出项，
                    # 但 get_final_response() 可能返回空输出列表。
                    # 从收集的项或合成的增量中回填。
                    _out = getattr(final_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            final_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex stream: backfilled %d output items from stream events",
                                len(collected_output_items),
                            )
                        elif self._codex_streamed_text_parts and not has_tool_calls:
                            assembled = "".join(self._codex_streamed_text_parts)
                            final_response.output = [SimpleNamespace(
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex stream: synthesized output from %d text deltas (%d chars)",
                                len(self._codex_streamed_text_parts), len(assembled),
                            )
                    return final_response
            except (_httpx.RemoteProtocolError, _httpx.ReadTimeout, _httpx.ConnectError, ConnectionError) as exc:
                if attempt < max_stream_retries:
                    logger.debug(
                        "Codex Responses stream transport failed (attempt %s/%s); retrying. %s error=%s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                        exc,
                    )
                    continue
                logger.debug(
                    "Codex Responses stream transport failed; falling back to create(stream=True). %s error=%s",
                    self._client_log_context(),
                    exc,
                )
                return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
            except RuntimeError as exc:
                err_text = str(exc)
                missing_completed = "response.completed" in err_text
                if missing_completed and attempt < max_stream_retries:
                    logger.debug(
                        "Responses stream closed before completion (attempt %s/%s); retrying. %s",
                        attempt + 1,
                        max_stream_retries + 1,
                        self._client_log_context(),
                    )
                    continue
                if missing_completed:
                    logger.debug(
                        "Responses stream did not emit response.completed; falling back to create(stream=True). %s",
                        self._client_log_context(),
                    )
                    return self._run_codex_create_stream_fallback(api_kwargs, client=active_client)
                raise

    def _run_codex_create_stream_fallback(self, api_kwargs: dict, client: Any = None):
        """Codex 风格 Responses 后端流式完成边缘情况的回退路径。"""
        active_client = client or self._ensure_primary_openai_client(reason="codex_create_stream_fallback")
        fallback_kwargs = dict(api_kwargs)
        fallback_kwargs["stream"] = True
        fallback_kwargs = self._preflight_codex_api_kwargs(fallback_kwargs, allow_stream=True)
        stream_or_response = active_client.responses.create(**fallback_kwargs)

        # 兼容性垫片，用于仍返回具体响应的模拟或提供商。
        if hasattr(stream_or_response, "output"):
            return stream_or_response
        if not hasattr(stream_or_response, "__iter__"):
            return stream_or_response

        terminal_response = None
        collected_output_items: list = []
        collected_text_deltas: list = []
        try:
            for event in stream_or_response:
                event_type = getattr(event, "type", None)
                if not event_type and isinstance(event, dict):
                    event_type = event.get("type")

                # 收集输出项和文本增量用于回填
                if event_type == "response.output_item.done":
                    done_item = getattr(event, "item", None)
                    if done_item is None and isinstance(event, dict):
                        done_item = event.get("item")
                    if done_item is not None:
                        collected_output_items.append(done_item)
                elif event_type in ("response.output_text.delta",):
                    delta = getattr(event, "delta", "")
                    if not delta and isinstance(event, dict):
                        delta = event.get("delta", "")
                    if delta:
                        collected_text_deltas.append(delta)

                if event_type not in {"response.completed", "response.incomplete", "response.failed"}:
                    continue

                terminal_response = getattr(event, "response", None)
                if terminal_response is None and isinstance(event, dict):
                    terminal_response = event.get("response")
                if terminal_response is not None:
                    # 从收集的流事件中回填空输出
                    _out = getattr(terminal_response, "output", None)
                    if isinstance(_out, list) and not _out:
                        if collected_output_items:
                            terminal_response.output = list(collected_output_items)
                            logger.debug(
                                "Codex fallback stream: backfilled %d output items",
                                len(collected_output_items),
                            )
                        elif collected_text_deltas:
                            assembled = "".join(collected_text_deltas)
                            terminal_response.output = [SimpleNamespace(
                                type="message", role="assistant",
                                status="completed",
                                content=[SimpleNamespace(type="output_text", text=assembled)],
                            )]
                            logger.debug(
                                "Codex fallback stream: synthesized from %d deltas (%d chars)",
                                len(collected_text_deltas), len(assembled),
                            )
                    return terminal_response
        finally:
            close_fn = getattr(stream_or_response, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass

        if terminal_response is not None:
            return terminal_response
        raise RuntimeError("Responses create(stream=True) fallback did not emit a terminal response.")

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider != "openai-codex":
            return False

        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="codex_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "chat_completions" or self.provider != "nous":
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.getenv("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.getenv("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous 请求不应继承 OpenRouter 专用的归属头。
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # 仅在原生 Anthropic 提供商时刷新凭证。
        # 其他 anthropic_messages 提供商（MiniMax、Alibaba 等）使用自己的密钥。
        if self.provider != "anthropic":
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(new_token, getattr(self, "_anthropic_base_url", None))
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # 更新 OAuth 标志 — 令牌类型可能已更改（API 密钥 ↔ OAuth）
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def _apply_client_headers_for_base_url(self, base_url: str) -> None:
        from agent.auxiliary_client import _OR_HEADERS

        normalized = (base_url or "").lower()
        if "openrouter" in normalized:
            self._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif "api.githubcopilot.com" in normalized:
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif "api.kimi.com" in normalized:
            self._client_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.30.0"}
        elif "portal.qwen.ai" in normalized:
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        else:
            self._client_kwargs.pop("default_headers", None)

    def _swap_credential(self, entry) -> None:
        runtime_key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
        runtime_base = getattr(entry, "runtime_base_url", None) or getattr(entry, "base_url", None) or self.base_url

        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_client, _is_oauth_token

            try:
                self._anthropic_client.close()
            except Exception:
                pass

            self._anthropic_api_key = runtime_key
            self._anthropic_base_url = runtime_base
            self._anthropic_client = build_anthropic_client(runtime_key, runtime_base)
            self._is_anthropic_oauth = _is_oauth_token(runtime_key)
            self.api_key = runtime_key
            self.base_url = runtime_base
            return

        self.api_key = runtime_key
        self.base_url = runtime_base.rstrip("/") if isinstance(runtime_base, str) else runtime_base
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(self.base_url)
        self._replace_primary_openai_client(reason="credential_rotation")

    def _recover_with_credential_pool(
        self,
        *,
        status_code: Optional[int],
        has_retried_429: bool,
        classified_reason: Optional[FailoverReason] = None,
        error_context: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, bool]:
        """通过池轮换尝试凭证恢复。

        返回 (recovered, has_retried_429)。
        速率限制时：首次重试相同凭证（设置标志为 True）。
                       第二次连续失败则轮换到下一凭证。
        账单耗尽时：立即轮换。
        认证失败时：尝试刷新令牌后再轮换。

        `classified_reason` 让恢复路径遵循结构化错误
        分类器，而非仅依赖原始 HTTP 状态码。这对那些
        将账单/速率限制/认证条件以不同状态码返回的提供商很重要，
        例如 Anthropic 对"超出额外使用量"返回 HTTP 400。
        """
        pool = self._credential_pool
        if pool is None:
            return False, has_retried_429

        effective_reason = classified_reason
        if effective_reason is None:
            if status_code == 402:
                effective_reason = FailoverReason.billing
            elif status_code == 429:
                effective_reason = FailoverReason.rate_limit
            elif status_code == 401:
                effective_reason = FailoverReason.auth

        if effective_reason == FailoverReason.billing:
            rotate_status = status_code if status_code is not None else 402
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (billing) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, has_retried_429

        if effective_reason == FailoverReason.rate_limit:
            if not has_retried_429:
                return False, True
            rotate_status = status_code if status_code is not None else 429
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (rate limit) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False
            return False, True

        if effective_reason == FailoverReason.auth:
            refreshed = pool.try_refresh_current()
            if refreshed is not None:
                logger.info(f"Credential auth failure — refreshed pool entry {getattr(refreshed, 'id', '?')}")
                self._swap_credential(refreshed)
                return True, has_retried_429
            # 刷新失败 — 轮换到下一凭证而非放弃。
            # 失败条目标记为耗尽由 try_refresh_current() 处理。
            rotate_status = status_code if status_code is not None else 401
            next_entry = pool.mark_exhausted_and_rotate(status_code=rotate_status, error_context=error_context)
            if next_entry is not None:
                logger.info(
                    "Credential %s (auth refresh failed) — rotated to pool entry %s",
                    rotate_status,
                    getattr(next_entry, "id", "?"),
                )
                self._swap_credential(next_entry)
                return True, False

        return False, has_retried_429

    def _anthropic_messages_create(self, api_kwargs: dict):
        if self.api_mode == "anthropic_messages":
            self._try_refresh_anthropic_client_credentials()
        return self._anthropic_client.messages.create(**api_kwargs)

    def _interruptible_api_call(self, api_kwargs: dict):
        """
        在后台线程中运行 API 调用，使主对话循环
        能够在不等待完整 HTTP 往返的情况下检测中断。

        每个工作线程获得自己的 OpenAI 客户端实例。中断仅
        关闭该线程本地的客户端，因此重试和其他请求永远不会
        继承已关闭的传输。
        """
        result = {"response": None, "error": None}
        request_client_holder = {"client": None}

        def _call():
            try:
                if self.api_mode == "codex_responses":
                    request_client_holder["client"] = self._create_request_openai_client(reason="codex_stream_request")
                    result["response"] = self._run_codex_stream(
                        api_kwargs,
                        client=request_client_holder["client"],
                        on_first_delta=getattr(self, "_codex_on_first_delta", None),
                    )
                elif self.api_mode == "anthropic_messages":
                    result["response"] = self._anthropic_messages_create(api_kwargs)
                else:
                    request_client_holder["client"] = self._create_request_openai_client(reason="chat_completion_request")
                    result["response"] = request_client_holder["client"].chat.completions.create(**api_kwargs)
            except Exception as e:
                result["error"] = e
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="request_complete")

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)
            if self._interrupt_requested:
                # 强制关闭进行中的工作线程本地 HTTP 连接以停止
                # 令牌生成，同时不污染用于后续重试的共享客户端。
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during API call")
        if result["error"] is not None:
            raise result["error"]
        return result["response"]

    # ── Unified streaming API call ─────────────────────────────────────────

    def _reset_stream_delivery_tracking(self) -> None:
        """充值跟踪当前模型响应期间发送的文本。"""
        self._current_streamed_assistant_text = ""

    def _record_streamed_assistant_text(self, text: str) -> None:
        """累积通过流回调发出的可见助手文本。"""
        if isinstance(text, str) and text:
            self._current_streamed_assistant_text = (
                getattr(self, "_current_streamed_assistant_text", "") + text
            )

    @staticmethod
    def _normalize_interim_visible_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _interim_content_was_streamed(self, content: str) -> bool:
        visible_content = self._normalize_interim_visible_text(
            self._strip_think_blocks(content or "")
        )
        if not visible_content:
            return False
        streamed = self._normalize_interim_visible_text(
            self._strip_think_blocks(getattr(self, "_current_streamed_assistant_text", "") or "")
        )
        return bool(streamed) and streamed == visible_content

    def _emit_interim_assistant_message(self, assistant_msg: Dict[str, Any]) -> None:
        """向 UI 层推送真实的轮次中助手评论消息。"""
        cb = getattr(self, "interim_assistant_callback", None)
        if cb is None or not isinstance(assistant_msg, dict):
            return
        content = assistant_msg.get("content")
        visible = self._strip_think_blocks(content or "").strip()
        if not visible or visible == "(empty)":
            return
        already_streamed = self._interim_content_was_streamed(visible)
        try:
            cb(visible, already_streamed=already_streamed)
        except Exception:
            logger.debug("interim_assistant_callback error", exc_info=True)

    def _fire_stream_delta(self, text: str) -> None:
        """触发所有已注册的流增量回调（显示 + TTS）。"""
        # 如果工具迭代设置了断点标记，在第一个真实文本增量前
        # 加上单个段落分隔。这样可以避免原始问题（跨工具边界的
        # 文本拼接），而不会在多个工具迭代连续运行时堆积空行。
        if getattr(self, "_stream_needs_break", False) and text and text.strip():
            self._stream_needs_break = False
            text = "\n\n" + text
        callbacks = [cb for cb in (self.stream_delta_callback, self._stream_callback) if cb is not None]
        delivered = False
        for cb in callbacks:
            try:
                cb(text)
                delivered = True
            except Exception:
                pass
        if delivered:
            self._record_streamed_assistant_text(text)

    def _fire_reasoning_delta(self, text: str) -> None:
        """Fire reasoning callback if registered."""
        cb = self.reasoning_callback
        if cb is not None:
            try:
                cb(text)
            except Exception:
                pass

    def _fire_tool_gen_started(self, tool_name: str) -> None:
        """通知显示层模型正在生成工具调用参数。

        当流式响应开始生成 tool_call / tool_use 令牌时，
        每个工具名称触发一次。让 TUI 有机会显示旋转器
        或状态行，这样用户在大型工具载荷（例如 45 KB 的 write_file）
        生成时不会盯着冻结的屏幕。
        """
        cb = self.tool_gen_callback
        if cb is not None:
            try:
                cb(tool_name)
            except Exception:
                pass

    def _has_stream_consumers(self) -> bool:
        """如果有流消费者注册则返回 True。"""
        return (
            self.stream_delta_callback is not None
            or getattr(self, "_stream_callback", None) is not None
        )

    def _interruptible_streaming_api_call(
        self, api_kwargs: dict, *, on_first_delta: callable = None
    ):
        """_interruptible_api_call 的流式变体，用于实时令牌传递。

        处理所有三种 api_modes：
        - chat_completions：在 OpenAI 兼容端点上 stream=True
        - anthropic_messages：通过 Anthropic SDK 的 client.messages.stream()
        - codex_responses：委托给 _run_codex_stream（已在流式传输）

        为每个文本令牌触发 stream_delta_callback 和 _stream_callback。
        工具调用轮次会抑制回调 — 只有纯文本的最终响应
        才流向消费者。返回一个 SimpleNamespace，模拟
        非流式响应形状，使代理循环其余部分保持不变。

        当提供商错误表明不支持流式传输时，
        回退到 _interruptible_api_call。
        """
        if self.api_mode == "codex_responses":
            # Codex 通过 _run_codex_stream 在内部流式传输。_interruptible_api_call
            # 中的主分发已经调用它；我们只需要确保 on_first_delta 能触达它。
            # 暂时存储在实例上，让 _run_codex_stream 来获取。
            self._codex_on_first_delta = on_first_delta
            try:
                return self._interruptible_api_call(api_kwargs)
            finally:
                self._codex_on_first_delta = None

        result = {"response": None, "error": None}
        request_client_holder = {"client": None}
        first_delta_fired = {"done": False}
        deltas_were_sent = {"yes": False}  # 追踪是否有增量被触发（用于回退）
        # 最后一个真实流式数据块的墙上时钟时间戳。外部
        # 轮询循环用它来检测保持接收 SSE keep-alive ping
        # 但没有实际数据的过时连接。
        last_chunk_time = {"t": time.time()}

        def _fire_first_delta():
            if not first_delta_fired["done"] and on_first_delta:
                first_delta_fired["done"] = True
                try:
                    on_first_delta()
                except Exception:
                    pass

        def _call_chat_completions():
            """流式传输 chat completions 响应。"""
            import httpx as _httpx
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            # 本地提供商（Ollama、llama.cpp、vLLM）在大型上下文上
            # 进行预填充时可能需要数分钟才能产生第一个令牌。
            # 自动增加 httpx 读取超时，除非用户明确
            # 覆盖了 HERMES_STREAM_READ_TIMEOUT。
            if _stream_read_timeout == 120.0 and self.base_url and is_local_endpoint(self.base_url):
                _stream_read_timeout = _base_timeout
                logger.debug(
                    "Local provider detected (%s) — stream read timeout raised to %.0fs",
                    self.base_url, _stream_read_timeout,
                )
            stream_kwargs = {
                **api_kwargs,
                "stream": True,
                "stream_options": {"include_usage": True},
                "timeout": _httpx.Timeout(
                    connect=30.0,
                    read=_stream_read_timeout,
                    write=_base_timeout,
                    pool=30.0,
                ),
            }
            request_client_holder["client"] = self._create_request_openai_client(
                reason="chat_completion_stream_request"
            )
            # 重置过时流定时器，使检测器从这次
            # 尝试的开始时间测量，而非上一次尝试的最后一块时间。
            last_chunk_time["t"] = time.time()
            self._touch_activity("waiting for provider response (streaming)")
            stream = request_client_holder["client"].chat.completions.create(**stream_kwargs)

            # 从初始 HTTP 响应中捕获速率限制头。
            # OpenAI SDK Stream 对象通过 .response 在任何
            # 数据块被消费之前公开底层 httpx 响应。
            self._capture_rate_limits(getattr(stream, "response", None))

            content_parts: list = []
            tool_calls_acc: dict = {}
            tool_gen_notified: set = set()
            # Ollama 兼容端点在并行批次中为每个工具调用重用索引 0，
            # 仅通过 id 来区分。追踪每个原始索引最后看到的 id，
            # 以便检测在同一索引开始的新工具调用并将其重定向到新的槽位。
            _last_id_at_idx: dict = {}      # raw_index -> 最后看到的非空 id
            _active_slot_by_idx: dict = {}  # raw_index -> tool_calls_acc 中的当前槽位
            finish_reason = None
            model_name = None
            role = "assistant"
            reasoning_parts: list = []
            usage_obj = None
            _first_chunk_seen = False
            for chunk in stream:
                last_chunk_time["t"] = time.time()
                if not _first_chunk_seen:
                    _first_chunk_seen = True
                    self._touch_activity("receiving stream response")

                if self._interrupt_requested:
                    break

                if not chunk.choices:
                    if hasattr(chunk, "model") and chunk.model:
                        model_name = chunk.model
                    # 用法信息在最终块（空 choices）中到达
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_obj = chunk.usage
                    continue

                delta = chunk.choices[0].delta
                if hasattr(chunk, "model") and chunk.model:
                    model_name = chunk.model

                # 累积推理内容
                reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    _fire_first_delta()
                    self._fire_reasoning_delta(reasoning_text)

                # 累积文本内容 — 仅在无工具调用时触发回调
                if delta and delta.content:
                    content_parts.append(delta.content)
                    if not tool_calls_acc:
                        _fire_first_delta()
                        self._fire_stream_delta(delta.content)
                        deltas_were_sent["yes"] = True
                    else:
                        # 工具调用会抑制常规内容流式传输（避免在
                        # 工具调用旁边显示啰嗦的"I'll use the tool..."文本）。
                        # 但嵌入在抑制内容中的推理标签应该仍然到达显示 —
                        # 否则推理框仅作为响应后回退出现，
                        # 在已流式传输的响应之后令人困惑地呈现。
                        # 通过流增量回调路由被抑制的内容，
                        # 以便其标签提取可以触发推理显示。
                        # 非推理文本会被 CLI 的 _stream_delta
                        # 无害地抑制（当流式框已关闭时 — 工具边界刷新）。
                        if self.stream_delta_callback:
                            try:
                                self.stream_delta_callback(delta.content)
                                self._record_streamed_assistant_text(delta.content)
                            except Exception:
                                pass

                # 累积工具调用增量 — 在首个名称时通知显示
                if delta and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        raw_idx = tc_delta.index if tc_delta.index is not None else 0
                        delta_id = tc_delta.id or ""

                        # Ollama 修复：检测重用相同原始索引的新工具调用
                        #（不同的 id）并将其重定向到新的槽位。
                        if raw_idx not in _active_slot_by_idx:
                            _active_slot_by_idx[raw_idx] = raw_idx
                        if (
                            delta_id
                            and raw_idx in _last_id_at_idx
                            and delta_id != _last_id_at_idx[raw_idx]
                        ):
                            new_slot = max(tool_calls_acc, default=-1) + 1
                            _active_slot_by_idx[raw_idx] = new_slot
                        if delta_id:
                            _last_id_at_idx[raw_idx] = delta_id
                        idx = _active_slot_by_idx[raw_idx]

                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {
                                "id": tc_delta.id or "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                                "extra_content": None,
                            }
                        entry = tool_calls_acc[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["function"]["arguments"] += tc_delta.function.arguments
                        extra = getattr(tc_delta, "extra_content", None)
                        if extra is None and hasattr(tc_delta, "model_extra"):
                            extra = (tc_delta.model_extra or {}).get("extra_content")
                        if extra is not None:
                            if hasattr(extra, "model_dump"):
                                extra = extra.model_dump()
                            entry["extra_content"] = extra
                        # Fire once per tool when the full name is available
                        name = entry["function"]["name"]
                        if name and idx not in tool_gen_notified:
                            tool_gen_notified.add(idx)
                            _fire_first_delta()
                            self._fire_tool_gen_started(name)

                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

                # 最终块中的用法信息
                if hasattr(chunk, "usage") and chunk.usage:
                    usage_obj = chunk.usage

            # 构建模拟响应，匹配非流式形状
            full_content = "".join(content_parts) or None
            mock_tool_calls = None
            has_truncated_tool_args = False
            if tool_calls_acc:
                mock_tool_calls = []
                for idx in sorted(tool_calls_acc):
                    tc = tool_calls_acc[idx]
                    arguments = tc["function"]["arguments"]
                    if arguments and arguments.strip():
                        try:
                            json.loads(arguments)
                        except json.JSONDecodeError:
                            has_truncated_tool_args = True
                    mock_tool_calls.append(SimpleNamespace(
                        id=tc["id"],
                        type=tc["type"],
                        extra_content=tc.get("extra_content"),
                        function=SimpleNamespace(
                            name=tc["function"]["name"],
                            arguments=arguments,
                        ),
                    ))

            effective_finish_reason = finish_reason or "stop"
            if has_truncated_tool_args:
                effective_finish_reason = "length"

            full_reasoning = "".join(reasoning_parts) or None
            mock_message = SimpleNamespace(
                role=role,
                content=full_content,
                tool_calls=mock_tool_calls,
                reasoning_content=full_reasoning,
            )
            mock_choice = SimpleNamespace(
                index=0,
                message=mock_message,
                finish_reason=effective_finish_reason,
            )
            return SimpleNamespace(
                id="stream-" + str(uuid.uuid4()),
                model=model_name,
                choices=[mock_choice],
                usage=usage_obj,
            )

        def _call_anthropic():
            """流式传输 Anthropic Messages API 响应。

            为实时令牌传递触发增量回调，但返回
            来自 get_final_message() 的原生 Anthropic Message 对象，
            以便代理循环的其余部分（验证、工具提取等）
            保持不变地工作。
            """
            has_tool_use = False

            # 重置此尝试的过时流定时器
            last_chunk_time["t"] = time.time()
            # 使用 Anthropic SDK 的流式上下文管理器
            with self._anthropic_client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    # 每个事件都更新过时流定时器，以便
                    # 外部轮询循环知道数据正在流动。没有这个，
                    # 检测器会在 180 秒后终止健康的长时运行
                    # Opus 流，即使事件正在积极到达
                    #（chat_completions 路径已在
                    # 其块循环顶部执行此操作）。
                    last_chunk_time["t"] = time.time()

                    if self._interrupt_requested:
                        break

                    event_type = getattr(event, "type", None)

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block and getattr(block, "type", None) == "tool_use":
                            has_tool_use = True
                            tool_name = getattr(block, "name", None)
                            if tool_name:
                                _fire_first_delta()
                                self._fire_tool_gen_started(tool_name)

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", None)
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text and not has_tool_use:
                                    _fire_first_delta()
                                    self._fire_stream_delta(text)
                                    deltas_were_sent["yes"] = True
                            elif delta_type == "thinking_delta":
                                thinking_text = getattr(delta, "thinking", "")
                                if thinking_text:
                                    _fire_first_delta()
                                    self._fire_reasoning_delta(thinking_text)

                # 返回原生 Anthropic Message 用于下游处理
                return stream.get_final_message()

        def _call():
            import httpx as _httpx

            _max_stream_retries = int(os.getenv("HERMES_STREAM_RETRIES", 2))

            try:
                for _stream_attempt in range(_max_stream_retries + 1):
                    try:
                        if self.api_mode == "anthropic_messages":
                            self._try_refresh_anthropic_client_credentials()
                            result["response"] = _call_anthropic()
                        else:
                            result["response"] = _call_chat_completions()
                        return  # success
                    except Exception as e:
                        if deltas_were_sent["yes"]:
                            # 在一些令牌已经发送后流式传输失败。不要
                            # 重试或回退 — 部分内容已到达用户。
                            logger.warning(
                                "Streaming failed after partial delivery, not retrying: %s", e
                            )
                            result["error"] = e
                            return

                        _is_timeout = isinstance(
                            e, (_httpx.ReadTimeout, _httpx.ConnectTimeout, _httpx.PoolTimeout)
                        )
                        _is_conn_err = isinstance(
                            e, (_httpx.ConnectError, _httpx.RemoteProtocolError, ConnectionError)
                        )

                        # 来自代理的 SSE 错误事件（例如 OpenRouter 发送的
                        # {"error":{"message":"Network connection lost."}}）
                        # 被 OpenAI SDK 作为 APIError 引发。这些
                        # 在语义上与 httpx 连接断开相同 —
                        # 上游流已死亡 — 应该用新连接重试。
                        # 与 HTTP 错误区分：SSE 的 APIError 没有 status_code，
                        # 而 APIStatusError（4xx/5xx）始终有。
                        _is_sse_conn_err = False
                        if not _is_timeout and not _is_conn_err:
                            from openai import APIError as _APIError
                            if isinstance(e, _APIError) and not getattr(e, "status_code", None):
                                _err_lower_sse = str(e).lower()
                                _SSE_CONN_PHRASES = (
                                    "connection lost",
                                    "connection reset",
                                    "connection closed",
                                    "connection terminated",
                                    "network error",
                                    "network connection",
                                    "terminated",
                                    "peer closed",
                                    "broken pipe",
                                    "upstream connect error",
                                )
                                _is_sse_conn_err = any(
                                    phrase in _err_lower_sse
                                    for phrase in _SSE_CONN_PHRASES
                                )

                        if _is_timeout or _is_conn_err or _is_sse_conn_err:
                            # Transient network / timeout error. Retry the
                            # streaming request with a fresh connection first.
                            if _stream_attempt < _max_stream_retries:
                                logger.info(
                                    "Streaming attempt %s/%s failed (%s: %s), "
                                    "retrying with fresh connection...",
                                    _stream_attempt + 1,
                                    _max_stream_retries + 1,
                                    type(e).__name__,
                                    e,
                                )
                                self._emit_status(
                                    f"⚠️ Connection to provider dropped "
                                    f"({type(e).__name__}). Reconnecting… "
                                    f"(attempt {_stream_attempt + 2}/{_max_stream_retries + 1})"
                                )
                                # 在重试前关闭过时的请求客户端
                                stale = request_client_holder.get("client")
                                if stale is not None:
                                    self._close_request_openai_client(
                                        stale, reason="stream_retry_cleanup"
                                    )
                                    request_client_holder["client"] = None
                                # 同时重建主客户端以清除
                                # 池中的任何死亡连接。
                                try:
                                    self._replace_primary_openai_client(
                                        reason="stream_retry_pool_cleanup"
                                    )
                                except Exception:
                                    pass
                                continue
                            self._emit_status(
                                "❌ Connection to provider failed after "
                                f"{_max_stream_retries + 1} attempts. "
                                "The provider may be experiencing issues — "
                                "try again in a moment."
                            )
                            logger.warning(
                                "Streaming exhausted %s retries on transient error, "
                                "falling back to non-streaming: %s",
                                _max_stream_retries + 1,
                                e,
                            )
                        else:
                            _err_lower = str(e).lower()
                            _is_stream_unsupported = (
                                "stream" in _err_lower
                                and "not supported" in _err_lower
                            )
                            if _is_stream_unsupported:
                                self._safe_print(
                                    "\n⚠  Streaming is not supported for this "
                                    "model/provider. Falling back to non-streaming.\n"
                                    "   To avoid this delay, set display.streaming: false "
                                    "in config.yaml\n"
                                )
                            logger.info(
                                "Streaming failed before delivery, falling back to non-streaming: %s",
                                e,
                            )

                        try:
                            # 重置过时定时器 — 非流式回退
                            # 使用自己的客户端；防止过时检测器
                            # 对失败流的过时时间戳触发。
                            last_chunk_time["t"] = time.time()
                            result["response"] = self._interruptible_api_call(api_kwargs)
                        except Exception as fallback_err:
                            result["error"] = fallback_err
                        return
            finally:
                request_client = request_client_holder.get("client")
                if request_client is not None:
                    self._close_request_openai_client(request_client, reason="stream_request_complete")

        _stream_stale_timeout_base = float(os.getenv("HERMES_STREAM_STALE_TIMEOUT", 180.0))
        # 本地提供商（Ollama、oMLX、llama-cpp）在大型上下文上
        # 的预填充可能需要 300+ 秒。除非用户明确
        # 设置了 HERMES_STREAM_STALE_TIMEOUT，否则禁用过时检测器。
        if _stream_stale_timeout_base == 180.0 and self.base_url and is_local_endpoint(self.base_url):
            _stream_stale_timeout = float("inf")
            logger.debug("Local provider detected (%s) — stale stream timeout disabled", self.base_url)
        else:
            # 为大型上下文缩放过时超时：慢速模型（如 Opus）
            # 在上下文很大时可能在产生第一个令牌前合法地思考数分钟。
            # 没有这个，过时检测器会在模型的思考阶段杀死
            # 健康连接，产生虚假的 RemoteProtocolError（"对端关闭连接"）。
            _est_tokens = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
            if _est_tokens > 100_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 300.0)
            elif _est_tokens > 50_000:
                _stream_stale_timeout = max(_stream_stale_timeout_base, 240.0)
            else:
                _stream_stale_timeout = _stream_stale_timeout_base

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        while t.is_alive():
            t.join(timeout=0.3)

            # 检测过时流：保持活跃但
            # 没有传递真实数据块的连接。杀死客户端以便
            # 内部重试循环可以开始新的连接。
            _stale_elapsed = time.time() - last_chunk_time["t"]
            if _stale_elapsed > _stream_stale_timeout:
                _est_ctx = sum(len(str(v)) for v in api_kwargs.get("messages", [])) // 4
                logger.warning(
                    "Stream stale for %.0fs (threshold %.0fs) — no chunks received. "
                    "model=%s context=~%s tokens. Killing connection.",
                    _stale_elapsed, _stream_stale_timeout,
                    api_kwargs.get("model", "unknown"), f"{_est_ctx:,}",
                )
                self._emit_status(
                    f"⚠️ No response from provider for {int(_stale_elapsed)}s "
                    f"(model: {api_kwargs.get('model', 'unknown')}, "
                    f"context: ~{_est_ctx:,} tokens). "
                    f"Reconnecting..."
                )
                try:
                    rc = request_client_holder.get("client")
                    if rc is not None:
                        self._close_request_openai_client(rc, reason="stale_stream_kill")
                except Exception:
                    pass
                # 也重建主客户端 — 其连接池
                # 可能持有来自同一提供商故障的死亡套接字。
                try:
                    self._replace_primary_openai_client(reason="stale_stream_pool_cleanup")
                except Exception:
                    pass
                # 重置定时器，这样在内部线程处理关闭时
                # 不会重复杀死。
                last_chunk_time["t"] = time.time()

            if self._interrupt_requested:
                try:
                    if self.api_mode == "anthropic_messages":
                        from agent.anthropic_adapter import build_anthropic_client

                        self._anthropic_client.close()
                        self._anthropic_client = build_anthropic_client(
                            self._anthropic_api_key,
                            getattr(self, "_anthropic_base_url", None),
                        )
                    else:
                        request_client = request_client_holder.get("client")
                        if request_client is not None:
                            self._close_request_openai_client(request_client, reason="stream_interrupt_abort")
                except Exception:
                    pass
                raise InterruptedError("Agent interrupted during streaming API call")
        if result["error"] is not None:
            if deltas_were_sent["yes"]:
                # 流式传输失败时一些令牌已传递到
                # 平台。重新引发会让外部重试循环发起
                # 新的 API 调用，产生重复消息。改为返回
                # 部分的 "stop" 响应，使外部循环将此
                # 轮视为完成（不重试、不回退）。
                logger.warning(
                    "Partial stream delivered before error; returning stub "
                    "response to prevent duplicate messages: %s",
                    result["error"],
                )
                _stub_msg = SimpleNamespace(
                    role="assistant", content=None, tool_calls=None,
                    reasoning_content=None,
                )
                return SimpleNamespace(
                    id="partial-stream-stub",
                    model=getattr(self, "model", "unknown"),
                    choices=[SimpleNamespace(
                        index=0, message=_stub_msg, finish_reason="stop",
                    )],
                    usage=None,
                )
            raise result["error"]
        return result["response"]

    # ── 提供商回退 ──────────────────────────────────────────────────

    def _try_activate_fallback(self) -> bool:
        """切换到链中的下一个回退模型/提供商。

        在当前模型重试后失败时调用。在原地交换
        OpenAI 客户端、模型 slug 和提供商，使重试循环
        能够继续使用新的后端。每次调用推进链；
        用尽时返回 False。

        使用集中式提供商路由器（resolve_provider_client）进行
        认证解析和客户端构建 — 无重复的提供商→密钥映射。
        """
        if self._fallback_index >= len(self._fallback_chain):
            return False

        fb = self._fallback_chain[self._fallback_index]
        self._fallback_index += 1
        fb_provider = (fb.get("provider") or "").strip().lower()
        fb_model = (fb.get("model") or "").strip()
        if not fb_provider or not fb_model:
            return self._try_activate_fallback()  # skip invalid, try next

        # 使用集中式路由器进行客户端构建。
        # raw_codex=True，因为主代理需要直接的 responses.stream()
        # 访问权限用于 Codex 提供商。
        try:
            from agent.auxiliary_client import resolve_provider_client
            # 从回退配置中传递 base_url 和 api_key，使自定义
            # 端点（如 Ollama Cloud）能正确解析，而非
            # 回退到 OpenRouter 默认值。
            fb_base_url_hint = (fb.get("base_url") or "").strip() or None
            fb_api_key_hint = (fb.get("api_key") or "").strip() or None
            # 对于 Ollama Cloud 端点，当回退配置中没有显式密钥时，
            # 从环境变量中获取 OLLAMA_API_KEY。
            if fb_base_url_hint and "ollama.com" in fb_base_url_hint.lower() and not fb_api_key_hint:
                fb_api_key_hint = os.getenv("OLLAMA_API_KEY") or None
            fb_client, _resolved_fb_model = resolve_provider_client(
                fb_provider, model=fb_model, raw_codex=True,
                explicit_base_url=fb_base_url_hint,
                explicit_api_key=fb_api_key_hint)
            if fb_client is None:
                logging.warning(
                    "Fallback to %s failed: provider not configured",
                    fb_provider)
                return self._try_activate_fallback()  # try next in chain
            try:
                from hermes_cli.model_normalize import normalize_model_for_provider

                fb_model = normalize_model_for_provider(fb_model, fb_provider)
            except Exception:
                pass

            # 根据提供商 / base URL / model 确定 api_mode
            fb_api_mode = "chat_completions"
            fb_base_url = str(fb_client.base_url)
            if fb_provider == "openai-codex":
                fb_api_mode = "codex_responses"
            elif fb_provider == "anthropic" or fb_base_url.rstrip("/").lower().endswith("/anthropic"):
                fb_api_mode = "anthropic_messages"
            elif self._is_direct_openai_url(fb_base_url):
                fb_api_mode = "codex_responses"
            elif self._model_requires_responses_api(fb_model):
                # GPT-5.x 模型在每个提供商上都需要 Responses API
                #（OpenRouter、Copilot、直接 OpenAI 等）
                fb_api_mode = "codex_responses"

            old_model = self.model
            self.model = fb_model
            self.provider = fb_provider
            self.base_url = fb_base_url
            self.api_mode = fb_api_mode
            self._fallback_activated = True

            if fb_api_mode == "anthropic_messages":
                # 构建原生 Anthropic 客户端，而非使用 OpenAI 客户端
                from agent.anthropic_adapter import build_anthropic_client, resolve_anthropic_token, _is_oauth_token
                effective_key = (fb_client.api_key or resolve_anthropic_token() or "") if fb_provider == "anthropic" else (fb_client.api_key or "")
                self.api_key = effective_key
                self._anthropic_api_key = effective_key
                self._anthropic_base_url = fb_base_url
                self._anthropic_client = build_anthropic_client(effective_key, self._anthropic_base_url)
                self._is_anthropic_oauth = _is_oauth_token(effective_key)
                self.client = None
                self._client_kwargs = {}
            else:
                # 就地交换 OpenAI 客户端和配置
                self.api_key = fb_client.api_key
                self.client = fb_client
                # 保留 resolve_provider_client() 可能通过 default_headers kwarg
                # 烘焙到 fb_client 中的提供商特定头。OpenAI
                # SDK 将这些存储在 _custom_headers 中。没有这个，
                # 后续的请求客户端重建（通过
                # _create_request_openai_client）会丢失这些头，
                # 导致需要 User-Agent 哨兵的提供商（如 Kimi Coding）
                # 出现 403。
                fb_headers = getattr(fb_client, "_custom_headers", None)
                if not fb_headers:
                    fb_headers = getattr(fb_client, "default_headers", None)
                self._client_kwargs = {
                    "api_key": fb_client.api_key,
                    "base_url": fb_base_url,
                    **({"default_headers": dict(fb_headers)} if fb_headers else {}),
                }

            # Re-evaluate prompt caching for the new provider/model
            is_native_anthropic = fb_api_mode == "anthropic_messages" and fb_provider == "anthropic"
            self._use_prompt_caching = (
                ("openrouter" in fb_base_url.lower() and "claude" in fb_model.lower())
                or is_native_anthropic
            )

            # 为回退模型更新上下文压缩器限制。
            # 没有这个，压缩决策会使用主模型的
            # 上下文窗口（例如 200K）而非回退的（例如 32K），
            # 导致超大会话溢出回退。
            if hasattr(self, 'context_compressor') and self.context_compressor:
                from agent.model_metadata import get_model_context_length
                fb_context_length = get_model_context_length(
                    self.model, base_url=self.base_url,
                    api_key=self.api_key, provider=self.provider,
                )
                self.context_compressor.update_model(
                    model=self.model,
                    context_length=fb_context_length,
                    base_url=self.base_url,
                    api_key=getattr(self, "api_key", ""),
                    provider=self.provider,
                )

            self._emit_status(
                f"🔄 Primary model failed — switching to fallback: "
                f"{fb_model} via {fb_provider}"
            )
            logging.info(
                "Fallback activated: %s → %s (%s)",
                old_model, fb_model, fb_provider,
            )
            return True
        except Exception as e:
            logging.error("Failed to activate fallback %s: %s", fb_model, e)
            return self._try_activate_fallback()  # try next in chain

    # ── 每轮主运行时恢复 ─────────────────────────────────────

    def _restore_primary_runtime(self) -> bool:
        """在新轮开始时恢复主运行时。

        在长寿的 CLI 会话中，单个 AIAgent 实例跨越多轮。
        没有恢复，一次瞬时失败会将会话固定到
        后续每轮的回退提供商。在 ``run_conversation()``
        顶部调用此方法使回退成为轮作用域的。

        Gateway 在消息之间缓存代理（``gateway/run.py`` 中的
        ``_agent_cache``），所以在那里也需要恢复。
        """
        if not self._fallback_activated:
            return False

        rt = self._primary_runtime
        try:
            # ── 核心运行时状态 ──
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]           # setter 更新 _base_url_lower
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]
            self._client_kwargs = dict(rt["client_kwargs"])
            self._use_prompt_caching = rt["use_prompt_caching"]

            # ── 为主提供商重建客户端 ──
            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="restore_primary",
                    shared=True,
                )

            # ── 恢复上下文引擎状态 ──
            cc = self.context_compressor
            cc.update_model(
                model=rt["compressor_model"],
                context_length=rt["compressor_context_length"],
                base_url=rt["compressor_base_url"],
                api_key=rt["compressor_api_key"],
                provider=rt["compressor_provider"],
            )

            # ── 为新轮重置回退链 ──
            self._fallback_activated = False
            self._fallback_index = 0

            logging.info(
                "Primary runtime restored for new turn: %s (%s)",
                self.model, self.provider,
            )
            return True
        except Exception as e:
            logging.warning("Failed to restore primary runtime: %s", e)
            return False

    # 哪些错误类型表示瞬态传输失败，值得
    # 用重建的客户端 / 连接池再尝试一次。
    _TRANSIENT_TRANSPORT_ERRORS = frozenset({
        "ReadTimeout", "ConnectTimeout", "PoolTimeout",
        "ConnectError", "RemoteProtocolError",
        "APIConnectionError", "APITimeoutError",
    })

    def _try_recover_primary_transport(
        self, api_error: Exception, *, retry_count: int, max_retries: int,
    ) -> bool:
        """为瞬态传输失败尝试一次额外的提供商家恢复周期。

        在 ``max_retries`` 耗尽后，重建主客户端（清除
        陈旧连接池）并再给一次尝试机会再回退。这对直接端点最有用
        （自定义、Z.AI、Anthropic、OpenAI、本地模型），在这些地方
        TCP 级别的小问题不意味着提供商宕机。

        对代理/聚合提供商（OpenRouter、Nous）跳过 —
        它们已在服务器端管理连接池和重试 — 如果我们通过
        它们的重试已耗尽，再重建一个客户端也无济于事。
        """
        if self._fallback_activated:
            return False

        # 仅针对瞬态传输错误
        error_type = type(api_error).__name__
        if error_type not in self._TRANSIENT_TRANSPORT_ERRORS:
            return False

        # 对聚合提供商跳过 — 它们管理自己的重试基础设施
        if self._is_openrouter_url():
            return False
        provider_lower = (self.provider or "").strip().lower()
        if provider_lower in ("nous", "nous-research"):
            return False

        try:
            # 关闭现有客户端以释放死亡连接
            if getattr(self, "client", None) is not None:
                try:
                    self._close_openai_client(
                        self.client, reason="primary_recovery", shared=True,
                    )
                except Exception:
                    pass

            # 从主快照重建
            rt = self._primary_runtime
            self._client_kwargs = dict(rt["client_kwargs"])
            self.model = rt["model"]
            self.provider = rt["provider"]
            self.base_url = rt["base_url"]
            self.api_mode = rt["api_mode"]
            self.api_key = rt["api_key"]

            if self.api_mode == "anthropic_messages":
                from agent.anthropic_adapter import build_anthropic_client
                self._anthropic_api_key = rt["anthropic_api_key"]
                self._anthropic_base_url = rt["anthropic_base_url"]
                self._anthropic_client = build_anthropic_client(
                    rt["anthropic_api_key"], rt["anthropic_base_url"],
                )
                self._is_anthropic_oauth = rt["is_anthropic_oauth"]
                self.client = None
            else:
                self.client = self._create_openai_client(
                    dict(rt["client_kwargs"]),
                    reason="primary_recovery",
                    shared=True,
                )

            wait_time = min(3 + retry_count, 8)
            self._vprint(
                f"{self.log_prefix}🔁 Transient {error_type} on {self.provider} — "
                f"rebuilt client, waiting {wait_time}s before one last primary attempt.",
                force=True,
            )
            time.sleep(wait_time)
            return True
        except Exception as e:
            logging.warning("Primary transport recovery failed: %s", e)
            return False

    # ── 提供商回退结束 ──────────────────────────────────────────────

    @staticmethod
    def _content_has_image_parts(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image"}:
                return True
        return False

    @staticmethod
    def _materialize_data_url_for_vision(image_url: str) -> tuple[str, Optional[Path]]:
        # 将 data url 处理为图片文件，并返回路径
        header, _, data = str(image_url or "").partition(",")
        mime = "image/jpeg"
        if header.startswith("data:"):
            mime_part = header[len("data:"):].split(";", 1)[0].strip()
            if mime_part.startswith("image/"):
                mime = mime_part
        suffix = {
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
        }.get(mime, ".jpg")
        tmp = tempfile.NamedTemporaryFile(prefix="anthropic_image_", suffix=suffix, delete=False)
        with tmp:
            tmp.write(base64.b64decode(data))
        path = Path(tmp.name)
        return str(path), path

    def _describe_image_for_anthropic_fallback(self, image_url: str, role: str) -> str:
        # Anthropic 不支持图片输入时，为图片生成文本描述用于回退
        cache_key = hashlib.sha256(str(image_url or "").encode("utf-8")).hexdigest()
        cached = self._anthropic_image_fallback_cache.get(cache_key)
        if cached:
            return cached

        role_label = {
            "assistant": "assistant",
            "tool": "tool result",
        }.get(role, "user")
        analysis_prompt = (
            "用详尽的细节描述这张图片里可见的所有内容。包括所有文字、代码、界面、数据、物体、人、布局、色彩及任何其他显著的视觉信息。"
        )

        vision_source = str(image_url or "")
        cleanup_path: Optional[Path] = None
        if vision_source.startswith("data:"):
            vision_source, cleanup_path = self._materialize_data_url_for_vision(vision_source)

        description = ""
        try:
            from tools.vision_tools import vision_analyze_tool

            result_json = asyncio.run(
                vision_analyze_tool(image_url=vision_source, user_prompt=analysis_prompt)
            )
            result = json.loads(result_json) if isinstance(result_json, str) else {}
            description = (result.get("analysis") or "").strip()
        except Exception as e:
            description = f"图片分析失败: {e}"
        finally:
            if cleanup_path and cleanup_path.exists():
                try:
                    cleanup_path.unlink()
                except OSError:
                    pass

        if not description:
            description = "图片分析失败。"

        note = f"[{role_label} 附加了一张图片。图片内容如下：\n{description}]"
        if vision_source and not str(image_url or "").startswith("data:"):
            note += (
                f"\n[如需更详细分析，可使用 vision_analyze，图片链接为: {vision_source}]"
            )

        self._anthropic_image_fallback_cache[cache_key] = note
        return note

    def _preprocess_anthropic_content(self, content: Any, role: str) -> Any:
        # 将 multimodal 内容转换为适用于 Anthropic 的文本格式
        if not self._content_has_image_parts(content):
            return content

        text_parts: List[str] = []
        image_notes: List[str] = []
        for part in content:
            if isinstance(part, str):
                if part.strip():
                    text_parts.append(part.strip())
                continue
            if not isinstance(part, dict):
                continue

            ptype = part.get("type")
            if ptype in {"text", "input_text"}:
                text = str(part.get("text", "") or "").strip()
                if text:
                    text_parts.append(text)
                continue

            if ptype in {"image_url", "input_image"}:
                image_data = part.get("image_url", {})
                image_url = image_data.get("url", "") if isinstance(image_data, dict) else str(image_data or "")
                if image_url:
                    image_notes.append(self._describe_image_for_anthropic_fallback(image_url, role))
                else:
                    image_notes.append("[已附加图片，但未提供图片来源。]")
                continue

            text = str(part.get("text", "") or "").strip()
            if text:
                text_parts.append(text)

        prefix = "\n\n".join(note for note in image_notes if note).strip()
        suffix = "\n".join(text for text in text_parts if text).strip()
        if prefix and suffix:
            return f"{prefix}\n\n{suffix}"
        if prefix:
            return prefix
        if suffix:
            return suffix
        return "[多模态消息已被转换为文本，以兼容 Anthropic。]"

    def _prepare_anthropic_messages_for_api(self, api_messages: list) -> list:
        # 检查消息内容是否包含图片，必要时进行转换
        if not any(
            isinstance(msg, dict) and self._content_has_image_parts(msg.get("content"))
            for msg in api_messages
        ):
            return api_messages

        transformed = copy.deepcopy(api_messages)
        for msg in transformed:
            if not isinstance(msg, dict):
                continue
            msg["content"] = self._preprocess_anthropic_content(
                msg.get("content"),
                str(msg.get("role", "user") or "user"),
            )
        return transformed

    def _anthropic_preserve_dots(self) -> bool:
        """当使用保留模型名称中点号的 Anthropic 兼容端点时返回 True。
        阿里巴巴/DashScope（如 qwen3.5-plus）保留点；
        MiniMax（如 MiniMax-M2.7）保留点；
        OpenCode Go/Zen，非 Claude 模型（如 minimax-m2.5-free）保留点；
        ZAI/智谱（如 glm-4.7, glm-5.1）保留点。"""
        if (getattr(self, "provider", "") or "").lower() in {"alibaba", "minimax", "minimax-cn", "opencode-go", "opencode-zen", "zai"}:
            return True
        base = (getattr(self, "base_url", "") or "").lower()
        return "dashscope" in base or "aliyuncs" in base or "minimax" in base or "opencode.ai/zen/" in base or "bigmodel.cn" in base

    def _is_qwen_portal(self) -> bool:
        """基础 URL 指向 Qwen Portal 时返回 True。"""
        return "portal.qwen.ai" in self._base_url_lower

    def _qwen_prepare_chat_messages(self, api_messages: list) -> list:
        # 为 Qwen API 适配消息格式
        prepared = copy.deepcopy(api_messages)
        if not prepared:
            return prepared

        for msg in prepared:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                # 规范化：将字符串变为文本字典，保留字典原样。
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        # 在第一条 system 消息的最后一个部分，注入 cache_control
        for msg in prepared:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return prepared

    def _qwen_prepare_chat_messages_inplace(self, messages: list) -> None:
        """原地处理（修改传入的已复制消息列表）。"""
        if not messages:
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                normalized_parts = []
                for part in content:
                    if isinstance(part, str):
                        normalized_parts.append({"type": "text", "text": part})
                    elif isinstance(part, dict):
                        normalized_parts.append(part)
                if normalized_parts:
                    msg["content"] = normalized_parts

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content")
                if isinstance(content, list) and content and isinstance(content[-1], dict):
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    def _build_api_kwargs(self, api_messages: list) -> dict:
        """为当前 API 模式构建关键字参数字典。"""
        if self.api_mode == "anthropic_messages":
            from agent.anthropic_adapter import build_anthropic_kwargs
            anthropic_messages = self._prepare_anthropic_messages_for_api(api_messages)
            # 传递 context_length（总输入+输出窗口大小），以便当用户配置的
            # 上下文窗口小于模型原生输出限制时，适配器可以限制 max_tokens（输出上限）。
            ctx_len = getattr(self, "context_compressor", None)
            ctx_len = ctx_len.context_length if ctx_len else None
            # 当 API 返回 "max_tokens too large given prompt" 时，会为单次
            # 调用设置 _ephemeral_max_output_tokens——它会限制输出到可用窗口空间，
            # 而不改变 context_length。
            ephemeral_out = getattr(self, "_ephemeral_max_output_tokens", None)
            if ephemeral_out is not None:
                self._ephemeral_max_output_tokens = None  # consume immediately
            return build_anthropic_kwargs(
                model=self.model,
                messages=anthropic_messages,
                tools=self.tools,
                max_tokens=ephemeral_out if ephemeral_out is not None else self.max_tokens,
                reasoning_config=self.reasoning_config,
                is_oauth=self._is_anthropic_oauth,
                preserve_dots=self._anthropic_preserve_dots(),
                context_length=ctx_len,
                base_url=getattr(self, "_anthropic_base_url", None),
                fast_mode=(self.request_overrides or {}).get("speed") == "fast",
            )

        if self.api_mode == "codex_responses":
            instructions = ""
            payload_messages = api_messages
            if api_messages and api_messages[0].get("role") == "system":
                instructions = str(api_messages[0].get("content") or "").strip()
                payload_messages = api_messages[1:]
            if not instructions:
                instructions = DEFAULT_AGENT_IDENTITY

            is_github_responses = (
                "models.github.ai" in self.base_url.lower()
                or "api.githubcopilot.com" in self.base_url.lower()
            )
            is_codex_backend = (
                self.provider == "openai-codex"
                or "chatgpt.com/backend-api/codex" in self.base_url.lower()
            )

            # 推理强度解析优先级：配置参数 > 默认值（medium）
     
            reasoning_effort = "medium"
            reasoning_enabled = True
            if self.reasoning_config and isinstance(self.reasoning_config, dict):
                if self.reasoning_config.get("enabled") is False:
                    reasoning_enabled = False
                elif self.reasoning_config.get("effort"):
                    reasoning_effort = self.reasoning_config["effort"]

            kwargs = {
                "model": self.model,
                "instructions": instructions,
                "input": self._chat_messages_to_responses_input(payload_messages),
                "tools": self._responses_tools(),
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "store": False,
            }

            if not is_github_responses:
                kwargs["prompt_cache_key"] = self.session_id

            if reasoning_enabled:
                if is_github_responses:
                    # Copilot 的 Responses 路由支持 reasoning-effort（推理强度），
                    # 但不支持 OpenAI 特有的 prompt 缓存或加密推理字段。
                    # 只保留文档中规定的字段到请求体中。
         
                    github_reasoning = self._github_models_reasoning_extra_body()
                    if github_reasoning is not None:
                        kwargs["reasoning"] = github_reasoning
                else:
                    kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                    kwargs["include"] = ["reasoning.encrypted_content"]
            elif not is_github_responses:
                kwargs["include"] = []

            if self.request_overrides:
                kwargs.update(self.request_overrides)

            if self.max_tokens is not None and not is_codex_backend:
                kwargs["max_output_tokens"] = self.max_tokens

            return kwargs

        sanitized_messages = api_messages
        needs_sanitization = False
        for msg in api_messages:
            if not isinstance(msg, dict):
                continue
            if "codex_reasoning_items" in msg:
                needs_sanitization = True
                break

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    if "call_id" in tool_call or "response_item_id" in tool_call:
                        needs_sanitization = True
                        break
                if needs_sanitization:
                    break

        if needs_sanitization:
            # 如果需要清理，深拷贝一份 api_messages，避免直接修改原始数据。
            sanitized_messages = copy.deepcopy(api_messages)
            for msg in sanitized_messages:
                if not isinstance(msg, dict):
                    continue

                # 只有 Codex 重放相关的状态，不能泄漏到严格的 chat-completions API。
                msg.pop("codex_reasoning_items", None)

                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if isinstance(tool_call, dict):
                            tool_call.pop("call_id", None)
                            tool_call.pop("response_item_id", None)

        # Qwen 门户：将 content 标准化为字典的列表，注入 cache_control 字段。
        # 必须在 Codex 清理之后执行，这样才能处理最终的消息内容。
        # 如果清理过程已经做了深拷贝，这里就直接在副本上原地修改。
        if self._is_qwen_portal():
            if sanitized_messages is api_messages:
                # 没有经过清理，需要自己制作一份副本。
                sanitized_messages = self._qwen_prepare_chat_messages(sanitized_messages)
            else:
                # 已经做过深拷贝，直接就地处理，避免再次深拷贝。
                self._qwen_prepare_chat_messages_inplace(sanitized_messages)

        # GPT-5 和 Codex 系列模型在执行指令时，
        # "developer" 作为身份比 "system" 效果更好。
        # 因此在 API 调用边界处将角色从 "system" 替换为 "developer"，
        # 这样内部消息表示依然统一（"system"）。
        _model_lower = (self.model or "").lower()
        if (
            sanitized_messages
            and sanitized_messages[0].get("role") == "system"
            and any(p in _model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            # Shallow-copy the list + first message only — rest stays shared.
            sanitized_messages = list(sanitized_messages)
            sanitized_messages[0] = {**sanitized_messages[0], "role": "developer"}

        provider_preferences = {}
        if self.providers_allowed:
            provider_preferences["only"] = self.providers_allowed
        if self.providers_ignored:
            provider_preferences["ignore"] = self.providers_ignored
        if self.providers_order:
            provider_preferences["order"] = self.providers_order
        if self.provider_sort:
            provider_preferences["sort"] = self.provider_sort
        if self.provider_require_parameters:
            provider_preferences["require_parameters"] = True
        if self.provider_data_collection:
            provider_preferences["data_collection"] = self.provider_data_collection

        api_kwargs = {
            "model": self.model,
            "messages": sanitized_messages,
            "timeout": float(os.getenv("HERMES_API_TIMEOUT", 1800.0)),
        }
        if self._is_qwen_portal():
            api_kwargs["metadata"] = {
                "sessionId": self.session_id or "hermes",
                "promptId": str(uuid.uuid4()),
            }
        if self.tools:
            api_kwargs["tools"] = self.tools

        if self.max_tokens is not None:
            api_kwargs.update(self._max_tokens_param(self.max_tokens))
        elif self._is_qwen_portal():
            # Qwen Portal defaults to a very low max_tokens when omitted.
            # Reasoning models (qwen3-coder-plus) exhaust that budget on
            # thinking tokens alone, causing the portal to return
            # finish_reason="stop" with truncated output — the agent sees
            # this as an intentional stop and exits the loop.  Send 65536
            # (the documented max output for qwen3-coder models) so the
            # model has adequate output budget for tool calls.
            api_kwargs.update(self._max_tokens_param(65536))
        elif (self._is_openrouter_url() or "nousresearch" in self._base_url_lower) and "claude" in (self.model or "").lower():
            # OpenRouter and Nous Portal translate requests to Anthropic's
            # Messages API, which requires max_tokens as a mandatory field.
            # When we omit it, the proxy picks a default that can be too
            # low — the model spends its output budget on thinking and has
            # almost nothing left for the actual response (especially large
            # tool calls like write_file).  Sending the model's real output
            # limit ensures full capacity.
            try:
                from agent.anthropic_adapter import _get_anthropic_max_output
                _model_output_limit = _get_anthropic_max_output(self.model)
                api_kwargs["max_tokens"] = _model_output_limit
            except Exception:
                pass  # fail open — let the proxy pick its default

        extra_body = {}

        _is_openrouter = self._is_openrouter_url()
        _is_github_models = (
            "models.github.ai" in self._base_url_lower
            or "api.githubcopilot.com" in self._base_url_lower
        )

        # Provider preferences (only, ignore, order, sort) are OpenRouter-
        # specific.  Only send to OpenRouter-compatible endpoints.
        # TODO: Nous Portal will add transparent proxy support — re-enable
        # for _is_nous when their backend is updated.
        if provider_preferences and _is_openrouter:
            extra_body["provider"] = provider_preferences
        _is_nous = "nousresearch" in self._base_url_lower

        if self._supports_reasoning_extra_body():
            if _is_github_models:
                github_reasoning = self._github_models_reasoning_extra_body()
                if github_reasoning is not None:
                    extra_body["reasoning"] = github_reasoning
            else:
                if self.reasoning_config is not None:
                    rc = dict(self.reasoning_config)
                    # Nous Portal requires reasoning enabled — don't send
                    # enabled=false to it (would cause 400).
                    if _is_nous and rc.get("enabled") is False:
                        pass  # omit reasoning entirely for Nous when disabled
                    else:
                        extra_body["reasoning"] = rc
                else:
                    extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }

        # Nous Portal product attribution
        if _is_nous:
            extra_body["tags"] = ["product=hermes-agent"]

        # Ollama num_ctx: override the 2048 default so the model actually
        # uses the context window it was trained for.  Passed via the OpenAI
        # SDK's extra_body → options.num_ctx, which Ollama's OpenAI-compat
        # endpoint forwards to the runner as --ctx-size.
        if self._ollama_num_ctx:
            options = extra_body.get("options", {})
            options["num_ctx"] = self._ollama_num_ctx
            extra_body["options"] = options

        if self._is_qwen_portal():
            extra_body["vl_high_resolution_images"] = True

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # xAI prompt caching: send x-grok-conv-id header to route requests
        # to the same server, maximizing automatic cache hits.
        # https://docs.x.ai/developers/advanced-api-usage/prompt-caching
        if "x.ai" in self._base_url_lower and hasattr(self, "session_id") and self.session_id:
            api_kwargs["extra_headers"] = {"x-grok-conv-id": self.session_id}

        # Priority Processing / generic request overrides (e.g. service_tier).
        # Applied last so overrides win over any defaults set above.
        if self.request_overrides:
            api_kwargs.update(self.request_overrides)

        return api_kwargs

    def _supports_reasoning_extra_body(self) -> bool:
        """Return True when reasoning extra_body is safe to send for this route/model.

        OpenRouter forwards unknown extra_body fields to upstream providers.
        Some providers/routes reject `reasoning` with 400s, so gate it to
        known reasoning-capable model families and direct Nous Portal.
        """
        if "nousresearch" in self._base_url_lower:
            return True
        if "ai-gateway.vercel.sh" in self._base_url_lower:
            return True
        if "models.github.ai" in self._base_url_lower or "api.githubcopilot.com" in self._base_url_lower:
            try:
                from hermes_cli.models import github_model_reasoning_efforts

                return bool(github_model_reasoning_efforts(self.model))
            except Exception:
                return False
        if "openrouter" not in self._base_url_lower:
            return False
        if "api.mistral.ai" in self._base_url_lower:
            return False

        model = (self.model or "").lower()
        reasoning_model_prefixes = (
            "deepseek/",
            "anthropic/",
            "openai/",
            "x-ai/",
            "google/gemini-2",
            "qwen/qwen3",
        )
        return any(model.startswith(prefix) for prefix in reasoning_model_prefixes)

    def _github_models_reasoning_extra_body(self) -> dict | None:
        """Format reasoning payload for GitHub Models/OpenAI-compatible routes."""
        try:
            from hermes_cli.models import github_model_reasoning_efforts
        except Exception:
            return None

        supported_efforts = github_model_reasoning_efforts(self.model)
        if not supported_efforts:
            return None

        if self.reasoning_config and isinstance(self.reasoning_config, dict):
            if self.reasoning_config.get("enabled") is False:
                return None
            requested_effort = str(
                self.reasoning_config.get("effort", "medium")
            ).strip().lower()
        else:
            requested_effort = "medium"

        if requested_effort == "xhigh" and "high" in supported_efforts:
            requested_effort = "high"
        elif requested_effort not in supported_efforts:
            if requested_effort == "minimal" and "low" in supported_efforts:
                requested_effort = "low"
            elif "medium" in supported_efforts:
                requested_effort = "medium"
            else:
                requested_effort = supported_efforts[0]

        return {"effort": requested_effort}

    def _build_assistant_message(self, assistant_message, finish_reason: str) -> dict:
        """Build a normalized assistant message dict from an API response message.

        Handles reasoning extraction, reasoning_details, and optional tool_calls
        so both the tool-call path and the final-response path share one builder.
        """
        reasoning_text = self._extract_reasoning(assistant_message)
        _from_structured = bool(reasoning_text)

        # Fallback: extract inline <think> blocks from content when no structured
        # reasoning fields are present (some models/providers embed thinking
        # directly in the content rather than returning separate API fields).
        if not reasoning_text:
            content = assistant_message.content or ""
            think_blocks = re.findall(r'<think>(.*?)</think>', content, flags=re.DOTALL)
            if think_blocks:
                combined = "\n\n".join(b.strip() for b in think_blocks if b.strip())
                reasoning_text = combined or None

        if reasoning_text and self.verbose_logging:
            logging.debug(f"Captured reasoning ({len(reasoning_text)} chars): {reasoning_text}")

        if reasoning_text and self.reasoning_callback:
            # Skip callback when streaming is active — reasoning was already
            # displayed during the stream via one of two paths:
            #   (a) _fire_reasoning_delta (structured reasoning_content deltas)
            #   (b) _stream_delta tag extraction (<think>/<REASONING_SCRATCHPAD>)
            # When streaming is NOT active, always fire so non-streaming modes
            # (gateway, batch, quiet) still get reasoning.
            # Any reasoning that wasn't shown during streaming is caught by the
            # CLI post-response display fallback (cli.py _reasoning_shown_this_turn).
            if not self.stream_delta_callback:
                try:
                    self.reasoning_callback(reasoning_text)
                except Exception:
                    pass

        msg = {
            "role": "assistant",
            "content": assistant_message.content or "",
            "reasoning": reasoning_text,
            "finish_reason": finish_reason,
        }

        if hasattr(assistant_message, 'reasoning_details') and assistant_message.reasoning_details:
            # Pass reasoning_details back unmodified so providers (OpenRouter,
            # Anthropic, OpenAI) can maintain reasoning continuity across turns.
            # Each provider may include opaque fields (signature, encrypted_content)
            # that must be preserved exactly.
            raw_details = assistant_message.reasoning_details
            preserved = []
            for d in raw_details:
                if isinstance(d, dict):
                    preserved.append(d)
                elif hasattr(d, "__dict__"):
                    preserved.append(d.__dict__)
                elif hasattr(d, "model_dump"):
                    preserved.append(d.model_dump())
            if preserved:
                msg["reasoning_details"] = preserved

        # Codex Responses API: preserve encrypted reasoning items for
        # multi-turn continuity. These get replayed as input on the next turn.
        codex_items = getattr(assistant_message, "codex_reasoning_items", None)
        if codex_items:
            msg["codex_reasoning_items"] = codex_items

        if assistant_message.tool_calls:
            tool_calls = []
            for tool_call in assistant_message.tool_calls:
                raw_id = getattr(tool_call, "id", None)
                call_id = getattr(tool_call, "call_id", None)
                if not isinstance(call_id, str) or not call_id.strip():
                    embedded_call_id, _ = self._split_responses_tool_id(raw_id)
                    call_id = embedded_call_id
                if not isinstance(call_id, str) or not call_id.strip():
                    if isinstance(raw_id, str) and raw_id.strip():
                        call_id = raw_id.strip()
                    else:
                        _fn = getattr(tool_call, "function", None)
                        _fn_name = getattr(_fn, "name", "") if _fn else ""
                        _fn_args = getattr(_fn, "arguments", "{}") if _fn else "{}"
                        call_id = self._deterministic_call_id(_fn_name, _fn_args, len(tool_calls))
                call_id = call_id.strip()

                response_item_id = getattr(tool_call, "response_item_id", None)
                if not isinstance(response_item_id, str) or not response_item_id.strip():
                    _, embedded_response_item_id = self._split_responses_tool_id(raw_id)
                    response_item_id = embedded_response_item_id

                response_item_id = self._derive_responses_function_call_id(
                    call_id,
                    response_item_id if isinstance(response_item_id, str) else None,
                )

                tc_dict = {
                    "id": call_id,
                    "call_id": call_id,
                    "response_item_id": response_item_id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    },
                }
                # Preserve extra_content (e.g. Gemini thought_signature) so it
                # is sent back on subsequent API calls.  Without this, Gemini 3
                # thinking models reject the request with a 400 error.
                extra = getattr(tool_call, "extra_content", None)
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        extra = extra.model_dump()
                    tc_dict["extra_content"] = extra
                tool_calls.append(tc_dict)
            msg["tool_calls"] = tool_calls

        return msg

    @staticmethod
    def _sanitize_tool_calls_for_strict_api(api_msg: dict) -> dict:
        """Strip Codex Responses API fields from tool_calls for strict providers.

        Providers like Mistral, Fireworks, and other strict OpenAI-compatible APIs
        validate the Chat Completions schema and reject unknown fields (call_id,
        response_item_id) with 400 or 422 errors. These fields are preserved in
        the internal message history — this method only modifies the outgoing
        API copy.

        Creates new tool_call dicts rather than mutating in-place, so the
        original messages list retains call_id/response_item_id for Codex
        Responses API compatibility (e.g. if the session falls back to a
        Codex provider later).

        Fields stripped: call_id, response_item_id
        """
        tool_calls = api_msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            return api_msg
        _STRIP_KEYS = {"call_id", "response_item_id"}
        api_msg["tool_calls"] = [
            {k: v for k, v in tc.items() if k not in _STRIP_KEYS}
            if isinstance(tc, dict) else tc
            for tc in tool_calls
        ]
        return api_msg

    def _should_sanitize_tool_calls(self) -> bool:
        """Determine if tool_calls need sanitization for strict APIs.

        Codex Responses API uses fields like call_id and response_item_id
        that are not part of the standard Chat Completions schema. These
        fields must be stripped when calling any other API to avoid
        validation errors (400 Bad Request).

        Returns:
            bool: True if sanitization is needed (non-Codex API), False otherwise.
        """
        return self.api_mode != "codex_responses"

    def flush_memories(self, messages: list = None, min_turns: int = None):
        """Give the model one turn to persist memories before context is lost.

        Called before compression, session reset, or CLI exit. Injects a flush
        message, makes one API call, executes any memory tool calls, then
        strips all flush artifacts from the message list.

        Args:
            messages: The current conversation messages. If None, uses
                      self._session_messages (last run_conversation state).
            min_turns: Minimum user turns required to trigger the flush.
                       None = use config value (flush_min_turns).
                       0 = always flush (used for compression).
        """
        if self._memory_flush_min_turns == 0 and min_turns is None:
            return
        if "memory" not in self.valid_tool_names or not self._memory_store:
            return
        effective_min = min_turns if min_turns is not None else self._memory_flush_min_turns
        if self._user_turn_count < effective_min:
            return

        if messages is None:
            messages = getattr(self, '_session_messages', None)
        if not messages or len(messages) < 3:
            return

        flush_content = (
            "[System: The session is being compressed. "
            "Save anything worth remembering — prioritize user preferences, "
            "corrections, and recurring patterns over task-specific details.]"
        )
        _sentinel = f"__flush_{id(self)}_{time.monotonic()}"
        flush_msg = {"role": "user", "content": flush_content, "_flush_sentinel": _sentinel}
        messages.append(flush_msg)

        try:
            # Build API messages for the flush call
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                if msg.get("role") == "assistant":
                    reasoning = msg.get("reasoning")
                    if reasoning:
                        api_msg["reasoning_content"] = reasoning
                api_msg.pop("reasoning", None)
                api_msg.pop("finish_reason", None)
                api_msg.pop("_flush_sentinel", None)
                api_msg.pop("_thinking_prefill", None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            if self._cached_system_prompt:
                api_messages = [{"role": "system", "content": self._cached_system_prompt}] + api_messages

            # Make one API call with only the memory tool available
            memory_tool_def = None
            for t in (self.tools or []):
                if t.get("function", {}).get("name") == "memory":
                    memory_tool_def = t
                    break

            if not memory_tool_def:
                messages.pop()  # remove flush msg
                return

            # Use auxiliary client for the flush call when available --
            # it's cheaper and avoids Codex Responses API incompatibility.
            from agent.auxiliary_client import call_llm as _call_llm
            _aux_available = True
            try:
                response = _call_llm(
                    task="flush_memories",
                    messages=api_messages,
                    tools=[memory_tool_def],
                    temperature=0.3,
                    max_tokens=5120,
                    # timeout resolved from auxiliary.flush_memories.timeout config
                )
            except RuntimeError:
                _aux_available = False
                response = None

            if not _aux_available and self.api_mode == "codex_responses":
                # No auxiliary client -- use the Codex Responses path directly
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs["tools"] = self._responses_tools([memory_tool_def])
                codex_kwargs["temperature"] = 0.3
                if "max_output_tokens" in codex_kwargs:
                    codex_kwargs["max_output_tokens"] = 5120
                response = self._run_codex_stream(codex_kwargs)
            elif not _aux_available and self.api_mode == "anthropic_messages":
                # Native Anthropic — use the Anthropic client directly
                from agent.anthropic_adapter import build_anthropic_kwargs as _build_ant_kwargs
                ant_kwargs = _build_ant_kwargs(
                    model=self.model, messages=api_messages,
                    tools=[memory_tool_def], max_tokens=5120,
                    reasoning_config=None,
                    preserve_dots=self._anthropic_preserve_dots(),
                )
                response = self._anthropic_messages_create(ant_kwargs)
            elif not _aux_available:
                api_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                    "tools": [memory_tool_def],
                    "temperature": 0.3,
                    **self._max_tokens_param(5120),
                }
                from agent.auxiliary_client import _get_task_timeout
                response = self._ensure_primary_openai_client(reason="flush_memories").chat.completions.create(
                    **api_kwargs, timeout=_get_task_timeout("flush_memories")
                )

            # Extract tool calls from the response, handling all API formats
            tool_calls = []
            if self.api_mode == "codex_responses" and not _aux_available:
                assistant_msg, _ = self._normalize_codex_response(response)
                if assistant_msg and assistant_msg.tool_calls:
                    tool_calls = assistant_msg.tool_calls
            elif self.api_mode == "anthropic_messages" and not _aux_available:
                from agent.anthropic_adapter import normalize_anthropic_response as _nar_flush
                _flush_msg, _ = _nar_flush(response, strip_tool_prefix=self._is_anthropic_oauth)
                if _flush_msg and _flush_msg.tool_calls:
                    tool_calls = _flush_msg.tool_calls
            elif hasattr(response, "choices") and response.choices:
                assistant_message = response.choices[0].message
                if assistant_message.tool_calls:
                    tool_calls = assistant_message.tool_calls

            for tc in tool_calls:
                if tc.function.name == "memory":
                    try:
                        args = json.loads(tc.function.arguments)
                        flush_target = args.get("target", "memory")
                        from tools.memory_tool import memory_tool as _memory_tool
                        _memory_tool(
                            action=args.get("action"),
                            target=flush_target,
                            content=args.get("content"),
                            old_text=args.get("old_text"),
                            store=self._memory_store,
                        )
                        if not self.quiet_mode:
                            print(f"  🧠 Memory flush: saved to {args.get('target', 'memory')}")
                    except Exception as e:
                        logger.debug("Memory flush tool call failed: %s", e)
        except Exception as e:
            logger.debug("Memory flush API call failed: %s", e)
        finally:
            # Strip flush artifacts: remove everything from the flush message onward.
            # Use sentinel marker instead of identity check for robustness.
            while messages and messages[-1].get("_flush_sentinel") != _sentinel:
                messages.pop()
                if not messages:
                    break
            if messages and messages[-1].get("_flush_sentinel") == _sentinel:
                messages.pop()

    def _compress_context(self, messages: list, system_message: str, *, approx_tokens: int = None, task_id: str = "default", focus_topic: str = None) -> tuple:
        """Compress conversation context and split the session in SQLite.

        Args:
            focus_topic: Optional focus string for guided compression — the
                summariser will prioritise preserving information related to
                this topic.  Inspired by Claude Code's ``/compact <focus>``.

        Returns:
            (compressed_messages, new_system_prompt) tuple
        """
        _pre_msg_count = len(messages)
        logger.info(
            "context compression started: session=%s messages=%d tokens=~%s model=%s focus=%r",
            self.session_id or "none", _pre_msg_count,
            f"{approx_tokens:,}" if approx_tokens else "unknown", self.model,
            focus_topic,
        )
        # Pre-compression memory flush: let the model save memories before they're lost
        self.flush_memories(messages, min_turns=0)

        # Notify external memory provider before compression discards context
        if self._memory_manager:
            try:
                self._memory_manager.on_pre_compress(messages)
            except Exception:
                pass

        compressed = self.context_compressor.compress(messages, current_tokens=approx_tokens, focus_topic=focus_topic)

        todo_snapshot = self._todo_store.format_for_injection()
        if todo_snapshot:
            compressed.append({"role": "user", "content": todo_snapshot})

        self._invalidate_system_prompt()
        new_system_prompt = self._build_system_prompt(system_message)
        self._cached_system_prompt = new_system_prompt

        if self._session_db:
            try:
                # Propagate title to the new session with auto-numbering
                old_title = self._session_db.get_session_title(self.session_id)
                self._session_db.end_session(self.session_id, "compression")
                old_session_id = self.session_id
                self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
                # Update session_log_file to point to the new session's JSON file
                self.session_log_file = self.logs_dir / f"session_{self.session_id}.json"
                self._session_db.create_session(
                    session_id=self.session_id,
                    source=self.platform or os.environ.get("HERMES_SESSION_SOURCE", "cli"),
                    model=self.model,
                    parent_session_id=old_session_id,
                )
                # Auto-number the title for the continuation session
                if old_title:
                    try:
                        new_title = self._session_db.get_next_title_in_lineage(old_title)
                        self._session_db.set_session_title(self.session_id, new_title)
                    except (ValueError, Exception) as e:
                        logger.debug("Could not propagate title on compression: %s", e)
                self._session_db.update_system_prompt(self.session_id, new_system_prompt)
                # Reset flush cursor — new session starts with no messages written
                self._last_flushed_db_idx = 0
            except Exception as e:
                logger.warning("Session DB compression split failed — new session will NOT be indexed: %s", e)

        # Warn on repeated compressions (quality degrades with each pass)
        _cc = self.context_compressor.compression_count
        if _cc >= 2:
            self._vprint(
                f"{self.log_prefix}⚠️  Session compressed {_cc} times — "
                f"accuracy may degrade. Consider /new to start fresh.",
                force=True,
            )

        # Update token estimate after compaction so pressure calculations
        # use the post-compression count, not the stale pre-compression one.
        _compressed_est = (
            estimate_tokens_rough(new_system_prompt)
            + estimate_messages_tokens_rough(compressed)
        )
        self.context_compressor.last_prompt_tokens = _compressed_est
        self.context_compressor.last_completion_tokens = 0

        # Only reset the pressure warning if compression actually brought
        # us below the warning level (85% of threshold).  When compression
        # can't reduce enough (e.g. threshold is very low, or system prompt
        # alone exceeds the warning level), keep the tier set to prevent
        # spamming the user with repeated warnings every loop iteration.
        if self.context_compressor.threshold_tokens > 0:
            _post_progress = _compressed_est / self.context_compressor.threshold_tokens
            if _post_progress < 0.85:
                self._context_pressure_warned_at = 0.0
                # Clear class-level dedup for this session so a fresh
                # warning cycle can start if context grows again.
                _sid = self.session_id or "default"
                AIAgent._context_pressure_last_warned.pop(_sid, None)

        # Clear the file-read dedup cache.  After compression the original
        # read content is summarised away — if the model re-reads the same
        # file it needs the full content, not a "file unchanged" stub.
        try:
            from tools.file_tools import reset_file_dedup
            reset_file_dedup(task_id)
        except Exception:
            pass

        logger.info(
            "context compression done: session=%s messages=%d->%d tokens=~%s",
            self.session_id or "none", _pre_msg_count, len(compressed),
            f"{_compressed_est:,}",
        )
        return compressed, new_system_prompt

    def _execute_tool_calls(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls from the assistant message and append results to messages.

        Dispatches to concurrent execution only for batches that look
        independent: read-only tools may always share the parallel path, while
        file reads/writes may do so only when their target paths do not overlap.
        """
        tool_calls = assistant_message.tool_calls

        # Allow _vprint during tool execution even with stream consumers
        self._executing_tools = True
        try:
            if not _should_parallelize_tool_batch(tool_calls):
                return self._execute_tool_calls_sequential(
                    assistant_message, messages, effective_task_id, api_call_count
                )

            return self._execute_tool_calls_concurrent(
                assistant_message, messages, effective_task_id, api_call_count
            )
        finally:
            self._executing_tools = False

    def _invoke_tool(self, function_name: str, function_args: dict, effective_task_id: str,
                     tool_call_id: Optional[str] = None) -> str:
        """Invoke a single tool and return the result string. No display logic.

        Handles both agent-level tools (todo, memory, etc.) and registry-dispatched
        tools. Used by the concurrent execution path; the sequential path retains
        its own inline invocation for backward-compatible display handling.
        """
        if function_name == "todo":
            from tools.todo_tool import todo_tool as _todo_tool
            return _todo_tool(
                todos=function_args.get("todos"),
                merge=function_args.get("merge", False),
                store=self._todo_store,
            )
        elif function_name == "session_search":
            if not self._session_db:
                return json.dumps({"success": False, "error": "Session database not available."})
            from tools.session_search_tool import session_search as _session_search
            return _session_search(
                query=function_args.get("query", ""),
                role_filter=function_args.get("role_filter"),
                limit=function_args.get("limit", 3),
                db=self._session_db,
                current_session_id=self.session_id,
            )
        elif function_name == "memory":
            target = function_args.get("target", "memory")
            from tools.memory_tool import memory_tool as _memory_tool
            result = _memory_tool(
                action=function_args.get("action"),
                target=target,
                content=function_args.get("content"),
                old_text=function_args.get("old_text"),
                store=self._memory_store,
            )
            # Bridge: notify external memory provider of built-in memory writes
            if self._memory_manager and function_args.get("action") in ("add", "replace"):
                try:
                    self._memory_manager.on_memory_write(
                        function_args.get("action", ""),
                        target,
                        function_args.get("content", ""),
                    )
                except Exception:
                    pass
            return result
        elif self._memory_manager and self._memory_manager.has_tool(function_name):
            return self._memory_manager.handle_tool_call(function_name, function_args)
        elif function_name == "clarify":
            from tools.clarify_tool import clarify_tool as _clarify_tool
            return _clarify_tool(
                question=function_args.get("question", ""),
                choices=function_args.get("choices"),
                callback=self.clarify_callback,
            )
        elif function_name == "delegate_task":
            from tools.delegate_tool import delegate_task as _delegate_task
            return _delegate_task(
                goal=function_args.get("goal"),
                context=function_args.get("context"),
                toolsets=function_args.get("toolsets"),
                tasks=function_args.get("tasks"),
                max_iterations=function_args.get("max_iterations"),
                parent_agent=self,
            )
        else:
            return handle_function_call(
                function_name, function_args, effective_task_id,
                tool_call_id=tool_call_id,
                session_id=self.session_id or "",
                enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
            )

    def _execute_tool_calls_concurrent(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute multiple tool calls concurrently using a thread pool.

        Results are collected in the original tool-call order and appended to
        messages so the API sees them in the expected sequence.
        """
        tool_calls = assistant_message.tool_calls
        num_tools = len(tool_calls)

        # ── Pre-flight: interrupt check ──────────────────────────────────
        if self._interrupt_requested:
            print(f"{self.log_prefix}⚡ Interrupt: skipping {num_tools} tool call(s)")
            for tc in tool_calls:
                messages.append({
                    "role": "tool",
                    "content": f"[Tool execution cancelled — {tc.function.name} was skipped due to user interrupt]",
                    "tool_call_id": tc.id,
                })
            return

        # ── Parse args + pre-execution bookkeeping ───────────────────────
        parsed_calls = []  # list of (tool_call, function_name, function_args)
        for tool_call in tool_calls:
            function_name = tool_call.function.name

            # Reset nudge counters
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            # Checkpoint for file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(work_dir, f"before {function_name}")
                except Exception:
                    pass

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass

            parsed_calls.append((tool_call, function_name, function_args))

        # ── Logging / callbacks ──────────────────────────────────────────
        tool_names_str = ", ".join(name for _, name, _ in parsed_calls)
        if not self.quiet_mode:
            print(f"  ⚡ Concurrent: {num_tools} tool calls — {tool_names_str}")
            for i, (tc, name, args) in enumerate(parsed_calls, 1):
                args_str = json.dumps(args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {name}({list(args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {name}({list(args.keys())}) - {args_preview}")

        for tc, name, args in parsed_calls:
            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(name, args)
                    self.tool_progress_callback("tool.started", name, preview, args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

        for tc, name, args in parsed_calls:
            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tc.id, name, args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

        # ── Concurrent execution ─────────────────────────────────────────
        # Each slot holds (function_name, function_args, function_result, duration, error_flag)
        results = [None] * num_tools

        def _run_tool(index, tool_call, function_name, function_args):
            """Worker function executed in a thread."""
            start = time.time()
            try:
                result = self._invoke_tool(function_name, function_args, effective_task_id, tool_call.id)
            except Exception as tool_error:
                result = f"Error executing tool '{function_name}': {tool_error}"
                logger.error("_invoke_tool raised for %s: %s", function_name, tool_error, exc_info=True)
            duration = time.time() - start
            is_error, _ = _detect_tool_failure(function_name, result)
            if is_error:
                logger.info("tool %s failed (%.2fs): %s", function_name, duration, result[:200])
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, duration, len(result))
            results[index] = (function_name, function_args, result, duration, is_error)

        # Start spinner for CLI mode (skip when TUI handles tool progress)
        spinner = None
        if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
            face = random.choice(KawaiiSpinner.KAWAII_WAITING)
            spinner = KawaiiSpinner(f"{face} ⚡ running {num_tools} tools concurrently", spinner_type='dots', print_fn=self._print_fn)
            spinner.start()

        try:
            max_workers = min(num_tools, _MAX_TOOL_WORKERS)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for i, (tc, name, args) in enumerate(parsed_calls):
                    f = executor.submit(_run_tool, i, tc, name, args)
                    futures.append(f)

                # Wait for all to complete (exceptions are captured inside _run_tool)
                concurrent.futures.wait(futures)
        finally:
            if spinner:
                # Build a summary message for the spinner stop
                completed = sum(1 for r in results if r is not None)
                total_dur = sum(r[3] for r in results if r is not None)
                spinner.stop(f"⚡ {completed}/{num_tools} tools completed in {total_dur:.1f}s total")

        # ── Post-execution: display per-tool results ─────────────────────
        for i, (tc, name, args) in enumerate(parsed_calls):
            r = results[i]
            if r is None:
                # Shouldn't happen, but safety fallback
                function_result = f"Error executing tool '{name}': thread did not return a result"
                tool_duration = 0.0
            else:
                function_name, function_args, function_result, tool_duration, is_error = r

                if is_error:
                    result_preview = function_result[:200] if len(function_result) > 200 else function_result
                    logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)

                if self.tool_progress_callback:
                    try:
                        self.tool_progress_callback(
                            "tool.completed", function_name, None, None,
                            duration=tool_duration, is_error=is_error,
                        )
                    except Exception as cb_err:
                        logging.debug(f"Tool progress callback error: {cb_err}")

                if self.verbose_logging:
                    logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                    logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            # Print cute message per tool
            if self._should_emit_quiet_tool_messages():
                cute_msg = _get_cute_tool_message_impl(name, args, tool_duration, result=function_result)
                self._safe_print(f"  {cute_msg}")
            elif not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i+1} completed in {tool_duration:.2f}s - {response_preview}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {name} ({tool_duration:.1f}s)")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tc.id, name, args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=name,
                tool_use_id=tc.id,
                env=get_active_env(effective_task_id),
            )

            subdir_hints = self._subdirectory_hints.check_tool_call(name, args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tc.id,
            }
            messages.append(tool_msg)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools = len(parsed_calls)
        if num_tools > 0:
            turn_tool_msgs = messages[-num_tools:]
            enforce_turn_budget(turn_tool_msgs, env=get_active_env(effective_task_id))

    def _execute_tool_calls_sequential(self, assistant_message, messages: list, effective_task_id: str, api_call_count: int = 0) -> None:
        """Execute tool calls sequentially (original behavior). Used for single calls or interactive tools."""
        for i, tool_call in enumerate(assistant_message.tool_calls, 1):
            # SAFETY: check interrupt BEFORE starting each tool.
            # If the user sent "stop" during a previous tool's execution,
            # do NOT start any more tools -- skip them all immediately.
            if self._interrupt_requested:
                remaining_calls = assistant_message.tool_calls[i-1:]
                if remaining_calls:
                    self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {len(remaining_calls)} tool call(s)", force=True)
                for skipped_tc in remaining_calls:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution cancelled — {skipped_name} was skipped due to user interrupt]",
                        "tool_call_id": skipped_tc.id,
                    }
                    messages.append(skip_msg)
                break

            function_name = tool_call.function.name

            # Reset nudge counters when the relevant tool is actually used
            if function_name == "memory":
                self._turns_since_memory = 0
            elif function_name == "skill_manage":
                self._iters_since_skill = 0

            try:
                function_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                logging.warning(f"Unexpected JSON error after validation: {e}")
                function_args = {}
            if not isinstance(function_args, dict):
                function_args = {}

            if not self.quiet_mode:
                args_str = json.dumps(function_args, ensure_ascii=False)
                if self.verbose_logging:
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())})")
                    print(f"     Args: {args_str}")
                else:
                    args_preview = args_str[:self.log_prefix_chars] + "..." if len(args_str) > self.log_prefix_chars else args_str
                    print(f"  📞 Tool {i}: {function_name}({list(function_args.keys())}) - {args_preview}")

            self._current_tool = function_name
            self._touch_activity(f"executing tool: {function_name}")

            # Set activity callback for long-running tool execution (terminal
            # commands, etc.) so the gateway's inactivity monitor doesn't kill
            # the agent while a command is running.
            try:
                from tools.environments.base import set_activity_callback
                set_activity_callback(self._touch_activity)
            except Exception:
                pass

            if self.tool_progress_callback:
                try:
                    preview = _build_tool_preview(function_name, function_args)
                    self.tool_progress_callback("tool.started", function_name, preview, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            if self.tool_start_callback:
                try:
                    self.tool_start_callback(tool_call.id, function_name, function_args)
                except Exception as cb_err:
                    logging.debug(f"Tool start callback error: {cb_err}")

            # Checkpoint: snapshot working dir before file-mutating tools
            if function_name in ("write_file", "patch") and self._checkpoint_mgr.enabled:
                try:
                    file_path = function_args.get("path", "")
                    if file_path:
                        work_dir = self._checkpoint_mgr.get_working_dir_for_path(file_path)
                        self._checkpoint_mgr.ensure_checkpoint(
                            work_dir, f"before {function_name}"
                        )
                except Exception:
                    pass  # never block tool execution

            # Checkpoint before destructive terminal commands
            if function_name == "terminal" and self._checkpoint_mgr.enabled:
                try:
                    cmd = function_args.get("command", "")
                    if _is_destructive_command(cmd):
                        cwd = function_args.get("workdir") or os.getenv("TERMINAL_CWD", os.getcwd())
                        self._checkpoint_mgr.ensure_checkpoint(
                            cwd, f"before terminal: {cmd[:60]}"
                        )
                except Exception:
                    pass  # never block tool execution

            tool_start_time = time.time()

            if function_name == "todo":
                from tools.todo_tool import todo_tool as _todo_tool
                function_result = _todo_tool(
                    todos=function_args.get("todos"),
                    merge=function_args.get("merge", False),
                    store=self._todo_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('todo', function_args, tool_duration, result=function_result)}")
            elif function_name == "session_search":
                if not self._session_db:
                    function_result = json.dumps({"success": False, "error": "Session database not available."})
                else:
                    from tools.session_search_tool import session_search as _session_search
                    function_result = _session_search(
                        query=function_args.get("query", ""),
                        role_filter=function_args.get("role_filter"),
                        limit=function_args.get("limit", 3),
                        db=self._session_db,
                        current_session_id=self.session_id,
                    )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('session_search', function_args, tool_duration, result=function_result)}")
            elif function_name == "memory":
                target = function_args.get("target", "memory")
                from tools.memory_tool import memory_tool as _memory_tool
                function_result = _memory_tool(
                    action=function_args.get("action"),
                    target=target,
                    content=function_args.get("content"),
                    old_text=function_args.get("old_text"),
                    store=self._memory_store,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('memory', function_args, tool_duration, result=function_result)}")
            elif function_name == "clarify":
                from tools.clarify_tool import clarify_tool as _clarify_tool
                function_result = _clarify_tool(
                    question=function_args.get("question", ""),
                    choices=function_args.get("choices"),
                    callback=self.clarify_callback,
                )
                tool_duration = time.time() - tool_start_time
                if self._should_emit_quiet_tool_messages():
                    self._vprint(f"  {_get_cute_tool_message_impl('clarify', function_args, tool_duration, result=function_result)}")
            elif function_name == "delegate_task":
                from tools.delegate_tool import delegate_task as _delegate_task
                tasks_arg = function_args.get("tasks")
                if tasks_arg and isinstance(tasks_arg, list):
                    spinner_label = f"🔀 delegating {len(tasks_arg)} tasks"
                else:
                    goal_preview = (function_args.get("goal") or "")[:30]
                    spinner_label = f"🔀 {goal_preview}" if goal_preview else "🔀 delegating"
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    spinner = KawaiiSpinner(f"{face} {spinner_label}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                self._delegate_spinner = spinner
                _delegate_result = None
                try:
                    function_result = _delegate_task(
                        goal=function_args.get("goal"),
                        context=function_args.get("context"),
                        toolsets=function_args.get("toolsets"),
                        tasks=tasks_arg,
                        max_iterations=function_args.get("max_iterations"),
                        parent_agent=self,
                    )
                    _delegate_result = function_result
                finally:
                    self._delegate_spinner = None
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl('delegate_task', function_args, tool_duration, result=_delegate_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self._context_engine_tool_names and function_name in self._context_engine_tool_names:
                # Context engine tools (lcm_grep, lcm_describe, lcm_expand, etc.)
                spinner = None
                if self.quiet_mode and not self.tool_progress_callback:
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _ce_result = None
                try:
                    function_result = self.context_compressor.handle_tool_call(function_name, function_args, messages=messages)
                    _ce_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Context engine tool '{function_name}' failed: {tool_error}"})
                    logger.error("context_engine.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_ce_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self.quiet_mode:
                        self._vprint(f"  {cute_msg}")
            elif self._memory_manager and self._memory_manager.has_tool(function_name):
                # Memory provider tools (hindsight_retain, honcho_search, etc.)
                # These are not in the tool registry — route through MemoryManager.
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _mem_result = None
                try:
                    function_result = self._memory_manager.handle_tool_call(function_name, function_args)
                    _mem_result = function_result
                except Exception as tool_error:
                    function_result = json.dumps({"error": f"Memory tool '{function_name}' failed: {tool_error}"})
                    logger.error("memory_manager.handle_tool_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_mem_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            elif self.quiet_mode:
                spinner = None
                if self._should_emit_quiet_tool_messages() and self._should_start_quiet_spinner():
                    face = random.choice(KawaiiSpinner.KAWAII_WAITING)
                    emoji = _get_tool_emoji(function_name)
                    preview = _build_tool_preview(function_name, function_args) or function_name
                    spinner = KawaiiSpinner(f"{face} {emoji} {preview}", spinner_type='dots', print_fn=self._print_fn)
                    spinner.start()
                _spinner_result = None
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                    _spinner_result = function_result
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                finally:
                    tool_duration = time.time() - tool_start_time
                    cute_msg = _get_cute_tool_message_impl(function_name, function_args, tool_duration, result=_spinner_result)
                    if spinner:
                        spinner.stop(cute_msg)
                    elif self._should_emit_quiet_tool_messages():
                        self._vprint(f"  {cute_msg}")
            else:
                try:
                    function_result = handle_function_call(
                        function_name, function_args, effective_task_id,
                        tool_call_id=tool_call.id,
                        session_id=self.session_id or "",
                        enabled_tools=list(self.valid_tool_names) if self.valid_tool_names else None,
                    )
                except Exception as tool_error:
                    function_result = f"Error executing tool '{function_name}': {tool_error}"
                    logger.error("handle_function_call raised for %s: %s", function_name, tool_error, exc_info=True)
                tool_duration = time.time() - tool_start_time

            result_preview = function_result if self.verbose_logging else (
                function_result[:200] if len(function_result) > 200 else function_result
            )

            # Log tool errors to the persistent error log so [error] tags
            # in the UI always have a corresponding detailed entry on disk.
            _is_error_result, _ = _detect_tool_failure(function_name, function_result)
            if _is_error_result:
                logger.warning("Tool %s returned error (%.2fs): %s", function_name, tool_duration, result_preview)
            else:
                logger.info("tool %s completed (%.2fs, %d chars)", function_name, tool_duration, len(function_result))

            if self.tool_progress_callback:
                try:
                    self.tool_progress_callback(
                        "tool.completed", function_name, None, None,
                        duration=tool_duration, is_error=_is_error_result,
                    )
                except Exception as cb_err:
                    logging.debug(f"Tool progress callback error: {cb_err}")

            self._current_tool = None
            self._touch_activity(f"tool completed: {function_name} ({tool_duration:.1f}s)")

            if self.verbose_logging:
                logging.debug(f"Tool {function_name} completed in {tool_duration:.2f}s")
                logging.debug(f"Tool result ({len(function_result)} chars): {function_result}")

            if self.tool_complete_callback:
                try:
                    self.tool_complete_callback(tool_call.id, function_name, function_args, function_result)
                except Exception as cb_err:
                    logging.debug(f"Tool complete callback error: {cb_err}")

            function_result = maybe_persist_tool_result(
                content=function_result,
                tool_name=function_name,
                tool_use_id=tool_call.id,
                env=get_active_env(effective_task_id),
            )

            # Discover subdirectory context files from tool arguments
            subdir_hints = self._subdirectory_hints.check_tool_call(function_name, function_args)
            if subdir_hints:
                function_result += subdir_hints

            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)

            if not self.quiet_mode:
                if self.verbose_logging:
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s")
                    print(f"     Result: {function_result}")
                else:
                    response_preview = function_result[:self.log_prefix_chars] + "..." if len(function_result) > self.log_prefix_chars else function_result
                    print(f"  ✅ Tool {i} completed in {tool_duration:.2f}s - {response_preview}")

            if self._interrupt_requested and i < len(assistant_message.tool_calls):
                remaining = len(assistant_message.tool_calls) - i
                self._vprint(f"{self.log_prefix}⚡ Interrupt: skipping {remaining} remaining tool call(s)", force=True)
                for skipped_tc in assistant_message.tool_calls[i:]:
                    skipped_name = skipped_tc.function.name
                    skip_msg = {
                        "role": "tool",
                        "content": f"[Tool execution skipped — {skipped_name} was not started. User sent a new message]",
                        "tool_call_id": skipped_tc.id
                    }
                    messages.append(skip_msg)
                break

            if self.tool_delay > 0 and i < len(assistant_message.tool_calls):
                time.sleep(self.tool_delay)

        # ── Per-turn aggregate budget enforcement ─────────────────────────
        num_tools_seq = len(assistant_message.tool_calls)
        if num_tools_seq > 0:
            enforce_turn_budget(messages[-num_tools_seq:], env=get_active_env(effective_task_id))



    def _emit_context_pressure(self, compaction_progress: float, compressor) -> None:
        """Notify the user that context is approaching the compaction threshold.

        Args:
            compaction_progress: How close to compaction (0.0–1.0, where 1.0 = fires).
            compressor: The ContextCompressor instance (for threshold/context info).

        Purely user-facing — does NOT modify the message stream.
        For CLI: prints a formatted line with a progress bar.
        For gateway: fires status_callback so the platform can send a chat message.
        """
        from agent.display import format_context_pressure, format_context_pressure_gateway

        threshold_pct = compressor.threshold_tokens / compressor.context_length if compressor.context_length else 0.5

        # CLI output — always shown (these are user-facing status notifications,
        # not verbose debug output, so they bypass quiet_mode).
        # Gateway users also get the callback below.
        if self.platform in (None, "cli"):
            line = format_context_pressure(
                compaction_progress=compaction_progress,
                threshold_tokens=compressor.threshold_tokens,
                threshold_percent=threshold_pct,
                compression_enabled=self.compression_enabled,
            )
            self._safe_print(line)

        # Gateway / external consumers
        if self.status_callback:
            try:
                msg = format_context_pressure_gateway(
                    compaction_progress=compaction_progress,
                    threshold_percent=threshold_pct,
                    compression_enabled=self.compression_enabled,
                )
                self.status_callback("context_pressure", msg)
            except Exception:
                logger.debug("status_callback error in context pressure", exc_info=True)

    def _handle_max_iterations(self, messages: list, api_call_count: int) -> str:
        """Request a summary when max iterations are reached. Returns the final response text."""
        print(f"⚠️  Reached maximum iterations ({self.max_iterations}). Requesting summary...")

        summary_request = (
            "You've reached the maximum number of tool-calling iterations allowed. "
            "Please provide a final response summarizing what you've found and accomplished so far, "
            "without calling any more tools."
        )
        messages.append({"role": "user", "content": summary_request})

        try:
            # Build API messages, stripping internal-only fields
            # (finish_reason, reasoning) that strict APIs like Mistral reject with 422
            _needs_sanitize = self._should_sanitize_tool_calls()
            api_messages = []
            for msg in messages:
                api_msg = msg.copy()
                for internal_field in ("reasoning", "finish_reason", "_thinking_prefill"):
                    api_msg.pop(internal_field, None)
                if _needs_sanitize:
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                api_messages.append(api_msg)

            effective_system = self._cached_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            summary_extra_body = {}
            _is_nous = "nousresearch" in self._base_url_lower
            if self._supports_reasoning_extra_body():
                if self.reasoning_config is not None:
                    summary_extra_body["reasoning"] = self.reasoning_config
                else:
                    summary_extra_body["reasoning"] = {
                        "enabled": True,
                        "effort": "medium"
                    }
            if _is_nous:
                summary_extra_body["tags"] = ["product=hermes-agent"]

            if self.api_mode == "codex_responses":
                codex_kwargs = self._build_api_kwargs(api_messages)
                codex_kwargs.pop("tools", None)
                summary_response = self._run_codex_stream(codex_kwargs)
                assistant_message, _ = self._normalize_codex_response(summary_response)
                final_response = (assistant_message.content or "").strip() if assistant_message else ""
            else:
                summary_kwargs = {
                    "model": self.model,
                    "messages": api_messages,
                }
                if self.max_tokens is not None:
                    summary_kwargs.update(self._max_tokens_param(self.max_tokens))

                # Include provider routing preferences
                provider_preferences = {}
                if self.providers_allowed:
                    provider_preferences["only"] = self.providers_allowed
                if self.providers_ignored:
                    provider_preferences["ignore"] = self.providers_ignored
                if self.providers_order:
                    provider_preferences["order"] = self.providers_order
                if self.provider_sort:
                    provider_preferences["sort"] = self.provider_sort
                if provider_preferences:
                    summary_extra_body["provider"] = provider_preferences

                if summary_extra_body:
                    summary_kwargs["extra_body"] = summary_extra_body

                if self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak, normalize_anthropic_response as _nar
                    _ant_kw = _bak(model=self.model, messages=api_messages, tools=None,
                                   max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                   is_oauth=self._is_anthropic_oauth,
                                   preserve_dots=self._anthropic_preserve_dots())
                    summary_response = self._anthropic_messages_create(_ant_kw)
                    _msg, _ = _nar(summary_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_msg.content or "").strip()
                else:
                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

            if final_response:
                if "<think>" in final_response:
                    final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                if final_response:
                    messages.append({"role": "assistant", "content": final_response})
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."
            else:
                # Retry summary generation
                if self.api_mode == "codex_responses":
                    codex_kwargs = self._build_api_kwargs(api_messages)
                    codex_kwargs.pop("tools", None)
                    retry_response = self._run_codex_stream(codex_kwargs)
                    retry_msg, _ = self._normalize_codex_response(retry_response)
                    final_response = (retry_msg.content or "").strip() if retry_msg else ""
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import build_anthropic_kwargs as _bak2, normalize_anthropic_response as _nar2
                    _ant_kw2 = _bak2(model=self.model, messages=api_messages, tools=None,
                                    is_oauth=self._is_anthropic_oauth,
                                    max_tokens=self.max_tokens, reasoning_config=self.reasoning_config,
                                    preserve_dots=self._anthropic_preserve_dots())
                    retry_response = self._anthropic_messages_create(_ant_kw2)
                    _retry_msg, _ = _nar2(retry_response, strip_tool_prefix=self._is_anthropic_oauth)
                    final_response = (_retry_msg.content or "").strip()
                else:
                    summary_kwargs = {
                        "model": self.model,
                        "messages": api_messages,
                    }
                    if self.max_tokens is not None:
                        summary_kwargs.update(self._max_tokens_param(self.max_tokens))
                    if summary_extra_body:
                        summary_kwargs["extra_body"] = summary_extra_body

                    summary_response = self._ensure_primary_openai_client(reason="iteration_limit_summary_retry").chat.completions.create(**summary_kwargs)

                    if summary_response.choices and summary_response.choices[0].message.content:
                        final_response = summary_response.choices[0].message.content
                    else:
                        final_response = ""

                if final_response:
                    if "<think>" in final_response:
                        final_response = re.sub(r'<think>.*?</think>\s*', '', final_response, flags=re.DOTALL).strip()
                    if final_response:
                        messages.append({"role": "assistant", "content": final_response})
                    else:
                        final_response = "I reached the iteration limit and couldn't generate a summary."
                else:
                    final_response = "I reached the iteration limit and couldn't generate a summary."

        except Exception as e:
            logging.warning(f"Failed to get summary response: {e}")
            final_response = f"I reached the maximum iterations ({self.max_iterations}) but couldn't summarize. Error: {str(e)}"

        return final_response

    def run_conversation(
        self,
        user_message: str,
        system_message: str = None,
        conversation_history: List[Dict[str, Any]] = None,
        task_id: str = None,
        stream_callback: Optional[callable] = None,
        persist_user_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        运行完整的对话流程，包含工具调用，直到对话完成。

        参数说明:
            user_message (str): 用户输入的问题或消息
            system_message (str): 自定义系统消息（可选，如果提供将覆盖临时系统提示）
            conversation_history (List[Dict]): 先前的对话历史（可选）
            task_id (str): 本次任务的唯一标识符，用于在并发任务间隔离虚拟机（可选，如果未提供会自动生成）
            stream_callback: 可选回调，流式输出每一段文本时被调用。
                用于 TTS 管道，在完整回复生成前就开始音频合成。
                为空（默认）时，API 使用标准的非流式模式。
            persist_user_message: 可选的已清洗用户消息，用于当 user_message 含有仅API使用的前缀时
                存入对话历史或用于后续的预抓取队列工作。

        返回值:
            Dict: 包含最终回复和消息历史的完整对话结果
        """
   
        # 防止 stdio 因管道断裂（如 systemd/headless/daemon 环境下）导致 OSError。
        # 安装一次，若流正常则无影响，能避免写入时崩溃。
        _install_safe_stdio()

        # 给本线程的所有日志打上 session ID 标签，方便用
        # ``hermes logs --session <id>`` 过滤单次会话日志。
        from hermes_logging import set_session_context
        set_session_context(self.session_id)

        # 如果上一轮触发了降级机制，这里恢复主模型环境，
        # 让本轮能够用期望模型重新尝试。
        # 如果未激活降级（如 gateway、第一轮等），则无影响。
        self._restore_primary_runtime()

        # 清理用户输入中的代理字符（如单独出现的代理对）。
        # 从富文本编辑器（如 Google Docs、Word）粘贴内容可能包含无效代理字符，
        # 这些会导致 OpenAI SDK 进行 JSON 序列化时崩溃。
        if isinstance(user_message, str):
            user_message = _sanitize_surrogates(user_message)
        if isinstance(persist_user_message, str):
            persist_user_message = _sanitize_surrogates(persist_user_message)

        # 存储流式回调，让 _interruptible_api_call 能读取到
        self._stream_callback = stream_callback
        self._persist_user_message_idx = None
        self._persist_user_message_override = persist_user_message
        # 如果没有传 task_id，为本次任务生成唯一标识，保证并发任务的隔离
        effective_task_id = task_id or str(uuid.uuid4())
        
        # 每回合重置重试计数与迭代预算，
        # 避免上轮 subagent 消耗影响下一轮正常额度。
        self._invalid_tool_retries = 0
        self._invalid_json_retries = 0
        self._empty_content_retries = 0
        self._incomplete_scratchpad_retries = 0
        self._codex_incomplete_retries = 0
        self._thinking_prefill_retries = 0
        self._last_content_with_tools = None
        self._mute_post_response = False
        self._unicode_sanitization_passes = 0

        # 回合前检查连接健康，发现残留的 TCP 连接（如上游故障遗留）就主动清理，
        # 避免下次 API 请求卡死在僵尸 socket 上。
        if self.api_mode != "anthropic_messages":
            try:
                if self._cleanup_dead_connections():
                    self._emit_status(
                        "🔌 检测到上轮出现供应商连接遗留，已自动清理，继续用新连接。"
                    )
            except Exception:
                pass
        # 若 gateway 模式，回放压缩警告（因为 __init__ 时未完成回调 wiring）。
        if self._compression_warning:
            self._replay_compression_warning()
            self._compression_warning = None  # 只发一次

        # 注意：_turns_since_memory 与 _iters_since_skill 不在此处重置。
        # 它们在 __init__ 初始化，需跨 run_conversation 多次调用累加（如 CLI 模式下 nudge 逻辑）。
        self.iteration_budget = IterationBudget(self.max_iterations)

        # 记录对话回合开始，方便调试/可观测
        _msg_preview = (user_message[:80] + "...") if len(user_message) > 80 else user_message
        _msg_preview = _msg_preview.replace("\n", " ")
        logger.info(
            "conversation turn: session=%s model=%s provider=%s platform=%s history=%d msg=%r",
            self.session_id or "none", self.model, self.provider or "unknown",
            self.platform or "unknown", len(conversation_history or []),
            _msg_preview,
        )

        # 初始化对话消息（副本，避免修改调用者 List）
        messages = list(conversation_history) if conversation_history else []

        # 从历史记录恢复 todo 状态（如 gateway 每条消息创建新 AIAgent，本地 store 为空，需要恢复最近的 todo 工具回复部分）
        if conversation_history and not self._todo_store.has_items():
            self._hydrate_todo_store(conversation_history)
        
        # 首轮 few-shot/priming 消息，仅在 API 调用时注入，不进 messages。
        # 保持临时性，不写入 session DB、日志或批量轨迹；续会话/多次 API 调用也自动重用。
        
        # 追踪用户回合数，用于记忆刷新和定期触发提示
        self._user_turn_count += 1

        # 保存未注入提示的原始用户消息
        original_user_message = persist_user_message if persist_user_message is not None else user_message

        # 判断是否触发记忆提示 nudge（基于轮次，技能相关则在 agent 主循环后再处理）
        _should_review_memory = False
        if (self._memory_nudge_interval > 0
                and "memory" in self.valid_tool_names
                and self._memory_store):
            self._turns_since_memory += 1
            if self._turns_since_memory >= self._memory_nudge_interval:
                _should_review_memory = True
                self._turns_since_memory = 0

        # 添加用户消息
        user_msg = {"role": "user", "content": user_message}
        messages.append(user_msg)
        current_turn_user_idx = len(messages) - 1
        self._persist_user_message_idx = current_turn_user_idx
        
        if not self.quiet_mode:
            self._safe_print(f"💬 开始新会话: '{user_message[:60]}{'...' if len(user_message) > 60 else ''}'")
        
        # ── 系统提示语缓存（每会话一次，做前缀缓存）──
        # 首次构建后，每次自动重用。只有在上下文压缩触发（会清除缓存并重载记忆）时才会重新生成。
        #
        # gateway 场景续会话（每条消息新建 agent），
        # 系统提示来自 session DB，不直接重新构建。否则，如从磁盘新加载记忆，
        # 会生成不一样的提示，导致 Anthropic 前缀缓存失效。
        if self._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and self._session_db:
                try:
                    session_row = self._session_db.get_session(self.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass  # 如果失败则全新构建

            if stored_prompt:
                # 续会话 -- 重用上一轮保存的 system prompt，保证 Anthropic 缓存前缀吻合
                self._cached_system_prompt = stored_prompt
            else:
                # 新会话首轮 -- 全新生成
                self._cached_system_prompt = self._build_system_prompt(system_message)
                # 插件钩子：on_session_start
                # 仅新会话第一次触发。插件可用此初始化会话范围的状态（如预热内存缓存）。
                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _invoke_hook(
                        "on_session_start",
                        session_id=self.session_id,
                        model=self.model,
                        platform=getattr(self, "platform", None) or "",
                    )
                except Exception as exc:
                    logger.warning("on_session_start hook failed: %s", exc)

                # 把当前系统 prompt 快照进 SQLite
                if self._session_db:
                    try:
                        self._session_db.update_system_prompt(self.session_id, self._cached_system_prompt)
                    except Exception as e:
                        logger.debug("Session DB update_system_prompt failed: %s", e)

        active_system_prompt = self._cached_system_prompt

        # ── 主循环前预压缩上下文 ──
        # 进入主循环前，先检查历史消息是否已超模型上下文上限。
        # 适配用户切换到上下文窗口更小的模型且会话很大，主动压缩，比收到 API 错误（通常 4xx 不可重试直接中止）更友好。
        if (
            self.compression_enabled
            and len(messages) > self.context_compressor.protect_first_n
                                + self.context_compressor.protect_last_n + 1
        ):
            # 计入工具 schema token，多个工具情况下这些可能占用 2-3 万 token（旧方法只计 sys+msg 时极易漏算）
            _preflight_tokens = estimate_request_tokens_rough(
                messages,
                system_prompt=active_system_prompt or "",
                tools=self.tools or None,
            )

            if _preflight_tokens >= self.context_compressor.threshold_tokens:
                logger.info(
                    "预压缩: ~%s tokens >= %s 上限 (model %s, ctx %s)",
                    f"{_preflight_tokens:,}",
                    f"{self.context_compressor.threshold_tokens:,}",
                    self.model,
                    f"{self.context_compressor.context_length:,}",
                )
                if not self.quiet_mode:
                    self._safe_print(
                        f"📦 预压缩: ~{_preflight_tokens:,} tokens "
                        f">= {self.context_compressor.threshold_tokens:,} 上限"
                    )
                # 特别大的会话+很小的 context 可能需多轮压缩（每次只汇总中间 N 轮）
                for _pass in range(3):
                    _orig_len = len(messages)
                    messages, active_system_prompt = self._compress_context(
                        messages, system_message, approx_tokens=_preflight_tokens,
                        task_id=effective_task_id,
                    )
                    if len(messages) >= _orig_len:
                        break  # 已无法再压缩
                    # 一旦产生了新会话，则要清空历史引用（保证 _flush_messages_to_session_db 能把所有消息都写到新会话 sqlite，不因旧消息还在 conversation_history 而跳过）
                    conversation_history = None
                    # 压缩后重新估算 token
                    _preflight_tokens = estimate_request_tokens_rough(
                        messages,
                        system_prompt=active_system_prompt or "",
                        tools=self.tools or None,
                    )
                    if _preflight_tokens < self.context_compressor.threshold_tokens:
                        break  # 已低于阈值
        
        # 插件钩子：pre_llm_call
        # 每轮主循环前触发，插件可返回 {'context': ...} 或纯字符串，内容会附加到本回合用户消息后。
        #
        # 该 context 只会注入用户消息，不会出现在 system prompt，
        # 保证提示缓存的前缀一致系统提示不被污染。system prompt 属于 Hermes 内部，插件只能追加用户输入上下文。
        #
        # 所有注入 context 都是临时的，不进 session DB。
        _plugin_user_context = ""
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _pre_results = _invoke_hook(
                "pre_llm_call",
                session_id=self.session_id,
                user_message=original_user_message,
                conversation_history=list(messages),
                is_first_turn=(not bool(conversation_history)),
                model=self.model,
                platform=getattr(self, "platform", None) or "",
                sender_id=getattr(self, "_user_id", None) or "",
            )
            _ctx_parts: list[str] = []
            for r in _pre_results:
                if isinstance(r, dict) and r.get("context"):
                    _ctx_parts.append(str(r["context"]))
                elif isinstance(r, str) and r.strip():
                    _ctx_parts.append(r)
            if _ctx_parts:
                _plugin_user_context = "\n\n".join(_ctx_parts)
        except Exception as exc:
            logger.warning("pre_llm_call hook failed: %s", exc)

        # 主对话循环
        api_call_count = 0
        final_response = None
        interrupted = False
        codex_ack_continuations = 0
        length_continue_retries = 0
        truncated_tool_call_retries = 0
        truncated_response_prefix = ""
        compression_attempts = 0
        _turn_exit_reason = "unknown"  # 记录循环结束原因
        
        # 记录当前执行线程ID，便于 interrupt()/clear_interrupt() 只作用于本 agent 线程。
        # 必须在 clear_interrupt() 前设置（后者要用）。
        self._execution_thread_id = threading.current_thread().ident

        # 开始前清空遗留的中断信号
        self.clear_interrupt()

        # 外部记忆供应商：主循环前预取一遍，缓存后每次都复用，避免每次工具调用都重复请求（会极大提高延迟与费用）。
        # 用 original_user_message（纯净输入），因为 user_message 可能混入技能 context 影响查询效果。
        _ext_prefetch_cache = ""
        if self._memory_manager:
            try:
                _query = original_user_message if isinstance(original_user_message, str) else ""
                _ext_prefetch_cache = self._memory_manager.prefetch_all(_query) or ""
            except Exception:
                pass

        while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) or self._budget_grace_call:
            # 每轮重置 checkpoint 去重，方便每次迭代都快照一次
            self._checkpoint_mgr.new_turn()

            # 检查是否有中断请求（如用户输入新消息等）
            if self._interrupt_requested:
                interrupted = True
                _turn_exit_reason = "interrupted_by_user"
                if not self.quiet_mode:
                    self._safe_print("\n⚡ 检测到中断，跳出工具调用循环...")
                break
            
            api_call_count += 1
            self._api_call_count = api_call_count
            self._touch_activity(f"开始 API 请求第 {api_call_count} 次")

            # 宽限调用：已耗尽预算但给模型最后一次机会，用后标记 flag 用尽，下轮直接退出
            if self._budget_grace_call:
                self._budget_grace_call = False
            elif not self.iteration_budget.consume():
                _turn_exit_reason = "budget_exhausted"
                if not self.quiet_mode:
                    self._safe_print(f"\n⚠️  迭代预算耗尽 ({self.iteration_budget.used}/{self.iteration_budget.max_total} 次)")
                break

            # gateway-hook 回调，每步都能收到（如 agent:step 事件）
            if self.step_callback is not None:
                try:
                    prev_tools = []
                    for _idx, _m in enumerate(reversed(messages)):
                        if _m.get("role") == "assistant" and _m.get("tool_calls"):
                            _fwd_start = len(messages) - _idx
                            _results_by_id = {}
                            for _tm in messages[_fwd_start:]:
                                if _tm.get("role") != "tool":
                                    break
                                _tcid = _tm.get("tool_call_id")
                                if _tcid:
                                    _results_by_id[_tcid] = _tm.get("content", "")
                            prev_tools = [
                                {
                                    "name": tc["function"]["name"],
                                    "result": _results_by_id.get(tc.get("id")),
                                }
                                for tc in _m["tool_calls"]
                                if isinstance(tc, dict)
                            ]
                            break
                    self.step_callback(api_call_count, prev_tools)
                except Exception as _step_err:
                    logger.debug("step_callback error (iteration %s): %s", api_call_count, _step_err)

            # 工具调用计数，用于技能提示间隔（每次真正 skill_manage 后会重置）
            if (self._skill_nudge_interval > 0
                    and "skill_manage" in self.valid_tool_names):
                self._iters_since_skill += 1
            
            # 构造 API 请求消息
            # 如果有临时 system prompt，提前插入
            # 说明：推理中间内容用 <think> 嵌入，方便轨迹存档；
            # 有些供应商（如 Moonshot AI）要求工具调用消息里 reasoning_content 字段，
            # 两种场景都兼容。
            api_messages = []
            for idx, msg in enumerate(messages):
                api_msg = msg.copy()

                # 当前轮用户消息注入临时 context（外部记忆预取+插件），
                # 只作用于 API 调用副本，messages 不变，避免 session DB 污染。
                if idx == current_turn_user_idx and msg.get("role") == "user":
                    _injections = []
                    if _ext_prefetch_cache:
                        _fenced = build_memory_context_block(_ext_prefetch_cache)
                        if _fenced:
                            _injections.append(_fenced)
                    if _plugin_user_context:
                        _injections.append(_plugin_user_context)
                    if _injections:
                        _base = api_msg.get("content", "")
                        if isinstance(_base, str):
                            api_msg["content"] = _base + "\n\n" + "\n\n".join(_injections)

                # 所有 assistant 消息，把 reasoning 传递到 API（保留多轮推理上下文）
                if msg.get("role") == "assistant":
                    reasoning_text = msg.get("reasoning")
                    if reasoning_text:
                        # 为 Moonshot AI、Novita、OpenRouter 兼容性加 reasoning_content 字段
                        api_msg["reasoning_content"] = reasoning_text

                # 移除 reasoning 字段（仅内部存轨迹），已拷贝到 reasoning_content
                if "reasoning" in api_msg:
                    api_msg.pop("reasoning")
                # 移除 finish_reason 字段（Mistral 等 API 不接收）
                if "finish_reason" in api_msg:
                    api_msg.pop("finish_reason")
                # 清理内部思维引导标记
                api_msg.pop("_thinking_prefill", None)
                # 清除 Codex Responses 的其它字段（Mistral/Fireworks 这样严格 API 会拒绝不认识的字段）
                # 用新字典保持内部 messages 字段不变便于 Codex Responses 兼容
                if self._should_sanitize_tool_calls():
                    self._sanitize_tool_calls_for_strict_api(api_msg)
                # reasoning_details 可保留（OpenRouter 用于多轮推理签名）
                api_messages.append(api_msg)

            # 生成最终 system 消息（缓存 prompt + 临时 prompt），
            # 临时内容只在 API 调用时注入，不入 session DB。
            # 外部召回 context 注入用户消息，不入 system prompt，保持前缀一致。
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            # 注意：插件 pre_llm_call 返回的 context 仅进 user message，不进 system prompt
            # 这是有意保留 system prompt 前缀缓存一致性，Hermes 只自身管控 system prompt。
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages

            # 临时 prefill message 在 system prompt 后、历史前插入（只在 API 调用时注入）
            if self.prefill_messages:
                sys_offset = 1 if effective_system else 0
                for idx, pfm in enumerate(self.prefill_messages):
                    api_messages.insert(sys_offset + idx, pfm.copy())

            # Anthropic 前缀缓存，Claude 模型在 OpenRouter 自动检测
            # （名字带 claude 且 base_url 符合），
            # 插入 cache_control 节点提升多轮 token 利用率（降约 75%）。
            if self._use_prompt_caching:
                api_messages = apply_anthropic_cache_control(api_messages, cache_ttl=self._cache_ttl, native_anthropic=(self.api_mode == 'anthropic_messages'))

            # 最终安全校验：清理孤立工具回复/补全缺少 stub，避免会话/历史加载/手动改 history 时出错
            # （无论 context_compressor 状态，每次都处理，保证安全）
            api_messages = self._sanitize_api_messages(api_messages)

            # 归一化消息内容/工具 JSON（空白移除 & 工具参数 key 排序），保证跨回合前缀 bit 一致。
            # 能用 KV 缓存和提升云推理缓存命中率，对本地服务端/云推理均有加速作用。仅操作 api_messages，messages 原始副本不动。
            for am in api_messages:
                if isinstance(am.get("content"), str):
                    am["content"] = am["content"].strip()
            for am in api_messages:
                tcs = am.get("tool_calls")
                if not tcs:
                    continue
                new_tcs = []
                for tc in tcs:
                    if isinstance(tc, dict) and "function" in tc:
                        try:
                            args_obj = json.loads(tc["function"]["arguments"])
                            tc = {**tc, "function": {
                                **tc["function"],
                                "arguments": json.dumps(
                                    args_obj, separators=(",", ":"),
                                    sort_keys=True,
                                ),
                            }}
                        except Exception:
                            pass
                    new_tcs.append(tc)
                am["tool_calls"] = new_tcs

            # 估算本次请求大小，方便日志和调优
            total_chars = sum(len(str(msg)) for msg in api_messages)
            approx_tokens = estimate_messages_tokens_rough(api_messages)
            
            # 静默模式下思考转圈动画（API 调用时显示）
            thinking_spinner = None
            
            if not self.quiet_mode:
                self._vprint(f"\n{self.log_prefix}🔄 发起 API 调用 #{api_call_count}/{self.max_iterations}...")
                self._vprint(f"{self.log_prefix}   📊 消息 {len(api_messages)} 条，~{approx_tokens:,} tokens (~{total_chars:,} 字符)")
                self._vprint(f"{self.log_prefix}   🔧 可用工具: {len(self.tools) if self.tools else 0}")
            else:
                # 安静模式下随机表情&动作转圈（如 CLI 动态 loading）
                face = random.choice(KawaiiSpinner.KAWAII_THINKING)
                verb = random.choice(KawaiiSpinner.THINKING_VERBS)
                if self.thinking_callback:
                    # CLI TUI 模式用 prompt_toolkit 组件，不用原始 spinner（流/非流都适用）
                    self.thinking_callback(f"{face} {verb}...")
                elif not self._has_stream_consumers() and self._should_start_quiet_spinner():
                    # 无流式消费则本地用原生 KawaiiSpinner（stdout 已处理），
                    # 防止输出乱套。
                    spinner_type = random.choice(['brain', 'sparkle', 'pulse', 'moon', 'star'])
                    thinking_spinner = KawaiiSpinner(f"{face} {verb}...", spinner_type=spinner_type, print_fn=self._print_fn)
                    thinking_spinner.start()
            
            # Verbose 日志：详细记录请求细节
            if self.verbose_logging:
                logging.debug(f"API Request - Model: {self.model}, Messages: {len(messages)}, Tools: {len(self.tools) if self.tools else 0}")
                logging.debug(f"Last message role: {messages[-1]['role'] if messages else 'none'}")
                logging.debug(f"Total message size: ~{approx_tokens:,} tokens")
            
            api_start_time = time.time()
            retry_count = 0
            max_retries = 3
            primary_recovery_attempted = False
            max_compression_attempts = 3
            codex_auth_retry_attempted = False
            anthropic_auth_retry_attempted = False
            nous_auth_retry_attempted = False
            thinking_sig_retry_attempted = False
            has_retried_429 = False
            restart_with_compressed_messages = False
            restart_with_length_continuation = False

            finish_reason = "stop"
            response = None  # 防止所有重试都失败时报 UnboundLocalError
            api_kwargs = None  # except 分支引用

            while retry_count < max_retries:
                try:
                    self._reset_stream_delivery_tracking()
                    api_kwargs = self._build_api_kwargs(api_messages)
                    if self.api_mode == "codex_responses":
                        api_kwargs = self._preflight_codex_api_kwargs(api_kwargs, allow_stream=False)

                    try:
                        from hermes_cli.plugins import invoke_hook as _invoke_hook
                        _invoke_hook(
                            "pre_api_request",
                            task_id=effective_task_id,
                            session_id=self.session_id or "",
                            platform=self.platform or "",
                            model=self.model,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_mode=self.api_mode,
                            api_call_count=api_call_count,
                            message_count=len(api_messages),
                            tool_count=len(self.tools or []),
                            approx_input_tokens=approx_tokens,
                            request_char_count=total_chars,
                            max_tokens=self.max_tokens,
                        )
                    except Exception:
                        pass

                    if env_var_enabled("HERMES_DUMP_REQUESTS"):
                        self._dump_api_request_debug(api_kwargs, reason="preflight")

                    # 始终优先选择流式路径（即便没有流式消费，也能健康探测），
                    # 能检测超时等问题，避免 gateway/subagent 安静模式死等。
                    # 若 provider 不支持则自动降级为普通调用。
                    def _stop_spinner():
                        nonlocal thinking_spinner
                        if thinking_spinner:
                            thinking_spinner.stop("")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")

                    _use_streaming = True
                    if not self._has_stream_consumers():
                        # 没有前端/TTS 就不需要 spinning，但单元测试 mock client（返回 SimpleNamespace 而不是迭代器）要跳过流模式。
                        from unittest.mock import Mock
                        if isinstance(getattr(self, "client", None), Mock):
                            _use_streaming = False

                    if _use_streaming:
                        response = self._interruptible_streaming_api_call(
                            api_kwargs, on_first_delta=_stop_spinner
                        )
                    else:
                        response = self._interruptible_api_call(api_kwargs)
                    
                    api_duration = time.time() - api_start_time
                    
                    # API 返回后静默关闭思考圈圈（后续回复或错误更重要）。
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}⏱️  API 调用完成，用时 {api_duration:.2f}s")
                    
                    if self.verbose_logging:
                        # 日志记录回复模型信息
                        resp_model = getattr(response, 'model', 'N/A') if response else 'N/A'
                        logging.debug(f"API Response received - Model: {resp_model}, Usage: {response.usage if hasattr(response, 'usage') else 'N/A'}")
                    
                    # 校验回复格式
                    response_invalid = False
                    error_details = []
                    if self.api_mode == "codex_responses":
                        output_items = getattr(response, "output", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(output_items, list):
                            response_invalid = True
                            error_details.append("response.output 不是列表")
                        elif not output_items:
                            # stream 回流失败但 _normalize_codex_response 还能用 output_text 兜底
                            _out_text = getattr(response, "output_text", None)
                            _out_text_stripped = _out_text.strip() if isinstance(_out_text, str) else ""
                            if _out_text_stripped:
                                logger.debug(
                                    "Codex response.output 为空但 output_text 存在 "
                                    "(%d 字符)，交由 normalize 兜底。",
                                    len(_out_text_stripped),
                                )
                            else:
                                _resp_status = getattr(response, "status", None)
                                _resp_incomplete = getattr(response, "incomplete_details", None)
                                logger.warning(
                                    "Codex response.output 回流失败 "
                                    "(status=%s, incomplete_details=%s, model=%s). %s",
                                    _resp_status, _resp_incomplete,
                                    getattr(response, "model", None),
                                    f"api_mode={self.api_mode} provider={self.provider}",
                                )
                                response_invalid = True
                                error_details.append("response.output 为空")
                    elif self.api_mode == "anthropic_messages":
                        content_blocks = getattr(response, "content", None) if response is not None else None
                        if response is None:
                            response_invalid = True
                            error_details.append("response is None")
                        elif not isinstance(content_blocks, list):
                            response_invalid = True
                            error_details.append("response.content 不是列表")
                        elif not content_blocks:
                            response_invalid = True
                            error_details.append("response.content 为空")
                    else:
                        if response is None or not hasattr(response, 'choices') or response.choices is None or not response.choices:
                            response_invalid = True
                            if response is None:
                                error_details.append("response is None")
                            elif not hasattr(response, 'choices'):
                                error_details.append("response 没有 choices 属性")
                            elif response.choices is None:
                                error_details.append("response.choices 为 None")
                            else:
                                error_details.append("response.choices 为空")

                    if response_invalid:
                        # 先关思维圈圈再打印错误
                        if thinking_spinner:
                            thinking_spinner.stop("(´;ω;`) 错误，重试中...")
                            thinking_spinner = None
                        if self.thinking_callback:
                            self.thinking_callback("")
                        
                        # 无效回复：可能是限流、上游超时、服务器异常或格式不符
                        retry_count += 1
                        
                        # 积极降级：空/格式错误回复通常是限流 尽快切 fallback 而不是盲目延长重试。
                        if self._fallback_index < len(self._fallback_chain):
                            self._emit_status("⚠️ 空/异常回复，切换到备份模型...")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue

                        # 部分上游有 error 字段（优先用 error 字段诊断）
                        error_msg = "Unknown"
                        provider_name = "Unknown"
                        if response and hasattr(response, 'error') and response.error:
                            error_msg = str(response.error)
                            # 有可能 error 里自带 provider 信息
                            if hasattr(response.error, 'metadata') and response.error.metadata:
                                provider_name = response.error.metadata.get('provider_name', 'Unknown')
                        elif response and hasattr(response, 'message') and response.message:
                            error_msg = str(response.message)
                        
                        # 有的（如 OpenRouter）可用 model 字段补足 provider
                        if provider_name == "Unknown" and response and hasattr(response, 'model') and response.model:
                            provider_name = f"model={response.model}"
                        
                        # 某些元数据字段额外打印，便于调试
                        if provider_name == "Unknown" and response:
                            resp_attrs = {k: str(v)[:100] for k, v in vars(response).items() if not k.startswith('_')}
                            if self.verbose_logging:
                                logging.debug(f"无效回复属性: {resp_attrs}")
                        
                        # 错误码解析，方便给不同的诊断提示
                        _resp_error_code = None
                        if response and hasattr(response, 'error') and response.error:
                            _code_raw = getattr(response.error, 'code', None)
                            if _code_raw is None and isinstance(response.error, dict):
                                _code_raw = response.error.get('code')
                            if _code_raw is not None:
                                try:
                                    _resp_error_code = int(_code_raw)
                                except (TypeError, ValueError):
                                    pass

                        # 人类可读诊断提示
                        if _resp_error_code == 524:
                            _failure_hint = f"上游超时 (Cloudflare 524, {api_duration:.0f}s)"
                        elif _resp_error_code == 504:
                            _failure_hint = f"网关超时 (504, {api_duration:.0f}s)"
                        elif _resp_error_code == 429:
                            _failure_hint = f"上游限流 (429)"
                        elif _resp_error_code in (500, 502):
                            _failure_hint = f"上游服务故障 ({_resp_error_code}, {api_duration:.0f}s)"
                        elif _resp_error_code in (503, 529):
                            _failure_hint = f"上游超载 ({_resp_error_code})"
                        elif _resp_error_code is not None:
                            _failure_hint = f"上游错误 (code {_resp_error_code}, {api_duration:.0f}s)"
                        elif api_duration < 10:
                            _failure_hint = f"响应极快（{api_duration:.1f}s）—很可能被限流"
                        elif api_duration > 60:
                            _failure_hint = f"响应极慢（{api_duration:.0f}s）—很可能超时"
                        else:
                            _failure_hint = f"响应时间 {api_duration:.1f}s"

                        self._vprint(f"{self.log_prefix}⚠️  API 回复无效（第 {retry_count}/{max_retries} 次）：{', '.join(error_details)}", force=True)
                        self._vprint(f"{self.log_prefix}   🏢 供应商: {provider_name}", force=True)
                        cleaned_provider_error = self._clean_error_message(error_msg)
                        self._vprint(f"{self.log_prefix}   📝 上游返回信息: {cleaned_provider_error}", force=True)
                        self._vprint(f"{self.log_prefix}   ⏱️  {_failure_hint}", force=True)
                        
                        if retry_count >= max_retries:
                            # 用尽重试仍无效，最后再试一次 fallback，否则宣告失败
                            self._emit_status(f"⚠️ API 连续无效达到 {max_retries} 次，尝试降级...")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue
                            self._emit_status(f"❌ 连续无效达到 {max_retries} 次，已放弃。")
                            logging.error(f"{self.log_prefix}API 调用无效重试 {max_retries} 次仍失败")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"API 调用无效超出最大重试次数（{max_retries}）：{_failure_hint}",
                                "failed": True  # 标为失败，方便筛查
                            }
                        
                        # Jitter 指数退避，基准 5s，最大 120s
                        wait_time = jittered_backoff(retry_count, base_delay=5.0, max_delay=120.0)
                        self._vprint(f"{self.log_prefix}⏳ 等待 {wait_time:.1f}s 后自动重试（{_failure_hint}）...", force=True)
                        logging.warning(f"API 返回无效（重试 {retry_count}/{max_retries}）：{', '.join(error_details)} | Provider: {provider_name}")
                        
                        # 细粒度 sleep，期间若检测到新中断可及时响应
                        sleep_end = time.time() + wait_time
                        while time.time() < sleep_end:
                            if self._interrupt_requested:
                                self._vprint(f"{self.log_prefix}⚡ 等待期间检测到中断，终止重试。", force=True)
                                self._persist_session(messages, conversation_history)
                                self.clear_interrupt()
                                return {
                                    "final_response": f"重试等待期间被中断（{_failure_hint}，第 {retry_count}/{max_retries} 次）。",
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "interrupted": True,
                                }
                            time.sleep(0.2)
                        continue  # 继续重试本次 API 调用

                    # 根据 finish_reason 做分支
                    if self.api_mode == "codex_responses":
                        status = getattr(response, "status", None)
                        incomplete_details = getattr(response, "incomplete_details", None)
                        incomplete_reason = None
                        if isinstance(incomplete_details, dict):
                            incomplete_reason = incomplete_details.get("reason")
                        else:
                            incomplete_reason = getattr(incomplete_details, "reason", None)
                        if status == "incomplete" and incomplete_reason in {"max_output_tokens", "length"}:
                            finish_reason = "length"
                        else:
                            finish_reason = "stop"
                    elif self.api_mode == "anthropic_messages":
                        stop_reason_map = {"end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop"}
                        finish_reason = stop_reason_map.get(response.stop_reason, "stop")
                    else:
                        finish_reason = response.choices[0].finish_reason

                    if finish_reason == "length":
                        self._vprint(f"{self.log_prefix}⚠️  回复被截断（finish_reason='length'）- 达到模型输出 token 上限", force=True)

                        # ── 检查推理预算已耗尽 ──────────────
                        # 若模型把所有输出 token 都用在了 reasoning，正文一丁点未生成，则无需再连着重试 3 次 API，只需一次性报明确错误提示即可。
                        _trunc_content = None
                        _trunc_has_tool_calls = False
                        if self.api_mode == "chat_completions":
                            _trunc_msg = response.choices[0].message if (hasattr(response, "choices") and response.choices) else None
                            _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
                            _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False
                        elif self.api_mode == "anthropic_messages":
                            # Anthropic 的 response.content 是区块列表
                            _text_parts = []
                            for _blk in getattr(response, "content", []):
                                if getattr(_blk, "type", None) == "text":
                                    _text_parts.append(getattr(_blk, "text", ""))
                            _trunc_content = "\n".join(_text_parts) if _text_parts else None

                        # 只要模型真输出过 <think> 且没有正文，就判定 reasoning 耗光（如 GLM-4.7/no <think> 时出现空内容属于正常截断，不是预算耗尽，不要误判）
                        _has_think_tags = bool(
                            _trunc_content and re.search(
                                r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>',
                                _trunc_content,
                                re.IGNORECASE,
                            )
                        )
                        _thinking_exhausted = (
                            not _trunc_has_tool_calls
                            and _has_think_tags
                            and (
                                (_trunc_content is not None and not self._has_content_after_think_block(_trunc_content))
                                or _trunc_content is None
                            )
                        )

                        if _thinking_exhausted:
                            _exhaust_error = (
                                "模型所有 token 都用于推理，没有留给正文。请尝试降低推理强度或提升 max_tokens。"
                            )
                            self._vprint(
                                f"{self.log_prefix}💭 推理耗尽了输出 token，上下文没有正文被生成。",
                                force=True,
                            )
                            # 用友好提示覆盖 CLI/gateway 用户显示，不直接抛 error
                            _exhaust_response = (
                                "⚠️ **推理预算耗尽**\n\n"
                                "模型把全部输出 token 都用在了推理步骤，没有剩余 token 生成正文回答。\n\n"
                                "修复建议：\n"
                                "→ 降低推理强度：输入 /thinkon low 或 /thinkon minimal\n"
                                "→ 提高模型输出 token 上限：编辑 config.yaml 里的 model.max_tokens 配置"
                            )
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": _exhaust_response,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": _exhaust_error,
                            }

                        if self.api_mode == "chat_completions":
                            assistant_message = response.choices[0].message
                            if not assistant_message.tool_calls:
                                length_continue_retries += 1
                                interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                                messages.append(interim_msg)
                                if assistant_message.content:
                                    truncated_response_prefix += assistant_message.content

                                if length_continue_retries < 3:
                                    self._vprint(
                                        f"{self.log_prefix}↻ 系统自动发起连续第 {length_continue_retries}/3 次继续回复...",
                                    )
                                    continue_msg = {
                                        "role": "user",
                                        "content": (
                                            "[System: 你刚才的回复因输出长度限制被截断。请直接从上次结束位置继续，无需复读前文/重启，直接补充完成即可。]"
                                        ),
                                    }
                                    messages.append(continue_msg)
                                    self._session_messages = messages
                                    self._save_session_log(messages)
                                    restart_with_length_continuation = True
                                    break

                                partial_response = self._strip_think_blocks(truncated_response_prefix).strip()
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": partial_response or None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "3次尝试后回复仍被截断",
                                }

                        if self.api_mode == "chat_completions":
                            assistant_message = response.choices[0].message
                            if assistant_message.tool_calls:
                                if truncated_tool_call_retries < 1:
                                    truncated_tool_call_retries += 1
                                    self._vprint(
                                        f"{self.log_prefix}⚠️  检测到被截断的工具调用 - 正在重试 ...",
                                        force=True,
                                    )
                                    # 不追加当前损坏回复，直接重跑一遍 API，给模型机会修复
                                    continue
                                self._vprint(
                                    f"{self.log_prefix}⚠️  再次检测到被截断的工具调用 - 放弃执行不完整参数。",
                                    force=True,
                                )
                                self._cleanup_task_resources(effective_task_id)
                                self._persist_session(messages, conversation_history)
                                return {
                                    "final_response": None,
                                    "messages": messages,
                                    "api_calls": api_call_count,
                                    "completed": False,
                                    "partial": True,
                                    "error": "工具回复因长度限制被截断",
                                }

                        # 如有历史，回滚到上一次完整助手回复后再给用户（更友善）
                        if len(messages) > 1:
                            self._vprint(f"{self.log_prefix}   ⏪ 已回滚至最后完整的助手回复")
                            rolled_back_messages = self._get_messages_up_to_last_assistant(messages)

                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)

                            return {
                                "final_response": None,
                                "messages": rolled_back_messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "回复因输出长度限制被截断，已回滚"
                            }
                        else:
                            # 如首轮就被截断，直接宣告失败
                            self._vprint(f"{self.log_prefix}❌ 首次回复就被截断，无法恢复", force=True)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "failed": True,
                                "error": "首次回复因输出长度被截断"
                            }
                    
                    # 实际 token 用量写入 context 管理，对后续自动压缩有关键作用
                    if hasattr(response, 'usage') and response.usage:
                        canonical_usage = normalize_usage(
                            response.usage,
                            provider=self.provider,
                            api_mode=self.api_mode,
                        )
                        prompt_tokens = canonical_usage.prompt_tokens
                        completion_tokens = canonical_usage.output_tokens
                        total_tokens = canonical_usage.total_tokens
                        usage_dict = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        self.context_compressor.update_from_response(usage_dict)

                        # 记录自动探知的 context 长度，只缓存 provider 真返回的数据（直接解析 error），不缓存猜测/探测出来的 tier
                        if getattr(self.context_compressor, "_context_probed", False):
                            ctx = self.context_compressor.context_length
                            if getattr(self.context_compressor, "_context_probe_persistable", False):
                                save_context_length(self.model, self.base_url, ctx)
                                self._safe_print(f"{self.log_prefix}💾 已缓存上下文长度：{ctx:,} tokens for {self.model}")
                            self.context_compressor._context_probed = False
                            self.context_compressor._context_probe_persistable = False

                        self.session_prompt_tokens += prompt_tokens
                        self.session_completion_tokens += completion_tokens
                        self.session_total_tokens += total_tokens
                        self.session_api_calls += 1
                        self.session_input_tokens += canonical_usage.input_tokens
                        self.session_output_tokens += canonical_usage.output_tokens
                        self.session_cache_read_tokens += canonical_usage.cache_read_tokens
                        self.session_cache_write_tokens += canonical_usage.cache_write_tokens
                        self.session_reasoning_tokens += canonical_usage.reasoning_tokens

                        # 详细日志 API 调用状态
                        _cache_pct = ""
                        if canonical_usage.cache_read_tokens and prompt_tokens:
                            _cache_pct = f" cache={canonical_usage.cache_read_tokens}/{prompt_tokens} ({100*canonical_usage.cache_read_tokens/prompt_tokens:.0f}%)"
                        logger.info(
                            "API 调用 #%d: model=%s provider=%s in=%d out=%d total=%d latency=%.1fs%s",
                            self.session_api_calls, self.model, self.provider or "unknown",
                            prompt_tokens, completion_tokens, total_tokens,
                            api_duration, _cache_pct,
                        )

                        cost_result = estimate_usage_cost(
                            self.model,
                            canonical_usage,
                            provider=self.provider,
                            base_url=self.base_url,
                            api_key=getattr(self, "api_key", ""),
                        )
                        if cost_result.amount_usd is not None:
                            self.session_estimated_cost_usd += float(cost_result.amount_usd)
                        self.session_cost_status = cost_result.status
                        self.session_cost_source = cost_result.source

                        # 每次都把 token 计数写入 session DB，保障无论在 gateway/cron/委托运行哪，都能持久保存
                        # gateway/session-store 只存绝对总量，所以不会重复累加导致数据脏
                        if self._session_db and self.session_id:
                            try:
                                self._session_db.update_token_counts(
                                    self.session_id,
                                    input_tokens=canonical_usage.input_tokens,
                                    output_tokens=canonical_usage.output_tokens,
                                    cache_read_tokens=canonical_usage.cache_read_tokens,
                                    cache_write_tokens=canonical_usage.cache_write_tokens,
                                    reasoning_tokens=canonical_usage.reasoning_tokens,
                                    estimated_cost_usd=float(cost_result.amount_usd)
                                    if cost_result.amount_usd is not None else None,
                                    cost_status=cost_result.status,
                                    cost_source=cost_result.source,
                                    billing_provider=self.provider,
                                    billing_base_url=self.base_url,
                                    billing_mode="subscription_included"
                                    if cost_result.status == "included" else None,
                                    model=self.model,
                                )
                            except Exception:
                                pass  # 不让异常影响 agent 主循环
                        
                        if self.verbose_logging:
                            logging.debug(f"Token 使用量: prompt={usage_dict['prompt_tokens']:,}, completion={usage_dict['completion_tokens']:,}, total={usage_dict['total_tokens']:,}")
                        
                        # 启动前缀缓存时额外记录缓存命中情况
                        if self._use_prompt_caching:
                            if self.api_mode == "anthropic_messages":
                                # Anthropic 专用统计字段
                                cached = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
                                written = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0
                            else:
                                # OpenRouter 统计字段
                                details = getattr(response.usage, 'prompt_tokens_details', None)
                                cached = getattr(details, 'cached_tokens', 0) or 0 if details else 0
                                written = getattr(details, 'cache_write_tokens', 0) or 0 if details else 0
                            prompt = usage_dict["prompt_tokens"]
                            hit_pct = (cached / prompt * 100) if prompt > 0 else 0
                            if not self.quiet_mode:
                                self._vprint(f"{self.log_prefix}   💾 缓存: {cached:,}/{prompt:,} tokens ({hit_pct:.0f}% 命中, {written:,} 已写入)")
                    
                    has_retried_429 = False  # 成功后重置
                    self._touch_activity(f"API 调用 #{api_call_count} 完成")
                    break  # 成功退出 retry 循环

                except InterruptedError:
                    if thinking_spinner:
                        thinking_spinner.stop("")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")
                    api_elapsed = time.time() - api_start_time
                    self._vprint(f"{self.log_prefix}⚡ API 调用被中断.", force=True)
                    self._persist_session(messages, conversation_history)
                    interrupted = True
                    final_response = f"操作被中断：等待模型回复期间（已耗时 {api_elapsed:.1f}s）"
                    break

                except Exception as api_error:
                    # 关思维圈圈再提示
                    if thinking_spinner:
                        thinking_spinner.stop("(╥_╥) 出错，重试中...")
                        thinking_spinner = None
                    if self.thinking_callback:
                        self.thinking_callback("")

                    # -----------------------------------------------------------
                    # UnicodeEncodeError 错误恢复，常见两种情况：
                    #   1. 剪切板粘贴含代理对（U+D800..U+DFFF），如富文本编辑器 — strip 干净后重试
                    #   2. LANG=C/编码不是 UTF-8 的系统（如 Chromebook），任何非 ASCII 都 fail
                    #      可通过异常 message 包含 'ascii' codec 检测
                    #   本地对 messages 就地 sanitize，最多重试两次：先 proxy，后 ASCII-only
                    # -----------------------------------------------------------
                    if isinstance(api_error, UnicodeEncodeError) and getattr(self, '_unicode_sanitization_passes', 0) < 2:
                        _err_str = str(api_error).lower()
                        _is_ascii_codec = "'ascii'" in _err_str or "ascii" in _err_str
                        _surrogates_found = _sanitize_messages_surrogates(messages)
                        if _surrogates_found:
                            self._unicode_sanitization_passes += 1
                            self._vprint(
                                f"{self.log_prefix}⚠️  已移除消息中的无效代理字符，重试中...",
                                force=True,
                            )
                            continue
                        if _is_ascii_codec:
                            # 系统只支持 ASCII，移除所有非 ASCII 字符再重试
                            if _sanitize_messages_non_ascii(messages):
                                self._unicode_sanitization_passes += 1
                                self._vprint(
                                    f"{self.log_prefix}⚠️  系统编码为 ASCII，已移除所有非 ASCII 内容，重试中...",
                                    force=True,
                                )
                                continue
                        # 都没找到要 sanitize 的，可能问题在 system prompt 或 prefill，继续走常规错误处理

                    status_code = getattr(api_error, "status_code", None)
                    error_context = self._extract_api_error_context(api_error)

                    # ── 错误归类，判定是否可自动恢复/降级/重试 ──
                    _compressor = getattr(self, "context_compressor", None)
                    _ctx_len = getattr(_compressor, "context_length", 200000) if _compressor else 200000
                    classified = classify_api_error(
                        api_error,
                        provider=getattr(self, "provider", "") or "",
                        model=getattr(self, "model", "") or "",
                        approx_tokens=approx_tokens,
                        context_length=_ctx_len,
                        num_messages=len(api_messages) if api_messages else 0,
                    )
                    logger.debug(
                        "错误分类: reason=%s status=%s 可重试=%s 要压缩=%s 换密钥=%s 需要降级=%s",
                        classified.reason.value, classified.status_code,
                        classified.retryable, classified.should_compress,
                        classified.should_rotate_credential, classified.should_fallback,
                    )

                    recovered_with_pool, has_retried_429 = self._recover_with_credential_pool(
                        status_code=status_code,
                        has_retried_429=has_retried_429,
                        classified_reason=classified.reason,
                        error_context=error_context,
                    )
                    if recovered_with_pool:
                        continue
                    if (
                        self.api_mode == "codex_responses"
                        and self.provider == "openai-codex"
                        and status_code == 401
                        and not codex_auth_retry_attempted
                    ):
                        codex_auth_retry_attempted = True
                        if self._try_refresh_codex_client_credentials(force=True):
                            self._vprint(f"{self.log_prefix}🔐 Codex 401 认证已刷新，重试请求...")
                            continue
                    if (
                        self.api_mode == "chat_completions"
                        and self.provider == "nous"
                        and status_code == 401
                        and not nous_auth_retry_attempted
                    ):
                        nous_auth_retry_attempted = True
                        if self._try_refresh_nous_client_credentials(force=True):
                            print(f"{self.log_prefix}🔐 Nous agent key 认证已刷新，重试请求...")
                            continue
                    if (
                        self.api_mode == "anthropic_messages"
                        and status_code == 401
                        and hasattr(self, '_anthropic_api_key')
                        and not anthropic_auth_retry_attempted
                    ):
                        anthropic_auth_retry_attempted = True
                        from agent.anthropic_adapter import _is_oauth_token
                        if self._try_refresh_anthropic_client_credentials():
                            print(f"{self.log_prefix}🔐 Anthropic 401 认证已刷新，重试请求...")
                            continue
                        # 认证仍失败，显示详细诊断
                        key = self._anthropic_api_key
                        auth_method = "Bearer (OAuth/setup-token)" if _is_oauth_token(key) else "x-api-key（API key）"
                        print(f"{self.log_prefix}🔐 Anthropic 401 未通过认证。")
                        print(f"{self.log_prefix}   认证方式: {auth_method}")
                        print(f"{self.log_prefix}   Token 前缀: {key[:12]}..." if key and len(key) > 12 else f"{self.log_prefix}   Token: (空或太短)")
                        print(f"{self.log_prefix}   故障排查:")
                        from hermes_constants import display_hermes_home as _dhh_fn
                        _dhh = _dhh_fn()
                        print(f"{self.log_prefix}     • 检查 {_dhh}/.env 里的 ANTHROPIC_TOKEN（Hermes OAuth/setup 专用 token）")
                        print(f"{self.log_prefix}     • 检查 {_dhh}/.env 里的 ANTHROPIC_API_KEY（普通 API key/历史 token）")
                        print(f"{self.log_prefix}     • 普通 API key 可去 https://console.anthropic.com/settings/keys 校验")
                        print(f"{self.log_prefix}     • Claude Code 用户可用 'claude /login' 刷新再试")
                        print(f"{self.log_prefix}     • 清理遗留 token: hermes config set ANTHROPIC_TOKEN \"\"")
                        print(f"{self.log_prefix}     • 清理无效 key: hermes config set ANTHROPIC_API_KEY \"\"")

                    # ── 推理块签名失效自动恢复 ─────────────────
                    # Anthropic 会把 <think> 块做全回合签名，若 context 被压缩/合并/截断 -- 签名必然失效 HTTP400
                    # 恢复方式：剥离所有消息的 reasoning_details 字段（不带任何推理块），只重试一次
                    # ── 推理块签名失效自动恢复 ─────────────────
                    if (
                        classified.reason == FailoverReason.thinking_signature
                        and not thinking_sig_retry_attempted
                    ):
                        thinking_sig_retry_attempted = True
                        for _m in messages:
                            if isinstance(_m, dict):
                                _m.pop("reasoning_details", None)
                        self._vprint(
                            f"{self.log_prefix}⚠️  推理块签名无效——已去除所有推理块，正在重试……",
                            force=True,
                        )
                        logging.warning(
                            "%s推理块签名恢复：已从 %d 条消息中去除 reasoning_details",
                            self.log_prefix, len(messages),
                        )
                        continue

                    retry_count += 1
                    elapsed_time = time.time() - api_start_time
                    
                    error_type = type(api_error).__name__
                    error_msg = str(api_error).lower()
                    _error_summary = self._summarize_api_error(api_error)
                    logger.warning(
                        "API 请求失败（第 %s/%s 次尝试） error_type=%s %s summary=%s",
                        retry_count,
                        max_retries,
                        error_type,
                        self._client_log_context(),
                        _error_summary,
                    )

                    _provider = getattr(self, "provider", "unknown")
                    _base = getattr(self, "base_url", "unknown")
                    _model = getattr(self, "model", "unknown")
                    _status_code_str = f" [HTTP {status_code}]" if status_code else ""
                    self._vprint(f"{self.log_prefix}⚠️  API 调用失败（第 {retry_count}/{max_retries} 次）：{error_type}{_status_code_str}", force=True)
                    self._vprint(f"{self.log_prefix}   🔌 提供商: {_provider}  模型: {_model}", force=True)
                    self._vprint(f"{self.log_prefix}   🌐 终端: {_base}", force=True)
                    self._vprint(f"{self.log_prefix}   📝 错误: {_error_summary}", force=True)
                    if status_code and status_code < 500:
                        _err_body = getattr(api_error, "body", None)
                        _err_body_str = str(_err_body)[:300] if _err_body else None
                        if _err_body_str:
                            self._vprint(f"{self.log_prefix}   📋 详情: {_err_body_str}", force=True)
                    self._vprint(f"{self.log_prefix}   ⏱️  用时: {elapsed_time:.2f}s  对话上下文: {len(api_messages)} 条, ~{approx_tokens:,} tokens")

                    # OpenRouter“无工具端点”错误的可操作提示。
                    # 无论是否成功fallback都会触发——用户需要知道自己的模型为什么失败了，以便修正配置，而不是静默fallback。
                    if (
                        self._is_openrouter_url()
                        and "support tool use" in error_msg
                    ):
                        self._vprint(
                            f"{self.log_prefix}   💡 当前设置下没有支持工具调用的 OpenRouter 提供商可用于 {_model}。",
                            force=True,
                        )
                        if self.providers_allowed:
                            self._vprint(
                                f"{self.log_prefix}      你的 provider_routing.only 限制过滤掉了支持工具的提供商。",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      请移除该限制或添加支持该模型工具调用的提供商。",
                                force=True,
                            )
                        self._vprint(
                            f"{self.log_prefix}      查看哪些提供商支持工具调用: https://openrouter.ai/models/{_model}",
                            force=True,
                        )

                    # 在决定重试前检查中断
                    if self._interrupt_requested:
                        self._vprint(f"{self.log_prefix}⚡ 错误处理期间检测到中断，终止重试。", force=True)
                        self._persist_session(messages, conversation_history)
                        self.clear_interrupt()
                        return {
                            "final_response": f"操作被中断：处理中 API 错误时（{error_type}: {self._clean_error_message(str(api_error))}）。",
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "interrupted": True,
                        }
                    
                    # 在通用 4xx 处理前，先检查 413（Payload Too Large）。
                    # 413 表示内容过大，正确做法是压缩历史并重试，而不是直接中断。
                    status_code = getattr(api_error, "status_code", None)

                    # ── Anthropic Sonnet 长上下文订阅等级限制 ───────────
                    # Anthropic 当 Claude Max（或类似套餐）未开通 1M context 时，长上下文请求返回 HTTP 429“需要额外费用”。
                    # 这不是短暂的限流，重试或切换授权无效。应将 context 降为 20W 并压缩。
                    if classified.reason == FailoverReason.long_context_tier:
                        _reduced_ctx = 200000
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length
                        if old_ctx > _reduced_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=_reduced_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            # context 探测标记，只对内置 compressor 生效（插件引擎自管理）
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # 不持久化——此为套餐限制非模型能力。如果用户后续升级套餐，1M 限额应自动恢复。
                                compressor._context_probe_persistable = False
                            self._vprint(
                                f"{self.log_prefix}⚠️  Anthropic 长上下文订阅受限，需要额外配额——context 由 {old_ctx:,} → {_reduced_ctx:,} tokens",
                                force=True,
                            )

                        compression_attempts += 1
                        if compression_attempts <= max_compression_attempts:
                            original_len = len(messages)
                            messages, active_system_prompt = self._compress_context(
                                messages, system_message,
                                approx_tokens=approx_tokens,
                                task_id=effective_task_id,
                            )
                            # 压缩会生成新 session——清空历史，以便 _flush_messages_to_session_db 正确写入。
                            conversation_history = None
                            if len(messages) < original_len or old_ctx > _reduced_ctx:
                                self._emit_status(
                                    f"🗜️ context 缩减至 {_reduced_ctx:,} tokens（原 {old_ctx:,}），重试中……"
                                )
                                time.sleep(2)
                                restart_with_compressed_messages = True
                                break
                        # 如果压缩用尽或无效，则进入常规错误处理

                    # Eager fallback for rate-limit errors (429 or quota exhaustion).
                    # When a fallback model is configured, switch immediately instead
                    # of burning through retries with exponential backoff -- the
                    # primary provider won't recover within the retry window.
                    is_rate_limited = classified.reason in (
                        FailoverReason.rate_limit,
                        FailoverReason.billing,
                    )
                    if is_rate_limited and self._fallback_index < len(self._fallback_chain):
                        # Don't eagerly fallback if credential pool rotation may
                        # still recover.  The pool's retry-then-rotate cycle needs
                        # at least one more attempt to fire — jumping to a fallback
                        # provider here short-circuits it.
                        pool = self._credential_pool
                        pool_may_recover = pool is not None and pool.has_available()
                        if not pool_may_recover:
                            self._emit_status("⚠️ 触发限流，切换到备用提供商…")
                            if self._try_activate_fallback():
                                retry_count = 0
                                compression_attempts = 0
                                primary_recovery_attempted = False
                                continue

                    is_payload_too_large = (
                        classified.reason == FailoverReason.payload_too_large
                    )

                    if is_payload_too_large:
                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ 超过最大压缩次数（{max_compression_attempts}），依然 payload-too-large。", force=True)
                            self._vprint(f"{self.log_prefix}   💡 可用 /new 开启新对话，或 /compress 重试压缩。", force=True)
                            logging.error(f"{self.log_prefix}413 payload 编码压缩失败，已达 {max_compression_attempts} 次。")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"请求内容过大。最大压缩次数（{max_compression_attempts}）已达。",
                                "partial": True
                            }
                        self._emit_status(f"⚠️  请求内容过大（413）—第 {compression_attempts}/{max_compression_attempts} 次压缩尝试…")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None

                        if len(messages) < original_len:
                            self._emit_status(f"🗜️ 已压缩消息数：{original_len} → {len(messages)}，重试中……")
                            time.sleep(2)
                            restart_with_compressed_messages = True
                            break
                        else:
                            self._vprint(f"{self.log_prefix}❌ 仍然内容溢出，无法进一步压缩。", force=True)
                            self._vprint(f"{self.log_prefix}   💡 可用 /new 开启新对话，或 /compress 重试压缩。", force=True)
                            logging.error(f"{self.log_prefix}413 内容超限，无法进一步压缩。")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": "请求内容过大（413），无法进一步压缩。",
                                "partial": True
                            }

                    # 优先处理 context-overflow（超长上下文）错误，早于通用 4xx
                    # 分类器能从显式错误消息、泛 400+大 session、服务器断连+大 session 等情况辨别 context overflow
                    is_context_length_error = (
                        classified.reason == FailoverReason.context_overflow
                    )

                    if is_context_length_error:
                        compressor = self.context_compressor
                        old_ctx = compressor.context_length

                        # ── 两类 context 错误区别严格 ───────────
                        # 1. "Prompt 太长": 输入超出 context window。需减小 context_length+压缩历史。
                        # 2. "max_tokens 太大": 输入没溢出，上限=输入+输出。需减小 output cap。
                        #    不要缩小 context_length，本次窗口总长度不变。
                        #
                        # max_tokens = 输出上限
                        # context_length = 总上下文窗口
                        available_out = parse_available_output_tokens_from_error(error_msg)
                        if available_out is not None:
                            # 错误是输出上限太高——只调整 max_tokens，无需压缩
                            safe_out = max(1, available_out - 64)  # 安全冗余
                            self._ephemeral_max_output_tokens = safe_out
                            self._vprint(
                                f"{self.log_prefix}⚠️  当前 prompt 输出上限过大—自动尝试 max_tokens={safe_out:,}（可用={available_out:,}，context 长度不变 {old_ctx:,}）",
                                force=True,
                            )
                            # 这类情况依然计入压缩次数，避免循环
                            compression_attempts += 1
                            if compression_attempts > max_compression_attempts:
                                self._vprint(f"{self.log_prefix}❌ 超过最大压缩次数（{max_compression_attempts}）。", force=True)
                                self._vprint(f"{self.log_prefix}   💡 可用 /new 开启新对话，或 /compress 重试压缩。", force=True)
                                logging.error(f"{self.log_prefix}context 编码压缩失败，已达 {max_compression_attempts} 次。")
                                self._persist_session(messages, conversation_history)
                                return {
                                    "messages": messages,
                                    "completed": False,
                                    "api_calls": api_call_count,
                                    "error": f"上下文过长：最大压缩次数（{max_compression_attempts}）已达。",
                                    "partial": True
                                }
                            restart_with_compressed_messages = True
                            break

                        # 错误为输入过大——需缩小 context_length
                        parsed_limit = parse_context_limit_from_error(error_msg)
                        if parsed_limit and parsed_limit < old_ctx:
                            new_ctx = parsed_limit
                            self._vprint(f"{self.log_prefix}⚠️  API 检测到 context 限额：{new_ctx:,}（原 {old_ctx:,}）", force=True)
                        else:
                            new_ctx = get_next_probe_tier(old_ctx)

                        if new_ctx and new_ctx < old_ctx:
                            compressor.update_model(
                                model=self.model,
                                context_length=new_ctx,
                                base_url=self.base_url,
                                api_key=getattr(self, "api_key", ""),
                                provider=self.provider,
                            )
                            if hasattr(compressor, "_context_probed"):
                                compressor._context_probed = True
                                # 只有真实解析出来的新限额才持久化；降级猜测 tier 不存 cache
                                compressor._context_probe_persistable = bool(
                                    parsed_limit and parsed_limit == new_ctx
                                )
                            self._vprint(f"{self.log_prefix}⚠️  上下文长度超限—step down: {old_ctx:,} → {new_ctx:,} tokens", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}⚠️  已达 context 最小 tier，尝试压缩历史…", force=True)

                        compression_attempts += 1
                        if compression_attempts > max_compression_attempts:
                            self._vprint(f"{self.log_prefix}❌ 超过最大压缩次数（{max_compression_attempts}）。", force=True)
                            self._vprint(f"{self.log_prefix}   💡 可用 /new 开启新对话，或 /compress 重试压缩。", force=True)
                            logging.error(f"{self.log_prefix}context 编码压缩失败，已达 {max_compression_attempts} 次。")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"上下文过长：最大压缩次数（{max_compression_attempts}）已达。",
                                "partial": True
                            }
                        self._emit_status(f"🗜️ 历史内容过大（约 {approx_tokens:,} tokens）—第 {compression_attempts}/{max_compression_attempts} 次压缩…")

                        original_len = len(messages)
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message, approx_tokens=approx_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history
                        # so _flush_messages_to_session_db writes compressed
                        # messages to the new session, not skipping them.
                        conversation_history = None

                        if len(messages) < original_len or new_ctx and new_ctx < old_ctx:
                            if len(messages) < original_len:
                                self._emit_status(f"🗜️ 已压缩消息数：{original_len} → {len(messages)}，重试中……")
                            time.sleep(2)
                            restart_with_compressed_messages = True
                            break
                        else:
                            # 不能再压缩且已到最小 tier，报错退出
                            self._vprint(f"{self.log_prefix}❌ 上下文长度超限且无法进一步压缩。", force=True)
                            self._vprint(f"{self.log_prefix}   💡 内容已积累过多。可 /new 重新开始或 /compress 强制压缩。", force=True)
                            logging.error(f"{self.log_prefix}context 超长：{approx_tokens:,} tokens。无法再压缩。")
                            self._persist_session(messages, conversation_history)
                            return {
                                "messages": messages,
                                "completed": False,
                                "api_calls": api_call_count,
                                "error": f"上下文过长（{approx_tokens:,} tokens）。无法进一步压缩。",
                                "partial": True
                            }

                    # 检查非可重试客户端错误。分类器已覆盖 413、429、529(transient)、context overflow、通用400等。ValueError、TypeError为本地校验bug。
                    is_local_validation_error = (
                        isinstance(api_error, (ValueError, TypeError))
                        and not isinstance(api_error, UnicodeEncodeError)
                    )
                    is_client_error = (
                        is_local_validation_error
                        or (
                            not classified.retryable
                            and not classified.should_compress
                            and classified.reason not in (
                                FailoverReason.rate_limit,
                                FailoverReason.billing,
                                FailoverReason.overloaded,
                                FailoverReason.context_overflow,
                                FailoverReason.payload_too_large,
                                FailoverReason.long_context_tier,
                                FailoverReason.thinking_signature,
                            )
                        )
                    ) and not is_context_length_error

                    if is_client_error:
                        # 优先尝试 fallback
                        self._emit_status(f"⚠️ 非可重试错误（HTTP {status_code}）—尝试备用…")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="non_retryable_client_error", error=api_error,
                            )
                        self._emit_status(
                            f"❌ 非可重试错误（HTTP {status_code}）：{self._summarize_api_error(api_error)}"
                        )
                        self._vprint(f"{self.log_prefix}❌ 非可重试客户端错误（HTTP {status_code}）。终止。", force=True)
                        self._vprint(f"{self.log_prefix}   🔌 提供商: {_provider}  模型: {_model}", force=True)
                        self._vprint(f"{self.log_prefix}   🌐 终端: {_base}", force=True)
                        # 针对常见认证类错误的指引
                        if classified.is_auth or classified.reason == FailoverReason.billing:
                            if _provider == "openai-codex" and status_code == 401:
                                self._vprint(f"{self.log_prefix}   💡 Codex OAuth token 被拒绝（HTTP 401）。你的 token 可能被其他客户端刷新了（Codex CLI、VS Code）。修复方法：", force=True)
                                self._vprint(f"{self.log_prefix}      1. 终端运行 codex 生成新 token。", force=True)
                                self._vprint(f"{self.log_prefix}      2. 运行 hermes auth 重新认证。", force=True)
                            else:
                                self._vprint(f"{self.log_prefix}   💡 API key 被提供商拒绝，请检查：", force=True)
                                self._vprint(f"{self.log_prefix}      • KEY 是否有效？hermes setup", force=True)
                                self._vprint(f"{self.log_prefix}      • 账户是否有权限访问 {_model}？", force=True)
                                if "openrouter" in str(_base).lower():
                                    self._vprint(f"{self.log_prefix}      • 查看余额：https://openrouter.ai/settings/credits", force=True)
                        else:
                            self._vprint(f"{self.log_prefix}   💡 该类错误重试无效。", force=True)
                        logging.error(f"{self.log_prefix}非可重试客户端错误：{api_error}")
                        # 大 session 的 context-overflow 错误（status=400且消息超长）跳过持久化，避免增长循环
                        if status_code == 400 and (approx_tokens > 50000 or len(api_messages) > 80):
                            self._vprint(
                                f"{self.log_prefix}⚠️  大 session 错误跳过持久化，防止死循环增长。",
                                force=True,
                            )
                        else:
                            self._persist_session(messages, conversation_history)
                        return {
                            "final_response": None,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": str(api_error),
                        }

                    if retry_count >= max_retries:
                        # 用尽重试前，对短暂传输类错误（连接池/断链）重建主 client，仅一次
                        if not primary_recovery_attempted and self._try_recover_primary_transport(
                            api_error, retry_count=retry_count, max_retries=max_retries,
                        ):
                            primary_recovery_attempted = True
                            retry_count = 0
                            continue
                        self._emit_status(f"⚠️ 达最大重试次数（{max_retries}），尝试备用…")
                        if self._try_activate_fallback():
                            retry_count = 0
                            compression_attempts = 0
                            primary_recovery_attempted = False
                            continue
                        _final_summary = self._summarize_api_error(api_error)
                        if is_rate_limited:
                            self._emit_status(f"❌ 多次重试后被限流 — {_final_summary}")
                        else:
                            self._emit_status(f"❌ 多次重试失败 — {_final_summary}")
                        self._vprint(f"{self.log_prefix}   💀 最终错误：{_final_summary}", force=True)

                        # 检测 SSE 流丢失模式，给出操作建议
                        _is_stream_drop = (
                            not getattr(api_error, "status_code", None)
                            and any(p in error_msg for p in (
                                "connection lost", "connection reset",
                                "connection closed", "network connection",
                                "network error", "terminated",
                            ))
                        )
                        if _is_stream_drop:
                            self._vprint(
                                f"{self.log_prefix}   💡 服务端 SSE 流连接反复丢失，通常是模型生成了非常大的工具调用（如巨型文件），被代理/CDN 中断。",
                                force=True,
                            )
                            self._vprint(
                                f"{self.log_prefix}      可请求模型用 execute_code 和 Python open() 分批写大文件，或改为分段输出。",
                                force=True,
                            )

                        logging.error(
                            "%sAPI 调用重试 %s 次后失败。%s | provider=%s model=%s msgs=%s tokens=~%s",
                            self.log_prefix, max_retries, _final_summary,
                            _provider, _model, len(api_messages), f"{approx_tokens:,}",
                        )
                        if api_kwargs is not None:
                            self._dump_api_request_debug(
                                api_kwargs, reason="max_retries_exhausted", error=api_error,
                            )
                        self._persist_session(messages, conversation_history)
                        _final_response = f"API 调用多次（{max_retries}）重试后失败: {_final_summary}"
                        if _is_stream_drop:
                            _final_response += (
                                "\n\n服务端 SSE 流连接反复被中断——多为生成超大工具/文件（如 write_file）造成。"
                                "可请求我用 execute_code/Python open() 分批写大文件，或拆小多段写。"
                            )
                        return {
                            "final_response": _final_response,
                            "messages": messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "failed": True,
                            "error": _final_summary,
                        }

                    # 针对限流，若返回 Header 里有 Retry-After，则遵循
                    _retry_after = None
                    if is_rate_limited:
                        _resp_headers = getattr(getattr(api_error, "response", None), "headers", None)
                        if _resp_headers and hasattr(_resp_headers, "get"):
                            _ra_raw = _resp_headers.get("retry-after") or _resp_headers.get("Retry-After")
                            if _ra_raw:
                                try:
                                    _retry_after = min(int(_ra_raw), 120)  # 最多等两分钟
                                except (TypeError, ValueError):
                                    pass
                    wait_time = _retry_after if _retry_after else jittered_backoff(retry_count, base_delay=2.0, max_delay=60.0)
                    if is_rate_limited:
                        self._emit_status(f"⏱️ 已限流，等待 {wait_time}s 后重试（第 {retry_count + 1}/{max_retries} 次）…")
                    else:
                        self._emit_status(f"⏳ {wait_time}s 后重试（第 {retry_count}/{max_retries} 次）…")
                    logger.warning(
                        "API 调用将在 %ss 后重试（第 %s/%s 次）%s error=%s",
                        wait_time,
                        retry_count,
                        max_retries,
                        self._client_log_context(),
                        api_error,
                    )
                    # 小步 sleep，允许快速响应中断
                    sleep_end = time.time() + wait_time
                    while time.time() < sleep_end:
                        if self._interrupt_requested:
                            self._vprint(f"{self.log_prefix}⚡ 重试等待中检测到中断，终止。", force=True)
                            self._persist_session(messages, conversation_history)
                            self.clear_interrupt()
                            return {
                                "final_response": f"操作被中断：等待重试期间（第 {retry_count}/{max_retries} 次）。",
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "interrupted": True,
                            }
                        time.sleep(0.2)  # 每 200ms 检查一次中断
            
            # 如果 API 调用被中断，跳过后续处理
            if interrupted:
                _turn_exit_reason = "interrupted_during_api_call"
                break

            if restart_with_compressed_messages:
                api_call_count -= 1
                self.iteration_budget.refund()
                # 计入 retry 以防压缩反复失败后死循环
                retry_count += 1
                restart_with_compressed_messages = False
                continue

            if restart_with_length_continuation:
                continue

            # 若多次重试依然返回 None，则退出
            if response is None:
                _turn_exit_reason = "all_retries_exhausted_no_response"
                print(f"{self.log_prefix}❌ 已用完所有 API 重试机会，未获成功响应。")
                self._persist_session(messages, conversation_history)
                break

            try:
                if self.api_mode == "codex_responses":
                    assistant_message, finish_reason = self._normalize_codex_response(response)
                elif self.api_mode == "anthropic_messages":
                    from agent.anthropic_adapter import normalize_anthropic_response
                    assistant_message, finish_reason = normalize_anthropic_response(
                        response, strip_tool_prefix=self._is_anthropic_oauth
                    )
                else:
                    assistant_message = response.choices[0].message
                
                # content 字段统一为字符串——有些 OpenAI 兼容 LLM （llama-server等）会返回 dict 或 list，
                # 下游如果 .strip() 直接用会报错。
                if assistant_message.content is not None and not isinstance(assistant_message.content, str):
                    raw = assistant_message.content
                    if isinstance(raw, dict):
                        assistant_message.content = raw.get("text", "") or raw.get("content", "") or json.dumps(raw)
                    elif isinstance(raw, list):
                        # 多模态 content list，抽取文字部分
                        parts = []
                        for part in raw:
                            if isinstance(part, str):
                                parts.append(part)
                            elif isinstance(part, dict) and part.get("type") == "text":
                                parts.append(part.get("text", ""))
                            elif isinstance(part, dict) and "text" in part:
                                parts.append(str(part["text"]))
                        assistant_message.content = "\n".join(parts)
                    else:
                        assistant_message.content = str(raw)

                try:
                    from hermes_cli.plugins import invoke_hook as _invoke_hook
                    _assistant_tool_calls = getattr(assistant_message, "tool_calls", None) or []
                    _assistant_text = assistant_message.content or ""
                    _invoke_hook(
                        "post_api_request",
                        task_id=effective_task_id,
                        session_id=self.session_id or "",
                        platform=self.platform or "",
                        model=self.model,
                        provider=self.provider,
                        base_url=self.base_url,
                        api_mode=self.api_mode,
                        api_call_count=api_call_count,
                        api_duration=api_duration,
                        finish_reason=finish_reason,
                        message_count=len(api_messages),
                        response_model=getattr(response, "model", None),
                        usage=self._usage_summary_for_api_request_hook(response),
                        assistant_content_chars=len(_assistant_text),
                        assistant_tool_call_count=len(_assistant_tool_calls),
                    )
                except Exception:
                    pass

                # 处理助手响应
                if assistant_message.content and not self.quiet_mode:
                    if self.verbose_logging:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content}")
                    else:
                        self._vprint(f"{self.log_prefix}🤖 Assistant: {assistant_message.content[:100]}{'...' if len(assistant_message.content) > 100 else ''}")

                # 通知进度回调，转发“推理”内容用于子代理/父展示等
                if (assistant_message.content and self.tool_progress_callback):
                    _think_text = assistant_message.content.strip()
                    # 去除 reasoning/think 等标签，避免泄漏到父代理端展示
                    _think_text = re.sub(
                        r'</?(?:REASONING_SCRATCHPAD|think|reasoning)>', '', _think_text
                    ).strip()
                    # 子代理：第一行转发到父端（保持原逻辑）。
                    # 所有结构化回调则发 reasoning.available 事件。
                    first_line = _think_text.split('\n')[0][:80] if _think_text else ""
                    if first_line and getattr(self, '_delegate_depth', 0) > 0:
                        try:
                            self.tool_progress_callback("_thinking", first_line)
                        except Exception:
                            pass
                    elif _think_text:
                        try:
                            self.tool_progress_callback("reasoning.available", "_thinking", _think_text[:500], None)
                        except Exception:
                            pass
                
                # Check for incomplete <REASONING_SCRATCHPAD> (opened but never closed)
                # This means the model ran out of output tokens mid-reasoning — retry up to 2 times
                if has_incomplete_scratchpad(assistant_message.content or ""):
                    self._incomplete_scratchpad_retries += 1
                    
                    self._vprint(f"{self.log_prefix}⚠️  Incomplete <REASONING_SCRATCHPAD> detected (opened but never closed)")
                    
                    if self._incomplete_scratchpad_retries <= 2:
                        self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._incomplete_scratchpad_retries}/2)...")
                        # Don't add the broken message, just retry
                        continue
                    else:
                        # Max retries - discard this turn and save as partial
                        self._vprint(f"{self.log_prefix}❌ Max retries (2) for incomplete scratchpad. Saving as partial.", force=True)
                        self._incomplete_scratchpad_retries = 0
                        
                        rolled_back_messages = self._get_messages_up_to_last_assistant(messages)
                        self._cleanup_task_resources(effective_task_id)
                        self._persist_session(messages, conversation_history)
                        
                        return {
                            "final_response": None,
                            "messages": rolled_back_messages,
                            "api_calls": api_call_count,
                            "completed": False,
                            "partial": True,
                            "error": "Incomplete REASONING_SCRATCHPAD after 2 retries"
                        }
                
                # Reset incomplete scratchpad counter on clean response
                self._incomplete_scratchpad_retries = 0

                if self.api_mode == "codex_responses" and finish_reason == "incomplete":
                    self._codex_incomplete_retries += 1

                    interim_msg = self._build_assistant_message(assistant_message, finish_reason)
                    interim_has_content = bool((interim_msg.get("content") or "").strip())
                    interim_has_reasoning = bool(interim_msg.get("reasoning", "").strip()) if isinstance(interim_msg.get("reasoning"), str) else False
                    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))

                    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning:
                        last_msg = messages[-1] if messages else None
                        # Duplicate detection: two consecutive incomplete assistant
                        # messages with identical content AND reasoning are collapsed.
                        # For reasoning-only messages (codex_reasoning_items differ but
                        # visible content/reasoning are both empty), we also compare
                        # the encrypted items to avoid silently dropping new state.
                        last_codex_items = last_msg.get("codex_reasoning_items") if isinstance(last_msg, dict) else None
                        interim_codex_items = interim_msg.get("codex_reasoning_items")
                        duplicate_interim = (
                            isinstance(last_msg, dict)
                            and last_msg.get("role") == "assistant"
                            and last_msg.get("finish_reason") == "incomplete"
                            and (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                            and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
                            and last_codex_items == interim_codex_items
                        )
                        if not duplicate_interim:
                            messages.append(interim_msg)
                            self._emit_interim_assistant_message(interim_msg)

                    if self._codex_incomplete_retries < 3:
                        if not self.quiet_mode:
                            self._vprint(f"{self.log_prefix}↻ Codex response incomplete; continuing turn ({self._codex_incomplete_retries}/3)")
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    self._codex_incomplete_retries = 0
                    self._persist_session(messages, conversation_history)
                    return {
                        "final_response": None,
                        "messages": messages,
                        "api_calls": api_call_count,
                        "completed": False,
                        "partial": True,
                        "error": "Codex response remained incomplete after 3 continuation attempts",
                    }
                elif hasattr(self, "_codex_incomplete_retries"):
                    self._codex_incomplete_retries = 0
                
                # Check for tool calls
                if assistant_message.tool_calls:
                    if not self.quiet_mode:
                        self._vprint(f"{self.log_prefix}🔧 Processing {len(assistant_message.tool_calls)} tool call(s)...")
                    
                    if self.verbose_logging:
                        for tc in assistant_message.tool_calls:
                            logging.debug(f"Tool call: {tc.function.name} with args: {tc.function.arguments[:200]}...")
                    
                    # Validate tool call names - detect model hallucinations
                    # Repair mismatched tool names before validating
                    for tc in assistant_message.tool_calls:
                        if tc.function.name not in self.valid_tool_names:
                            repaired = self._repair_tool_call(tc.function.name)
                            if repaired:
                                print(f"{self.log_prefix}🔧 Auto-repaired tool name: '{tc.function.name}' -> '{repaired}'")
                                tc.function.name = repaired
                    invalid_tool_calls = [
                        tc.function.name for tc in assistant_message.tool_calls
                        if tc.function.name not in self.valid_tool_names
                    ]
                    if invalid_tool_calls:
                        # Track retries for invalid tool calls
                        self._invalid_tool_retries += 1

                        # Return helpful error to model — model can self-correct next turn
                        available = ", ".join(sorted(self.valid_tool_names))
                        invalid_name = invalid_tool_calls[0]
                        invalid_preview = invalid_name[:80] + "..." if len(invalid_name) > 80 else invalid_name
                        self._vprint(f"{self.log_prefix}⚠️  Unknown tool '{invalid_preview}' — sending error to model for self-correction ({self._invalid_tool_retries}/3)")

                        if self._invalid_tool_retries >= 3:
                            self._vprint(f"{self.log_prefix}❌ Max retries (3) for invalid tool calls exceeded. Stopping as partial.", force=True)
                            self._invalid_tool_retries = 0
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": f"Model generated invalid tool call: {invalid_preview}"
                            }

                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        messages.append(assistant_msg)
                        for tc in assistant_message.tool_calls:
                            if tc.function.name not in self.valid_tool_names:
                                content = f"Tool '{tc.function.name}' does not exist. Available tools: {available}"
                            else:
                                content = "Skipped: another tool call in this turn used an invalid name. Please retry this tool call."
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": content,
                            })
                        continue
                    # Reset retry counter on successful tool call validation
                    self._invalid_tool_retries = 0
                    
                    # Validate tool call arguments are valid JSON
                    # Handle empty strings as empty objects (common model quirk)
                    invalid_json_args = []
                    for tc in assistant_message.tool_calls:
                        args = tc.function.arguments
                        if isinstance(args, (dict, list)):
                            tc.function.arguments = json.dumps(args)
                            continue
                        if args is not None and not isinstance(args, str):
                            tc.function.arguments = str(args)
                            args = tc.function.arguments
                        # Treat empty/whitespace strings as empty object
                        if not args or not args.strip():
                            tc.function.arguments = "{}"
                            continue
                        try:
                            json.loads(args)
                        except json.JSONDecodeError as e:
                            invalid_json_args.append((tc.function.name, str(e)))
                    
                    if invalid_json_args:
                        # Check if the invalid JSON is due to truncation rather
                        # than a model formatting mistake.  Routers sometimes
                        # rewrite finish_reason from "length" to "tool_calls",
                        # hiding the truncation from the length handler above.
                        # Detect truncation: args that don't end with } or ]
                        # (after stripping whitespace) are cut off mid-stream.
                        _truncated = any(
                            not (tc.function.arguments or "").rstrip().endswith(("}", "]"))
                            for tc in assistant_message.tool_calls
                            if tc.function.name in {n for n, _ in invalid_json_args}
                        )
                        if _truncated:
                            self._vprint(
                                f"{self.log_prefix}⚠️  Truncated tool call arguments detected "
                                f"(finish_reason={finish_reason!r}) — refusing to execute.",
                                force=True,
                            )
                            self._invalid_json_retries = 0
                            self._cleanup_task_resources(effective_task_id)
                            self._persist_session(messages, conversation_history)
                            return {
                                "final_response": None,
                                "messages": messages,
                                "api_calls": api_call_count,
                                "completed": False,
                                "partial": True,
                                "error": "Response truncated due to output length limit",
                            }

                        # Track retries for invalid JSON arguments
                        self._invalid_json_retries += 1

                        tool_name, error_msg = invalid_json_args[0]
                        self._vprint(f"{self.log_prefix}⚠️  Invalid JSON in tool call arguments for '{tool_name}': {error_msg}")

                        if self._invalid_json_retries < 3:
                            self._vprint(f"{self.log_prefix}🔄 Retrying API call ({self._invalid_json_retries}/3)...")
                            # Don't add anything to messages, just retry the API call
                            continue
                        else:
                            # Instead of returning partial, inject tool error results so the model can recover.
                            # Using tool results (not user messages) preserves role alternation.
                            self._vprint(f"{self.log_prefix}⚠️  Injecting recovery tool results for invalid JSON...")
                            self._invalid_json_retries = 0  # Reset for next attempt
                            
                            # Append the assistant message with its (broken) tool_calls
                            recovery_assistant = self._build_assistant_message(assistant_message, finish_reason)
                            messages.append(recovery_assistant)
                            
                            # Respond with tool error results for each tool call
                            invalid_names = {name for name, _ in invalid_json_args}
                            for tc in assistant_message.tool_calls:
                                if tc.function.name in invalid_names:
                                    err = next(e for n, e in invalid_json_args if n == tc.function.name)
                                    tool_result = (
                                        f"Error: Invalid JSON arguments. {err}. "
                                        f"For tools with no required parameters, use an empty object: {{}}. "
                                        f"Please retry with valid JSON."
                                    )
                                else:
                                    tool_result = "Skipped: other tool call in this response had invalid JSON."
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                })
                            continue
                    
                    # Reset retry counter on successful JSON validation
                    self._invalid_json_retries = 0

                    # ── Post-call guardrails ──────────────────────────
                    assistant_message.tool_calls = self._cap_delegate_task_calls(
                        assistant_message.tool_calls
                    )
                    assistant_message.tool_calls = self._deduplicate_tool_calls(
                        assistant_message.tool_calls
                    )

                    assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                    
                    # If this turn has both content AND tool_calls, capture the content
                    # as a fallback final response. Common pattern: model delivers its
                    # answer and calls memory/skill tools as a side-effect in the same
                    # turn. If the follow-up turn after tools is empty, we use this.
                    turn_content = assistant_message.content or ""
                    if turn_content and self._has_content_after_think_block(turn_content):
                        self._last_content_with_tools = turn_content
                        # Only mute subsequent output when EVERY tool call in
                        # this turn is post-response housekeeping (memory, todo,
                        # skill_manage, etc.).  If any substantive tool is present
                        # (search_files, read_file, write_file, terminal, ...),
                        # keep output visible so the user sees progress.
                        _HOUSEKEEPING_TOOLS = frozenset({
                            "memory", "todo", "skill_manage", "session_search",
                        })
                        _all_housekeeping = all(
                            tc.function.name in _HOUSEKEEPING_TOOLS
                            for tc in assistant_message.tool_calls
                        )
                        if _all_housekeeping and self._has_stream_consumers():
                            self._mute_post_response = True
                        elif self.quiet_mode:
                            clean = self._strip_think_blocks(turn_content).strip()
                            if clean:
                                self._vprint(f"  ┊ 💬 {clean}")
                    
                    # Pop thinking-only prefill message(s) before appending
                    # (tool-call path — same rationale as the final-response path).
                    _had_prefill = False
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()
                        _had_prefill = True

                    # Reset prefill counter when tool calls follow a prefill
                    # recovery.  Without this, the counter accumulates across
                    # the whole conversation — a model that intermittently
                    # empties (empty → prefill → tools → empty → prefill →
                    # tools) burns both prefill attempts and the third empty
                    # gets zero recovery.  Resetting here treats each tool-
                    # call success as a fresh start.
                    if _had_prefill:
                        self._thinking_prefill_retries = 0
                        self._empty_content_retries = 0

                    messages.append(assistant_msg)
                    self._emit_interim_assistant_message(assistant_msg)

                    # Close any open streaming display (response box, reasoning
                    # box) before tool execution begins.  Intermediate turns may
                    # have streamed early content that opened the response box;
                    # flushing here prevents it from wrapping tool feed lines.
                    # Only signal the display callback — TTS (_stream_callback)
                    # should NOT receive None (it uses None as end-of-stream).
                    if self.stream_delta_callback:
                        try:
                            self.stream_delta_callback(None)
                        except Exception:
                            pass

                    self._execute_tool_calls(assistant_message, messages, effective_task_id, api_call_count)

                    # Reset per-turn retry counters after successful tool
                    # execution so a single truncation doesn't poison the
                    # entire conversation.
                    truncated_tool_call_retries = 0

                    # Signal that a paragraph break is needed before the next
                    # streamed text.  We don't emit it immediately because
                    # multiple consecutive tool iterations would stack up
                    # redundant blank lines.  Instead, _fire_stream_delta()
                    # will prepend a single "\n\n" the next time real text
                    # arrives.
                    self._stream_needs_break = True

                    # Refund the iteration if the ONLY tool(s) called were
                    # execute_code (programmatic tool calling).  These are
                    # cheap RPC-style calls that shouldn't eat the budget.
                    _tc_names = {tc.function.name for tc in assistant_message.tool_calls}
                    if _tc_names == {"execute_code"}:
                        self.iteration_budget.refund()
                    
                    # Use real token counts from the API response to decide
                    # compression.  prompt_tokens + completion_tokens is the
                    # actual context size the provider reported plus the
                    # assistant turn — a tight lower bound for the next prompt.
                    # Tool results appended above aren't counted yet, but the
                    # threshold (default 50%) leaves ample headroom; if tool
                    # results push past it, the next API call will report the
                    # real total and trigger compression then.
                    #
                    # If last_prompt_tokens is 0 (stale after API disconnect
                    # or provider returned no usage data), fall back to rough
                    # estimate to avoid missing compression.  Without this,
                    # a session can grow unbounded after disconnects because
                    # should_compress(0) never fires.  (#2153)
                    _compressor = self.context_compressor
                    if _compressor.last_prompt_tokens > 0:
                        _real_tokens = (
                            _compressor.last_prompt_tokens
                            + _compressor.last_completion_tokens
                        )
                    else:
                        _real_tokens = estimate_messages_tokens_rough(messages)

                    # ── Context pressure warnings (user-facing only) ──────────
                    # Notify the user (NOT the LLM) as context approaches the
                    # compaction threshold.  Thresholds are relative to where
                    # compaction fires, not the raw context window.
                    # Does not inject into messages — just prints to CLI output
                    # and fires status_callback for gateway platforms.
                    # Tiered: 85% (orange) and 95% (red/critical).
                    if _compressor.threshold_tokens > 0:
                        _compaction_progress = _real_tokens / _compressor.threshold_tokens
                        # Determine the warning tier for this progress level
                        _warn_tier = 0.0
                        if _compaction_progress >= 0.95:
                            _warn_tier = 0.95
                        elif _compaction_progress >= 0.85:
                            _warn_tier = 0.85
                        if _warn_tier > self._context_pressure_warned_at:
                            # Class-level dedup: check if this session was already
                            # warned at this tier within the cooldown window.
                            _sid = self.session_id or "default"
                            _last = AIAgent._context_pressure_last_warned.get(_sid)
                            _now = time.time()
                            if _last is None or _last[0] < _warn_tier or (_now - _last[1]) >= self._CONTEXT_PRESSURE_COOLDOWN:
                                self._context_pressure_warned_at = _warn_tier
                                AIAgent._context_pressure_last_warned[_sid] = (_warn_tier, _now)
                                self._emit_context_pressure(_compaction_progress, _compressor)
                                # Evict stale entries (older than 2x cooldown)
                                _cutoff = _now - self._CONTEXT_PRESSURE_COOLDOWN * 2
                                AIAgent._context_pressure_last_warned = {
                                    k: v for k, v in AIAgent._context_pressure_last_warned.items()
                                    if v[1] > _cutoff
                                }

                    if self.compression_enabled and _compressor.should_compress(_real_tokens):
                        self._safe_print("  ⟳ compacting context…")
                        messages, active_system_prompt = self._compress_context(
                            messages, system_message,
                            approx_tokens=self.context_compressor.last_prompt_tokens,
                            task_id=effective_task_id,
                        )
                        # Compression created a new session — clear history so
                        # _flush_messages_to_session_db writes compressed messages
                        # to the new session (see preflight compression comment).
                        conversation_history = None
                    
                    # Save session log incrementally (so progress is visible even if interrupted)
                    self._session_messages = messages
                    self._save_session_log(messages)
                    
                    # Continue loop for next response
                    continue
                
                else:
                    # No tool calls - this is the final response
                    final_response = assistant_message.content or ""
                    
                    # Check if response only has think block with no actual content after it
                    if not self._has_content_after_think_block(final_response):
                        # If the previous turn already delivered real content alongside
                        # tool calls (e.g. "You're welcome!" + memory save), the model
                        # has nothing more to say. Use the earlier content immediately
                        # instead of wasting API calls on retries that won't help.
                        fallback = getattr(self, '_last_content_with_tools', None)
                        if fallback:
                            _turn_exit_reason = "fallback_prior_turn_content"
                            logger.info("Empty follow-up after tool calls — using prior turn content as final response")
                            self._emit_status("↻ Empty response after tool calls — using earlier content as final answer")
                            self._last_content_with_tools = None
                            self._empty_content_retries = 0
                            for i in range(len(messages) - 1, -1, -1):
                                msg = messages[i]
                                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                    tool_names = []
                                    for tc in msg["tool_calls"]:
                                        if not tc or not isinstance(tc, dict): continue
                                        fn = tc.get("function", {})
                                        tool_names.append(fn.get("name", "unknown"))
                                    msg["content"] = f"Calling the {', '.join(tool_names)} tool{'s' if len(tool_names) > 1 else ''}..."
                                    break
                            final_response = self._strip_think_blocks(fallback).strip()
                            self._response_was_previewed = True
                            break

                        # ── Thinking-only prefill continuation ──────────
                        # The model produced structured reasoning (via API
                        # fields) but no visible text content.  Rather than
                        # giving up, append the assistant message as-is and
                        # continue — the model will see its own reasoning
                        # on the next turn and produce the text portion.
                        # Inspired by clawdbot's "incomplete-text" recovery.
                        _has_structured = bool(
                            getattr(assistant_message, "reasoning", None)
                            or getattr(assistant_message, "reasoning_content", None)
                            or getattr(assistant_message, "reasoning_details", None)
                        )
                        if _has_structured and self._thinking_prefill_retries < 2:
                            self._thinking_prefill_retries += 1
                            logger.info(
                                "Thinking-only response (no visible content) — "
                                "prefilling to continue (%d/2)",
                                self._thinking_prefill_retries,
                            )
                            self._emit_status(
                                f"↻ Thinking-only response — prefilling to continue "
                                f"({self._thinking_prefill_retries}/2)"
                            )
                            interim_msg = self._build_assistant_message(
                                assistant_message, "incomplete"
                            )
                            interim_msg["_thinking_prefill"] = True
                            messages.append(interim_msg)
                            self._session_messages = messages
                            self._save_session_log(messages)
                            continue

                        # ── Empty response retry ──────────────────────
                        # Model returned nothing usable.  Retry up to 3
                        # times before attempting fallback.  This covers
                        # both truly empty responses (no content, no
                        # reasoning) AND reasoning-only responses after
                        # prefill exhaustion — models like mimo-v2-pro
                        # always populate reasoning fields via OpenRouter,
                        # so the old `not _has_structured` guard blocked
                        # retries for every reasoning model after prefill.
                        _truly_empty = not self._strip_think_blocks(
                            final_response
                        ).strip()
                        _prefill_exhausted = (
                            _has_structured
                            and self._thinking_prefill_retries >= 2
                        )
                        if _truly_empty and (not _has_structured or _prefill_exhausted) and self._empty_content_retries < 3:
                            self._empty_content_retries += 1
                            logger.warning(
                                "Empty response (no content or reasoning) — "
                                "retry %d/3 (model=%s)",
                                self._empty_content_retries, self.model,
                            )
                            self._emit_status(
                                f"⚠️ Empty response from model — retrying "
                                f"({self._empty_content_retries}/3)"
                            )
                            continue

                        # ── Exhausted retries — try fallback provider ──
                        # Before giving up with "(empty)", attempt to
                        # switch to the next provider in the fallback
                        # chain.  This covers the case where a model
                        # (e.g. GLM-4.5-Air) consistently returns empty
                        # due to context degradation or provider issues.
                        if _truly_empty and self._fallback_chain:
                            logger.warning(
                                "Empty response after %d retries — "
                                "attempting fallback (model=%s, provider=%s)",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "⚠️ Model returning empty responses — "
                                "switching to fallback provider..."
                            )
                            if self._try_activate_fallback():
                                self._empty_content_retries = 0
                                self._emit_status(
                                    f"↻ Switched to fallback: {self.model} "
                                    f"({self.provider})"
                                )
                                logger.info(
                                    "Fallback activated after empty responses: "
                                    "now using %s on %s",
                                    self.model, self.provider,
                                )
                                continue

                        # Exhausted retries and fallback chain (or no
                        # fallback configured).  Fall through to the
                        # "(empty)" terminal.
                        _turn_exit_reason = "empty_response_exhausted"
                        reasoning_text = self._extract_reasoning(assistant_message)
                        assistant_msg = self._build_assistant_message(assistant_message, finish_reason)
                        assistant_msg["content"] = "(empty)"
                        messages.append(assistant_msg)

                        if reasoning_text:
                            reasoning_preview = reasoning_text[:500] + "..." if len(reasoning_text) > 500 else reasoning_text
                            logger.warning(
                                "Reasoning-only response (no visible content) "
                                "after exhausting retries and fallback. "
                                "Reasoning: %s", reasoning_preview,
                            )
                            self._emit_status(
                                "⚠️ Model produced reasoning but no visible "
                                "response after all retries. Returning empty."
                            )
                        else:
                            logger.warning(
                                "Empty response (no content or reasoning) "
                                "after %d retries. No fallback available. "
                                "model=%s provider=%s",
                                self._empty_content_retries, self.model,
                                self.provider,
                            )
                            self._emit_status(
                                "❌ Model returned no content after all retries"
                                + (" and fallback attempts." if self._fallback_chain else
                                   ". No fallback providers configured.")
                            )

                        final_response = "(empty)"
                        break
                    
                    # 成功生成内容后重置重试计数器/标记
             
                    self._empty_content_retries = 0
                    self._thinking_prefill_retries = 0

                    if (
                        self.api_mode == "codex_responses"
                        and self.valid_tool_names
                        and codex_ack_continuations < 2
                        and self._looks_like_codex_intermediate_ack(
                            user_message=user_message,
                            assistant_content=final_response,
                            messages=messages,
                        )
                    ):
                        codex_ack_continuations += 1
                        interim_msg = self._build_assistant_message(assistant_message, "incomplete")
                        messages.append(interim_msg)
                        self._emit_interim_assistant_message(interim_msg)

                        continue_msg = {
                            "role": "user",
                            "content": (
                                "[System: Continue now. Execute the required tool calls and only "
                                "send your final answer after completing the task.]"
                            ),
                        }
                        messages.append(continue_msg)
                        self._session_messages = messages
                        self._save_session_log(messages)
                        continue

                    codex_ack_continuations = 0

                    if truncated_response_prefix:
                        final_response = truncated_response_prefix + final_response
                        truncated_response_prefix = ""
                        length_continue_retries = 0
                    
                    # 从面向用户的响应中剥离 <think> 块（在消息中保持原始用于轨迹）
                    final_response = self._strip_think_blocks(final_response).strip()
                    
                    final_msg = self._build_assistant_message(assistant_message, finish_reason)

                    # 在附加最终响应之前弹出仅思考的预填充消息。
                    # 这避免了连续的助手消息，这些消息会破坏严格交替的提供商
                    #（Anthropic Messages API）并保持历史记录干净。
                    while (
                        messages
                        and isinstance(messages[-1], dict)
                        and messages[-1].get("_thinking_prefill")
                    ):
                        messages.pop()

                    messages.append(final_msg)
                    
                    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
                    if not self.quiet_mode:
                        self._safe_print(f"🎉 在 {api_call_count} 个 OpenAI 兼容的 API 调用后对话完成")
                    break
                
            except Exception as e:
                error_msg = f"OpenAI 兼容的 API 调用 #{api_call_count} 期间出错：{str(e)}"
                try:
                    print(f"❌ {error_msg}")
                except (OSError, ValueError):
                    logger.error(error_msg)
                
                logger.debug("API 调用 #%d 中的外循环错误", api_call_count, exc_info=True)
                
                # 如果已附加带有 tool_calls 的助手消息，
                # API 期望每个 tool_call_id 都有一个 role="tool" 的结果。
                # 为尚未回答的任何内容填写错误结果。
                for idx in range(len(messages) - 1, -1, -1):
                    msg = messages[idx]
                    if not isinstance(msg, dict):
                        break
                    if msg.get("role") == "tool":
                        continue
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        answered_ids = {
                            m["tool_call_id"]
                            for m in messages[idx + 1:]
                            if isinstance(m, dict) and m.get("role") == "tool"
                        }
                        for tc in msg["tool_calls"]:
                            if not tc or not isinstance(tc, dict): continue
                            if tc["id"] not in answered_ids:
                                err_msg = {
                                    "role": "tool",
                                    "tool_call_id": tc["id"],
                                    "content": f"执行工具时出错：{error_msg}",
                                }
                                messages.append(err_msg)
                    break
                
                # 非工具错误不需要注入合成消息。
                # 错误已经打印给用户（上面的行），并且重试循环继续。
                # 注入假的用户/助手消息会污染历史、消耗 token，并冒着违反角色交替不变量的风险。

                # 如果我们接近限制，中断以避免无限循环
                if api_call_count >= self.max_iterations - 1:
                    _turn_exit_reason = f"error_near_max_iterations({error_msg[:80]})"
                    final_response = f"抱歉，但我遇到了重复的错误：{error_msg}"
                    # 作为助手附加，以便历史记录对会话恢复保持有效（避免连续的用户消息）。
                    messages.append({"role": "assistant", "content": final_response})
                    break
        
        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ) and not self._budget_exhausted_injected:
            # 预算已用尽，但我们尚未尝试要求模型总结。
            # 注入一条用户消息并给它一次优雅的 API 调用以产生文本响应。
            self._budget_exhausted_injected = True
            self._budget_grace_call = True
            _grace_msg = (
                "你的工具预算用完了。请给我你到目前为止完成的信息或操作。"
            )
            messages.append({"role": "user", "content": _grace_msg})
            self._emit_status(
                f"⚠️ 迭代预算已用尽 ({api_call_count}/{self.max_iterations}) — 要求模型总结"
            )
            if not self.quiet_mode:
                self._safe_print(
                    f"\n⚠️  迭代预算已用尽 ({api_call_count}/{self.max_iterations}) — 请求总结..."
                )

        if final_response is None and (
            api_call_count >= self.max_iterations
            or self.iteration_budget.remaining <= 0
        ) and not self._budget_grace_call:
            _turn_exit_reason = f"max_iterations_reached({api_call_count}/{self.max_iterations})"
            if self.iteration_budget.remaining <= 0 and not self.quiet_mode:
                print(f"\n⚠️  迭代预算已用尽（已使用 {self.iteration_budget.used}/{self.iteration_budget.max_total} 次迭代）")
            final_response = self._handle_max_iterations(messages, api_call_count)
        
        # 确定对话是否成功完成
        completed = final_response is not None and api_call_count < self.max_iterations

        # 如果启用则保存轨迹
        self._save_trajectory(messages, user_message, completed)

        # 对话完成后清理此任务的 VM 和浏览器
        self._cleanup_task_resources(effective_task_id)

        # 将会话持久化到 JSON 日志和 SQLite
        self._persist_session(messages, conversation_history)

        # ── 轮次退出诊断日志 ─────────────────────────────────────
        # 始终以 INFO 级别记录，以便 agent.log 捕获每次轮次结束的原因。
        # 当最后一条消息是工具结果时（代理正在工作中），
        # 以 WARNING 级别记录——这是用户报告的"刚刚停止"场景。
        _last_msg_role = messages[-1].get("role") if messages else None
        _last_tool_name = None
        if _last_msg_role == "tool":
            # 向后查找以查找带有工具调用的助手消息
            for _m in reversed(messages):
                if _m.get("role") == "assistant" and _m.get("tool_calls"):
                    _tcs = _m["tool_calls"]
                    if _tcs and isinstance(_tcs[0], dict):
                        _last_tool_name = _tcs[-1].get("function", {}).get("name")
                    break

        _turn_tool_count = sum(
            1 for m in messages
            if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")
        )
        _resp_len = len(final_response) if final_response else 0
        _budget_used = self.iteration_budget.used if self.iteration_budget else 0
        _budget_max = self.iteration_budget.max_total if self.iteration_budget else 0

        _diag_msg = (
            "轮次结束：reason=%s model=%s api_calls=%d/%d budget=%d/%d "
            "tool_turns=%d last_msg_role=%s response_len=%d session=%s"
        )
        _diag_args = (
            _turn_exit_reason, self.model, api_call_count, self.max_iterations,
            _budget_used, _budget_max,
            _turn_tool_count, _last_msg_role, _resp_len,
            self.session_id or "none",
        )

        if _last_msg_role == "tool" and not interrupted:
            # 代理正在工作中——这是"刚刚停止"的情况。
            logger.warning(
                "轮次结束时有挂起的工具结果（代理可能看起来卡住了）。"
                + _diag_msg + " last_tool=%s",
                *_diag_args, _last_tool_name,
            )
        else:
            logger.info(_diag_msg, *_diag_args)

        # 插件钩子：post_llm_call
        # 在工具调用循环完成后每轮次触发一次。
        # 插件可以使用此来持久化对话数据（例如同步到外部记忆系统）。
        if final_response and not interrupted:
            try:
                from hermes_cli.plugins import invoke_hook as _invoke_hook
                _invoke_hook(
                    "post_llm_call",
                    session_id=self.session_id,
                    user_message=original_user_message,
                    assistant_response=final_response,
                    conversation_history=list(messages),
                    model=self.model,
                    platform=getattr(self, "platform", None) or "",
                )
            except Exception as exc:
                logger.warning("post_llm_call hook failed: %s", exc)

        # 从最后一个助手消息中提取推理（如果有）
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break

        # 如果适用，构建带有中断信息的结果
        result = {
            "final_response": final_response,
            "last_reasoning": last_reasoning,
            "messages": messages,
            "api_calls": api_call_count,
            "completed": completed,
            "partial": False,  # 仅在因无效工具调用而停止时为 True
            "interrupted": interrupted,
            "response_previewed": getattr(self, "_response_was_previewed", False),
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "input_tokens": self.session_input_tokens,
            "output_tokens": self.session_output_tokens,
            "cache_read_tokens": self.session_cache_read_tokens,
            "cache_write_tokens": self.session_cache_write_tokens,
            "reasoning_tokens": self.session_reasoning_tokens,
            "prompt_tokens": self.session_prompt_tokens,
            "completion_tokens": self.session_completion_tokens,
            "total_tokens": self.session_total_tokens,
            "last_prompt_tokens": getattr(self.context_compressor, "last_prompt_tokens", 0) or 0,
            "estimated_cost_usd": self.session_estimated_cost_usd,
            "cost_status": self.session_cost_status,
            "cost_source": self.session_cost_source,
        }
        self._response_was_previewed = False
        
        # 如果触发了中断，则包含中断消息
        if interrupted and self._interrupt_message:
            result["interrupt_message"] = self._interrupt_message
        
        # 处理后清除中断状态
        self.clear_interrupt()

        # 清除流回调，以免泄漏到未来的调用中
        self._stream_callback = None

        # 现在检查技能触发器——基于本轮次使用了多少工具迭代。
        _should_review_skills = False
        if (self._skill_nudge_interval > 0
                and self._iters_since_skill >= self._skill_nudge_interval
                and "skill_manage" in self.valid_tool_names):
            _should_review_skills = True
            self._iters_since_skill = 0

        # 外部记忆提供程序：同步已完成的轮次 + 排队下一次预取。
        # 使用 original_user_message（干净的输入）——user_message 可能包含
        # 会膨胀/破坏提供程序查询的注入技能内容。
        if self._memory_manager and final_response and original_user_message:
            try:
                self._memory_manager.sync_all(original_user_message, final_response)
                self._memory_manager.queue_prefetch_all(original_user_message)
            except Exception:
                pass

        # 后台记忆/技能审查——在响应交付后运行，
        # 因此它永远不会与用户的任务竞争模型注意力。
        if final_response and not interrupted and (_should_review_memory or _should_review_skills):
            try:
                self._spawn_background_review(
                    messages_snapshot=list(messages),
                    review_memory=_should_review_memory,
                    review_skills=_should_review_skills,
                )
            except Exception:
                pass  # 后台审查是尽力而为的

        # 注意：记忆提供程序 on_session_end() + shutdown_all() 不在这里调用——
        # run_conversation() 在多轮会话中每条用户消息调用一次。在每轮后关闭会
        # 在第二条消息之前杀死提供程序。实际的会话结束清理由
        # CLI（atexit / /reset）和 gateway（会话到期 / _reset_session）处理。

        # 插件钩子：on_session_end
        # 在每次 run_conversation 调用的最后触发。
        # 插件可以使用此来清理、刷新缓冲区等。
        try:
            from hermes_cli.plugins import invoke_hook as _invoke_hook
            _invoke_hook(
                "on_session_end",
                session_id=self.session_id,
                completed=completed,
                interrupted=interrupted,
                model=self.model,
                platform=getattr(self, "platform", None) or "",
            )
        except Exception as exc:
            logger.warning("on_session_end hook failed: %s", exc)

        return result

    def chat(self, message: str, stream_callback: Optional[callable] = None) -> str:
        """
        仅返回最终响应的简单聊天接口。

        参数:
            message (str): 用户消息
            stream_callback: 流式传输期间用每个文本增量调用的可选回调。

        返回:
            str: 最终助手响应
        """
        result = self.run_conversation(message, stream_callback=stream_callback)
        return result["final_response"]


def main(
    query: str = None,
    model: str = "",
    api_key: str = None,
    base_url: str = "",
    max_turns: int = 10,
    enabled_toolsets: str = None,
    disabled_toolsets: str = None,
    list_tools: bool = False,
    save_trajectories: bool = False,
    save_sample: bool = False,
    verbose: bool = False,
    log_prefix_chars: int = 20
):
    """
    直接运行代理的主函数。

    参数:
        query (str): 代理的自然语言查询。默认为 Python 3.13 示例。
        model (str): 要使用的模型名称（OpenRouter 格式：provider/model）。默认为 anthropic/claude-sonnet-4.6。
        api_key (str): 用于身份验证的 API 密钥。如果未提供，则使用 OPENROUTER_API_KEY 环境变量。
        base_url (str): 模型 API 的基础 URL。默认为 https://openrouter.ai/api/v1
        max_turns (int): API 调用迭代的最大次数。默认为 10。
        enabled_toolsets (str): 要启用的工具集的逗号分隔列表。支持预定义工具集（例如，"research"、"development"、"safe"）。
                              可以组合多个工具集："web,vision"
        disabled_toolsets (str): 要禁用的工具集的逗号分隔列表（例如，"terminal"）
        list_tools (bool): 仅列出可用工具并退出
        save_trajectories (bool): 将对话轨迹保存到 JSONL 文件（附加到 trajectory_samples.jsonl）。默认为 False。
        save_sample (bool): 将单个轨迹样本保存到 UUID 命名的 JSONL 文件以供检查。默认为 False。
        verbose (bool): 启用详细日志记录以进行调试。默认为 False。
        log_prefix_chars (int): 在工具调用/响应的日志预览中显示的字符数。默认为 20。

    工具集示例：
        - "research"：网络搜索、提取、抓取 + 视觉工具
    """
    print("🤖 带有工具调用的 AI 代理")
    print("=" * 50)
    
    # 处理工具列表
    if list_tools:
        from model_tools import get_all_tool_names, get_toolset_for_tool, get_available_toolsets
        from toolsets import get_all_toolsets, get_toolset_info
        
        print("📋 可用工具和工具集：")
        print("-" * 50)
        
        # 显示新工具集系统
        print("\n🎯 预定义工具集（新系统）：")
        print("-" * 40)
        all_toolsets = get_all_toolsets()
        
        # 按类别分组
        basic_toolsets = []
        composite_toolsets = []
        scenario_toolsets = []
        
        for name, toolset in all_toolsets.items():
            info = get_toolset_info(name)
            if info:
                entry = (name, info)
                if name in ["web", "terminal", "vision", "creative", "reasoning"]:
                    basic_toolsets.append(entry)
                elif name in ["research", "development", "analysis", "content_creation", "full_stack"]:
                    composite_toolsets.append(entry)
                else:
                    scenario_toolsets.append(entry)
        
        # 打印基本工具集
        print("\n📌 基本工具集：")
        for name, info in basic_toolsets:
            tools_str = ', '.join(info['resolved_tools']) if info['resolved_tools'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    工具：{tools_str}")
        
        # 打印复合工具集
        print("\n📂 复合工具集（由其他工具集构建）：")
        for name, info in composite_toolsets:
            includes_str = ', '.join(info['includes']) if info['includes'] else 'none'
            print(f"  • {name:15} - {info['description']}")
            print(f"    包含：{includes_str}")
            print(f"    总工具数：{info['tool_count']}")
        
        # 打印场景特定工具集
        print("\n🎭 场景特定工具集：")
        for name, info in scenario_toolsets:
            print(f"  • {name:20} - {info['description']}")
            print(f"    总工具数：{info['tool_count']}")
        
        
        # 显示旧工具集兼容性
        print("\n📦 旧工具集（用于向后兼容）：")
        legacy_toolsets = get_available_toolsets()
        for name, info in legacy_toolsets.items():
            status = "✅" if info["available"] else "❌"
            print(f"  {status} {name}: {info['description']}")
            if not info["available"]:
                print(f"    Requirements: {', '.join(info['requirements'])}")
        
        # Show individual tools
        all_tools = get_all_tool_names()
        print(f"\n🔧 Individual Tools ({len(all_tools)} available):")
        for tool_name in sorted(all_tools):
            toolset = get_toolset_for_tool(tool_name)
            print(f"  📌 {tool_name} (from {toolset})")
        
        print("\n💡 Usage Examples:")
        print("  # Use predefined toolsets")
        print("  python run_agent.py --enabled_toolsets=research --query='search for Python news'")
        print("  python run_agent.py --enabled_toolsets=development --query='debug this code'")
        print("  python run_agent.py --enabled_toolsets=safe --query='analyze without terminal'")
        print("  ")
        print("  # Combine multiple toolsets")
        print("  python run_agent.py --enabled_toolsets=web,vision --query='analyze website'")
        print("  ")
        print("  # Disable toolsets")
        print("  python run_agent.py --disabled_toolsets=terminal --query='no command execution'")
        print("  ")
        print("  # Run with trajectory saving enabled")
        print("  python run_agent.py --save_trajectories --query='your question here'")
        return
    
    # Parse toolset selection arguments
    enabled_toolsets_list = None
    disabled_toolsets_list = None
    
    if enabled_toolsets:
        enabled_toolsets_list = [t.strip() for t in enabled_toolsets.split(",")]
        print(f"🎯 Enabled toolsets: {enabled_toolsets_list}")
    
    if disabled_toolsets:
        disabled_toolsets_list = [t.strip() for t in disabled_toolsets.split(",")]
        print(f"🚫 Disabled toolsets: {disabled_toolsets_list}")
    
    if save_trajectories:
        print("💾 Trajectory saving: ENABLED")
        print("   - Successful conversations → trajectory_samples.jsonl")
        print("   - Failed conversations → failed_trajectories.jsonl")
    
    # Initialize agent with provided parameters
    try:
        agent = AIAgent(
            base_url=base_url,
            model=model,
            api_key=api_key,
            max_iterations=max_turns,
            enabled_toolsets=enabled_toolsets_list,
            disabled_toolsets=disabled_toolsets_list,
            save_trajectories=save_trajectories,
            verbose_logging=verbose,
            log_prefix_chars=log_prefix_chars
        )
    except RuntimeError as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    # Use provided query or default to Python 3.13 example
    if query is None:
        user_query = (
            "Tell me about the latest developments in Python 3.13 and what new features "
            "developers should know about. Please search for current information and try it out."
        )
    else:
        user_query = query
    
    print(f"\n📝 User Query: {user_query}")
    print("\n" + "=" * 50)
    
    # Run conversation
    result = agent.run_conversation(user_query)
    
    print("\n" + "=" * 50)
    print("📋 CONVERSATION SUMMARY")
    print("=" * 50)
    print(f"✅ Completed: {result['completed']}")
    print(f"📞 API Calls: {result['api_calls']}")
    print(f"💬 Messages: {len(result['messages'])}")
    
    if result['final_response']:
        print("\n🎯 FINAL RESPONSE:")
        print("-" * 30)
        print(result['final_response'])
    
    # Save sample trajectory to UUID-named file if requested
    if save_sample:
        sample_id = str(uuid.uuid4())[:8]
        sample_filename = f"sample_{sample_id}.json"
        
        # Convert messages to trajectory format (same as batch_runner)
        trajectory = agent._convert_to_trajectory_format(
            result['messages'], 
            user_query, 
            result['completed']
        )
        
        entry = {
            "conversations": trajectory,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "completed": result['completed'],
            "query": user_query
        }
        
        try:
            with open(sample_filename, "w", encoding="utf-8") as f:
                # Pretty-print JSON with indent for readability
                f.write(json.dumps(entry, ensure_ascii=False, indent=2))
            print(f"\n💾 Sample trajectory saved to: {sample_filename}")
        except Exception as e:
            print(f"\n⚠️ Failed to save sample: {e}")
    
    print("\n👋 Agent execution completed!")


if __name__ == "__main__":
    fire.Fire(main)
