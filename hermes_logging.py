"""Hermes Agent 的集中日志系统配置。

提供了一个统一的 ``setup_logging()`` 入口，无论是 CLI 还是 gateway，
都会在启动早期调用。所有日志文件都存放在 ``~/.hermes/logs/`` 目录下（通过
``get_hermes_home()`` 支持 profile 感知）。

产生日志文件包括：
    agent.log   — INFO+，所有 agent/tool/session 活动（主日志）
    errors.log  — WARNING+，仅记录错误和警告（便于快速排查）
    gateway.log — INFO+，仅记录 gateway 相关事件（mode="gateway" 时创建）

所有日志文件都使用 ``RotatingFileHandler`` 结合 ``RedactingFormatter``，
保证敏感信息绝不会被写入磁盘。

组件区分：
    gateway.log 只接收 ``gateway.*`` logger 的日志记录——
    包括平台适配器、会话管理、斜杠指令、分发。
    agent.log 作为兜底日志（所有内容都会记录）。

会话上下文：
    会话开始时调用 ``set_session_context(session_id)``，
    会话结束后调用 ``clear_session_context()``。
    此线程所有日志行都会包含 ``[session_id]``，方便过滤/追踪。
"""

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Sequence

from hermes_constants import get_config_path, get_hermes_home

# 标记 setup_logging() 是否已经初始化。该函数具备幂等性——
# 多次调用安全，第二次调用除非 force=True，否则不会再次执行。
_logging_initialized = False

# 每会话线程本地存储，用于保存会话上下文。
_session_context = threading.local()

# 默认日志格式——包含时间戳、日志级别、可选 session_tag、logger 名称和消息内容。
# 保证所有 LogRecord 都有 ``%(session_tag)s`` 字段，见下方 _install_session_record_factory()。
_LOG_FORMAT = "%(asctime)s %(levelname)s%(session_tag)s %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = "%(asctime)s - %(name)s - %(levelname)s%(session_tag)s - %(message)s"

# 第三方 noisy 日志（低级别 DEBUG/INFO 下输出过多），需屏蔽。
_NOISY_LOGGERS = (
    "openai",
    "openai._base_client",
    "httpx",
    "httpcore",
    "asyncio",
    "hpack",
    "hpack.hpack",
    "grpc",
    "modal",
    "urllib3",
    "urllib3.connectionpool",
    "websockets",
    "charset_normalizer",
    "markdown_it",
)


# ---------------------------------------------------------------------------
# 会话上下文相关公有 API
# ---------------------------------------------------------------------------

def set_session_context(session_id: str) -> None:
    """为当前线程设置会话 ID。

    之后该线程所有日志记录输出将会带有 ``[session_id]``。
    一般在 ``run_conversation()`` 开头调用。
    """
    _session_context.session_id = session_id


def clear_session_context() -> None:
    """清除当前线程的会话 ID。

    非必须——``set_session_context()`` 会覆盖之前的值，仅当同一线程
    后续用于非会话相关任务时才需手动清除。
    """
    _session_context.session_id = None


# ---------------------------------------------------------------------------
# 日志记录工厂——在创建 LogRecord 时注入 session_tag
# ---------------------------------------------------------------------------

def _install_session_record_factory() -> None:
    """安装全局 LogRecord 工厂，在每个记录中增加 ``session_tag``。

    与 handler/logger 上的 ``logging.Filter`` 不同，record factory
    会在每条日志创建时运行——包括子 logger、第三方 handler 产生的日志。
    这样可确保格式化字符串中始终可以用 ``%(session_tag)s``，
    避免 handler 未加 ``_SessionFilter`` 时 KeyError。

    具备幂等性——带标志属性避免重复包装。
    """
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_hermes_session_injector", False):
        return  # 已安装，无需再次安装

    def _session_record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        sid = getattr(_session_context, "session_id", None)
        record.session_tag = f" [{sid}]" if sid else ""  # type: ignore[attr-defined]
        return record

    _session_record_factory._hermes_session_injector = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(_session_record_factory)


# 模块导入时立即安装——保证 session_tag 从此刻起对所有日志记录可用，
# 即使在 setup_logging() 调用之前。
_install_session_record_factory()


# ---------------------------------------------------------------------------
# 过滤器
# ---------------------------------------------------------------------------

class _ComponentFilter(logging.Filter):
    """只允许 logger 名字以 *prefixes* 中任一前缀开头的日志通过。

    用于把 gateway 相关日志导入 ``gateway.log``，其余都能被 ``agent.log`` 兜底。
    """

    def __init__(self, prefixes: Sequence[str]) -> None:
        super().__init__()
        self._prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


# 不同组件对应的 logger 前缀。
# 既用于 _ComponentFilter 也用于 cli 命令 ``hermes logs --component``。
COMPONENT_PREFIXES = {
    "gateway": ("gateway",),
    "agent": ("agent", "run_agent", "model_tools", "batch_runner"),
    "tools": ("tools",),
    "cli": ("hermes_cli", "cli"),
    "cron": ("cron",),
}


# ---------------------------------------------------------------------------
# 主日志配置
# ---------------------------------------------------------------------------

def setup_logging(
    *,
    hermes_home: Optional[Path] = None,
    log_level: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    mode: Optional[str] = None,
    force: bool = False,
) -> Path:
    """配置 Hermes 日志子系统。

    可安全多次调用——第二次调用除非 *force* 为 True，否则无操作。

    参数
    ----------
    hermes_home
        指定 Hermes 主目录，默认为 ``get_hermes_home()``（支持 profile）。
    log_level
        ``agent.log`` 的最低日志级别，接受任何标准 Python 等级名
        （如 ``"DEBUG"``, ``"INFO"``, ``"WARNING"``）。
        默认为 ``"INFO"``，也可以在 config.yaml ``logging.level`` 配置。
    max_size_mb
        每个日志文件的最大大小（MB），达到后自动轮转。
        默认为 5，也可从 config.yaml ``logging.max_size_mb`` 获取。
    backup_count
        日志轮转时最多保留的历史文件数量。
        默认为 3，也可从 config.yaml ``logging.backup_count`` 读取。
    mode
        调用上下文：``"cli"``、``"gateway"``、``"cron"``。
        若为 ``"gateway"``，额外生成 ``gateway.log``，仅记录 gateway 组件日志。
    force
        即使已经执行过，也强制重新配置。

    返回值
    -------
    Path
        实际写入日志文件的 ``logs/`` 目录路径。
    """
    global _logging_initialized
    if _logging_initialized and not force:
        home = hermes_home or get_hermes_home()
        return home / "logs"

    home = hermes_home or get_hermes_home()
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 尝试读取配置文件作为默认参数（注意：此时 config.yaml 可能还未加载）。
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()

    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    # 懒导入，防止模块加载时循环依赖。
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # --- agent.log (INFO+) ——主日志 -------------------------
    _add_rotating_handler(
        root,
        log_dir / "agent.log",
        level=level,
        max_bytes=max_bytes,
        backup_count=backups,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # --- errors.log (WARNING+) ——快速排查日志 ----------------
    _add_rotating_handler(
        root,
        log_dir / "errors.log",
        level=logging.WARNING,
        max_bytes=2 * 1024 * 1024,
        backup_count=2,
        formatter=RedactingFormatter(_LOG_FORMAT),
    )

    # --- gateway.log (INFO+，仅 gateway 组件) -----------------
    if mode == "gateway":
        _add_rotating_handler(
            root,
            log_dir / "gateway.log",
            level=logging.INFO,
            max_bytes=5 * 1024 * 1024,
            backup_count=3,
            formatter=RedactingFormatter(_LOG_FORMAT),
            log_filter=_ComponentFilter(COMPONENT_PREFIXES["gateway"]),
        )

    # 保证 root logger 级别足够低，能被 handler 捕获所有记录
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    # 屏蔽部分第三方 noisy 的 logger 输出
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_initialized = True
    return log_dir


def setup_verbose_logging() -> None:
    """为 ``--verbose`` / ``-v`` 模式启用 DEBUG 级别终端日志输出。

    由 ``AIAgent.__init__()`` 在 ``verbose_logging=True`` 时调用。
    """
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # 避免重复添加 stream handler
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler):
            if getattr(h, "_hermes_verbose", False):
                return

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(RedactingFormatter(_LOG_FORMAT_VERBOSE, datefmt="%H:%M:%S"))
    handler._hermes_verbose = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # 降低 root logger 级别，确保 DEBUG 记录能被所有 handler 捕获
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    # 对第三方 noisy 库保持 WARNING 级别，减少控制台噪音
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # rex-deploy 日志级别为 INFO，便于沙箱状态监测
    logging.getLogger("rex-deploy").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

class _ManagedRotatingFileHandler(RotatingFileHandler):
    """在受管模式下，确保日志文件可被群组写入的 RotatingFileHandler。

    受管模式下（如 NixOS），stateDir 目录采用 setgid (2770) 权限，
    新建文件会继承 hermes 群组。
    但 _open()（初次创建）和 doRollover() 都用 open()，
    这会受到当前进程的 umask 影响，通常是 0022，得到 0644 权限。
    该子类在上述两种情况下会自动 chmod 0660，
    以便 gateway 与交互用户均可访问日志文件。
    """

    def __init__(self, *args, **kwargs):
        from hermes_cli.config import is_managed
        self._managed = is_managed()
        super().__init__(*args, **kwargs)

    def _chmod_if_managed(self):
        if self._managed:
            try:
                os.chmod(self.baseFilename, 0o660)
            except OSError:
                pass

    def _open(self):
        stream = super()._open()
        self._chmod_if_managed()
        return stream

    def doRollover(self):
        super().doRollover()
        self._chmod_if_managed()


def _add_rotating_handler(
    logger: logging.Logger,
    path: Path,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    log_filter: Optional[logging.Filter] = None,
) -> None:
    """为 *logger* 添加 ``RotatingFileHandler``，如果指定文件已关联 handler 则跳过
    （幂等性）。

    参数
    ----------
    log_filter
        可选过滤器（如 ``_ComponentFilter``，专为 gateway.log）。
    """
    resolved = path.resolve()
    for existing in logger.handlers:
        if (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return  # 已经加过相同的 handler，跳过

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = _ManagedRotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count,
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    if log_filter is not None:
        handler.addFilter(log_filter)
    logger.addHandler(handler)


def _read_logging_config():
    """尽力读取 config.yaml 的 ``logging.*`` 配置项。

    返回 ``(level, max_size_mb, backup_count)``，如未配置则返回 None。
    """
    try:
        import yaml
        config_path = get_config_path()
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            log_cfg = cfg.get("logging", {})
            if isinstance(log_cfg, dict):
                return (
                    log_cfg.get("level"),
                    log_cfg.get("max_size_mb"),
                    log_cfg.get("backup_count"),
                )
    except Exception:
        pass
    return (None, None, None)
