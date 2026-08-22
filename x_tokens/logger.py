"""Unified logging configuration for xTokens."""

from __future__ import annotations

import copy
import json
import logging
import logging.config
import os
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from types import MethodType
from typing import Any, Literal, cast

LogScope = Literal["process", "local", "global"]

_FORMAT = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
_ACCESS_FORMAT = (
    "%(levelname)s %(asctime)s [access] %(client_addr)s "
    '"%(request_line)s" %(status_code)s'
)
_DATE_FORMAT = "%m-%d %H:%M:%S"
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PACKAGE_ROOT = Path(__file__).resolve().parent
_ONCE_CACHE_SIZE = 2048

_RESET = "\033[0m"
_GRAY = "\033[90m"
_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[31;1m",
}


class XTokensLogger(logging.Logger):
    """A typing facade for standard loggers extended with once methods."""

    def debug_once(
        self,
        msg: object,
        *args: object,
        scope: LogScope = "process",
        key: object | None = None,
        **kwargs: Any,
    ) -> None: ...

    def info_once(
        self,
        msg: object,
        *args: object,
        scope: LogScope = "process",
        key: object | None = None,
        **kwargs: Any,
    ) -> None: ...

    def warning_once(
        self,
        msg: object,
        *args: object,
        scope: LogScope = "process",
        key: object | None = None,
        **kwargs: Any,
    ) -> None: ...


class NewLineFormatter(logging.Formatter):
    """Add source information and align continuation lines."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        style: Literal["%", "{", "$"] = "%",
        validate: bool = True,
        *,
        defaults: dict[str, Any] | None = None,
        prefix: str = "",
        format: str | None = None,
    ) -> None:
        if fmt is None:
            fmt = format
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self._prefix = prefix

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        record.fileinfo = _fileinfo(record)
        message = f"{self._prefix}{record.getMessage()}"
        record.msg = message
        record.args = ()
        rendered = super().format(record)
        lines = rendered.splitlines()
        if len(lines) <= 1:
            return rendered

        message_first_line = message.splitlines()[0] if message else ""
        prefix_width = max(
            0,
            len(_strip_ansi(lines[0])) - len(_strip_ansi(message_first_line)),
        )
        indentation = " " * prefix_width
        return "\n".join((lines[0], *(f"{indentation}{line}" for line in lines[1:])))


class ColoredFormatter(NewLineFormatter):
    """Color the level and timestamp while preserving the message text."""

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        color = _LEVEL_COLORS.get(record.levelno)
        if color is not None:
            record.levelname = f"{color}{record.levelname}{_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname

    def formatTime(
        self,
        record: logging.LogRecord,
        datefmt: str | None = None,
    ) -> str:
        timestamp = super().formatTime(record, datefmt)
        return f"{_GRAY}{timestamp}{_RESET}"


class _UvicornAccessMixin:
    """Populate fields from Uvicorn's access log argument tuple."""

    def format(self, record: logging.LogRecord) -> str:
        record = copy.copy(record)
        if isinstance(record.args, tuple) and len(record.args) == 5:
            client_addr, method, path, http_version, status_code = record.args
            record.client_addr = client_addr
            record.request_line = f"{method} {path} HTTP/{http_version}"
            record.status_code = status_code
        else:
            record.client_addr = "-"
            record.request_line = record.getMessage()
            record.status_code = "-"
        return super().format(record)  # type: ignore[misc]


class UvicornAccessFormatter(_UvicornAccessMixin, NewLineFormatter):
    """Format Uvicorn access records without depending on Uvicorn internals."""


class ColoredUvicornAccessFormatter(_UvicornAccessMixin, ColoredFormatter):
    """Colorized Uvicorn access formatter."""


DEFAULT_LOGGING_CONFIG: dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "xtokens": {
            "()": NewLineFormatter,
            "format": _FORMAT,
            "datefmt": _DATE_FORMAT,
        },
        "xtokens_color": {
            "()": ColoredFormatter,
            "format": _FORMAT,
            "datefmt": _DATE_FORMAT,
        },
    },
    "handlers": {
        "xtokens": {
            "class": "logging.StreamHandler",
            "formatter": "xtokens",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "x_tokens": {
            "handlers": ["xtokens"],
            "level": "INFO",
            "propagate": False,
        },
        "transformers": {
            "level": "WARNING",
        },
        "httpx": {
            "level": "WARNING",
        },
        "httpcore": {
            "level": "WARNING",
        },
    },
}

_configure_lock = threading.Lock()
_method_install_lock = threading.Lock()
_once_lock = threading.Lock()
_rank_lock = threading.Lock()
_configured = False
_active_config: dict[str, Any] | None = None
_using_custom_config = False
_rank_context: tuple[int, int] | None = None


def _strip_ansi(value: str) -> str:
    return _ANSI_RE.sub("", value)


def _fileinfo(record: logging.LogRecord) -> str:
    if record.levelno > logging.DEBUG:
        return record.filename
    try:
        relative = Path(record.pathname).resolve().relative_to(_PACKAGE_ROOT)
    except ValueError:
        return record.filename
    parts = relative.parts
    if len(parts) > 4:
        parts = (parts[0], "...", *parts[-3:])
    return "/".join(parts)


def _logging_level(value: str) -> str:
    normalized = value.upper()
    level = logging.getLevelNamesMapping().get(normalized)
    if level is None:
        raise ValueError(f"invalid XTOKENS_LOGGING_LEVEL: {value!r}")
    return normalized


def _logging_stream(value: str) -> tuple[str, Any]:
    normalized = value.lower()
    if normalized == "stdout":
        return "ext://sys.stdout", sys.stdout
    if normalized == "stderr":
        return "ext://sys.stderr", sys.stderr
    raise ValueError(
        f"invalid XTOKENS_LOGGING_STREAM: {value!r}; expected 'stdout' or 'stderr'"
    )


def _use_color(value: str, stream: Any) -> bool:
    if "NO_COLOR" in os.environ:
        return False
    normalized = value.lower()
    if normalized == "0":
        return False
    if normalized == "1":
        return True
    if normalized == "auto":
        return bool(getattr(stream, "isatty", lambda: False)())
    raise ValueError(
        f"invalid XTOKENS_LOGGING_COLOR: {value!r}; expected 'auto', '0', or '1'"
    )


def _load_custom_config(path_value: str) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"failed to load logging config from {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"logging config in {path} must be a JSON object")
    return value


def _build_default_config() -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_LOGGING_CONFIG)
    level = _logging_level(os.environ.get("XTOKENS_LOGGING_LEVEL", "INFO"))
    stream_name, stream = _logging_stream(
        os.environ.get("XTOKENS_LOGGING_STREAM", "stdout")
    )
    color = _use_color(os.environ.get("XTOKENS_LOGGING_COLOR", "auto"), stream)
    prefix = os.environ.get("XTOKENS_LOGGING_PREFIX", "")

    config["handlers"]["xtokens"]["stream"] = stream_name
    config["handlers"]["xtokens"]["formatter"] = "xtokens_color" if color else "xtokens"
    config["loggers"]["x_tokens"]["level"] = level
    config["formatters"]["xtokens"]["prefix"] = prefix
    config["formatters"]["xtokens_color"]["prefix"] = prefix
    return config


def _configure_xtokens_root_logger() -> None:
    global _active_config, _configured, _using_custom_config

    if _configured:
        return
    with _configure_lock:
        if _configured:
            return
        config_path = os.environ.get("XTOKENS_LOGGING_CONFIG_PATH")
        try:
            config = (
                _load_custom_config(config_path)
                if config_path
                else _build_default_config()
            )
            logging.config.dictConfig(config)
        except (ValueError, TypeError, AttributeError, ImportError) as exc:
            raise RuntimeError("failed to configure xTokens logging") from exc
        _active_config = copy.deepcopy(config)
        _using_custom_config = config_path is not None
        _configured = True


def create_uvicorn_log_config() -> dict[str, Any]:
    """Return the active configuration extended for Uvicorn loggers."""

    _configure_xtokens_root_logger()
    assert _active_config is not None
    config = copy.deepcopy(_active_config)
    if _using_custom_config:
        return config

    formatter_name = config["handlers"]["xtokens"]["formatter"]
    color = formatter_name == "xtokens_color"
    access_formatter = "xtokens_access_color" if color else "xtokens_access"
    formatter_class = ColoredUvicornAccessFormatter if color else UvicornAccessFormatter
    config["formatters"][access_formatter] = {
        "()": formatter_class,
        "format": _ACCESS_FORMAT,
        "datefmt": _DATE_FORMAT,
    }
    config["handlers"]["xtokens_access"] = {
        "class": "logging.StreamHandler",
        "formatter": access_formatter,
        "stream": config["handlers"]["xtokens"]["stream"],
    }
    level = config["loggers"]["x_tokens"]["level"]
    config["loggers"].update(
        {
            "uvicorn": {
                "handlers": ["xtokens"],
                "level": level,
                "propagate": False,
            },
            "uvicorn.error": {
                "level": level,
                "propagate": True,
            },
            "uvicorn.access": {
                "handlers": ["xtokens_access"],
                "level": level,
                "propagate": False,
            },
        }
    )
    return config


def set_logging_rank_context(*, global_rank: int, local_rank: int) -> None:
    """Set rank values used by local and global once scopes."""

    if global_rank < 0 or local_rank < 0:
        raise ValueError("logging ranks must be non-negative")
    global _rank_context
    with _rank_lock:
        _rank_context = (global_rank, local_rank)


def _current_rank_context() -> tuple[int, int]:
    with _rank_lock:
        if _rank_context is not None:
            return _rank_context
    try:
        return int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        return 0, 0


def _scope_allows(scope: LogScope) -> bool:
    if scope == "process":
        return True
    global_rank, local_rank = _current_rank_context()
    if scope == "local":
        return local_rank == 0
    if scope == "global":
        return global_rank == 0
    raise ValueError(f"invalid logging scope: {scope!r}")


def _cache_value(value: object) -> object:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


@lru_cache(maxsize=_ONCE_CACHE_SIZE)
def _remember_once(key: tuple[object, ...]) -> None:
    del key


def _claim_once(key: tuple[object, ...]) -> bool:
    with _once_lock:
        misses = _remember_once.cache_info().misses
        _remember_once(key)
        return _remember_once.cache_info().misses > misses


def _log_once(
    logger: logging.Logger,
    level: int,
    msg: object,
    args: tuple[object, ...],
    *,
    scope: LogScope,
    key: object | None,
    kwargs: dict[str, Any],
) -> None:
    if not logger.isEnabledFor(level) or not _scope_allows(scope):
        return
    identity: tuple[object, ...]
    if key is None:
        identity = (
            logger.name,
            level,
            scope,
            _cache_value(msg),
            *(_cache_value(arg) for arg in args),
        )
    else:
        identity = (logger.name, level, scope, _cache_value(key))
    if not _claim_once(identity):
        return

    call_kwargs = dict(kwargs)
    call_kwargs["stacklevel"] = int(call_kwargs.get("stacklevel", 1)) + 3
    logger.log(level, msg, *args, **call_kwargs)


def _debug_once(
    self: logging.Logger,
    msg: object,
    *args: object,
    scope: LogScope = "process",
    key: object | None = None,
    **kwargs: Any,
) -> None:
    _log_once(
        self,
        logging.DEBUG,
        msg,
        args,
        scope=scope,
        key=key,
        kwargs=kwargs,
    )


def _info_once(
    self: logging.Logger,
    msg: object,
    *args: object,
    scope: LogScope = "process",
    key: object | None = None,
    **kwargs: Any,
) -> None:
    _log_once(
        self,
        logging.INFO,
        msg,
        args,
        scope=scope,
        key=key,
        kwargs=kwargs,
    )


def _warning_once(
    self: logging.Logger,
    msg: object,
    *args: object,
    scope: LogScope = "process",
    key: object | None = None,
    **kwargs: Any,
) -> None:
    _log_once(
        self,
        logging.WARNING,
        msg,
        args,
        scope=scope,
        key=key,
        kwargs=kwargs,
    )


def init_logger(name: str) -> XTokensLogger:
    """Return a standard logger extended with xTokens once methods."""

    logger = logging.getLogger(name)
    with _method_install_lock:
        if (
            getattr(logger.__dict__.get("debug_once"), "__func__", None)
            is not _debug_once
        ):
            logger.debug_once = MethodType(_debug_once, logger)  # type: ignore[attr-defined]
        if (
            getattr(logger.__dict__.get("info_once"), "__func__", None)
            is not _info_once
        ):
            logger.info_once = MethodType(_info_once, logger)  # type: ignore[attr-defined]
        if (
            getattr(logger.__dict__.get("warning_once"), "__func__", None)
            is not _warning_once
        ):
            logger.warning_once = MethodType(  # type: ignore[attr-defined]
                _warning_once, logger
            )
    return cast(XTokensLogger, logger)


_configure_xtokens_root_logger()
