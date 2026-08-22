from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

import x_tokens.logger as logger_module
from x_tokens.logger import (
    ColoredFormatter,
    NewLineFormatter,
    UvicornAccessFormatter,
    create_uvicorn_log_config,
    init_logger,
    set_logging_rank_context,
)

FORMAT = "%(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"
ANSI_ESCAPE = "\033["
REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(
    level: int,
    message: str,
    *,
    pathname: str | None = None,
    lineno: int = 42,
) -> logging.LogRecord:
    return logging.LogRecord(
        "x_tokens.test",
        level,
        pathname or str(REPO_ROOT / "x_tokens" / "test_module.py"),
        lineno,
        message,
        (),
        None,
    )


def _isolated_logger(name: str) -> tuple[logging.Logger, StringIO]:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    init_logger(name)
    return logger, stream


def _run_python(
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("XTOKENS_LOGGING_") or name == "NO_COLOR":
            environment.pop(name)
    environment.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_newline_formatter_aligns_continuation_lines() -> None:
    formatter = NewLineFormatter(FORMAT, datefmt="%m-%d %H:%M:%S")

    output = formatter.format(_record(logging.INFO, "first line\nsecond line"))
    first, second = output.splitlines()

    assert "[test_module.py:42] first line" in first
    assert second.index("second line") == first.index("first line")


def test_newline_formatter_aligns_exception_traceback() -> None:
    formatter = NewLineFormatter(FORMAT)
    record = _record(logging.ERROR, "failed")
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record.exc_info = sys.exc_info()

    output = formatter.format(record)
    lines = output.splitlines()

    assert lines[1].index("Traceback") == lines[0].index("failed")
    assert "RuntimeError: boom" in output


def test_formatter_folds_debug_paths_only() -> None:
    pathname = REPO_ROOT / "x_tokens/model_executor/layers/quantization/utils/fp8.py"
    formatter = NewLineFormatter(FORMAT)

    debug_output = formatter.format(
        _record(logging.DEBUG, "debug", pathname=str(pathname))
    )
    info_output = formatter.format(
        _record(logging.INFO, "info", pathname=str(pathname))
    )

    assert "[model_executor/.../quantization/utils/fp8.py:42]" in debug_output
    assert "[fp8.py:42]" in info_output


def test_colored_formatter_only_colors_the_prefix() -> None:
    formatter = ColoredFormatter(FORMAT)

    output = formatter.format(_record(logging.WARNING, "plain message"))

    assert ANSI_ESCAPE in output
    assert f"{ANSI_ESCAPE}33mWARNING" in output
    assert f"{ANSI_ESCAPE}90m" in output
    assert output.endswith("plain message")


def test_uvicorn_access_formatter_extracts_record_arguments() -> None:
    formatter = UvicornAccessFormatter(
        '%(levelname)s [access] %(client_addr)s "%(request_line)s" %(status_code)s'
    )
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "httptools_impl.py",
        1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:1234", "POST", "/v1/completions", "1.1", 200),
        None,
    )

    output = formatter.format(record)

    assert output == (
        'INFO [access] 127.0.0.1:1234 "POST /v1/completions HTTP/1.1" 200'
    )


def test_once_methods_deduplicate_and_honor_explicit_keys() -> None:
    logger, stream = _isolated_logger("x_tokens.test.once")

    logger.info_once("value %s", "one", key="same")  # type: ignore[attr-defined]
    logger.info_once("value %s", "two", key="same")  # type: ignore[attr-defined]
    logger.info_once("value %s", "two", key="different")  # type: ignore[attr-defined]

    assert stream.getvalue().splitlines() == ["value one", "value two"]
    assert logger_module._remember_once.cache_parameters()["maxsize"] == 2048


def test_init_logger_overrides_foreign_class_once_methods(monkeypatch) -> None:
    def foreign_info_once(self, *args, **kwargs) -> None:
        del self, args, kwargs

    monkeypatch.setattr(
        logging.Logger,
        "info_once",
        foreign_info_once,
        raising=False,
    )
    logger, stream = _isolated_logger("x_tokens.test.foreign_once")

    logger.info_once("xTokens method", key="foreign")  # type: ignore[attr-defined]

    assert stream.getvalue() == "xTokens method\n"


def test_disabled_once_message_is_not_cached() -> None:
    logger, stream = _isolated_logger("x_tokens.test.disabled_once")
    logger.setLevel(logging.WARNING)

    logger.info_once("enabled later", key="level-change")  # type: ignore[attr-defined]
    logger.setLevel(logging.INFO)
    logger.info_once("enabled later", key="level-change")  # type: ignore[attr-defined]

    assert stream.getvalue() == "enabled later\n"


def test_once_scope_suppresses_non_primary_ranks() -> None:
    logger, stream = _isolated_logger("x_tokens.test.scope")
    set_logging_rank_context(global_rank=1, local_rank=1)
    try:
        logger.warning_once("process", key="process")  # type: ignore[attr-defined]
        logger.warning_once("local", scope="local", key="local-1")  # type: ignore[attr-defined]
        logger.warning_once("global", scope="global", key="global-1")  # type: ignore[attr-defined]
        set_logging_rank_context(global_rank=1, local_rank=0)
        logger.warning_once("local", scope="local", key="local-2")  # type: ignore[attr-defined]
        logger.warning_once("global", scope="global", key="global-2")  # type: ignore[attr-defined]
        set_logging_rank_context(global_rank=0, local_rank=1)
        logger.warning_once("global", scope="global", key="global-3")  # type: ignore[attr-defined]
    finally:
        set_logging_rank_context(global_rank=0, local_rank=0)

    assert stream.getvalue().splitlines() == ["process", "local", "global"]


def test_invalid_once_scope_is_rejected() -> None:
    logger, _ = _isolated_logger("x_tokens.test.invalid_scope")

    with pytest.raises(ValueError, match="invalid logging scope"):
        logger.info_once("bad scope", scope="node")  # type: ignore[attr-defined,arg-type]


def test_environment_controls_level_prefix_color_and_stream() -> None:
    code = (
        "from x_tokens.logger import init_logger; "
        "init_logger('x_tokens.subprocess').info('hello')"
    )
    result = _run_python(
        code,
        extra_env={
            "XTOKENS_LOGGING_LEVEL": "INFO",
            "XTOKENS_LOGGING_PREFIX": "worker-0 ",
            "XTOKENS_LOGGING_COLOR": "1",
            "XTOKENS_LOGGING_STREAM": "stderr",
        },
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert "worker-0 hello" in result.stderr
    assert ANSI_ESCAPE in result.stderr


def test_no_color_overrides_forced_color() -> None:
    result = _run_python(
        "from x_tokens.logger import init_logger; "
        "init_logger('x_tokens.subprocess').warning('hello')",
        extra_env={"XTOKENS_LOGGING_COLOR": "1", "NO_COLOR": "1"},
    )

    assert result.returncode == 0
    assert ANSI_ESCAPE not in result.stdout


def test_custom_dict_config_takes_full_control(tmp_path: Path) -> None:
    config_path = tmp_path / "logging.json"
    config_path.write_text(
        json.dumps(
            {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {"plain": {"format": "CUSTOM %(message)s"}},
                "handlers": {
                    "custom": {
                        "class": "logging.StreamHandler",
                        "formatter": "plain",
                        "stream": "ext://sys.stdout",
                    }
                },
                "loggers": {
                    "x_tokens": {
                        "handlers": ["custom"],
                        "level": "INFO",
                        "propagate": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = _run_python(
        "from x_tokens.logger import init_logger; "
        "init_logger('x_tokens.subprocess').info('hello')",
        extra_env={"XTOKENS_LOGGING_CONFIG_PATH": str(config_path)},
    )

    assert result.returncode == 0
    assert result.stdout == "CUSTOM hello\n"


def test_invalid_custom_config_fails_import(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    result = _run_python(
        "import x_tokens.logger",
        extra_env={"XTOKENS_LOGGING_CONFIG_PATH": str(missing)},
    )

    assert result.returncode != 0
    assert "failed to load logging config" in result.stderr


def test_invalid_environment_value_fails_import() -> None:
    result = _run_python(
        "import x_tokens.logger",
        extra_env={"XTOKENS_LOGGING_LEVEL": "LOUD"},
    )

    assert result.returncode != 0
    assert "failed to configure xTokens logging" in result.stderr


def test_reloading_logger_does_not_duplicate_handlers() -> None:
    result = _run_python(
        "import importlib; import x_tokens.logger as module; "
        "before = len(module.logging.getLogger('x_tokens').handlers); "
        "importlib.reload(module); "
        "after = len(module.logging.getLogger('x_tokens').handlers); "
        "print(before, after)"
    )

    assert result.returncode == 0
    assert result.stdout == "1 1\n"


def test_uvicorn_config_uses_xtokens_handlers() -> None:
    config = create_uvicorn_log_config()

    assert config["loggers"]["uvicorn"]["handlers"] == ["xtokens"]
    assert config["loggers"]["uvicorn.error"]["propagate"] is True
    assert config["loggers"]["uvicorn.access"]["handlers"] == ["xtokens_access"]
    assert (
        config["handlers"]["xtokens_access"]["stream"]
        == (config["handlers"]["xtokens"]["stream"])
    )
