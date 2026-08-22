from __future__ import annotations

from typing import Any

from x_tokens.entrypoints.serve import cli


def test_cli_passes_unified_logging_config_to_uvicorn(monkeypatch) -> None:
    invocation: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        invocation["app"] = app
        invocation.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    cli.main(["--model", "test-model", "--no-access-log"])

    assert invocation["access_log"] is False
    assert invocation["log_config"]["loggers"]["uvicorn"]["handlers"] == ["xtokens"]
    assert invocation["log_config"]["loggers"]["uvicorn.access"]["handlers"] == [
        "xtokens_access"
    ]
