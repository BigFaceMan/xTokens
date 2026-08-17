"""Stable errors returned by the OpenAI-compatible HTTP layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OpenAIError(Exception):
    status_code: int
    message: str
    error_type: str = "server_error"
    param: str | None = None
    code: str | None = None

    def body(self) -> dict[str, object]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class EngineRequestError(OpenAIError):
    def __init__(
        self, message: str = "The engine failed to generate a response"
    ) -> None:
        super().__init__(500, message)


class EngineUnavailableError(OpenAIError):
    def __init__(self) -> None:
        super().__init__(
            503, "The engine is not ready", "server_error", code="engine_unavailable"
        )
