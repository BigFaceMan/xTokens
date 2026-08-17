"""Configuration owned by the HTTP Serve entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str = "x-tokens-mock"
    api_key: str | None = None
    engine_mode: str = "local"
    request_timeout_s: float | None = 600.0
    shutdown_timeout_s: float = 30.0
    shutdown_policy: Literal["abort", "drain"] = "abort"
    max_request_body_size: int = 4 * 1024 * 1024
    output_queue_size: int = 32
    cors_origins: tuple[str, ...] = ()
    access_log: bool = True

    def __post_init__(self) -> None:
        if not self.served_model_name:
            raise ValueError("served_model_name must not be empty")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.engine_mode != "local":
            raise ValueError(
                "only local engine_mode is available until EngineCore exists"
            )
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive or None")
        if self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if self.max_request_body_size <= 0:
            raise ValueError("max_request_body_size must be positive")
        if self.output_queue_size <= 0:
            raise ValueError("output_queue_size must be positive")
