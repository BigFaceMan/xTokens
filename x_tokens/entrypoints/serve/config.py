"""Configuration owned by the HTTP Serve entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ServeConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    served_model_name: str = "x-tokens-mock"
    hf_model: str | None = None
    hf_device: str = "auto"
    hf_dtype: str = "auto"
    hf_local_files_only: bool = False
    hf_max_num_seqs: int = 4
    api_key: str | None = None
    request_timeout_s: float | None = 600.0
    shutdown_timeout_s: float = 30.0
    shutdown_policy: Literal["abort", "drain"] = "abort"
    max_request_body_size: int = 4 * 1024 * 1024
    cors_origins: tuple[str, ...] = ()
    access_log: bool = True

    def __post_init__(self) -> None:
        if not self.served_model_name:
            raise ValueError("served_model_name must not be empty")
        if self.hf_model is not None and not self.hf_model:
            raise ValueError("hf_model must not be empty")
        if self.hf_max_num_seqs < 1:
            raise ValueError("hf_max_num_seqs must be positive")
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive or None")
        if self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if self.max_request_body_size <= 0:
            raise ValueError("max_request_body_size must be positive")
