"""Top-level xTokens configuration and structured sub-configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelConfig:
    served_model_name: str = "x-tokens-mock"
    model: str | None = None
    max_model_len: int = 4096

    def __post_init__(self) -> None:
        if not self.served_model_name:
            raise ValueError("served_model_name must not be empty")
        if self.model is not None and not self.model:
            raise ValueError("model must not be empty")
        if self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")

    @property
    def model_name(self) -> str:
        return self.model or self.served_model_name


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    policy: str = "naive"
    max_num_seqs: int = 4

    def __post_init__(self) -> None:
        if not self.policy:
            raise ValueError("scheduler policy must not be empty")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    backend: str = "naive_hf"
    device: str = "auto"
    dtype: str = "auto"
    local_files_only: bool = False

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("executor backend must not be empty")


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str | None = None
    request_timeout_s: float | None = 600.0
    shutdown_timeout_s: float = 30.0
    shutdown_policy: Literal["abort", "drain"] = "abort"
    max_request_body_size: int = 4 * 1024 * 1024
    cors_origins: tuple[str, ...] = ()
    access_log: bool = True

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive or None")
        if self.shutdown_timeout_s <= 0:
            raise ValueError("shutdown_timeout_s must be positive")
        if self.max_request_body_size <= 0:
            raise ValueError("max_request_body_size must be positive")


@dataclass(frozen=True, slots=True)
class XTokensConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    executor_config: ExecutorConfig = field(default_factory=ExecutorConfig)
    server_config: ServerConfig = field(default_factory=ServerConfig)

    def __post_init__(self) -> None:
        if self.executor_config.backend != "naive_hf":
            raise ValueError(
                f"unsupported executor backend: {self.executor_config.backend}"
            )
        if self.scheduler_config.policy != "naive":
            raise ValueError(
                f"unsupported scheduler policy: {self.scheduler_config.policy}"
            )
