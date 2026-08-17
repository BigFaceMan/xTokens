"""Shared request and response models used by the standalone benchmark."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleRequest:
    """A normalized benchmark sample independent of any API backend."""

    prompt: str | list[str] | list[dict[str, Any]]
    prompt_len: int
    expected_output_len: int = 0
    request_id: str | None = None
    timestamp: float | None = None
    multi_modal_data: dict[str, Any] | list[dict[str, Any]] | None = None
    chat_messages: list[dict[str, Any]] | None = None
    request_overrides: dict[str, Any] | None = None


@dataclass
class RequestFuncInput:
    """Backend-neutral input passed to one asynchronous request adapter."""

    prompt: str | list[str] | list[dict[str, Any]]
    api_url: str
    model: str
    output_len: int
    prompt_len: int
    model_name: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    logprobs: int | None = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, Any] | None = None
    chat_messages: list[dict[str, Any]] | None = None
    multi_modal_content: dict[str, Any] | list[dict[str, Any]] | None = None
    request_id: str | None = None


@dataclass
class RequestFuncOutput:
    """Raw timing and usage data collected for a single request."""

    success: bool = False
    generated_text: str = ""
    output_tokens: int = 0
    prompt_len: int = 0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    latency: float = 0.0
    start_time: float = 0.0
    error: str = ""
    status_code: int | None = None

    @property
    def tpot(self) -> float:
        """Average time per generated token after the first token."""
        if len(self.itl) == 0:
            return 0.0
        return sum(self.itl) / len(self.itl)
