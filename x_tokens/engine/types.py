"""Protocol-independent request and streaming event types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SamplingParams:
    """Canonical generation parameters accepted by the engine layer."""

    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    ignore_eos: bool = False


@dataclass(frozen=True, slots=True)
class GenerateRequest:
    """A protocol-independent text generation request."""

    request_id: str
    model: str
    prompt: str | tuple[int, ...]
    sampling: SamplingParams
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    token_id: int
    text: str


@dataclass(frozen=True, slots=True)
class FinishedEvent:
    request_id: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    request_id: str
    message: str
    retryable: bool = False


EngineEvent = TokenEvent | FinishedEvent | ErrorEvent


@dataclass(frozen=True, slots=True)
class CoreTokenEvent:
    """An unnormalized token event emitted by the Core transport."""

    request_id: str
    token_id: int


@dataclass(frozen=True, slots=True)
class CoreFinishedEvent:
    request_id: str
    finish_reason: FinishReason
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True, slots=True)
class CoreErrorEvent:
    request_id: str
    message: str
    retryable: bool = False


CoreEvent = CoreTokenEvent | CoreFinishedEvent | CoreErrorEvent


@dataclass(frozen=True, slots=True)
class EngineHealth:
    ready: bool
    detail: str = ""
