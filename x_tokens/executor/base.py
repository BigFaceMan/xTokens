"""Backend interface consumed by the backend-independent EngineCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from x_tokens.core.scheduler import ScheduledRequest, SchedulingBatch


class Executor(Protocol):
    """Tokenization, one-step execution, and decoding for a model backend."""

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    def encode(self, prompt: str | tuple[int, ...]) -> tuple[int, ...]: ...

    def execute(self, batch: SchedulingBatch) -> tuple[int, ...]: ...

    def decode_delta(self, request: ScheduledRequest) -> str: ...
