"""Backend interface consumed by the backend-independent EngineCore."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from x_tokens.core.scheduler import SchedulerOutput


class Executor(Protocol):
    """Model execution backend consumed by EngineCore."""

    @property
    def eos_token_ids(self) -> frozenset[int]: ...

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput: ...

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]: ...


@dataclass(frozen=True, slots=True)
class ModelForwardOutput:
    """Normalized output of one model forward pass.

    ``logits`` contains the final-position logits for each request in batch
    order. Optional model state is reserved for future KV-cache executors.
    """

    logits: Any
    past_key_values: Any | None = None
