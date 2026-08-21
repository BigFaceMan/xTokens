"""Backend-independent configuration for an EngineCore."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineCoreConfig:
    """Scheduling and model-routing settings owned by EngineCore."""

    model_aliases: tuple[str, ...]
    max_model_len: int = 4096
    max_num_seqs: int = 4

    def __post_init__(self) -> None:
        if not self.model_aliases:
            raise ValueError("model_aliases must not be empty")
        if self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be positive")

    def accepts_model(self, model: str) -> bool:
        return model in self.model_aliases
