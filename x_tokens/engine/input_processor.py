"""Engine-side prompt preprocessing."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from .types import GenerateRequest


class InputProcessor(Protocol):
    """Turn user-facing prompts into the token IDs accepted by EngineCore."""

    def process(self, request: GenerateRequest) -> GenerateRequest: ...


class TokenizerInputProcessor:
    """Adapt a Hugging Face-compatible tokenizer for the Engine boundary."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_config(
        cls, model: str, *, local_files_only: bool
    ) -> TokenizerInputProcessor:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "HF input processing requires x-tokens[hf] (transformers)"
            ) from error
        return cls(
            AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
        )

    @property
    def pad_token_id(self) -> int:
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        return int(pad_token_id)

    @property
    def eos_token_ids(self) -> frozenset[int]:
        value = self._tokenizer.eos_token_id
        if value is None:
            return frozenset()
        return frozenset((value,)) if isinstance(value, int) else frozenset(value)

    @property
    def tokenizer(self) -> Any:
        """Expose the shared tokenizer to the paired output processor."""
        return self._tokenizer

    def process(self, request: GenerateRequest) -> GenerateRequest:
        if isinstance(request.prompt, tuple):
            token_ids = request.prompt
        else:
            encoded = self._tokenizer(request.prompt, add_special_tokens=True)
            token_ids = tuple(int(token_id) for token_id in encoded["input_ids"])
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        return replace(request, prompt=token_ids)
