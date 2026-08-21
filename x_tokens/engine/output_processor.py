"""Engine-side conversion from generated token IDs to text."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol


class OutputProcessor(Protocol):
    """Maintain request-local detokenization state outside EngineCore."""

    def process_token(self, request_id: str, token_id: int) -> str: ...

    def finish(self, request_id: str) -> None: ...


class TokenizerOutputProcessor:
    """Incrementally detokenize generated IDs with a HF-compatible tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._token_ids: dict[str, list[int]] = defaultdict(list)
        self._decoded_text: dict[str, str] = defaultdict(str)

    def process_token(self, request_id: str, token_id: int) -> str:
        token_ids = self._token_ids[request_id]
        token_ids.append(token_id)
        previous_text = self._decoded_text[request_id]
        decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        text = decoded.removeprefix(previous_text)
        self._decoded_text[request_id] = previous_text + text
        return text

    def finish(self, request_id: str) -> None:
        self._token_ids.pop(request_id, None)
        self._decoded_text.pop(request_id, None)
