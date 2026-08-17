"""Frontend prompt rendering abstractions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .openai.protocol import ChatMessage


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str


class PromptRenderer(Protocol):
    async def render_chat(
        self, model: str, messages: Sequence[ChatMessage]
    ) -> RenderedPrompt: ...


class PlainTextPromptRenderer:
    """Temporary text renderer used until a model tokenizer is connected."""

    async def render_chat(
        self, model: str, messages: Sequence[ChatMessage]
    ) -> RenderedPrompt:
        del model
        return RenderedPrompt(
            "\n".join(f"{message.role}: {message.content}" for message in messages)
        )
