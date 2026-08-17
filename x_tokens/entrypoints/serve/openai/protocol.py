"""Pydantic DTOs for the supported OpenAI-compatible API surface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StreamOptions(BaseModel):
    include_usage: bool = False


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None

    @model_validator(mode="after")
    def validate_token_limits(self) -> ChatCompletionRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError(
                "only one of max_tokens or max_completion_tokens may be set"
            )
        return self
