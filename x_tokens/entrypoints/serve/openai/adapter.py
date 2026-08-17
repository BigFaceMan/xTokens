"""Conversion between OpenAI DTOs and protocol-independent Serve values."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from x_tokens.engine.types import (
    ErrorEvent,
    FinishedEvent,
    GenerateRequest,
    SamplingParams,
    TokenEvent,
)

from ..errors import EngineRequestError
from .protocol import ChatCompletionRequest, CompletionRequest
from .sse import encode_sse


def request_id(kind: str, supplied_request_id: str | None) -> str:
    if supplied_request_id:
        return supplied_request_id
    return f"{kind}-{uuid.uuid4().hex}"


def _stop_values(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    return (stop,) if isinstance(stop, str) else tuple(stop)


def sampling_from_request(
    request: CompletionRequest | ChatCompletionRequest,
) -> SamplingParams:
    max_tokens = (
        getattr(request, "max_completion_tokens", None) or request.max_tokens or 16
    )
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stop=_stop_values(request.stop),
    )


def completion_generate_request(
    request: CompletionRequest, request_id_value: str
) -> GenerateRequest:
    return GenerateRequest(
        request_id_value, request.model, request.prompt, sampling_from_request(request)
    )


def chat_generate_request(
    request: ChatCompletionRequest, request_id_value: str, prompt: str
) -> GenerateRequest:
    return GenerateRequest(
        request_id_value, request.model, prompt, sampling_from_request(request)
    )


def usage(event: FinishedEvent) -> dict[str, int]:
    return {
        "prompt_tokens": event.prompt_tokens,
        "completion_tokens": event.completion_tokens,
        "total_tokens": event.prompt_tokens + event.completion_tokens,
    }


async def collect_events(
    events: AsyncIterator[TokenEvent | FinishedEvent | ErrorEvent],
) -> tuple[str, FinishedEvent]:
    text: list[str] = []
    async for event in events:
        if isinstance(event, TokenEvent):
            text.append(event.text)
        elif isinstance(event, FinishedEvent):
            return "".join(text), event
        else:
            raise EngineRequestError(event.message)
    raise EngineRequestError("The engine ended without a terminal event")


def completion_response(
    request: GenerateRequest, text: str, finished: FinishedEvent
) -> dict[str, object]:
    return {
        "id": request.request_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {"index": 0, "text": text, "finish_reason": finished.finish_reason.value}
        ],
        "usage": usage(finished),
    }


def chat_response(
    request: GenerateRequest, text: str, finished: FinishedEvent
) -> dict[str, object]:
    return {
        "id": request.request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finished.finish_reason.value,
            }
        ],
        "usage": usage(finished),
    }


async def completion_sse(
    request: GenerateRequest,
    events: AsyncIterator[TokenEvent | FinishedEvent | ErrorEvent],
    include_usage: bool,
) -> AsyncIterator[str]:
    created = int(time.time())
    async for event in events:
        if isinstance(event, TokenEvent):
            yield encode_sse(
                {
                    "id": request.request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {"index": 0, "text": event.text, "finish_reason": None}
                    ],
                }
            )
        elif isinstance(event, FinishedEvent):
            yield encode_sse(
                {
                    "id": request.request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "finish_reason": event.finish_reason.value,
                        }
                    ],
                }
            )
            if include_usage:
                yield encode_sse(
                    {
                        "id": request.request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": request.model,
                        "choices": [],
                        "usage": usage(event),
                    }
                )
        else:
            yield encode_sse(EngineRequestError(event.message).body())
            break
    yield encode_sse("[DONE]")


async def chat_sse(
    request: GenerateRequest,
    events: AsyncIterator[TokenEvent | FinishedEvent | ErrorEvent],
    include_usage: bool,
) -> AsyncIterator[str]:
    created = int(time.time())
    sent_role = False
    async for event in events:
        if isinstance(event, TokenEvent):
            delta: dict[str, str] = {"content": event.text}
            if not sent_role:
                delta["role"] = "assistant"
                sent_role = True
            yield encode_sse(
                {
                    "id": request.request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )
        elif isinstance(event, FinishedEvent):
            yield encode_sse(
                {
                    "id": request.request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": event.finish_reason.value,
                        }
                    ],
                }
            )
            if include_usage:
                yield encode_sse(
                    {
                        "id": request.request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model,
                        "choices": [],
                        "usage": usage(event),
                    }
                )
        else:
            yield encode_sse(EngineRequestError(event.message).body())
            break
    yield encode_sse("[DONE]")
