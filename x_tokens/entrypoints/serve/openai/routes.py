"""Thin FastAPI routes for the supported OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from x_tokens.engine.llm_engine import LLMEngineProtocol
from x_tokens.engine.types import EngineEvent, ErrorEvent

from ..config import ServeConfig
from ..errors import EngineRequestError, OpenAIError
from ..generation import ChatCompletionService, GenerationService
from ..models import ModelRegistry
from .adapter import (
    chat_generate_request,
    chat_response,
    chat_sse,
    collect_events,
    completion_generate_request,
    completion_response,
    completion_sse,
    request_id,
)
from .protocol import ChatCompletionRequest, CompletionRequest
from .sse import SSE_HEADERS

logger = logging.getLogger(__name__)
router = APIRouter()


@dataclass(slots=True)
class ServeServices:
    config: ServeConfig
    engine: LLMEngineProtocol
    models: ModelRegistry
    completions: GenerationService
    chat: ChatCompletionService

    async def close(self) -> None:
        await self.completions.stop()
        await self.chat.stop()
        await self.engine.close()


def _services(request: Request) -> ServeServices:
    return request.app.state.serve_services


def _check_api_key(request: Request, config: ServeConfig) -> None:
    if config.api_key is None:
        return
    if request.headers.get("authorization") != f"Bearer {config.api_key}":
        raise OpenAIError(401, "Incorrect API key provided", "authentication_error")


def _response_headers(request_id_value: str) -> dict[str, str]:
    return {"X-Request-ID": request_id_value}


async def _prepend(
    first: EngineEvent, rest: AsyncIterator[EngineEvent]
) -> AsyncIterator[EngineEvent]:
    yield first
    async for event in rest:
        yield event


async def _prefetch(events: AsyncIterator[EngineEvent]) -> AsyncIterator[EngineEvent]:
    iterator = events.__aiter__()
    try:
        first = await anext(iterator)
    except StopAsyncIteration as exc:
        raise EngineRequestError("The engine ended without a terminal event") from exc
    if isinstance(first, ErrorEvent):
        await iterator.aclose()
        raise EngineRequestError(first.message)
    return _prepend(first, iterator)


async def _disconnect_aware(
    events: AsyncIterator[EngineEvent],
    request: Request,
    service: GenerationService,
    request_id_value: str,
) -> AsyncIterator[EngineEvent]:
    async for event in events:
        if await request.is_disconnected():
            await service.abort(request_id_value)
            return
        yield event


async def _stream_with_timeout(
    body: AsyncIterator[str],
    timeout_s: float | None,
) -> AsyncIterator[str]:
    try:
        if timeout_s is None:
            async for chunk in body:
                yield chunk
            return
        async with asyncio.timeout(timeout_s):
            async for chunk in body:
                yield chunk
    except TimeoutError:
        yield 'data: {"error":{"message":"Request timed out","type":"server_error","param":null,"code":"request_timeout"}}\n\n'
        yield "data: [DONE]\n\n"
    except Exception:
        logger.exception("stream generation failed")
        yield 'data: {"error":{"message":"The engine failed to generate a response","type":"server_error","param":null,"code":null}}\n\n'
        yield "data: [DONE]\n\n"


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(raw_request: Request) -> JSONResponse:
    services = _services(raw_request)
    is_ready = await services.completions.refresh_readiness()
    return JSONResponse(
        status_code=200 if is_ready else 503, content={"ready": is_ready}
    )


@router.get("/v1/models")
async def list_models(raw_request: Request) -> dict[str, object]:
    services = _services(raw_request)
    _check_api_key(raw_request, services.config)
    return services.models.list_models()


@router.post("/v1/completions")
async def completions(body: CompletionRequest, raw_request: Request):
    services = _services(raw_request)
    _check_api_key(raw_request, services.config)
    await services.completions.ensure_available(body.model)
    request_id_value = request_id("cmpl", raw_request.headers.get("x-request-id"))
    generation_request = completion_generate_request(body, request_id_value)
    headers = _response_headers(request_id_value)
    events = services.completions.events(generation_request)

    if body.stream:
        prefetched = await _prefetch(events)
        disconnected = _disconnect_aware(
            prefetched, raw_request, services.completions, request_id_value
        )
        stream = completion_sse(
            generation_request,
            disconnected,
            bool(body.stream_options and body.stream_options.include_usage),
        )
        return StreamingResponse(
            _stream_with_timeout(stream, services.config.request_timeout_s),
            media_type="text/event-stream",
            headers={**SSE_HEADERS, **headers},
        )

    try:
        async with asyncio.timeout(services.config.request_timeout_s):
            text, finished = await collect_events(events)
    except TimeoutError as exc:
        raise OpenAIError(504, "Request timed out", code="request_timeout") from exc
    return JSONResponse(
        completion_response(generation_request, text, finished), headers=headers
    )


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, raw_request: Request):
    services = _services(raw_request)
    _check_api_key(raw_request, services.config)
    await services.chat.ensure_available(body.model)
    request_id_value = request_id("chatcmpl", raw_request.headers.get("x-request-id"))
    prompt = await services.chat.render(body.model, body.messages)
    generation_request = chat_generate_request(body, request_id_value, prompt)
    headers = _response_headers(request_id_value)
    events = services.chat.events(generation_request)

    if body.stream:
        prefetched = await _prefetch(events)
        disconnected = _disconnect_aware(
            prefetched, raw_request, services.chat, request_id_value
        )
        stream = chat_sse(
            generation_request,
            disconnected,
            bool(body.stream_options and body.stream_options.include_usage),
        )
        return StreamingResponse(
            _stream_with_timeout(stream, services.config.request_timeout_s),
            media_type="text/event-stream",
            headers={**SSE_HEADERS, **headers},
        )

    try:
        async with asyncio.timeout(services.config.request_timeout_s):
            text, finished = await collect_events(events)
    except TimeoutError as exc:
        raise OpenAIError(504, "Request timed out", code="request_timeout") from exc
    return JSONResponse(
        chat_response(generation_request, text, finished), headers=headers
    )
