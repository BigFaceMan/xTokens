"""Serving-facing engine facade."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .core_client import EngineCoreClient
from .types import (
    CoreErrorEvent,
    CoreEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    EngineEvent,
    EngineHealth,
    ErrorEvent,
    FinishedEvent,
    GenerateRequest,
    TokenEvent,
)


class EngineClientProtocol(Protocol):
    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]: ...

    async def abort(self, request_id: str) -> None: ...

    async def health(self) -> EngineHealth: ...

    async def close(self) -> None: ...


class EngineClient:
    """Stable facade used by entrypoints regardless of Core transport."""

    def __init__(self, core_client: EngineCoreClient) -> None:
        self._core_client = core_client
        self._active_request_ids: set[str] = set()
        self._closed = False

    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        return self._generate(request)

    async def _generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        self._active_request_ids.add(request.request_id)
        try:
            async for event in self._core_client.submit(request):
                yield self._normalize_event(event)
        finally:
            self._active_request_ids.discard(request.request_id)

    @staticmethod
    def _normalize_event(event: CoreEvent) -> EngineEvent:
        if isinstance(event, CoreTokenEvent):
            return TokenEvent(event.request_id, event.token_id, event.text)
        if isinstance(event, CoreFinishedEvent):
            return FinishedEvent(
                event.request_id,
                event.finish_reason,
                event.prompt_tokens,
                event.completion_tokens,
            )
        if isinstance(event, CoreErrorEvent):
            return ErrorEvent(event.request_id, event.message, event.retryable)
        raise TypeError(f"unsupported Core event: {type(event)!r}")

    async def abort(self, request_id: str) -> None:
        await self._core_client.abort(request_id)

    async def health(self) -> EngineHealth:
        return await self._core_client.health()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for request_id in tuple(self._active_request_ids):
            await self.abort(request_id)
        await self._core_client.close()
