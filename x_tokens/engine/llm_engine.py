"""Serving-facing LLM engine facade."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
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


class LLMEngineProtocol(Protocol):
    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]: ...

    async def abort(self, request_id: str) -> None: ...

    async def health(self) -> EngineHealth: ...

    async def close(self) -> None: ...


class LLMEngine(LLMEngineProtocol):
    """Stable facade used by entrypoints regardless of Core transport."""

    def __init__(self, core_client: EngineCoreClient) -> None:
        self._core_client = core_client
        self._active_request_ids: set[str] = set()
        self._pending_events: dict[str, deque[CoreEvent]] = defaultdict(deque)
        self._closed = False

    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        return self._generate(request)

    async def _generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        self._active_request_ids.add(request.request_id)
        terminal = False
        try:
            self._core_client.add_request(request)
            while not terminal:
                events = self._pending_events.pop(request.request_id, ())
                if not events:
                    self._dispatch_outputs(self._core_client.get_output())
                    events = self._pending_events.pop(request.request_id, ())
                for event in events:
                    terminal = isinstance(event, (CoreErrorEvent, CoreFinishedEvent))
                    yield self._normalize_event(event)
                    if terminal:
                        break
                await asyncio.sleep(0)
        finally:
            if not terminal:
                self._core_client.abort_requests((request.request_id,))
            self._active_request_ids.discard(request.request_id)

    def _dispatch_outputs(self, events: tuple[CoreEvent, ...]) -> None:
        for event in events:
            self._pending_events[event.request_id].append(event)

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
        self._core_client.abort_requests((request_id,))

    async def health(self) -> EngineHealth:
        return self._core_client.health()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._core_client.abort_requests(tuple(self._active_request_ids))
        self._core_client.close()
