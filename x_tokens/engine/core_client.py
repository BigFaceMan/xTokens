"""Core transport contracts and request-scoped output dispatching."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from .types import (
    CoreErrorEvent,
    CoreEvent,
    CoreFinishedEvent,
    EngineHealth,
    GenerateRequest,
)


class EngineCoreClient(Protocol):
    """Submit canonical requests to an in-process or remote inference core."""

    def submit(self, request: GenerateRequest) -> AsyncIterator[CoreEvent]: ...

    async def abort(self, request_id: str) -> None: ...

    async def health(self) -> EngineHealth: ...

    async def close(self) -> None: ...


class CoreOutputSource(Protocol):
    """The small interface a local EngineCore or RPC transport must provide."""

    async def submit(self, request: GenerateRequest) -> None: ...

    async def abort(self, request_id: str) -> None: ...

    def outputs(self) -> AsyncIterator[CoreEvent]: ...

    async def health(self) -> EngineHealth: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class RequestState:
    """Per-request output ownership, private to one EngineCoreClient."""

    queue: asyncio.Queue[CoreEvent]
    terminal: bool = False
    cancel_reason: str | None = None
    blocked_started_at: float | None = None
    terminal_task: asyncio.Task[None] | None = None


@dataclass(slots=True)
class DispatchMetrics:
    active_requests: int = 0
    unknown_request_events: int = 0
    slow_consumer_cancellations: int = 0
    output_queue_blocked_seconds: float = 0.0


def _is_terminal(event: CoreEvent) -> bool:
    return isinstance(event, (CoreErrorEvent, CoreFinishedEvent))


class QueuedEngineCoreClient:
    """Demultiplex batched Core output into bounded request-local queues.

    The dispatcher is the only consumer of ``CoreOutputSource.outputs()``. HTTP
    request coroutines consume only their own queues, so their speed never blocks
    the Core output loop.
    """

    def __init__(self, core: CoreOutputSource, *, output_queue_size: int = 32) -> None:
        if output_queue_size < 1:
            raise ValueError("output_queue_size must be positive")
        self._core = core
        self._output_queue_size = output_queue_size
        self._requests: dict[str, RequestState] = {}
        self._dispatcher_task: asyncio.Task[None] | None = None
        self._closed = False
        self.metrics = DispatchMetrics()

    def submit(self, request: GenerateRequest) -> AsyncIterator[CoreEvent]:
        return self._stream(request)

    async def _stream(self, request: GenerateRequest) -> AsyncIterator[CoreEvent]:
        if self._closed:
            yield CoreErrorEvent(request.request_id, "EngineCoreClient is closed")
            return
        if request.request_id in self._requests:
            yield CoreErrorEvent(request.request_id, "Duplicate request ID")
            return

        self._ensure_dispatcher()
        state = RequestState(asyncio.Queue(maxsize=self._output_queue_size))
        self._requests[request.request_id] = state
        self.metrics.active_requests = len(self._requests)
        try:
            await self._core.submit(request)
            while True:
                event = await state.queue.get()
                yield event
                if _is_terminal(event):
                    return
        finally:
            if not state.terminal:
                await self.abort(request.request_id, reason="consumer_closed")
            self._remove_request(request.request_id, state)

    async def abort(self, request_id: str, *, reason: str = "cancelled") -> None:
        state = self._requests.get(request_id)
        if state is None:
            await self._core.abort(request_id)
            return
        if state.terminal:
            return

        state.terminal = True
        state.cancel_reason = reason
        await self._core.abort(request_id)
        self._schedule_terminal(
            request_id,
            state,
            CoreErrorEvent(request_id, f"Request aborted: {reason}"),
        )

    async def health(self) -> EngineHealth:
        return await self._core.health()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await asyncio.gather(
            *(
                self.abort(request_id, reason="engine_shutdown")
                for request_id in tuple(self._requests)
            ),
        )
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            await asyncio.gather(self._dispatcher_task, return_exceptions=True)
        await self._core.close()

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher_task is None:
            self._dispatcher_task = asyncio.create_task(self._dispatch_outputs())

    async def _dispatch_outputs(self) -> None:
        try:
            async for event in self._core.outputs():
                self._dispatch_event(event)
            if not self._closed:
                self._fail_pending_requests("EngineCore output stream ended")
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - Core transports can raise implementation-specific errors.
            self._fail_pending_requests("EngineCore output stream failed")

    def _fail_pending_requests(self, message: str) -> None:
        for request_id, state in tuple(self._requests.items()):
            if state.terminal:
                continue
            state.terminal = True
            self._schedule_terminal(
                request_id,
                state,
                CoreErrorEvent(request_id, message),
            )

    def _dispatch_event(self, event: CoreEvent) -> None:
        state = self._requests.get(event.request_id)
        if state is None:
            self.metrics.unknown_request_events += 1
            return
        if state.terminal:
            return

        if _is_terminal(event):
            state.terminal = True
            self._schedule_terminal(event.request_id, state, event)
            return

        try:
            state.queue.put_nowait(event)
        except asyncio.QueueFull:
            state.terminal = True
            state.cancel_reason = "output_queue_full"
            state.blocked_started_at = time.monotonic()
            self.metrics.slow_consumer_cancellations += 1
            state.terminal_task = asyncio.create_task(
                self._abort_slow_consumer(event.request_id, state)
            )

    async def _abort_slow_consumer(self, request_id: str, state: RequestState) -> None:
        await self._core.abort(request_id)
        await state.queue.put(
            CoreErrorEvent(request_id, "Request aborted: output queue is full")
        )
        self._record_blocked_duration(state)

    def _schedule_terminal(
        self, request_id: str, state: RequestState, event: CoreEvent
    ) -> None:
        try:
            state.queue.put_nowait(event)
            self._record_blocked_duration(state)
        except asyncio.QueueFull:
            if state.terminal_task is None or state.terminal_task.done():
                state.terminal_task = asyncio.create_task(
                    self._wait_and_signal_terminal(request_id, state, event)
                )

    async def _wait_and_signal_terminal(
        self, request_id: str, state: RequestState, event: CoreEvent
    ) -> None:
        await state.queue.put(event)
        self._record_blocked_duration(state)

    def _record_blocked_duration(self, state: RequestState) -> None:
        if state.blocked_started_at is not None:
            self.metrics.output_queue_blocked_seconds += (
                time.monotonic() - state.blocked_started_at
            )
            state.blocked_started_at = None

    def _remove_request(self, request_id: str, state: RequestState) -> None:
        if self._requests.get(request_id) is not state:
            return
        self._requests.pop(request_id)
        self.metrics.active_requests = len(self._requests)
        if state.terminal_task is not None and not state.terminal_task.done():
            state.terminal_task.cancel()
