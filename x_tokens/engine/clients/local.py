"""Deterministic local mock Core transport for the Serve API iteration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..core_client import CoreOutputSource, QueuedEngineCoreClient
from ..types import (
    CoreEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    EngineHealth,
    FinishReason,
    GenerateRequest,
)


class MockEngineCore:
    """Mock EngineCore that publishes events to one global output stream.

    It has the same producer shape as a future local EngineCore: submit starts
    independent work and ``outputs()`` is consumed once by the Core client
    dispatcher.
    """

    def __init__(self, response_text: str = "Mock response") -> None:
        self._response_text = response_text
        self._aborted: set[str] = set()
        self._closed = False
        self._outputs: asyncio.Queue[CoreEvent | None] = asyncio.Queue()
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def submit(self, request: GenerateRequest) -> None:
        if self._closed:
            raise RuntimeError("mock EngineCore is closed")
        self._tasks[request.request_id] = asyncio.create_task(self._produce(request))

    async def _produce(self, request: GenerateRequest) -> None:
        words = self._response_text.split(" ")[: request.sampling.max_tokens]
        emitted = 0
        try:
            for index, word in enumerate(words):
                if request.request_id in self._aborted or self._closed:
                    return
                text = word if index == 0 else f" {word}"
                emitted += 1
                await self._outputs.put(CoreTokenEvent(request.request_id, index, text))
                await asyncio.sleep(0)
            if request.request_id not in self._aborted and not self._closed:
                await self._outputs.put(
                    CoreFinishedEvent(
                        request_id=request.request_id,
                        finish_reason=FinishReason.STOP,
                        prompt_tokens=len(str(request.prompt).split()),
                        completion_tokens=emitted,
                    )
                )
        finally:
            self._tasks.pop(request.request_id, None)

    async def outputs(self) -> AsyncIterator[CoreEvent]:
        while True:
            event = await self._outputs.get()
            if event is None:
                return
            yield event

    async def abort(self, request_id: str) -> None:
        self._aborted.add(request_id)
        task = self._tasks.get(request_id)
        if task is not None:
            task.cancel()

    async def health(self) -> EngineHealth:
        return EngineHealth(ready=not self._closed, detail="mock engine")

    async def close(self) -> None:
        self._closed = True
        for task in tuple(self._tasks.values()):
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        await self._outputs.put(None)


class LocalEngineCoreClient(QueuedEngineCoreClient):
    """Local EngineCoreClient with a mock Core until real Core is available."""

    def __init__(
        self, response_text: str = "Mock response", *, output_queue_size: int = 32
    ) -> None:
        self._mock_core: CoreOutputSource = MockEngineCore(response_text)
        super().__init__(self._mock_core, output_queue_size=output_queue_size)
