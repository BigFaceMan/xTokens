from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from x_tokens.engine.client import EngineClient
from x_tokens.engine.core_client import QueuedEngineCoreClient
from x_tokens.engine.types import (
    CoreErrorEvent,
    CoreEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    EngineHealth,
    ErrorEvent,
    FinishReason,
    GenerateRequest,
    SamplingParams,
    TokenEvent,
)


class ManualCore:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.aborted: list[str] = []
        self._outputs: asyncio.Queue[CoreEvent | Exception | None] = asyncio.Queue()

    async def submit(self, request: GenerateRequest) -> None:
        self.submitted.append(request.request_id)

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    def outputs(self) -> AsyncIterator[CoreEvent]:
        return self._stream_outputs()

    async def _stream_outputs(self) -> AsyncIterator[CoreEvent]:
        while True:
            item = await self._outputs.get()
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item

    async def emit(self, event: CoreEvent) -> None:
        await self._outputs.put(event)

    async def fail_outputs(self) -> None:
        await self._outputs.put(RuntimeError("Core output failure"))

    async def health(self) -> EngineHealth:
        return EngineHealth(True)

    async def close(self) -> None:
        await self._outputs.put(None)


def request(request_id: str) -> GenerateRequest:
    return GenerateRequest(request_id, "mock", "hello", SamplingParams(3))


async def wait_for_submission(core: ManualCore, *request_ids: str) -> None:
    for _ in range(100):
        if all(request_id in core.submitted for request_id in request_ids):
            return
        await asyncio.sleep(0)
    raise AssertionError(f"requests were not submitted: {request_ids}")


async def wait_for_abort(core: ManualCore, request_id: str) -> None:
    for _ in range(100):
        if request_id in core.aborted:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"request was not aborted: {request_id}")


def test_dispatcher_demultiplexes_batched_core_output() -> None:
    async def scenario() -> None:
        core = ManualCore()
        client = QueuedEngineCoreClient(core, output_queue_size=2)
        first = client.submit(request("one"))
        second = client.submit(request("two"))
        first_next = asyncio.create_task(anext(first))
        second_next = asyncio.create_task(anext(second))
        await wait_for_submission(core, "one", "two")

        await core.emit(CoreTokenEvent("two", 2, "two-token"))
        await core.emit(CoreTokenEvent("one", 1, "one-token"))
        assert (await first_next).text == "one-token"
        assert (await second_next).text == "two-token"

        await core.emit(CoreFinishedEvent("one", FinishReason.STOP, 1, 1))
        await core.emit(CoreFinishedEvent("two", FinishReason.STOP, 1, 1))
        assert isinstance(await anext(first), CoreFinishedEvent)
        assert isinstance(await anext(second), CoreFinishedEvent)
        await first.aclose()
        await second.aclose()
        await client.close()

    asyncio.run(scenario())


def test_slow_consumer_is_cancelled_without_blocking_other_requests() -> None:
    async def scenario() -> None:
        core = ManualCore()
        client = QueuedEngineCoreClient(core, output_queue_size=1)
        slow = client.submit(request("slow"))
        slow_first = asyncio.create_task(anext(slow))
        await wait_for_submission(core, "slow")
        await core.emit(CoreTokenEvent("slow", 1, "one"))
        assert (await slow_first).text == "one"

        await core.emit(CoreTokenEvent("slow", 2, "two"))
        await asyncio.sleep(0)
        await core.emit(CoreTokenEvent("slow", 3, "three"))
        await wait_for_abort(core, "slow")

        fast = client.submit(request("fast"))
        fast_first = asyncio.create_task(anext(fast))
        await wait_for_submission(core, "fast")
        await core.emit(CoreTokenEvent("fast", 1, "fast-token"))
        assert (await fast_first).text == "fast-token"

        assert (await anext(slow)).text == "two"
        terminal = await anext(slow)
        assert isinstance(terminal, CoreErrorEvent)
        assert client.metrics.slow_consumer_cancellations == 1
        assert client.metrics.output_queue_blocked_seconds >= 0

        await fast.aclose()
        await slow.aclose()
        await client.close()

    asyncio.run(scenario())


def test_core_output_failure_terminates_pending_request() -> None:
    async def scenario() -> None:
        core = ManualCore()
        client = QueuedEngineCoreClient(core)
        stream = client.submit(request("failure"))
        next_event = asyncio.create_task(anext(stream))
        await wait_for_submission(core, "failure")
        await core.fail_outputs()
        event = await next_event
        assert isinstance(event, CoreErrorEvent)
        assert event.message == "EngineCore output stream failed"
        await stream.aclose()
        await client.close()

    asyncio.run(scenario())


def test_abort_is_idempotent_and_cleans_request_state() -> None:
    async def scenario() -> None:
        core = ManualCore()
        client = QueuedEngineCoreClient(core)
        stream = client.submit(request("cancel"))
        pending = asyncio.create_task(anext(stream))
        await wait_for_submission(core, "cancel")
        await client.abort("cancel")
        await client.abort("cancel")
        event = await pending
        assert isinstance(event, CoreErrorEvent)
        assert core.aborted == ["cancel"]
        await stream.aclose()
        assert client.metrics.active_requests == 0
        await client.close()

    asyncio.run(scenario())


def test_engine_client_normalizes_queued_core_events() -> None:
    async def scenario() -> None:
        core = ManualCore()
        client = EngineClient(QueuedEngineCoreClient(core))
        stream = client.generate(request("normalized"))
        next_event = asyncio.create_task(anext(stream))
        await wait_for_submission(core, "normalized")
        await core.emit(CoreTokenEvent("normalized", 7, "hello"))
        event = await next_event
        assert isinstance(event, TokenEvent)
        assert event.text == "hello"
        await core.emit(CoreErrorEvent("normalized", "failure"))
        terminal = await anext(stream)
        assert isinstance(terminal, ErrorEvent)
        await stream.aclose()
        await client.close()

    asyncio.run(scenario())
