from __future__ import annotations

import asyncio

from x_tokens.core import EngineCore, EngineCoreConfig
from x_tokens.core.scheduler import ScheduledRequest, SchedulingBatch
from x_tokens.engine.clients.inproc import InprocClient
from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.engine.types import (
    CoreFinishedEvent,
    FinishedEvent,
    GenerateRequest,
    SamplingParams,
    TokenEvent,
)
from x_tokens.executor.base import Executor


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))

    def encode(self, prompt: str | tuple[int, ...]) -> tuple[int, ...]:
        del prompt
        return (1,)

    def execute(self, batch: SchedulingBatch) -> tuple[int, ...]:
        return tuple(9 for _ in batch.requests)

    def decode_delta(self, request: ScheduledRequest) -> str:
        del request
        return ""


def test_inproc_client_owns_core_and_steps_in_the_calling_thread() -> None:
    client = InprocClient(EngineCoreConfig(("test-model",)), FakeExecutor)
    assert isinstance(client, InprocClient)
    assert isinstance(client.engine_core, EngineCore)
    client.add_request(
        GenerateRequest("inproc-request", "test-model", "hello", SamplingParams(2))
    )

    outputs = client.get_output()

    assert isinstance(outputs[-1], CoreFinishedEvent)
    assert outputs[-1].request_id == "inproc-request"
    client.close()
    assert not client.health().ready


def test_llm_engine_normalizes_direct_inproc_outputs() -> None:
    async def scenario() -> None:
        engine = LLMEngine(
            InprocClient(EngineCoreConfig(("test-model",)), FakeExecutor)
        )
        request = GenerateRequest("request", "test-model", "hello", SamplingParams(2))
        events = [event async for event in engine.generate(request)]
        assert not any(isinstance(event, TokenEvent) for event in events)
        assert isinstance(events[-1], FinishedEvent)
        await engine.close()

    asyncio.run(scenario())
