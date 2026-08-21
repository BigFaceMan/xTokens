from __future__ import annotations

import asyncio

from x_tokens.core import EngineCore, EngineCoreConfig
from x_tokens.core.scheduler import NaiveScheduler, SchedulerOutput
from x_tokens.engine.clients.inproc import InprocClient
from x_tokens.engine.input_processor import InputProcessor
from x_tokens.engine.llm_engine import LLMEngine
from x_tokens.engine.output_processor import OutputProcessor
from x_tokens.engine.types import (
    CoreFinishedEvent,
    FinishedEvent,
    GenerateRequest,
    SamplingParams,
    TokenEvent,
)
from x_tokens.executor.base import Executor, ModelForwardOutput


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        return ModelForwardOutput(None)

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        del output
        return tuple(9 for _ in batch.requests)


class FakeInputProcessor(InputProcessor):
    def process(self, request: GenerateRequest) -> GenerateRequest:
        return GenerateRequest(
            request.request_id, request.model, (1,), request.sampling
        )


class FakeOutputProcessor(OutputProcessor):
    def process_token(self, request_id: str, token_id: int) -> str:
        del request_id, token_id
        return ""

    def finish(self, request_id: str) -> None:
        del request_id


def test_inproc_client_owns_core_and_steps_in_the_calling_thread() -> None:
    client = InprocClient(EngineCoreConfig(("test-model",)), FakeExecutor)
    assert isinstance(client, InprocClient)
    assert isinstance(client.engine_core, EngineCore)
    client.add_request(
        GenerateRequest("inproc-request", "test-model", (1,), SamplingParams(2))
    )

    outputs = client.get_output()

    assert isinstance(outputs[-1], CoreFinishedEvent)
    assert outputs[-1].request_id == "inproc-request"
    client.close()
    assert not client.health().ready


def test_inproc_client_accepts_scheduler_factory() -> None:
    created: list[NaiveScheduler] = []

    def scheduler_factory(config: EngineCoreConfig) -> NaiveScheduler:
        scheduler = NaiveScheduler(max_num_seqs=1, max_model_len=config.max_model_len)
        created.append(scheduler)
        return scheduler

    client = InprocClient(
        EngineCoreConfig(("test-model",), max_num_seqs=2),
        FakeExecutor,
        scheduler_factory=scheduler_factory,
    )

    assert len(created) == 1
    assert client.engine_core._scheduler is created[0]
    client.close()


def test_llm_engine_normalizes_direct_inproc_outputs() -> None:
    async def scenario() -> None:
        engine = LLMEngine(
            InprocClient(EngineCoreConfig(("test-model",)), FakeExecutor),
            FakeInputProcessor(),
            FakeOutputProcessor(),
        )
        request = GenerateRequest("request", "test-model", "hello", SamplingParams(2))
        events = [event async for event in engine.generate(request)]
        assert not any(isinstance(event, TokenEvent) for event in events)
        assert isinstance(events[-1], FinishedEvent)
        await engine.close()

    asyncio.run(scenario())
