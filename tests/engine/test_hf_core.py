from __future__ import annotations

from x_tokens.core import EngineCore, EngineCoreConfig
from x_tokens.core.scheduler import ScheduledRequest, SchedulingBatch
from x_tokens.engine.types import (
    CoreErrorEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    FinishReason,
    GenerateRequest,
    SamplingParams,
)
from x_tokens.executor.base import Executor


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))

    def __init__(self, token_ids: list[int]) -> None:
        self._token_ids = iter(token_ids)
        self.batches: list[list[str]] = []

    def encode(self, prompt: str | tuple[int, ...]) -> tuple[int, ...]:
        return prompt if isinstance(prompt, tuple) else (1,)

    def execute(self, batch: SchedulingBatch) -> tuple[int, ...]:
        self.batches.append([request.request_id for request in batch.requests])
        return tuple(next(self._token_ids) for _ in batch.requests)

    def decode_delta(self, request: ScheduledRequest) -> str:
        return f"<{request.output_token_ids[-1]}>"


def request(request_id: str, *, model: str = "test-model") -> GenerateRequest:
    return GenerateRequest(request_id, model, "hello", SamplingParams(max_tokens=2))


def test_core_steps_batches_and_returns_events_directly() -> None:
    executor = FakeExecutor([2, 3, 9, 4])
    core = EngineCore(
        EngineCoreConfig(("test-model",), max_num_seqs=2), executor=executor
    )
    for request_id in ("one", "two"):
        req, wave = core.preprocess_add_request(request(request_id))
        core.add_request(req, wave)

    first, executed = core.step_fn()
    core.post_step(model_executed=executed)
    second, executed = core.step_fn()
    core.post_step(model_executed=executed)

    assert executor.batches == [["one", "two"], ["one", "two"]]
    assert first.get(0) == (
        CoreTokenEvent("one", 2, "<2>"),
        CoreTokenEvent("two", 3, "<3>"),
    )
    assert second.get(0) == (
        CoreFinishedEvent("one", FinishReason.STOP, 1, 2),
        CoreTokenEvent("two", 4, "<4>"),
        CoreFinishedEvent("two", FinishReason.LENGTH, 1, 2),
    )


def test_core_isolates_request_validation_errors_without_a_queue() -> None:
    core = EngineCore(EngineCoreConfig(("test-model",)), executor=FakeExecutor([9]))
    for item in (request("bad", model="other-model"), request("good")):
        req, wave = core.preprocess_add_request(item)
        core.add_request(req, wave)

    outputs, _ = core.step_fn()

    assert isinstance(outputs.get(0)[0], CoreErrorEvent)
    assert outputs.get(0)[0].request_id == "bad"
    assert outputs.get(0)[1] == CoreFinishedEvent("good", FinishReason.STOP, 1, 1)
