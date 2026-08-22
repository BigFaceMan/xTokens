from __future__ import annotations

from x_tokens.core import EngineCore, EngineCoreConfig
from x_tokens.core.scheduler import NaiveScheduler, SchedulerOutput
from x_tokens.engine.types import (
    CoreErrorEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    FinishReason,
    GenerateRequest,
    SamplingParams,
)
from x_tokens.executor.base import Executor, ModelForwardOutput


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))

    def __init__(self, token_ids: list[int]) -> None:
        self._token_ids = iter(token_ids)
        self.batches: list[list[str]] = []

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        self.batches.append([request.request_id for request in batch.requests])
        return ModelForwardOutput(None)

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        del output
        return tuple(next(self._token_ids) for _ in batch.requests)


def request(
    request_id: str,
    *,
    model: str = "test-model",
    ignore_eos: bool = False,
) -> GenerateRequest:
    return GenerateRequest(
        request_id,
        model,
        (1,),
        SamplingParams(max_tokens=2, ignore_eos=ignore_eos),
    )


def test_core_steps_batches_and_returns_events_directly() -> None:
    executor = FakeExecutor([2, 3, 9, 4])
    core = EngineCore(
        EngineCoreConfig(("test-model",), max_num_seqs=2), executor=executor
    )
    for request_id in ("one", "two"):
        core.add_request(request(request_id))

    first, executed = core.step_fn()
    core.post_step(model_executed=executed)
    second, executed = core.step_fn()
    core.post_step(model_executed=executed)

    assert executor.batches == [["one", "two"], ["one", "two"]]
    assert first.get(0) == (
        CoreTokenEvent("one", 2),
        CoreTokenEvent("two", 3),
    )
    assert second.get(0) == (
        CoreFinishedEvent("one", FinishReason.STOP, 1, 2),
        CoreTokenEvent("two", 4),
        CoreFinishedEvent("two", FinishReason.LENGTH, 1, 2),
    )


def test_core_emits_ignored_eos_until_request_length() -> None:
    core = EngineCore(
        EngineCoreConfig(("test-model",)),
        executor=FakeExecutor([9, 9]),
    )
    core.add_request(request("eval", ignore_eos=True))

    first, _ = core.step_fn()
    second, _ = core.step_fn()

    assert first.get(0) == (CoreTokenEvent("eval", 9),)
    assert second.get(0) == (
        CoreTokenEvent("eval", 9),
        CoreFinishedEvent("eval", FinishReason.LENGTH, 1, 2),
    )


def test_core_isolates_request_validation_errors_without_a_queue() -> None:
    core = EngineCore(EngineCoreConfig(("test-model",)), executor=FakeExecutor([9]))
    for item in (request("bad", model="other-model"), request("good")):
        core.add_request(item)

    outputs, _ = core.step_fn()

    assert isinstance(outputs.get(0)[0], CoreErrorEvent)
    assert outputs.get(0)[0].request_id == "bad"
    assert outputs.get(0)[1] == CoreFinishedEvent("good", FinishReason.STOP, 1, 1)


def test_core_rejects_text_prompts_at_the_backend_boundary() -> None:
    core = EngineCore(EngineCoreConfig(("test-model",)), executor=FakeExecutor([9]))
    text_request = GenerateRequest("text", "test-model", "hello", SamplingParams())

    core.add_request(text_request)

    outputs = core.step_fn()[0].get(0)
    assert outputs
    assert isinstance(outputs[0], CoreErrorEvent)
    assert "preprocessed prompt token IDs" in outputs[0].message


def test_core_accepts_an_injected_scheduler() -> None:
    scheduler = NaiveScheduler(max_num_seqs=1, max_model_len=8)
    executor = FakeExecutor([2, 3])
    core = EngineCore(
        EngineCoreConfig(("test-model",), max_num_seqs=2),
        executor=executor,
        scheduler=scheduler,
    )
    core.add_request(request("one"))
    core.add_request(request("two"))

    outputs, executed = core.step_fn()

    assert executed
    assert executor.batches == [["one"]]
    assert outputs.get(0) == (CoreTokenEvent("one", 2),)
