from __future__ import annotations

import logging
from unittest.mock import Mock

from x_tokens.core import scheduler as scheduler_module
from x_tokens.core.scheduler import NaiveScheduler, RequestStatus
from x_tokens.engine.types import FinishReason, GenerateRequest, SamplingParams


def request(
    request_id: str,
    *,
    max_tokens: int = 2,
    ignore_eos: bool = False,
) -> GenerateRequest:
    return GenerateRequest(
        request_id,
        "test-model",
        "prompt",
        SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos),
    )


def test_scheduler_admits_requests_in_fifo_batches() -> None:
    scheduler = NaiveScheduler(max_num_seqs=2, max_model_len=8)
    for request_id in ("one", "two", "three"):
        scheduler.add_request(request(request_id), (1,))

    first_batch = scheduler.schedule()
    assert [item.request_id for item in first_batch.requests] == ["one", "two"]

    updates = scheduler.update_from_output(
        first_batch, (2, 3), eos_token_ids=frozenset()
    )
    assert all(update.finish_reason is None for update in updates)

    second_batch = scheduler.schedule()
    assert [item.request_id for item in second_batch.requests] == ["one", "two"]
    scheduler.update_from_output(second_batch, (4, 5), eos_token_ids=frozenset())

    third_batch = scheduler.schedule()
    assert [item.request_id for item in third_batch.requests] == ["three"]


def test_scheduler_logs_every_scheduled_batch(monkeypatch) -> None:
    debug = Mock()
    monkeypatch.setattr(
        scheduler_module.logger,
        "isEnabledFor",
        lambda level: level == logging.DEBUG,
    )
    monkeypatch.setattr(scheduler_module.logger, "debug", debug)
    scheduler = NaiveScheduler(max_num_seqs=1, max_model_len=8)
    scheduler.add_request(request("one"), (1,))

    scheduler.schedule()
    scheduler.schedule()

    schedule_calls = [
        call
        for call in debug.call_args_list
        if call.args[0].startswith("scheduler step:")
    ]
    assert len(schedule_calls) == 2
    assert all(call.args[1] == 1 for call in schedule_calls)


def test_scheduler_finishes_on_eos_and_length() -> None:
    scheduler = NaiveScheduler(max_num_seqs=2, max_model_len=3)
    scheduler.add_request(request("eos", max_tokens=3), (1,))
    scheduler.add_request(request("length", max_tokens=1), (1,))
    batch = scheduler.schedule()

    updates = scheduler.update_from_output(batch, (9, 2), eos_token_ids=frozenset((9,)))

    assert updates[0].finish_reason is FinishReason.STOP
    assert updates[1].finish_reason is FinishReason.LENGTH
    assert not scheduler.has_work


def test_scheduler_applies_ignore_eos_per_request_until_max_tokens() -> None:
    scheduler = NaiveScheduler(max_num_seqs=2, max_model_len=8)
    scheduler.add_request(request("stop", max_tokens=2), (1,))
    scheduler.add_request(
        request("ignore", max_tokens=2, ignore_eos=True),
        (1,),
    )

    first = scheduler.update_from_output(
        scheduler.schedule(),
        (9, 9),
        eos_token_ids=frozenset((9,)),
    )
    second = scheduler.update_from_output(
        scheduler.schedule(),
        (9,),
        eos_token_ids=frozenset((9,)),
    )

    assert first[0].finish_reason is FinishReason.STOP
    assert first[1].finish_reason is None
    assert second[0].finish_reason is FinishReason.LENGTH
    assert second[0].request.output_token_ids == [9, 9]
    assert not scheduler.has_work


def test_scheduler_aborts_waiting_and_running_requests() -> None:
    scheduler = NaiveScheduler(max_num_seqs=1, max_model_len=8)
    scheduler.add_request(request("running"), (1,))
    scheduler.add_request(request("waiting"), (1,))
    running = scheduler.schedule().requests[0]

    assert scheduler.abort("waiting")
    assert scheduler.abort("running")
    assert running.status is RequestStatus.ABORTED
    assert not scheduler.has_work
    assert not scheduler.abort("unknown")
