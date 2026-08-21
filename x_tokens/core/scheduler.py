"""FIFO, no-KV-cache scheduler used by the naive HF EngineCore."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import StrEnum

from ..engine.types import FinishReason, GenerateRequest


class RequestStatus(StrEnum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass(slots=True)
class ScheduledRequest:
    request: GenerateRequest
    prompt_token_ids: tuple[int, ...]
    output_token_ids: list[int] = field(default_factory=list)
    decoded_text: str = ""
    status: RequestStatus = RequestStatus.WAITING

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def context_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.output_token_ids)

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def completion_tokens(self) -> int:
        return len(self.output_token_ids)


@dataclass(frozen=True, slots=True)
class SchedulingBatch:
    requests: tuple[ScheduledRequest, ...]


@dataclass(frozen=True, slots=True)
class SchedulerUpdate:
    request: ScheduledRequest
    token_id: int
    finish_reason: FinishReason | None


class NaiveScheduler:
    """Admit requests in FIFO order and execute every running request per step."""

    def __init__(self, *, max_num_seqs: int, max_model_len: int) -> None:
        self._max_num_seqs = max_num_seqs
        self._max_model_len = max_model_len
        self.waiting: deque[ScheduledRequest] = deque()
        self.running: OrderedDict[str, ScheduledRequest] = OrderedDict()
        self._request_ids: set[str] = set()

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def add_request(
        self, request: GenerateRequest, prompt_token_ids: tuple[int, ...]
    ) -> None:
        if request.request_id in self._request_ids:
            raise ValueError("duplicate request ID")
        if not prompt_token_ids:
            raise ValueError("prompt must contain at least one token")
        if len(prompt_token_ids) >= self._max_model_len:
            raise ValueError("prompt exceeds max_model_len")
        self.waiting.append(ScheduledRequest(request, prompt_token_ids))
        self._request_ids.add(request.request_id)

    def abort(self, request_id: str) -> bool:
        request = self.running.pop(request_id, None)
        if request is not None:
            request.status = RequestStatus.ABORTED
            self._request_ids.discard(request_id)
            return True
        for request in tuple(self.waiting):
            if request.request_id == request_id:
                self.waiting.remove(request)
                request.status = RequestStatus.ABORTED
                self._request_ids.discard(request_id)
                return True
        return False

    def schedule(self) -> SchedulingBatch:
        while self.waiting and len(self.running) < self._max_num_seqs:
            request = self.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.running[request.request_id] = request
        return SchedulingBatch(tuple(self.running.values()))

    def update_from_output(
        self,
        batch: SchedulingBatch,
        token_ids: tuple[int, ...],
        *,
        eos_token_ids: frozenset[int],
    ) -> tuple[SchedulerUpdate, ...]:
        if len(batch.requests) != len(token_ids):
            raise ValueError("batch size and token output size differ")
        updates: list[SchedulerUpdate] = []
        for request, token_id in zip(batch.requests, token_ids, strict=True):
            if self.running.get(request.request_id) is not request:
                continue
            request.output_token_ids.append(token_id)
            finish_reason: FinishReason | None = None
            if token_id in eos_token_ids:
                finish_reason = FinishReason.STOP
            elif (
                request.completion_tokens >= request.request.sampling.max_tokens
                or len(request.context_token_ids) >= self._max_model_len
            ):
                finish_reason = FinishReason.LENGTH
            if finish_reason is not None:
                request.status = RequestStatus.FINISHED
                self.running.pop(request.request_id)
                self._request_ids.discard(request.request_id)
            updates.append(SchedulerUpdate(request, token_id, finish_reason))
        return tuple(updates)

    def fail_batch(self, batch: SchedulingBatch) -> tuple[ScheduledRequest, ...]:
        failed: list[ScheduledRequest] = []
        for request in batch.requests:
            if self.running.get(request.request_id) is request:
                request.status = RequestStatus.FINISHED
                self.running.pop(request.request_id)
                self._request_ids.discard(request.request_id)
                failed.append(request)
        return tuple(failed)

    def abort_all(self) -> tuple[ScheduledRequest, ...]:
        requests = tuple(self.waiting) + tuple(self.running.values())
        self.waiting.clear()
        self.running.clear()
        self._request_ids.clear()
        for request in requests:
            request.status = RequestStatus.ABORTED
        return requests
