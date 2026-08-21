"""Backend-independent, single-threaded EngineCore."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..engine.types import (
    CoreErrorEvent,
    CoreEvent,
    CoreFinishedEvent,
    CoreTokenEvent,
    FinishReason,
    GenerateRequest,
)
from ..executor.base import Executor
from .config import EngineCoreConfig
from .scheduler import NaiveScheduler, ScheduledRequest


@dataclass(slots=True)
class RequestWave:
    """Preprocessed request state accepted by ``EngineCore.add_request``."""

    prompt_token_ids: tuple[int, ...] | None = None
    error: str | None = None


@dataclass(slots=True)
class EngineCoreOutputs:
    """Outputs produced by one Core step, grouped by output channel."""

    outputs: dict[int, list[CoreEvent]] = field(default_factory=dict)

    def add(self, event: CoreEvent, channel: int = 0) -> None:
        self.outputs.setdefault(channel, []).append(event)

    def get(
        self, channel: int, default: tuple[CoreEvent, ...] = ()
    ) -> tuple[CoreEvent, ...]:
        events = self.outputs.get(channel)
        return tuple(events) if events is not None else default


class EngineCore:
    """Owns scheduler and executor state for direct, synchronous execution."""

    def __init__(
        self,
        config: EngineCoreConfig,
        *,
        executor: Executor,
    ) -> None:
        self._config = config
        self._executor = executor
        self._scheduler = NaiveScheduler(
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
        )
        self._pending_outputs = EngineCoreOutputs()
        self._closed = False

    def preprocess_add_request(
        self, request: GenerateRequest
    ) -> tuple[GenerateRequest, RequestWave]:
        """Validate and tokenize a request before adding it to the scheduler."""
        try:
            self._validate_request(request)
            prompt_token_ids = self._executor.encode(request.prompt)
            return request, RequestWave(prompt_token_ids=prompt_token_ids)
        except Exception as error:  # noqa: BLE001 - request errors are isolated.
            return request, RequestWave(error=str(error))

    def add_request(self, request: GenerateRequest, request_wave: RequestWave) -> None:
        if self._closed:
            self._pending_outputs.add(
                CoreErrorEvent(request.request_id, "EngineCore is closed")
            )
            return
        if request_wave.error is not None:
            self._pending_outputs.add(
                CoreErrorEvent(request.request_id, request_wave.error)
            )
            return
        try:
            self._scheduler.add_request(request, request_wave.prompt_token_ids or ())
        except Exception as error:  # noqa: BLE001 - duplicate requests are isolated.
            self._pending_outputs.add(CoreErrorEvent(request.request_id, str(error)))

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        for request_id in request_ids:
            self._scheduler.abort(request_id)

    def step_fn(self) -> tuple[EngineCoreOutputs, bool]:
        """Run at most one scheduled model step in the caller's thread."""
        outputs = self._take_pending_outputs()
        if self._closed:
            return outputs, False
        batch = self._scheduler.schedule()
        if not batch.requests:
            return outputs, False
        try:
            token_ids = self._executor.execute(batch)
        except Exception as error:  # noqa: BLE001 - forward failure affects this batch.
            for request in self._scheduler.fail_batch(batch):
                outputs.add(CoreErrorEvent(request.request_id, str(error)))
            return outputs, True
        updates = self._scheduler.update_from_output(
            batch, token_ids, eos_token_ids=self._executor.eos_token_ids
        )
        for update in updates:
            request = update.request
            if update.token_id not in self._executor.eos_token_ids:
                outputs.add(
                    CoreTokenEvent(
                        request.request_id,
                        update.token_id,
                        self._executor.decode_delta(request),
                    )
                )
            if update.finish_reason is not None:
                outputs.add(self._finished_event(request, update.finish_reason))
        return outputs, True

    def post_step(self, *, model_executed: bool) -> None:
        """Lifecycle hook reserved for post-forward maintenance."""
        del model_executed

    def close(self) -> None:
        self._closed = True
        self._scheduler.abort_all()

    def _validate_request(self, request: GenerateRequest) -> None:
        if not self._config.accepts_model(request.model):
            raise ValueError("requested model is not loaded by this EngineCore")
        sampling = request.sampling
        if sampling.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if sampling.temperature < 0:
            raise ValueError("temperature must not be negative")
        if not 0 < sampling.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if sampling.top_k is not None and sampling.top_k < 1:
            raise ValueError("top_k must be positive")
        if sampling.stop:
            raise ValueError("stop strings are not supported by the HF EngineCore")

    def _finished_event(
        self, request: ScheduledRequest, finish_reason: FinishReason
    ) -> CoreFinishedEvent:
        return CoreFinishedEvent(
            request.request_id,
            finish_reason,
            request.prompt_tokens,
            request.completion_tokens,
        )

    def _take_pending_outputs(self) -> EngineCoreOutputs:
        outputs = self._pending_outputs
        self._pending_outputs = EngineCoreOutputs()
        return outputs
