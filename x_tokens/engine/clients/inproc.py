"""Direct same-process implementation of ``EngineCoreClient``."""

from __future__ import annotations

from collections.abc import Callable

from x_tokens.core import EngineCore, EngineCoreConfig, NaiveScheduler, Scheduler
from x_tokens.engine.core_client import EngineCoreClient
from x_tokens.engine.types import CoreEvent, EngineHealth, GenerateRequest
from x_tokens.executor.base import Executor

ExecutorFactory = Callable[[], Executor]
SchedulerFactory = Callable[[EngineCoreConfig], Scheduler]


class InprocClient(EngineCoreClient):
    """An adapter that owns an EngineCore and invokes it directly.

    No queues, serialization, background threads, or transport state are used in
    this path. ``get_output`` advances scheduling and model execution once in
    the caller's thread.
    """

    def __init__(
        self,
        config: EngineCoreConfig,
        executor_factory: ExecutorFactory,
        scheduler_factory: SchedulerFactory | None = None,
    ) -> None:
        scheduler = (
            scheduler_factory(config)
            if scheduler_factory is not None
            else NaiveScheduler(
                max_num_seqs=config.max_num_seqs,
                max_model_len=config.max_model_len,
            )
        )
        self.engine_core = EngineCore(
            config, executor=executor_factory(), scheduler=scheduler
        )
        self._closed = False

    def add_request(self, request: GenerateRequest) -> None:
        self.engine_core.add_request(request)

    def get_output(self) -> tuple[CoreEvent, ...]:
        if self._closed:
            return ()
        outputs, model_executed = self.engine_core.step_fn()
        self.engine_core.post_step(model_executed=model_executed)
        return outputs.get(0)

    def abort_requests(self, request_ids: tuple[str, ...]) -> None:
        self.engine_core.abort_requests(request_ids)

    def health(self) -> EngineHealth:
        return EngineHealth(
            not self._closed,
            "Inproc EngineCore is ready"
            if not self._closed
            else "Inproc EngineCore is closed",
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.engine_core.close()
