"""Protocol-independent generation services for the Serve entrypoint."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

from x_tokens.engine.llm_engine import LLMEngineProtocol
from x_tokens.engine.types import (
    EngineEvent,
    ErrorEvent,
    FinishedEvent,
    GenerateRequest,
)
from x_tokens.logger import init_logger

from .errors import EngineUnavailableError
from .models import ModelRegistry
from .openai.protocol import ChatMessage
from .renderer import PromptRenderer

logger = init_logger(__name__)


class GenerationService:
    """Own request admission, event cleanup, and Engine interaction."""

    def __init__(
        self,
        engine: LLMEngineProtocol,
        models: ModelRegistry,
        *,
        shutdown_policy: str = "abort",
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        self._engine = engine
        self._models = models
        self._accepting = False
        self._active_request_ids: set[str] = set()
        self._shutdown_policy = shutdown_policy
        self._shutdown_timeout_s = shutdown_timeout_s

    async def refresh_readiness(self) -> bool:
        self._accepting = (await self._engine.health()).ready
        return self._accepting

    async def ensure_available(self, model: str) -> None:
        self._models.validate(model)
        if not await self.refresh_readiness():
            raise EngineUnavailableError()

    async def events(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        completed = False
        started = time.perf_counter()
        self._active_request_ids.add(request.request_id)
        logger.info(
            "request started: request_id=%s model=%s max_tokens=%d "
            "ignore_eos=%s active_requests=%d",
            request.request_id,
            request.model,
            request.sampling.max_tokens,
            request.sampling.ignore_eos,
            len(self._active_request_ids),
        )
        try:
            async for event in self._engine.generate(request):
                if isinstance(event, FinishedEvent):
                    completed = True
                    logger.info(
                        "request finished: request_id=%s finish_reason=%s "
                        "prompt_tokens=%d completion_tokens=%d duration_ms=%.2f",
                        request.request_id,
                        event.finish_reason.value,
                        event.prompt_tokens,
                        event.completion_tokens,
                        (time.perf_counter() - started) * 1000,
                    )
                elif isinstance(event, ErrorEvent):
                    completed = True
                    logger.warning(
                        "request failed: request_id=%s retryable=%s error=%s "
                        "duration_ms=%.2f",
                        request.request_id,
                        event.retryable,
                        event.message,
                        (time.perf_counter() - started) * 1000,
                    )
                yield event
        except Exception:
            logger.exception(
                "engine stream failed: request_id=%s duration_ms=%.2f",
                request.request_id,
                (time.perf_counter() - started) * 1000,
            )
            completed = True
            yield ErrorEvent(
                request.request_id,
                "The engine failed to generate a response",
            )
        finally:
            self._active_request_ids.discard(request.request_id)
            if not completed:
                await self._engine.abort(request.request_id)
                logger.info(
                    "request cancelled: request_id=%s duration_ms=%.2f "
                    "active_requests=%d",
                    request.request_id,
                    (time.perf_counter() - started) * 1000,
                    len(self._active_request_ids),
                )

    async def abort(self, request_id: str) -> None:
        logger.debug("request abort requested: request_id=%s", request_id)
        await self._engine.abort(request_id)

    async def stop(self) -> None:
        self._accepting = False
        if self._shutdown_policy == "drain":
            try:
                async with asyncio.timeout(self._shutdown_timeout_s):
                    while self._active_request_ids:
                        await asyncio.sleep(0.01)
            except TimeoutError:
                pass
        for request_id in tuple(self._active_request_ids):
            await self._engine.abort(request_id)


class ChatCompletionService(GenerationService):
    def __init__(
        self,
        engine: LLMEngineProtocol,
        models: ModelRegistry,
        renderer: PromptRenderer,
        *,
        shutdown_policy: str = "abort",
        shutdown_timeout_s: float = 30.0,
    ) -> None:
        super().__init__(
            engine,
            models,
            shutdown_policy=shutdown_policy,
            shutdown_timeout_s=shutdown_timeout_s,
        )
        self._renderer = renderer

    async def render(self, model: str, messages: list[ChatMessage]) -> str:
        rendered = await self._renderer.render_chat(model, messages)
        return rendered.text
