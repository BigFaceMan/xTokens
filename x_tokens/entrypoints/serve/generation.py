"""Protocol-independent generation services for the Serve entrypoint."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from x_tokens.engine.client import EngineClientProtocol
from x_tokens.engine.types import (
    EngineEvent,
    ErrorEvent,
    FinishedEvent,
    GenerateRequest,
)

from .errors import EngineUnavailableError
from .models import ModelRegistry
from .openai.protocol import ChatMessage
from .renderer import PromptRenderer

logger = logging.getLogger(__name__)


class GenerationService:
    """Own request admission, event cleanup, and Engine interaction."""

    def __init__(
        self,
        engine: EngineClientProtocol,
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
        self._active_request_ids.add(request.request_id)
        try:
            async for event in self._engine.generate(request):
                yield event
                if isinstance(event, (FinishedEvent, ErrorEvent)):
                    completed = True
        except Exception:
            logger.exception("engine stream failed for request %s", request.request_id)
            completed = True
            yield ErrorEvent(
                request.request_id,
                "The engine failed to generate a response",
            )
        finally:
            self._active_request_ids.discard(request.request_id)
            if not completed:
                await self._engine.abort(request.request_id)

    async def abort(self, request_id: str) -> None:
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
        engine: EngineClientProtocol,
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
