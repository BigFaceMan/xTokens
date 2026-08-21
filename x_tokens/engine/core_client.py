"""Synchronous client contract used by ``LLMEngine`` to drive an EngineCore."""

from __future__ import annotations

from typing import Protocol

from .types import CoreEvent, EngineHealth, GenerateRequest


class EngineCoreClient(Protocol):
    """Drive an EngineCore from the caller's thread."""

    def add_request(self, request: GenerateRequest) -> None: ...

    def get_output(self) -> tuple[CoreEvent, ...]: ...

    def abort_requests(self, request_ids: tuple[str, ...]) -> None: ...

    def health(self) -> EngineHealth: ...

    def close(self) -> None: ...
