"""FastAPI application factory for the OpenAI-compatible Serve entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from x_tokens.engine.client import EngineClient, EngineClientProtocol
from x_tokens.engine.clients.local import LocalEngineCoreClient

from .config import ServeConfig
from .errors import OpenAIError
from .generation import ChatCompletionService, GenerationService
from .models import ModelRegistry
from .openai.routes import ServeServices, router
from .renderer import PlainTextPromptRenderer

EngineFactory = Callable[[ServeConfig], EngineClientProtocol]


class RequestBodyLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, max_body_size: int) -> None:
        super().__init__(app)
        self._max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        content_length = request.headers.get("content-length")
        if content_length is not None and int(content_length) > self._max_body_size:
            return JSONResponse(
                OpenAIError(
                    413,
                    "Request body is too large",
                    "invalid_request_error",
                    code="request_too_large",
                ).body(),
                status_code=413,
            )
        return await call_next(request)


def default_engine_factory(config: ServeConfig) -> EngineClientProtocol:
    """Use a GPU-free Core transport mock until EngineCore is implemented."""
    return EngineClient(
        LocalEngineCoreClient(output_queue_size=config.output_queue_size)
    )


def _validation_error(exc: RequestValidationError) -> OpenAIError:
    first = exc.errors()[0] if exc.errors() else {}
    location = first.get("loc", ())
    param = str(location[-1]) if location else None
    return OpenAIError(
        422,
        "Invalid request parameters",
        "invalid_request_error",
        param,
        "validation_error",
    )


def create_app(
    config: ServeConfig | None = None,
    engine_factory: EngineFactory = default_engine_factory,
) -> FastAPI:
    """Create an app with injected Engine dependencies and no module-global Engine."""
    config = config or ServeConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = engine_factory(config)
        models = ModelRegistry(config.served_model_name)
        completions = GenerationService(
            engine,
            models,
            shutdown_policy=config.shutdown_policy,
            shutdown_timeout_s=config.shutdown_timeout_s,
        )
        chat = ChatCompletionService(
            engine,
            models,
            PlainTextPromptRenderer(),
            shutdown_policy=config.shutdown_policy,
            shutdown_timeout_s=config.shutdown_timeout_s,
        )
        app.state.serve_services = ServeServices(
            config, engine, models, completions, chat
        )
        await completions.refresh_readiness()
        try:
            yield
        finally:
            await app.state.serve_services.close()

    app = FastAPI(title="xTokens Serve API", lifespan=lifespan)
    app.add_middleware(
        RequestBodyLimitMiddleware, max_body_size=config.max_request_body_size
    )
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )

    @app.exception_handler(OpenAIError)
    async def openai_error(_: Request, exc: OpenAIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.body())

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        error = _validation_error(exc)
        return JSONResponse(status_code=error.status_code, content=error.body())

    app.include_router(router)
    return app
