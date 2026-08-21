"""FastAPI application factory for the OpenAI-compatible Serve entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from x_tokens.config import XTokensConfig
from x_tokens.core import EngineCoreConfig, NaiveScheduler, Scheduler
from x_tokens.engine.clients.inproc import InprocClient, SchedulerFactory
from x_tokens.engine.input_processor import TokenizerInputProcessor
from x_tokens.engine.llm_engine import LLMEngine, LLMEngineProtocol
from x_tokens.engine.output_processor import TokenizerOutputProcessor
from x_tokens.executor.base import Executor
from x_tokens.executor.naive_hf_executor import (
    NaiveHFExecutor,
    NaiveHFExecutorConfig,
)

from .config import ServeConfig
from .errors import OpenAIError
from .generation import ChatCompletionService, GenerationService
from .models import ModelRegistry
from .openai.routes import ServeServices, router
from .renderer import PlainTextPromptRenderer

EngineFactory = Callable[[XTokensConfig], LLMEngineProtocol]
ExecutorFactory = Callable[[NaiveHFExecutorConfig], Executor]


def _default_executor_factory(config: NaiveHFExecutorConfig) -> NaiveHFExecutor:
    return NaiveHFExecutor(config)


def _default_scheduler_factory(config: EngineCoreConfig) -> Scheduler:
    return NaiveScheduler(
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
    )


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


def default_engine_factory(
    config: XTokensConfig,
    *,
    executor_factory: ExecutorFactory = _default_executor_factory,
    scheduler_factory: SchedulerFactory = _default_scheduler_factory,
) -> LLMEngineProtocol:
    """Build the single-process Hugging Face EngineCore path."""
    model_config = config.model_config
    executor_config = config.executor_config
    processor = TokenizerInputProcessor.from_config(
        model_config.model_name,
        local_files_only=executor_config.local_files_only,
    )
    hf_executor_config = NaiveHFExecutorConfig(
        model_config.model_name,
        device=executor_config.device,
        dtype=executor_config.dtype,
        local_files_only=executor_config.local_files_only,
        pad_token_id=processor.pad_token_id,
        eos_token_ids=processor.eos_token_ids,
    )
    executor = executor_factory(hf_executor_config)
    core_config = EngineCoreConfig(
        (model_config.served_model_name,),
        max_model_len=model_config.max_model_len,
        max_num_seqs=config.scheduler_config.max_num_seqs,
    )
    return LLMEngine(
        InprocClient(
            core_config,
            lambda: executor,
            scheduler_factory=scheduler_factory,
        ),
        processor,
        TokenizerOutputProcessor(processor.tokenizer),
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
    config: XTokensConfig | ServeConfig | None = None,
    engine_factory: EngineFactory = default_engine_factory,
) -> FastAPI:
    """Create an app with injected Engine dependencies and no module-global Engine."""
    if config is None:
        config = XTokensConfig()
    elif isinstance(config, ServeConfig):
        config = config.to_xtokens_config()
    server_config = config.server_config

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = engine_factory(config)
        models = ModelRegistry(config.model_config.served_model_name)
        completions = GenerationService(
            engine,
            models,
            shutdown_policy=server_config.shutdown_policy,
            shutdown_timeout_s=server_config.shutdown_timeout_s,
        )
        chat = ChatCompletionService(
            engine,
            models,
            PlainTextPromptRenderer(),
            shutdown_policy=server_config.shutdown_policy,
            shutdown_timeout_s=server_config.shutdown_timeout_s,
        )
        app.state.serve_services = ServeServices(
            server_config, engine, models, completions, chat
        )
        await completions.refresh_readiness()
        try:
            yield
        finally:
            await app.state.serve_services.close()

    app = FastAPI(title="xTokens Serve API", lifespan=lifespan)
    app.add_middleware(
        RequestBodyLimitMiddleware, max_body_size=server_config.max_request_body_size
    )
    if server_config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(server_config.cors_origins),
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
