from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import Mock

from fastapi.testclient import TestClient

from x_tokens.core.scheduler import SchedulerOutput
from x_tokens.engine.input_processor import InputProcessor
from x_tokens.engine.output_processor import OutputProcessor
from x_tokens.engine.types import (
    EngineEvent,
    EngineHealth,
    ErrorEvent,
    FinishedEvent,
    FinishReason,
    GenerateRequest,
    SamplingParams,
    TokenEvent,
)
from x_tokens.entrypoints.serve import ServeConfig, create_app
from x_tokens.entrypoints.serve import app as serve_app
from x_tokens.entrypoints.serve import generation as generation_module
from x_tokens.entrypoints.serve.generation import GenerationService
from x_tokens.entrypoints.serve.models import ModelRegistry
from x_tokens.executor.base import Executor, ModelForwardOutput


class FakeEngine:
    def __init__(self, *, ready: bool = True, error_after_token: bool = False) -> None:
        self.ready = ready
        self.error_after_token = error_after_token
        self.aborted: list[str] = []
        self.closed = False
        self.calls: list[GenerateRequest] = []

    def generate(self, request: GenerateRequest) -> AsyncIterator[EngineEvent]:
        self.calls.append(request)

        async def stream() -> AsyncIterator[EngineEvent]:
            yield TokenEvent(request.request_id, 1, "Hello")
            if self.error_after_token:
                yield ErrorEvent(request.request_id, "mock failure")
                return
            yield TokenEvent(request.request_id, 2, " world")
            yield FinishedEvent(request.request_id, FinishReason.STOP, 3, 2)

        return stream()

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    async def health(self) -> EngineHealth:
        return EngineHealth(self.ready, "fake")

    async def close(self) -> None:
        self.closed = True


class FakeExecutor(Executor):
    eos_token_ids = frozenset((9,))

    def execute_model(self, batch: SchedulerOutput) -> ModelForwardOutput:
        return ModelForwardOutput(None)

    def sample_tokens(
        self, output: ModelForwardOutput, batch: SchedulerOutput
    ) -> tuple[int, ...]:
        del output
        return tuple(9 for _ in batch.requests)


class FakeInputProcessor(InputProcessor):
    pad_token_id = 0
    eos_token_ids = frozenset((9,))

    def process(self, request: GenerateRequest) -> GenerateRequest:
        return GenerateRequest(
            request.request_id, request.model, (1,), request.sampling
        )

    @property
    def tokenizer(self) -> object:
        return object()


class FakeOutputProcessor(OutputProcessor):
    def process_token(self, request_id: str, token_id: int) -> str:
        del request_id, token_id
        return ""

    def finish(self, request_id: str) -> None:
        del request_id


def client_for(engine: FakeEngine, **config: object) -> TestClient:
    app = create_app(
        ServeConfig(served_model_name="test-model", **config), lambda _: engine
    )
    return TestClient(app)


def test_completion_uses_client_request_id_and_openai_error_format() -> None:
    engine = FakeEngine()
    with client_for(engine) as client:
        response = client.post(
            "/v1/completions",
            headers={"X-Request-ID": "benchmark-1"},
            json={
                "model": "test-model",
                "prompt": "hello",
                "ignore_eos": True,
            },
        )
        assert response.status_code == 200
        assert response.json()["id"] == "benchmark-1"
        assert response.headers["x-request-id"] == "benchmark-1"
        assert response.json()["usage"] == {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        assert engine.calls[0].sampling.ignore_eos is True

        invalid = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hello", "temperature": -1},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["type"] == "invalid_request_error"


def test_ready_gate_and_api_key() -> None:
    engine = FakeEngine(ready=False)
    with client_for(engine, api_key="secret") as client:
        assert client.get("/ready").status_code == 503
        unauthorized = client.get("/v1/models")
        assert unauthorized.status_code == 401
        response = client.post(
            "/v1/completions",
            headers={"Authorization": "Bearer secret"},
            json={"model": "test-model", "prompt": "hello"},
        )
        assert response.status_code == 503
        assert engine.calls == []
    assert engine.closed


def test_chat_stream_includes_usage_and_late_error() -> None:
    engine = FakeEngine(error_after_token=True)
    with client_for(engine) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert response.status_code == 200
        assert '"content":"Hello"' in response.text
        assert '"message":"mock failure"' in response.text
        assert response.text.endswith("data: [DONE]\n\n")
        assert engine.calls[0].sampling.ignore_eos is True


def test_unfinished_event_stream_aborts_request(monkeypatch) -> None:
    engine = FakeEngine()
    info = Mock()
    monkeypatch.setattr(generation_module.logger, "info", info)
    service = GenerationService(engine, ModelRegistry("test-model"))
    request = GenerateRequest("cancel-me", "test-model", "hello", SamplingParams(2))

    async def close_early() -> None:
        stream = service.events(request)
        await anext(stream)
        await stream.aclose()

    asyncio.run(close_early())
    assert engine.aborted == ["cancel-me"]
    assert any(
        call.args[0].startswith("request cancelled:") for call in info.call_args_list
    )


def test_finished_event_stream_logs_request_lifecycle(monkeypatch) -> None:
    engine = FakeEngine()
    info = Mock()
    monkeypatch.setattr(generation_module.logger, "info", info)
    service = GenerationService(engine, ModelRegistry("test-model"))
    request = GenerateRequest("logged", "test-model", "hello", SamplingParams(2))

    async def consume() -> None:
        async for _ in service.events(request):
            pass

    asyncio.run(consume())

    messages = [call.args[0] for call in info.call_args_list]
    assert any(message.startswith("request started:") for message in messages)
    assert any(message.startswith("request finished:") for message in messages)


def test_default_factory_uses_inproc_llm_engine(monkeypatch) -> None:
    monkeypatch.setattr(
        serve_app.TokenizerInputProcessor,
        "from_config",
        lambda *args, **kwargs: FakeInputProcessor(),
    )
    monkeypatch.setattr(serve_app, "NaiveHFExecutor", lambda _: FakeExecutor())
    monkeypatch.setattr(
        serve_app,
        "TokenizerOutputProcessor",
        lambda _: FakeOutputProcessor(),
    )
    app = create_app(ServeConfig(served_model_name="test-model"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hello"},
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "stop"


def test_default_factory_ignores_eos_until_max_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        serve_app.TokenizerInputProcessor,
        "from_config",
        lambda *args, **kwargs: FakeInputProcessor(),
    )
    monkeypatch.setattr(serve_app, "NaiveHFExecutor", lambda _: FakeExecutor())
    monkeypatch.setattr(
        serve_app,
        "TokenizerOutputProcessor",
        lambda _: FakeOutputProcessor(),
    )
    app = create_app(ServeConfig(served_model_name="test-model"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 2,
                "ignore_eos": True,
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert response.json()["usage"]["completion_tokens"] == 2
