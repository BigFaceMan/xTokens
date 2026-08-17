"""Async HTTP adapters for common OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from .models import RequestFuncInput, RequestFuncOutput  # pyright: ignore[reportMissingImports]


class EndpointError(RuntimeError):
    """Raised when an endpoint response cannot be interpreted."""


async def _sse_events(response: aiohttp.ClientResponse) -> AsyncIterator[dict[str, Any]]:
    """Yield JSON SSE payloads, handling arbitrary TCP chunk boundaries."""
    buffer = ""
    async for raw in response.content.iter_chunked(8192):
        buffer += raw.decode("utf-8", errors="replace")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            for line in block.splitlines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    value = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    yield value
    if buffer.strip().startswith("data:"):
        data = buffer.strip()[5:].strip()
        if data != "[DONE]":
            try:
                value = json.loads(data)
            except json.JSONDecodeError:
                return
            if isinstance(value, dict):
                yield value


def _headers(request: RequestFuncInput, content_type: str = "application/json") -> dict[str, str]:
    headers = {"Content-Type": content_type}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if request.extra_headers:
        headers.update(request.extra_headers)
    if request.request_id:
        headers["x-request-id"] = request.request_id
    return headers


def _base_payload(request: RequestFuncInput) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": request.model_name or request.model,
        "max_tokens": request.output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    optional = {
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "logprobs": request.logprobs,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    if request.extra_body:
        payload.update(request.extra_body)
    return payload


def _chat_messages(request: RequestFuncInput) -> list[dict[str, Any]]:
    if request.chat_messages is not None:
        return request.chat_messages
    content: list[dict[str, Any]] = []
    if isinstance(request.prompt, str):
        content.append({"type": "text", "text": request.prompt})
    else:
        content.append({"type": "text", "text": str(request.prompt)})
    if request.multi_modal_content:
        items = request.multi_modal_content
        content.extend(items if isinstance(items, list) else [items])
    return [{"role": "user", "content": content}]


async def _request_generation(
    request: RequestFuncInput,
    session: aiohttp.ClientSession,
    *,
    chat: bool,
) -> RequestFuncOutput:
    output = RequestFuncOutput(prompt_len=request.prompt_len, start_time=time.perf_counter())
    payload = _base_payload(request)
    if chat:
        payload["messages"] = _chat_messages(request)
        payload.pop("max_tokens", None)
        payload["max_completion_tokens"] = request.output_len
    else:
        payload["prompt"] = request.prompt
    generated: list[str] = []
    last_token_time = output.start_time
    try:
        async with session.post(request.api_url, json=payload, headers=_headers(request)) as response:
            output.status_code = response.status
            if response.status >= 400:
                output.error = await response.text()
                return output
            first = True
            async for data in _sse_events(response):
                choices = data.get("choices")
                if choices:
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    delta = choice.get("delta", {}) if chat else choice
                    text = delta.get("content" if chat else "text", "") or ""
                    if text:
                        now = time.perf_counter()
                        if first:
                            output.ttft = now - output.start_time
                            first = False
                        else:
                            output.itl.append(now - last_token_time)
                        last_token_time = now
                        generated.append(str(text))
                usage = data.get("usage")
                if isinstance(usage, dict):
                    output.output_tokens = int(
                        usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                    )
                    output.prompt_len = int(
                        usage.get("prompt_tokens", usage.get("input_tokens", output.prompt_len))
                        or output.prompt_len
                    )
            output.generated_text = "".join(generated)
            output.latency = max(0.0, last_token_time - output.start_time)
            output.success = not first
            if first:
                output.error = "stream ended without a generation token"
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        output.error = str(exc)
    except Exception:
        output.error = traceback.format_exc()
    return output


async def _request_pooling(
    request: RequestFuncInput,
    session: aiohttp.ClientSession,
    payload: dict[str, Any],
) -> RequestFuncOutput:
    output = RequestFuncOutput(prompt_len=request.prompt_len, start_time=time.perf_counter())
    try:
        async with session.post(request.api_url, json=payload, headers=_headers(request)) as response:
            output.status_code = response.status
            if response.status >= 400:
                output.error = await response.text()
                return output
            data = await response.json(content_type=None)
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            if isinstance(usage, dict):
                output.prompt_len = int(usage.get("prompt_tokens", request.prompt_len) or 0)
            output.latency = time.perf_counter() - output.start_time
            output.ttft = output.latency
            output.success = True
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError) as exc:
        output.error = str(exc)
    except Exception:
        output.error = traceback.format_exc()
    return output


async def request_openai_completions(request: RequestFuncInput, session: aiohttp.ClientSession) -> RequestFuncOutput:
    return await _request_generation(request, session, chat=False)


async def request_openai_chat(request: RequestFuncInput, session: aiohttp.ClientSession) -> RequestFuncOutput:
    return await _request_generation(request, session, chat=True)


async def request_embeddings(request: RequestFuncInput, session: aiohttp.ClientSession) -> RequestFuncOutput:
    payload = {"model": request.model_name or request.model, "input": request.prompt}
    if request.extra_body:
        payload.update(request.extra_body)
    return await _request_pooling(request, session, payload)


async def request_rerank(request: RequestFuncInput, session: aiohttp.ClientSession) -> RequestFuncOutput:
    if not isinstance(request.prompt, list) or len(request.prompt) < 2:
        raise EndpointError("rerank requests require a list containing query and documents")
    payload = {
        "model": request.model_name or request.model,
        "query": request.prompt[0],
        "documents": request.prompt[1:],
    }
    if request.extra_body:
        payload.update(request.extra_body)
    return await _request_pooling(request, session, payload)


BACKENDS = {
    "openai": ("/v1/completions", request_openai_completions),
    "vllm": ("/v1/completions", request_openai_completions),
    "openai-chat": ("/v1/chat/completions", request_openai_chat),
    "chat": ("/v1/chat/completions", request_openai_chat),
    "openai-embeddings": ("/v1/embeddings", request_embeddings),
    "embeddings": ("/v1/embeddings", request_embeddings),
    "vllm-rerank": ("/v1/rerank", request_rerank),
    "rerank": ("/v1/rerank", request_rerank),
}
