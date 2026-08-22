"""Traffic generation and orchestration for the standalone benchmark client."""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
from collections.abc import AsyncIterator
from typing import Any

import aiohttp

from .lib.endpoint import BACKENDS  # pyright: ignore[reportMissingImports]
from .lib.metrics import calculate_metrics  # pyright: ignore[reportMissingImports]
from .lib.models import (  # pyright: ignore[reportMissingImports]
    RequestFuncInput,
    RequestFuncOutput,
    SampleRequest,
)

logger = logging.getLogger(__name__)


async def get_request(
    samples: list[SampleRequest],
    request_rate: float,
    *,
    burstiness: float = 1.0,
    self_timed: bool = False,
) -> AsyncIterator[tuple[SampleRequest, float]]:
    """Yield samples according to Poisson/gamma, fixed, or trace timing."""
    if not samples:
        raise ValueError("at least one sample is required")
    if request_rate <= 0 and request_rate != math.inf:
        raise ValueError("request-rate must be positive")
    if burstiness <= 0:
        raise ValueError("burstiness must be positive")

    target_times: list[float] = []
    elapsed = 0.0
    for sample in samples:
        if self_timed and sample.timestamp is not None:
            elapsed = max(0.0, sample.timestamp)
        elif request_rate == math.inf:
            elapsed = 0.0
        else:
            if burstiness == math.inf:
                delay = 1.0 / request_rate
            else:
                # random.gammavariate(shape, scale) has mean 1/rate.
                delay = random.gammavariate(
                    burstiness, 1.0 / (request_rate * burstiness)
                )
            elapsed += delay
        target_times.append(elapsed)

    started = time.perf_counter()
    for sample, target in zip(samples, target_times, strict=True):
        remaining = started + target - time.perf_counter()
        if remaining > 0:
            await asyncio.sleep(remaining)
        yield sample, request_rate


def _service_base(base_url: str) -> str:
    """Normalize a host, ``/v1`` URL, or a full backend endpoint to a host."""
    normalized = base_url.rstrip("/")
    for suffix in (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/rerank",
    ):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized.removesuffix("/v1")


async def get_first_model(base_url: str, session: aiohttp.ClientSession) -> str:
    """Return the first model advertised by a server or its PD proxy."""
    service_base = _service_base(base_url)
    models_url = service_base + "/v1/models"
    async with session.get(models_url) as response:
        if response.status == 200:
            data = await response.json(content_type=None)
            models = data.get("data", []) if isinstance(data, dict) else []
            if models and isinstance(models[0], dict) and models[0].get("id"):
                return str(models[0]["id"])
        elif response.status not in {404, 405}:
            response.raise_for_status()

    # The vLLM demo PD proxy exposes /status but not /v1/models. Its status
    # contains the backend addresses, so discover the model from a backend.
    status_url = service_base + "/status"
    async with session.get(status_url) as response:
        if response.status != 200:
            response.raise_for_status()
        status = await response.json(content_type=None)
    nodes = status.get("prefill_nodes", []) if isinstance(status, dict) else []
    nodes = nodes or (
        status.get("decode_nodes", []) if isinstance(status, dict) else []
    )
    for node in nodes:
        node_url = str(node)
        if not node_url.startswith(("http://", "https://")):
            node_url = "http://" + node_url
        async with session.get(node_url.rstrip("/") + "/v1/models") as response:
            if response.status != 200:
                continue
            data = await response.json(content_type=None)
            models = data.get("data", []) if isinstance(data, dict) else []
            if models and isinstance(models[0], dict) and models[0].get("id"):
                return str(models[0]["id"])
    raise RuntimeError(
        f"server returned no models from {models_url} or proxy backends listed by {status_url}"
    )


async def wait_for_endpoint(
    request_func: Any,
    test_input: RequestFuncInput,
    session: aiohttp.ClientSession,
    timeout_s: float,
    retry_interval_s: float = 2.0,
) -> RequestFuncOutput:
    """Retry a real request until the configured endpoint responds successfully."""
    deadline = time.perf_counter() + timeout_s
    last = RequestFuncOutput(error="endpoint did not respond")
    while time.perf_counter() < deadline:
        last = await request_func(test_input, session)
        if last.success:
            return last
        await asyncio.sleep(
            min(retry_interval_s, max(0.0, deadline - time.perf_counter()))
        )
    return last


def _merge_body(
    base: dict[str, Any] | None, override: dict[str, Any] | None
) -> dict[str, Any] | None:
    if not base and not override:
        return None
    merged = dict(base or {})
    merged.update(override or {})
    return merged


async def benchmark(
    samples: list[SampleRequest],
    *,
    backend: str,
    base_url: str,
    model: str | None = None,
    served_model_name: str | None = None,
    request_rate: float = math.inf,
    max_concurrency: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    logprobs: int | None = None,
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    num_warmups: int = 0,
    timeout_s: float = 600.0,
    ready_check: bool = True,
    self_timed: bool = False,
    burstiness: float = 1.0,
    percentiles: list[float] | None = None,
    goodput_config: dict[str, float] | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run requests against an already-running service and return JSON data."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}'; choose from {sorted(BACKENDS)}")
    if not samples:
        raise ValueError("no samples to benchmark")
    suffix, request_func = BACKENDS[backend]
    normalized_base = _service_base(base_url)
    api_url = normalized_base + suffix
    rate_label = "unlimited" if request_rate == math.inf else f"{request_rate:g}"
    concurrency_label = max_concurrency if max_concurrency is not None else "unlimited"
    logger.info(
        "benchmark task preparing: measured_requests=%d warmup_requests=%d "
        "backend=%s endpoint=%s model=%s request_rate=%s max_concurrency=%s",
        len(samples),
        num_warmups,
        backend,
        api_url,
        served_model_name or model or "auto",
        rate_label,
        concurrency_label,
    )
    connector = aiohttp.TCPConnector(
        limit=max_concurrency or 0, limit_per_host=max_concurrency or 0
    )
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    outputs: list[RequestFuncOutput] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if model is None:
            model = await get_first_model(base_url, session)
        logger.info("benchmark target selected: model=%s", served_model_name or model)

        def make_input(sample: SampleRequest) -> RequestFuncInput:
            return RequestFuncInput(
                prompt=sample.prompt,
                api_url=api_url,
                model=model,
                model_name=served_model_name,
                output_len=sample.expected_output_len,
                prompt_len=sample.prompt_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                logprobs=logprobs,
                extra_headers=extra_headers,
                extra_body=_merge_body(extra_body, sample.request_overrides),
                chat_messages=sample.chat_messages,
                multi_modal_content=sample.multi_modal_data,
                request_id=sample.request_id,
            )

        if ready_check:
            ready = await wait_for_endpoint(
                request_func, make_input(samples[0]), session, timeout_s
            )
            if not ready.success:
                raise RuntimeError(f"endpoint readiness check failed: {ready.error}")
            logger.info("benchmark readiness check passed")

        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        measured_sent = 0

        async def send(
            sample: SampleRequest,
            *,
            measured: bool,
        ) -> RequestFuncOutput:
            async def dispatch() -> RequestFuncOutput:
                nonlocal measured_sent
                if measured:
                    measured_sent += 1
                logger.debug(
                    "sending request: phase=%s request_id=%s input_tokens=%d "
                    "output_tokens=%d",
                    "measured" if measured else "warmup",
                    sample.request_id,
                    sample.prompt_len,
                    sample.expected_output_len,
                )
                return await request_func(make_input(sample), session)

            if semaphore is None:
                return await dispatch()
            async with semaphore:
                return await dispatch()

        if num_warmups:
            logger.info("sending warmup requests: count=%d", num_warmups)
            warmup_outputs = await asyncio.gather(
                *(send(samples[0], measured=False) for _ in range(num_warmups))
            )
            warmup_succeeded = sum(output.success for output in warmup_outputs)
            logger.info(
                "warmup requests finished: sent=%d succeeded=%d failed=%d",
                num_warmups,
                warmup_succeeded,
                num_warmups - warmup_succeeded,
            )

        tasks: list[asyncio.Task[RequestFuncOutput]] = []
        progress: Any | None = None
        if show_progress:
            try:
                from tqdm import tqdm
            except ImportError as exc:
                raise RuntimeError(
                    "progress display requires the 'tqdm' package"
                ) from exc
            progress = tqdm(total=len(samples), desc="Benchmark", unit="request")

        # Exclude model discovery, readiness checks, and warmup from the
        # reported measurement window, matching benchmark convention.
        logger.info("sending measured requests: count=%d", len(samples))
        started = time.perf_counter()
        try:
            async for sample, _ in get_request(
                samples, request_rate, burstiness=burstiness, self_timed=self_timed
            ):
                task = asyncio.create_task(send(sample, measured=True))
                if progress is not None:
                    task.add_done_callback(lambda _: progress.update())
                tasks.append(task)
            outputs = await asyncio.gather(*tasks)
        except Exception:
            logger.exception(
                "benchmark task failed: sent_requests=%d planned_requests=%d",
                measured_sent,
                len(samples),
            )
            raise
        finally:
            if progress is not None:
                progress.close()

    duration = time.perf_counter() - started
    result = calculate_metrics(
        outputs, duration, percentiles=percentiles, goodput_config=goodput_config
    )
    result["backend"] = backend
    result["model"] = served_model_name or model
    result["endpoint"] = api_url
    result["requests"] = [
        {
            "request_id": sample.request_id,
            "success": output.success,
            "latency_ms": output.latency * 1000,
            "ttft_ms": output.ttft * 1000,
            "itl_ms": [value * 1000 for value in output.itl],
            "input_tokens": output.prompt_len,
            "output_tokens": output.output_tokens,
            "error": output.error,
        }
        for sample, output in zip(samples, outputs, strict=True)
    ]
    logger.info(
        "benchmark task finished: sent_requests=%d succeeded=%d failed=%d "
        "duration_s=%.3f",
        measured_sent,
        result["completed"],
        result["failed"],
        duration,
    )
    return result


def print_metrics(result: dict[str, Any]) -> None:
    """Print a compact human-readable result while preserving JSON output."""
    print("=" * 58)
    print(" Serving Benchmark Result ".center(58, "="))
    for label, key, fmt in (
        ("Successful requests", "completed", "{}"),
        ("Failed requests", "failed", "{}"),
        ("Duration (s)", "duration_s", "{:.3f}"),
        ("Request throughput (req/s)", "request_throughput", "{:.2f}"),
        ("Output throughput (tok/s)", "output_throughput", "{:.2f}"),
        ("Total token throughput (tok/s)", "total_token_throughput", "{:.2f}"),
        ("Peak concurrency", "peak_concurrent_requests", "{}"),
    ):
        print(f"{label:<38} {fmt.format(result[key])}")
    for name in ("latency", "ttft", "tpot", "itl"):
        values = result[name]
        print(
            f"{name.upper():<38} mean={values['mean_ms']:.2f} ms p99={values['percentiles_ms'].get('99.0', 0.0):.2f} ms"
        )
