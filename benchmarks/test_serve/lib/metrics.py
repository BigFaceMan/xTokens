"""Metric aggregation for completed benchmark requests."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from .models import RequestFuncOutput  # pyright: ignore[reportMissingImports]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _summary(values: Iterable[float], percentiles: list[float]) -> dict[str, Any]:
    items = list(values)
    return {
        "mean_ms": statistics.fmean(items) * 1000 if items else 0.0,
        "median_ms": statistics.median(items) * 1000 if items else 0.0,
        "std_ms": statistics.pstdev(items) * 1000 if len(items) > 1 else 0.0,
        "percentiles_ms": {
            str(percentile): _percentile(items, percentile) * 1000
            for percentile in percentiles
        },
    }


def calculate_metrics(
    outputs: list[RequestFuncOutput],
    duration_s: float,
    *,
    percentiles: list[float] | None = None,
    goodput_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate throughput, latency, tail, concurrency, and goodput metrics."""
    percentiles = percentiles or [50.0, 90.0, 99.0]
    duration_s = max(duration_s, 1e-12)
    successful = [output for output in outputs if output.success]
    failed = [output for output in outputs if not output.success]
    e2el = [output.latency for output in successful]
    ttft = [output.ttft for output in successful if output.ttft > 0]
    itl = [item for output in successful for item in output.itl]
    tpot = [output.tpot for output in successful if output.tpot > 0]
    input_tokens = sum(max(0, output.prompt_len) for output in successful)
    output_tokens = sum(
        max(output.output_tokens, len(output.itl) + (1 if output.ttft > 0 else 0))
        for output in successful
    )

    good_requests = 0
    if goodput_config:
        for output in successful:
            checks = {
                "ttft": output.ttft,
                "tpot": output.tpot,
                "e2el": output.latency,
            }
            try:
                is_good = all(
                    key in checks and checks[key] <= float(limit) / 1000
                    for key, limit in goodput_config.items()
                )
            except (TypeError, ValueError):
                is_good = False
            if is_good:
                good_requests += 1

    peak_concurrency = 0
    if successful:
        events: list[tuple[float, int]] = []
        for output in successful:
            events.append((output.start_time, 1))
            events.append((output.start_time + output.latency, -1))
        active = 0
        for _, change in sorted(events, key=lambda item: (item[0], -item[1])):
            active += change
            peak_concurrency = max(peak_concurrency, active)

    result: dict[str, Any] = {
        "duration_s": duration_s,
        "completed": len(successful),
        "failed": len(failed),
        "total_requests": len(outputs),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "request_throughput": len(successful) / duration_s,
        "output_throughput": output_tokens / duration_s,
        "total_token_throughput": (input_tokens + output_tokens) / duration_s,
        "request_goodput": good_requests / duration_s if goodput_config else None,
        "peak_concurrent_requests": peak_concurrency,
        "latency": _summary(e2el, percentiles),
        "ttft": _summary(ttft, percentiles),
        "tpot": _summary(tpot, percentiles),
        "itl": _summary(itl, percentiles),
        "errors": [output.error for output in failed],
    }
    return result
