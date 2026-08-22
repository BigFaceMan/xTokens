"""Calculate per-GPU LLM inference HBM capacity estimates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

GIB = 1 << 30


def align_up(value: int, alignment: int) -> int:
    """Round a positive token count up to an allocation boundary."""
    if value <= 0:
        raise ValueError("value must be positive")
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


def gib_to_bytes(value: float) -> int:
    """Convert GiB to bytes using the binary unit definition."""
    if value < 0:
        raise ValueError("GiB value must not be negative")
    return round(value * GIB)


def memory_value(byte_count: float) -> dict[str, int | float]:
    """Represent a memory value in bytes and GiB without hiding precision."""
    normalized_bytes: int | float
    if float(byte_count).is_integer():
        normalized_bytes = int(byte_count)
    else:
        normalized_bytes = byte_count
    return {
        "bytes": normalized_bytes,
        "gib": round(byte_count / GIB, 6),
    }


def effective_limit(hbm_limit: int, scheduler_limit: int | None) -> int:
    """Apply an optional scheduler limit to an HBM-derived limit."""
    if scheduler_limit is None:
        return hbm_limit
    if scheduler_limit <= 0:
        raise ValueError("scheduler limit must be positive")
    return min(hbm_limit, scheduler_limit)


def calculate_with_kv(
    *,
    hbm_bytes: int,
    weight_bytes: int,
    runtime_bytes: int,
    activation_bytes: int,
    workspace_bytes: int,
    margin_bytes: int,
    local_layers: int,
    local_kv_heads: int,
    head_dim: int,
    kv_element_bytes: float,
    context_tokens: int,
    block_size: int,
    scheduler_limit: int | None = None,
) -> dict[str, Any]:
    """Calculate logical paged-KV capacity for one GPU."""
    for name, value in (
        ("hbm_bytes", hbm_bytes),
        ("local_layers", local_layers),
        ("local_kv_heads", local_kv_heads),
        ("head_dim", head_dim),
        ("context_tokens", context_tokens),
        ("block_size", block_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("weight_bytes", weight_bytes),
        ("runtime_bytes", runtime_bytes),
        ("activation_bytes", activation_bytes),
        ("workspace_bytes", workspace_bytes),
        ("margin_bytes", margin_bytes),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if kv_element_bytes <= 0:
        raise ValueError("kv_element_bytes must be positive")

    static_bytes = (
        weight_bytes
        + runtime_bytes
        + activation_bytes
        + workspace_bytes
        + margin_bytes
    )
    available_bytes = max(hbm_bytes - static_bytes, 0)
    kv_bytes_per_token = (
        2 * local_layers * local_kv_heads * head_dim * kv_element_bytes
    )
    aligned_context_tokens = align_up(context_tokens, block_size)
    kv_bytes_per_sequence = kv_bytes_per_token * aligned_context_tokens
    hbm_limit = math.floor(available_bytes / kv_bytes_per_sequence)
    cache_token_capacity = math.floor(available_bytes / kv_bytes_per_token)
    block_capacity = cache_token_capacity // block_size

    return {
        "mode": "with-kv",
        "status": "ok" if hbm_bytes > static_bytes else "does_not_fit",
        "memory": {
            "hbm": memory_value(hbm_bytes),
            "weights": memory_value(weight_bytes),
            "runtime": memory_value(runtime_bytes),
            "activation": memory_value(activation_bytes),
            "workspace": memory_value(workspace_bytes),
            "margin": memory_value(margin_bytes),
            "kv_budget": memory_value(available_bytes),
        },
        "kv_cache": {
            "bytes_per_token": memory_value(kv_bytes_per_token),
            "requested_context_tokens": context_tokens,
            "aligned_context_tokens": aligned_context_tokens,
            "block_size_tokens": block_size,
            "bytes_per_sequence": memory_value(kv_bytes_per_sequence),
            "cache_token_capacity": cache_token_capacity,
            "block_capacity": block_capacity,
        },
        "concurrency": {
            "hbm_limit": hbm_limit,
            "scheduler_limit": scheduler_limit,
            "effective_limit": effective_limit(hbm_limit, scheduler_limit),
        },
        "limitations": [
            "The result models one GPU and must be repeated for every GPU.",
            "KV metadata, allocator fragmentation, and transient cache growth are not included.",
            "The result is an OOM boundary, not an SLO-qualified concurrency limit.",
        ],
    }


def calculate_without_kv(
    *,
    hbm_bytes: int,
    weight_bytes: int,
    runtime_bytes: int,
    workspace_bytes: int,
    margin_bytes: int,
    seq_len: int,
    hidden_size: int,
    activation_element_bytes: float,
    activation_multiplier: float,
    vocab_size: int,
    logits_element_bytes: float,
    last_token_logits: bool,
    attention: str,
    num_query_heads: int | None,
    input_bytes_per_token: float = 16,
    scheduler_limit: int | None = None,
) -> dict[str, Any]:
    """Estimate linear per-sequence no-KV temporary memory for one GPU."""
    for name, value in (
        ("hbm_bytes", hbm_bytes),
        ("seq_len", seq_len),
        ("hidden_size", hidden_size),
        ("vocab_size", vocab_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("weight_bytes", weight_bytes),
        ("runtime_bytes", runtime_bytes),
        ("workspace_bytes", workspace_bytes),
        ("margin_bytes", margin_bytes),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    for name, value in (
        ("activation_element_bytes", activation_element_bytes),
        ("activation_multiplier", activation_multiplier),
        ("logits_element_bytes", logits_element_bytes),
        ("input_bytes_per_token", input_bytes_per_token),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if attention not in {"efficient", "eager"}:
        raise ValueError("attention must be 'efficient' or 'eager'")
    if attention == "eager" and (num_query_heads is None or num_query_heads <= 0):
        raise ValueError("eager attention requires positive num_query_heads")

    static_bytes = weight_bytes + runtime_bytes + workspace_bytes + margin_bytes
    available_bytes = max(hbm_bytes - static_bytes, 0)
    input_bytes = seq_len * input_bytes_per_token
    layer_activation_bytes = (
        seq_len
        * hidden_size
        * activation_element_bytes
        * activation_multiplier
    )
    logits_positions = 1 if last_token_logits else seq_len
    logits_bytes = logits_positions * vocab_size * logits_element_bytes
    attention_score_bytes = 0.0
    if attention == "eager":
        assert num_query_heads is not None
        attention_score_bytes = (
            num_query_heads * seq_len * seq_len * activation_element_bytes
        )
    modeled_bytes_per_sequence = (
        input_bytes
        + layer_activation_bytes
        + logits_bytes
        + attention_score_bytes
    )
    hbm_limit = math.floor(available_bytes / modeled_bytes_per_sequence)

    return {
        "mode": "without-kv",
        "status": "estimate" if hbm_bytes > static_bytes else "does_not_fit",
        "memory": {
            "hbm": memory_value(hbm_bytes),
            "weights": memory_value(weight_bytes),
            "runtime": memory_value(runtime_bytes),
            "workspace": memory_value(workspace_bytes),
            "margin": memory_value(margin_bytes),
            "dynamic_budget": memory_value(available_bytes),
        },
        "per_sequence_model": {
            "padded_sequence_tokens": seq_len,
            "inputs": memory_value(input_bytes),
            "layer_activations": memory_value(layer_activation_bytes),
            "logits": memory_value(logits_bytes),
            "attention_scores": memory_value(attention_score_bytes),
            "modeled_total": memory_value(modeled_bytes_per_sequence),
        },
        "concurrency": {
            "modeled_hbm_limit": hbm_limit,
            "scheduler_limit": scheduler_limit,
            "effective_modeled_limit": effective_limit(hbm_limit, scheduler_limit),
        },
        "assumptions": {
            "attention": attention,
            "last_token_logits": last_token_logits,
            "activation_multiplier": activation_multiplier,
            "input_bytes_per_token": input_bytes_per_token,
        },
        "limitations": [
            "The no-KV result is a static tensor model, not a safe maximum concurrency.",
            "Calibrate the activation multiplier and workspace with peak-memory profiling.",
            "Padding, allocator behavior, fused kernels, and MoE dispatch can be nonlinear in batch size.",
            "The result is an OOM estimate, not an SLO-qualified concurrency limit.",
        ],
    }


def calculate_without_kv_profile(
    profile: dict[str, Any],
    *,
    margin_bytes: int,
    scheduler_limit: int | None = None,
) -> dict[str, Any]:
    """Calculate conservative no-KV limits from a profiler report."""
    if margin_bytes < 0:
        raise ValueError("margin_bytes must not be negative")
    if scheduler_limit is not None and scheduler_limit <= 0:
        raise ValueError("scheduler limit must be positive")
    if profile.get("schema_version") != 1:
        raise ValueError("profile schema_version must be 1")
    if profile.get("mode") != "without-kv-profile":
        raise ValueError("profile mode must be 'without-kv-profile'")

    placement = profile.get("placement")
    if not isinstance(placement, dict):
        raise TypeError("profile placement must be an object")
    offload_targets = placement.get("offload_targets")
    if not isinstance(offload_targets, list):
        raise TypeError("profile placement.offload_targets must be a list")
    if offload_targets:
        raise ValueError("profile contains CPU, disk, or meta offload targets")

    baseline_devices = profile.get("baseline_devices")
    if not isinstance(baseline_devices, list) or not baseline_devices:
        raise ValueError("profile baseline_devices must be a non-empty list")
    baseline_free: dict[int, int] = {}
    for device in baseline_devices:
        if not isinstance(device, dict):
            raise TypeError("profile baseline device must be an object")
        visible_index = device.get("visible_index")
        free_bytes = device.get("free_bytes")
        if not isinstance(visible_index, int) or visible_index < 0:
            raise ValueError("baseline visible_index must be a non-negative integer")
        if not isinstance(free_bytes, int) or free_bytes < 0:
            raise ValueError("baseline free_bytes must be a non-negative integer")
        if visible_index in baseline_free:
            raise ValueError("baseline visible_index values must be unique")
        baseline_free[visible_index] = free_bytes

    cases = profile.get("cases")
    if not isinstance(cases, list):
        raise TypeError("profile cases must be a list")
    if any(not isinstance(case, dict) for case in cases):
        raise TypeError("each profile case must be an object")
    successful_cases = [case for case in cases if case.get("status") == "ok"]
    if not successful_cases:
        raise ValueError("profile must contain at least one successful case")

    slopes: dict[int, float] = {device: 0.0 for device in baseline_free}
    performance: list[dict[str, int | float]] = []
    successful_batches: list[int] = []
    for case in successful_cases:
        batch = case.get("batch")
        seconds = case.get("seconds")
        devices = case.get("devices")
        if not isinstance(batch, int) or batch <= 0:
            raise ValueError("successful case batch must be a positive integer")
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            raise ValueError("successful case seconds must be positive")
        if not isinstance(devices, list):
            raise TypeError("successful case devices must be a list")
        deltas: dict[int, int] = {}
        for device in devices:
            if not isinstance(device, dict):
                raise TypeError("case device must be an object")
            visible_index = device.get("visible_index")
            peak_delta_bytes = device.get("peak_delta_bytes")
            if visible_index not in baseline_free:
                raise ValueError("case contains an unknown visible_index")
            if not isinstance(peak_delta_bytes, int) or peak_delta_bytes < 0:
                raise ValueError("case peak_delta_bytes must be non-negative")
            deltas[visible_index] = peak_delta_bytes
        if set(deltas) != set(baseline_free):
            raise ValueError("each successful case must contain every baseline device")
        for visible_index, peak_delta_bytes in deltas.items():
            slopes[visible_index] = max(
                slopes[visible_index], peak_delta_bytes / batch
            )
        successful_batches.append(batch)
        performance.append(
            {
                "batch": batch,
                "seconds_per_step": round(float(seconds), 6),
                "output_tokens_per_second": round(batch / float(seconds), 6),
            }
        )

    if any(slope <= 0 for slope in slopes.values()):
        raise ValueError("every device must have a positive measured peak slope")

    search = profile.get("search")
    if not isinstance(search, dict):
        raise TypeError("profile search must be an object")
    max_successful = search.get("max_successful_batch")
    min_oom = search.get("min_oom_batch")
    if not isinstance(max_successful, int) or max_successful <= 0:
        raise ValueError("search max_successful_batch must be positive")
    if max_successful != max(successful_batches):
        raise ValueError("search max_successful_batch does not match successful cases")
    if min_oom is not None and (not isinstance(min_oom, int) or min_oom <= 0):
        raise ValueError("search min_oom_batch must be null or positive")
    exact_hard_limit = min_oom == max_successful + 1
    measured_hard_limit = max_successful if exact_hard_limit else None

    per_device: list[dict[str, Any]] = []
    safe_limits: list[int] = []
    for visible_index in sorted(baseline_free):
        free_bytes = baseline_free[visible_index]
        slope = slopes[visible_index]
        usable_bytes = max(free_bytes - margin_bytes, 0)
        safe_limit = math.floor(usable_bytes / slope)
        safe_limits.append(safe_limit)
        per_device.append(
            {
                "visible_index": visible_index,
                "post_warmup_free": memory_value(free_bytes),
                "margin": memory_value(margin_bytes),
                "usable_dynamic_memory": memory_value(usable_bytes),
                "conservative_peak_delta_per_sequence": memory_value(slope),
                "modeled_safe_limit": safe_limit,
            }
        )

    recommended_safe_limit = min(min(safe_limits), max_successful)
    if scheduler_limit is not None:
        recommended_safe_limit = min(recommended_safe_limit, scheduler_limit)
    recommended_safe_limit_sampled = recommended_safe_limit in successful_batches

    return {
        "mode": "without-kv-profile",
        "status": "ok" if recommended_safe_limit > 0 else "does_not_fit",
        "profile": {
            "model": profile.get("model"),
            "seq_len": profile.get("seq_len"),
            "attention_backend": placement.get("attention_backend"),
            "input_mode": profile.get("input_mode"),
            "dtype": profile.get("dtype"),
        },
        "per_device": per_device,
        "limits": {
            "configured_limit": scheduler_limit,
            "measured_hard_limit": measured_hard_limit,
            "measured_lower_bound": max_successful,
            "min_oom_batch": min_oom,
            "recommended_safe_limit": recommended_safe_limit,
            "recommended_safe_limit_sampled": recommended_safe_limit_sampled,
            "slo_limit": None,
        },
        "confidence": (
            "measured_adjacent_oom_boundary"
            if exact_hard_limit
            else "measured_lower_bound_only"
        ),
        "performance": sorted(performance, key=lambda item: item["batch"]),
        "limitations": [
            "The safe limit uses the largest measured peak-delta-per-batch slope on each GPU.",
            "The recommendation never exceeds a successfully measured batch.",
            "Directly sample the recommended batch when recommended_safe_limit_sampled is false.",
            "Random or repeated tokens do not replace a representative production workload.",
            "slo_limit remains null until TTFT, TPOT, throughput, and tail latency are benchmarked.",
        ],
    }


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hbm-gib", type=positive_float, required=True)
    parser.add_argument("--weight-gib", type=nonnegative_float, required=True)
    parser.add_argument("--runtime-gib", type=nonnegative_float, default=0)
    parser.add_argument("--workspace-gib", type=nonnegative_float, default=0)
    parser.add_argument("--margin-gib", type=nonnegative_float, default=0)
    parser.add_argument("--scheduler-max-seqs", type=positive_int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate a per-GPU LLM inference HBM capacity estimate."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    with_kv = subparsers.add_parser("with-kv", help="calculate KV-cache capacity")
    add_common_arguments(with_kv)
    with_kv.add_argument("--activation-gib", type=nonnegative_float, default=0)
    with_kv.add_argument("--local-layers", type=positive_int, required=True)
    with_kv.add_argument("--local-kv-heads", type=positive_int, required=True)
    with_kv.add_argument("--head-dim", type=positive_int, required=True)
    with_kv.add_argument("--kv-bytes", type=positive_float, required=True)
    with_kv.add_argument("--context-tokens", type=positive_int, required=True)
    with_kv.add_argument("--block-size", type=positive_int, default=1)

    without_kv = subparsers.add_parser(
        "without-kv", help="model no-KV per-forward temporary memory"
    )
    add_common_arguments(without_kv)
    without_kv.add_argument("--seq-len", type=positive_int, required=True)
    without_kv.add_argument("--hidden-size", type=positive_int, required=True)
    without_kv.add_argument(
        "--activation-bytes", type=positive_float, required=True
    )
    without_kv.add_argument(
        "--activation-multiplier", type=positive_float, required=True
    )
    without_kv.add_argument("--vocab-size", type=positive_int, required=True)
    without_kv.add_argument("--logits-bytes", type=positive_float, required=True)
    without_kv.add_argument("--last-token-logits", action="store_true")
    without_kv.add_argument(
        "--attention", choices=("efficient", "eager"), required=True
    )
    without_kv.add_argument("--num-query-heads", type=positive_int)
    without_kv.add_argument(
        "--input-bytes-per-token", type=positive_float, default=16
    )

    without_kv_profile = subparsers.add_parser(
        "without-kv-profile", help="calculate no-KV limits from a profile JSON"
    )
    without_kv_profile.add_argument(
        "--profile-json", type=Path, required=True
    )
    without_kv_profile.add_argument(
        "--margin-gib", type=nonnegative_float, required=True
    )
    without_kv_profile.add_argument("--scheduler-max-seqs", type=positive_int)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "without-kv-profile":
        try:
            with args.profile_json.open(encoding="utf-8") as profile_file:
                profile = json.load(profile_file)
            result = calculate_without_kv_profile(
                profile,
                margin_bytes=gib_to_bytes(args.margin_gib),
                scheduler_limit=args.scheduler_max_seqs,
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            parser.error(str(error))
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    common = {
        "hbm_bytes": gib_to_bytes(args.hbm_gib),
        "weight_bytes": gib_to_bytes(args.weight_gib),
        "runtime_bytes": gib_to_bytes(args.runtime_gib),
        "workspace_bytes": gib_to_bytes(args.workspace_gib),
        "margin_bytes": gib_to_bytes(args.margin_gib),
        "scheduler_limit": args.scheduler_max_seqs,
    }

    if args.mode == "with-kv":
        result = calculate_with_kv(
            **common,
            activation_bytes=gib_to_bytes(args.activation_gib),
            local_layers=args.local_layers,
            local_kv_heads=args.local_kv_heads,
            head_dim=args.head_dim,
            kv_element_bytes=args.kv_bytes,
            context_tokens=args.context_tokens,
            block_size=args.block_size,
        )
    else:
        try:
            result = calculate_without_kv(
                **common,
                seq_len=args.seq_len,
                hidden_size=args.hidden_size,
                activation_element_bytes=args.activation_bytes,
                activation_multiplier=args.activation_multiplier,
                vocab_size=args.vocab_size,
                logits_element_bytes=args.logits_bytes,
                last_token_logits=args.last_token_logits,
                attention=args.attention,
                num_query_heads=args.num_query_heads,
                input_bytes_per_token=args.input_bytes_per_token,
            )
        except ValueError as error:
            parser.error(str(error))

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
