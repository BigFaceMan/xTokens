"""Profile no-KV Hugging Face inference HBM and batch boundaries."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

PROFILE_SCHEMA_VERSION = 1
OFFLOAD_DEVICES = {"cpu", "disk", "meta"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must not be negative")
    return parsed


def initial_batch_candidates(max_batch: int) -> list[int]:
    """Return sparse probes before an OOM-triggered binary search."""
    if max_batch <= 0:
        raise ValueError("max_batch must be positive")
    candidates = [1]
    candidate = 4
    while candidate < max_batch:
        candidates.append(candidate)
        candidate *= 2
    if max_batch not in candidates:
        candidates.append(max_batch)
    return sorted(set(candidates))


def find_offload_targets(
    device_map: Mapping[str, object], parameter_devices: set[str]
) -> list[str]:
    """Find module or parameter placement that invalidates a pure-GPU profile."""
    targets: list[str] = []
    for module_name, target in device_map.items():
        normalized = str(target).lower()
        if normalized in OFFLOAD_DEVICES:
            targets.append(f"module:{module_name}={normalized}")
    for device in sorted(parameter_devices):
        normalized = device.lower()
        if normalized in OFFLOAD_DEVICES:
            targets.append(f"parameter_device:{normalized}")
    return sorted(set(targets))


def search_batch_boundary(
    max_batch: int,
    measure: Callable[[int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int | bool | None]]:
    """Probe batches and bisect until success and OOM are adjacent."""
    cases: list[dict[str, Any]] = []
    successful = 0
    failed: int | None = None

    for candidate in initial_batch_candidates(max_batch):
        case = measure(candidate)
        cases.append(case)
        if case.get("status") == "ok":
            successful = max(successful, candidate)
            continue
        if case.get("status") != "oom":
            raise ValueError("measure must return status 'ok' or 'oom'")
        failed = candidate
        break

    if failed is not None and successful > 0:
        while failed - successful > 1:
            candidate = (successful + failed) // 2
            case = measure(candidate)
            cases.append(case)
            if case.get("status") == "ok":
                successful = candidate
            elif case.get("status") == "oom":
                failed = candidate
            else:
                raise ValueError("measure must return status 'ok' or 'oom'")

    return cases, {
        "max_successful_batch": successful,
        "min_oom_batch": failed,
        "exact_hard_limit": failed == successful + 1 and successful > 0,
        "search_ceiling": max_batch,
    }


def device_snapshots(torch: Any) -> list[dict[str, int | str]]:
    """Read allocator and driver-visible memory for every visible CUDA device."""
    snapshots: list[dict[str, int | str]] = []
    for visible_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(visible_index)
        properties = torch.cuda.get_device_properties(visible_index)
        snapshots.append(
            {
                "visible_index": visible_index,
                "name": properties.name,
                "total_bytes": total_bytes,
                "free_bytes": free_bytes,
                "allocated_bytes": torch.cuda.memory_allocated(visible_index),
                "reserved_bytes": torch.cuda.memory_reserved(visible_index),
            }
        )
    return snapshots


def synchronize_all(torch: Any) -> None:
    """Wait for work on every visible CUDA device."""
    for visible_index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(visible_index)


def emit_progress(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), file=sys.stderr, flush=True)


def write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if output is not None:
        with output.open("w", encoding="utf-8") as report_file:
            report_file.write(rendered)
            report_file.write("\n")
    print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile a pure-GPU Hugging Face no-KV forward path."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq-len", type=positive_int, required=True)
    parser.add_argument("--max-batch", type=positive_int, required=True)
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--warmup-seq-len", type=positive_int, default=16)
    parser.add_argument(
        "--input-mode", choices=("random", "repeated"), default="random"
    )
    parser.add_argument("--seed", type=nonnegative_int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise SystemExit("profile_no_kv.py requires torch and transformers") from error

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise SystemExit("profile_no_kv.py requires at least one visible CUDA GPU")

    preload_devices = device_snapshots(torch)
    emit_progress("load_start", devices=preload_devices)
    requested_dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=requested_dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )
    model.eval()

    device_map = getattr(model, "hf_device_map", {})
    if not isinstance(device_map, Mapping):
        device_map = {}
    parameter_bytes: defaultdict[str, int] = defaultdict(int)
    parameter_devices: set[str] = set()
    parameter_dtypes: set[str] = set()
    for parameter in model.parameters():
        device = str(parameter.device)
        parameter_devices.add(device)
        parameter_dtypes.add(str(parameter.dtype))
        parameter_bytes[device] += parameter.numel() * parameter.element_size()

    offload_targets = find_offload_targets(device_map, parameter_devices)
    forward_parameters = inspect.signature(model.forward).parameters
    logits_to_keep = forward_parameters.get("logits_to_keep")
    placement = {
        "hf_device_map": {name: target for name, target in device_map.items()},
        "parameter_bytes": dict(sorted(parameter_bytes.items())),
        "parameter_dtypes": sorted(parameter_dtypes),
        "offload_targets": offload_targets,
        "attention_backend": getattr(
            model.config, "_attn_implementation", None
        ),
        "supports_logits_to_keep": logits_to_keep is not None,
        "logits_to_keep_default": (
            repr(logits_to_keep.default) if logits_to_keep is not None else None
        ),
    }
    emit_progress(
        "load_complete",
        seconds=round(time.perf_counter() - load_started, 6),
        placement=placement,
    )

    base_report: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "mode": "without-kv-profile",
        "model": args.model,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_mode": args.input_mode,
        "seed": args.seed,
        "preload_devices": preload_devices,
        "placement": placement,
        "baseline_devices": [],
        "cases": [],
        "search": {
            "max_successful_batch": 0,
            "min_oom_batch": None,
            "exact_hard_limit": False,
            "search_ceiling": args.max_batch,
        },
        "warnings": [],
    }
    if offload_targets:
        base_report["status"] = "invalid_placement"
        base_report["warnings"].append(
            "CPU, disk, or meta offload invalidates a pure-GPU capacity profile."
        )
        write_report(base_report, args.output)
        return 2

    embedding_device = model.get_input_embeddings().weight.device
    vocab_size = int(model.config.vocab_size)

    def make_inputs(batch: int, seq_len: int, *, seed: int) -> dict[str, Any]:
        torch.manual_seed(seed)
        if args.input_mode == "random":
            input_ids = torch.randint(
                0,
                vocab_size,
                (batch, seq_len),
                dtype=torch.long,
                device=embedding_device,
            )
        else:
            input_ids = torch.ones(
                (batch, seq_len), dtype=torch.long, device=embedding_device
            )
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    warmup_inputs = make_inputs(1, args.warmup_seq_len, seed=args.seed)
    with torch.inference_mode():
        warmup_logits = model(**warmup_inputs, use_cache=False).logits[:, -1, :]
    synchronize_all(torch)
    del warmup_inputs, warmup_logits
    gc.collect()
    for visible_index in range(torch.cuda.device_count()):
        torch.cuda.empty_cache()

    baseline_devices = device_snapshots(torch)
    baseline_allocated = {
        device["visible_index"]: device["allocated_bytes"]
        for device in baseline_devices
    }
    base_report["baseline_devices"] = baseline_devices
    emit_progress("warmup_complete", devices=baseline_devices)

    def measure(batch: int) -> dict[str, Any]:
        inputs = None
        logits = None
        gc.collect()
        for visible_index in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(visible_index)
        try:
            inputs = make_inputs(batch, args.seq_len, seed=args.seed + batch)
            started = time.perf_counter()
            with torch.inference_mode():
                logits = model(**inputs, use_cache=False).logits[:, -1, :]
            synchronize_all(torch)
            devices = []
            for visible_index in range(torch.cuda.device_count()):
                peak_bytes = torch.cuda.max_memory_allocated(visible_index)
                devices.append(
                    {
                        "visible_index": visible_index,
                        "peak_allocated_bytes": peak_bytes,
                        "peak_delta_bytes": max(
                            peak_bytes - baseline_allocated[visible_index], 0
                        ),
                        "reserved_bytes_after": torch.cuda.memory_reserved(
                            visible_index
                        ),
                    }
                )
            case = {
                "batch": batch,
                "status": "ok",
                "seconds": round(time.perf_counter() - started, 6),
                "logits": {
                    "view_shape": list(logits.shape),
                    "dtype": str(logits.dtype),
                    "storage_bytes": logits.untyped_storage().nbytes(),
                },
                "devices": devices,
            }
        except torch.OutOfMemoryError as error:
            case = {
                "batch": batch,
                "status": "oom",
                "error": str(error).splitlines()[0],
                "devices": device_snapshots(torch),
            }
        finally:
            del inputs, logits
            gc.collect()
            for visible_index in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
        emit_progress("case", **case)
        return case

    cases, search = search_batch_boundary(args.max_batch, measure)
    base_report["status"] = "ok"
    base_report["cases"] = cases
    base_report["search"] = search
    if search["min_oom_batch"] is None:
        base_report["warnings"].append(
            "No OOM was observed; max_successful_batch is only a lower bound."
        )
    if args.input_mode != "random":
        base_report["warnings"].append(
            "Repeated tokens may not represent production MoE routing."
        )
    write_report(base_report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
