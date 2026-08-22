"""Command-line entry point for the standalone serving benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

from ..dataset.datasets import load_samples
from ..dataset.tokenizer import load_tokenizer
from ..serve import benchmark, print_metrics


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON object: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("value must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench-serve",
        description=(
            "Benchmark an already-running OpenAI-compatible inference server."
        ),
    )
    parser.add_argument(
        "--backend",
        default="openai-chat",
        choices=[
            "openai",
            "vllm",
            "openai-chat",
            "chat",
            "openai-embeddings",
            "embeddings",
            "vllm-rerank",
            "rerank",
        ],
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", help="model id; omitted means use /v1/models")
    parser.add_argument("--served-model-name")
    parser.add_argument("--tokenizer")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dataset-name", "--dataset", default="random")
    parser.add_argument("--dataset-path")
    parser.add_argument(
        "--dataset-config",
        help="Hugging Face dataset configuration; ignored by built-in and local datasets",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Hugging Face dataset split; default: train",
    )
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--prefix-len", type=int, default=256)
    parser.add_argument("--suffix-len", type=int, default=256)
    parser.add_argument("--num-prefixes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--request-rate", type=float, default=math.inf)
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--num-warmups", type=int, default=0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument("--no-ready-check", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the completed-request progress bar",
    )
    parser.add_argument("--self-timed", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--logprobs", type=int)
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="ignore EOS and generate until the requested output length",
    )
    parser.add_argument("--extra-body", type=_json_object)
    parser.add_argument("--header", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--percentiles", default="50,90,99")
    parser.add_argument("--goodput", action="append", default=[], metavar="METRIC=MS")
    parser.add_argument("--save-result", type=Path)
    parser.add_argument("--save-requests", type=Path)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def _headers(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --header '{item}', expected NAME=VALUE")
        name, value = item.split("=", 1)
        if not name:
            raise ValueError("header name cannot be empty")
        result[name] = value
    return result


def _limits(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"invalid --goodput '{item}', expected METRIC=MS")
        name, value = item.split("=", 1)
        try:
            result[name] = float(value)
        except ValueError as exc:
            raise ValueError(f"invalid goodput limit '{value}'") from exc
    return result


def _extra_body(args: argparse.Namespace) -> dict[str, Any] | None:
    extra_body = dict(args.extra_body or {})
    if args.ignore_eos:
        extra_body["ignore_eos"] = True
    return extra_body or None


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = load_tokenizer(args.tokenizer, args.trust_remote_code)
    samples = load_samples(
        args.dataset_name,
        args.num_prompts,
        args.input_len,
        args.output_len,
        tokenizer,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        seed=args.seed,
        prefix_len=args.prefix_len,
        suffix_len=args.suffix_len,
        num_prefixes=args.num_prefixes,
        no_oversample=args.no_oversample,
    )
    try:
        percentiles = [
            float(value.strip())
            for value in args.percentiles.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "--percentiles must be a comma-separated list of numbers"
        ) from exc

    result = asyncio.run(
        benchmark(
            samples,
            backend=args.backend,
            base_url=args.endpoint,
            model=args.model,
            served_model_name=args.served_model_name,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            logprobs=args.logprobs,
            extra_headers=_headers(args.header),
            extra_body=_extra_body(args),
            num_warmups=args.num_warmups,
            timeout_s=args.ready_timeout,
            ready_check=not args.no_ready_check,
            self_timed=args.self_timed,
            burstiness=args.burstiness,
            percentiles=percentiles,
            goodput_config=_limits(args.goodput),
            show_progress=not args.no_progress,
        )
    )
    print_metrics(result)
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.save_result:
        args.save_result.write_text(serialized + "\n", encoding="utf-8")
        print(f"saved result: {args.save_result}")
    if args.save_requests:
        args.save_requests.write_text(
            json.dumps(result["requests"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved request details: {args.save_requests}")
    return result


def main() -> None:
    args = build_parser().parse_args()
    _configure_logging(args.log_level)
    run(args)
