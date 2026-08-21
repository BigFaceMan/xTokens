"""Composition root and command-line entrypoint for Serve."""

from __future__ import annotations

import argparse
import os

import uvicorn

from .app import create_app
from .config import ServeConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the xTokens Serve API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", dest="served_model_name", default="x-tokens-mock")
    parser.add_argument("--engine", choices=("inproc",), default="inproc")
    parser.add_argument("--hf-model")
    parser.add_argument("--hf-device", default="auto")
    parser.add_argument("--hf-dtype", default="auto")
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--hf-max-num-seqs", type=int, default=4)
    parser.add_argument("--api-key", default=os.environ.get("XTOKENS_API_KEY"))
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument(
        "--shutdown-policy", choices=("abort", "drain"), default="abort"
    )
    parser.add_argument("--cors-origin", action="append", default=[])
    parser.add_argument("--no-access-log", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = ServeConfig(
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        hf_model=args.hf_model,
        hf_device=args.hf_device,
        hf_dtype=args.hf_dtype,
        hf_local_files_only=args.hf_local_files_only,
        hf_max_num_seqs=args.hf_max_num_seqs,
        api_key=args.api_key,
        request_timeout_s=args.request_timeout,
        shutdown_timeout_s=args.shutdown_timeout,
        shutdown_policy=args.shutdown_policy,
        cors_origins=tuple(args.cors_origin),
        access_log=not args.no_access_log,
    )
    uvicorn.run(
        create_app(config),
        host=config.host,
        port=config.port,
        access_log=config.access_log,
    )


if __name__ == "__main__":
    main()
