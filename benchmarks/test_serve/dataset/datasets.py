"""Dataset generation and loading for the standalone serving benchmark."""

from __future__ import annotations

import csv
import importlib
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..lib.models import SampleRequest  # pyright: ignore[reportMissingImports]
from .tokenizer import (  # pyright: ignore[reportMissingImports]
    count_tokens,
    make_prompt,
)


class DatasetError(ValueError):
    """Raised for malformed or unsupported benchmark datasets."""


def _text_from_row(row: dict[str, Any]) -> str:
    for key in ("prompt", "text", "question", "input"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(
            str(item.get("content", "")) for item in messages if isinstance(item, dict)
        )
    raise DatasetError(
        "custom row must contain prompt, text, question, input, or messages"
    )


def _load_json(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"could not read JSON dataset {path}: {exc}") from exc
    if isinstance(value, dict):
        value = value.get("data", value.get("requests", []))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DatasetError(f"{path} must contain a JSON list of objects")
    return value


def _load_rows(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".json", ".jsonl"}:
        if source.suffix.lower() == ".jsonl":
            try:
                with source.open(encoding="utf-8") as stream:
                    return [json.loads(line) for line in stream if line.strip()]
            except (OSError, json.JSONDecodeError) as exc:
                raise DatasetError(
                    f"could not read JSONL dataset {source}: {exc}"
                ) from exc
        return _load_json(source)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    raise DatasetError("custom datasets must be .json, .jsonl, or .csv")


def _load_huggingface_rows(
    dataset_id: str,
    config: str | None,
    split: str,
) -> Iterable[dict[str, Any]]:
    """Load one Hugging Face Hub split without requiring it for local modes."""
    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = datasets_module.load_dataset
    except (ImportError, AttributeError) as exc:
        raise DatasetError(
            "Hugging Face datasets require the optional 'datasets' package; "
            "install it with 'pip install datasets'"
        ) from exc
    try:
        loaded = load_dataset(dataset_id, name=config, split=split)
    except Exception as exc:
        raise DatasetError(
            f"could not load Hugging Face dataset '{dataset_id}' "
            f"(config={config!r}, split={split!r}): {exc}"
        ) from exc
    return (dict(row) for row in loaded)


def _sample_rows(
    rows: Iterable[dict[str, Any]],
    tokenizer: Any,
    count: int,
    output_len: int,
    seed: int,
    no_oversample: bool,
) -> list[SampleRequest]:
    usable = list(rows)
    if not usable:
        raise DatasetError("dataset contains no usable rows")
    rng = random.Random(seed)
    rng.shuffle(usable)
    if no_oversample and len(usable) < count:
        count = len(usable)
    result: list[SampleRequest] = []
    for index in range(count):
        row = usable[index % len(usable)]
        prompt = _text_from_row(row)
        messages = (
            row.get("messages") if isinstance(row.get("messages"), list) else None
        )
        try:
            prompt_len = int(row.get("prompt_len", count_tokens(tokenizer, prompt)))
            row_output = int(row.get("output_len", output_len))
            timestamp = (
                float(row["timestamp"]) if row.get("timestamp") is not None else None
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise DatasetError(f"invalid numeric fields in row {index}: {exc}") from exc
        request_id = str(row.get("request_id", index))
        result.append(
            SampleRequest(
                prompt=prompt,
                prompt_len=prompt_len,
                expected_output_len=row_output,
                request_id=request_id,
                chat_messages=messages,
                request_overrides=row.get("request_overrides"),
                timestamp=timestamp,
            )
        )
    return result


def load_samples(
    dataset: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    tokenizer: Any,
    *,
    dataset_path: str | None = None,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    seed: int = 0,
    prefix_len: int = 256,
    suffix_len: int = 256,
    num_prefixes: int = 10,
    no_oversample: bool = False,
) -> list[SampleRequest]:
    """Load one of the built-in datasets and normalize it to samples.

    Supported names are ``random``, ``random-mm``, ``prefix_repetition``,
    ``sharegpt``, ``custom``, and ``trace``. Any other dataset name is treated
    as a Hugging Face Hub dataset ID and loaded with ``datasets.load_dataset``.
    ``dataset_config`` and ``dataset_split`` select its configuration and split.
    """
    name = dataset.lower().replace("-", "_")
    if name in {"random", "random_mm"}:
        samples = []
        for index in range(num_prompts):
            prompt = make_prompt(tokenizer, input_len, seed + index)
            samples.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    request_id=str(index),
                    multi_modal_data=(
                        [{"type": "text", "text": "synthetic"}]
                        if name == "random_mm"
                        else None
                    ),
                )
            )
        return samples
    if name in {"prefix_repetition", "prefix_repetition_random"}:
        if num_prefixes <= 0 or num_prompts < num_prefixes:
            raise DatasetError("num-prompts must be >= num-prefixes")
        rng = random.Random(seed)
        prefixes = [
            make_prompt(tokenizer, prefix_len, seed + i) for i in range(num_prefixes)
        ]
        samples = []
        for index in range(num_prompts):
            prefix = prefixes[index % num_prefixes]
            suffix = make_prompt(tokenizer, suffix_len, seed + 10000 + index)
            samples.append(
                SampleRequest(
                    prompt=f"{prefix} {suffix}".strip(),
                    prompt_len=prefix_len + suffix_len,
                    expected_output_len=output_len,
                    request_id=str(index),
                )
            )
        rng.shuffle(samples)
        return samples
    if name in {"sharegpt", "custom", "trace"}:
        if not dataset_path:
            raise DatasetError(f"--dataset-path is required for dataset '{dataset}'")
        rows = _load_rows(dataset_path)
        if name == "sharegpt":
            normalized = []
            for row in rows:
                conversations = row.get("conversations", [])
                if len(conversations) >= 2:
                    normalized.append(
                        {
                            "prompt": str(
                                conversations[-2].get(
                                    "value", conversations[-2].get("content", "")
                                )
                            ),
                            "output_len": output_len,
                        }
                    )
            rows = normalized
        return _sample_rows(
            rows, tokenizer, num_prompts, output_len, seed, no_oversample
        )
    if dataset_path:
        raise DatasetError(
            "--dataset-path is only supported by sharegpt, custom, and trace datasets; "
            "remove it to load a Hugging Face Hub dataset"
        )
    rows = _load_huggingface_rows(dataset, dataset_config, dataset_split)
    return _sample_rows(rows, tokenizer, num_prompts, output_len, seed, no_oversample)
