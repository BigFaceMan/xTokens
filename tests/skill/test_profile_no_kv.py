from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "docs"
    / "skill"
    / "calculate-inference-hbm"
    / "scripts"
    / "profile_no_kv.py"
)


def load_profiler() -> ModuleType:
    spec = importlib.util.spec_from_file_location("profile_no_kv", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profiler = load_profiler()


def test_initial_batch_candidates_are_sparse_and_include_ceiling() -> None:
    assert profiler.initial_batch_candidates(16) == [1, 4, 8, 16]
    assert profiler.initial_batch_candidates(6) == [1, 4, 6]
    assert profiler.initial_batch_candidates(1) == [1]


def test_find_offload_targets_checks_map_and_parameter_devices() -> None:
    targets = profiler.find_offload_targets(
        {"model.layers.0": 0, "model.layers.1": "cpu", "lm_head": "disk"},
        {"cuda:0", "meta"},
    )

    assert targets == [
        "module:lm_head=disk",
        "module:model.layers.1=cpu",
        "parameter_device:meta",
    ]


def test_search_batch_boundary_bisects_to_adjacent_oom() -> None:
    measured: list[int] = []

    def measure(batch: int) -> dict[str, int | str]:
        measured.append(batch)
        return {"batch": batch, "status": "ok" if batch <= 5 else "oom"}

    cases, search = profiler.search_batch_boundary(16, measure)

    assert measured == [1, 4, 8, 6, 5]
    assert [case["batch"] for case in cases] == measured
    assert search == {
        "max_successful_batch": 5,
        "min_oom_batch": 6,
        "exact_hard_limit": True,
        "search_ceiling": 16,
    }


def test_search_batch_boundary_marks_ceiling_as_lower_bound() -> None:
    _, search = profiler.search_batch_boundary(
        8, lambda batch: {"batch": batch, "status": "ok"}
    )

    assert search["max_successful_batch"] == 8
    assert search["min_oom_batch"] is None
    assert search["exact_hard_limit"] is False
