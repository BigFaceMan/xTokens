from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

GIB = 1 << 30
SCRIPT_PATH = (
    Path(__file__).parents[2]
    / "docs"
    / "skill"
    / "calculate-inference-hbm"
    / "scripts"
    / "calculate_capacity.py"
)


def load_calculator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("calculate_capacity", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calculator = load_calculator()


def make_profile(*, min_oom_batch: int | None = 16) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "without-kv-profile",
        "model": "/model",
        "seq_len": 4096,
        "dtype": "bfloat16",
        "input_mode": "random",
        "placement": {
            "offload_targets": [],
            "attention_backend": "sdpa",
        },
        "baseline_devices": [
            {"visible_index": 0, "free_bytes": 18 * GIB},
            {"visible_index": 1, "free_bytes": 18 * GIB},
        ],
        "cases": [
            {
                "batch": batch,
                "status": "ok",
                "seconds": float(batch),
                "devices": [
                    {"visible_index": 0, "peak_delta_bytes": batch * GIB},
                    {
                        "visible_index": 1,
                        "peak_delta_bytes": int(batch * 1.1 * GIB),
                    },
                ],
            }
            for batch in (1, 4, 8, 15)
        ]
        + (
            [{"batch": min_oom_batch, "status": "oom", "devices": []}]
            if min_oom_batch is not None
            else []
        ),
        "search": {
            "max_successful_batch": 15,
            "min_oom_batch": min_oom_batch,
            "exact_hard_limit": min_oom_batch == 16,
            "search_ceiling": 16,
        },
    }


def test_qwen3_kv_capacity() -> None:
    result = calculator.calculate_with_kv(
        hbm_bytes=32 * GIB,
        weight_bytes=0,
        runtime_bytes=0,
        activation_bytes=0,
        workspace_bytes=0,
        margin_bytes=0,
        local_layers=48,
        local_kv_heads=4,
        head_dim=128,
        kv_element_bytes=2,
        context_tokens=4096,
        block_size=16,
    )

    assert result["kv_cache"]["bytes_per_token"]["bytes"] == 96 * 1024
    assert result["kv_cache"]["bytes_per_sequence"]["bytes"] == 384 * 1024**2
    assert result["concurrency"]["hbm_limit"] == 85
    assert result["concurrency"]["effective_limit"] == 85


def test_kv_block_alignment_and_scheduler_limit() -> None:
    result = calculator.calculate_with_kv(
        hbm_bytes=32 * GIB,
        weight_bytes=0,
        runtime_bytes=0,
        activation_bytes=0,
        workspace_bytes=0,
        margin_bytes=0,
        local_layers=48,
        local_kv_heads=4,
        head_dim=128,
        kv_element_bytes=2,
        context_tokens=4097,
        block_size=16,
        scheduler_limit=3,
    )

    assert result["kv_cache"]["aligned_context_tokens"] == 4112
    assert result["concurrency"]["hbm_limit"] > 3
    assert result["concurrency"]["effective_limit"] == 3


def test_qwen3_no_kv_full_logits_and_eager_attention() -> None:
    result = calculator.calculate_without_kv(
        hbm_bytes=48 * GIB,
        weight_bytes=0,
        runtime_bytes=0,
        workspace_bytes=0,
        margin_bytes=0,
        seq_len=4096,
        hidden_size=2048,
        activation_element_bytes=2,
        activation_multiplier=8,
        vocab_size=151936,
        logits_element_bytes=2,
        last_token_logits=False,
        attention="eager",
        num_query_heads=32,
    )

    assert result["status"] == "estimate"
    assert result["per_sequence_model"]["logits"]["bytes"] == (
        4096 * 151936 * 2
    )
    assert result["per_sequence_model"]["attention_scores"]["bytes"] == GIB


def test_last_token_logits_only_materialize_one_position() -> None:
    result = calculator.calculate_without_kv(
        hbm_bytes=48 * GIB,
        weight_bytes=0,
        runtime_bytes=0,
        workspace_bytes=0,
        margin_bytes=0,
        seq_len=4096,
        hidden_size=2048,
        activation_element_bytes=2,
        activation_multiplier=8,
        vocab_size=151936,
        logits_element_bytes=2,
        last_token_logits=True,
        attention="efficient",
        num_query_heads=None,
    )

    assert result["per_sequence_model"]["logits"]["bytes"] == 151936 * 2
    assert result["per_sequence_model"]["attention_scores"]["bytes"] == 0


def test_eager_attention_requires_query_heads() -> None:
    with pytest.raises(ValueError, match="num_query_heads"):
        calculator.calculate_without_kv(
            hbm_bytes=48 * GIB,
            weight_bytes=0,
            runtime_bytes=0,
            workspace_bytes=0,
            margin_bytes=0,
            seq_len=4096,
            hidden_size=2048,
            activation_element_bytes=2,
            activation_multiplier=8,
            vocab_size=151936,
            logits_element_bytes=2,
            last_token_logits=False,
            attention="eager",
            num_query_heads=None,
        )


def test_static_memory_that_exceeds_hbm_does_not_fit() -> None:
    result = calculator.calculate_with_kv(
        hbm_bytes=48 * GIB,
        weight_bytes=49 * GIB,
        runtime_bytes=0,
        activation_bytes=0,
        workspace_bytes=0,
        margin_bytes=0,
        local_layers=48,
        local_kv_heads=4,
        head_dim=128,
        kv_element_bytes=2,
        context_tokens=4096,
        block_size=16,
    )

    assert result["status"] == "does_not_fit"
    assert result["concurrency"]["hbm_limit"] == 0


def test_profile_reports_hard_safe_configured_and_slo_limits() -> None:
    result = calculator.calculate_without_kv_profile(
        make_profile(),
        margin_bytes=2 * GIB,
        scheduler_limit=13,
    )

    assert result["limits"] == {
        "configured_limit": 13,
        "measured_hard_limit": 15,
        "measured_lower_bound": 15,
        "min_oom_batch": 16,
        "recommended_safe_limit": 13,
        "recommended_safe_limit_sampled": False,
        "slo_limit": None,
    }
    assert result["confidence"] == "measured_adjacent_oom_boundary"
    assert result["per_device"][1]["modeled_safe_limit"] == 14
    assert result["performance"][0]["output_tokens_per_second"] == 1.0


def test_profile_without_oom_reports_only_a_lower_bound() -> None:
    profile = make_profile(min_oom_batch=None)
    result = calculator.calculate_without_kv_profile(
        profile,
        margin_bytes=0,
    )

    assert result["limits"]["measured_hard_limit"] is None
    assert result["limits"]["measured_lower_bound"] == 15
    assert result["limits"]["recommended_safe_limit"] == 15
    assert result["limits"]["recommended_safe_limit_sampled"] is True
    assert result["confidence"] == "measured_lower_bound_only"


def test_profile_rejects_offload_and_missing_successful_cases() -> None:
    offloaded = make_profile()
    offloaded["placement"]["offload_targets"] = ["module:model.layers.2=cpu"]
    with pytest.raises(ValueError, match="offload"):
        calculator.calculate_without_kv_profile(offloaded, margin_bytes=0)

    empty = make_profile()
    empty["cases"] = [{"batch": 1, "status": "oom", "devices": []}]
    with pytest.raises(ValueError, match="successful"):
        calculator.calculate_without_kv_profile(empty, margin_bytes=0)
