"""R2 evaluation tests: provenance derivation + measured-vs-estimated token metrics.

Telemetry rows carry ``cost_features["actual_total_tokens"]`` (measured);
bootstrap rows only have estimates. Token metrics (tokens_per_task,
tokens_per_success) must use measured tokens whenever available, and the
results doc must state the mix so consumers know what they are looking at.
"""

from __future__ import annotations

from gentle_ai_model_router.dataset.schema import (
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_TELEMETRY,
    CandidateRef,
    DatasetExample,
    DatasetV1,
)
from gentle_ai_model_router.router.config import PhaseConfig, load_config
from gentle_ai_model_router.training.evaluate import (
    GroupOutcome,
    evaluate_routers,
)

AA = "artificial_analysis_intelligence_index"

# Phase floor: strong (0.9) succeeds, cheap (0.4) fails.
CONFIG = load_config(config_path="/nonexistent/router.yaml", data_dir="/nonexistent/data")
CONFIG = CONFIG.model_copy(
    update={"phases": {"design": PhaseConfig(threshold_quality=0.7, weights={AA: 1.0})}}
)


def _example(
    idx: int,
    task_id: str,
    model: str,
    utility: float,
    provenance: str,
    *,
    est_total_tokens: float,
    actual_total_tokens: float | None = None,
) -> DatasetExample:
    cost_features: dict[str, float | None] = {
        "est_input_tokens": est_total_tokens / 2,
        "est_output_tokens": est_total_tokens / 2,
        "est_total_tokens": est_total_tokens,
        "est_cost": 1.0,
        "actual_total_tokens": actual_total_tokens,
    }
    return DatasetExample(
        example_id=f"ex-{idx}",
        task_id=task_id,
        phase="design",
        task_type="feature",
        context_tokens=10_000,
        task_text="synthetic task",
        candidate=CandidateRef(model=model, deployment="default", effort="high"),
        model_features=[0.0] * 7,
        benchmark_features=[utility],
        benchmark_feature_names=(AA,),
        cost_features=cost_features,
        label_utility=utility,
        label_quality_estimate=utility,
        label_provenance=provenance,
        snapshot_date="2026-01-01",
        split="test",
    )


def _mixed_dataset() -> DatasetV1:
    """Two groups; each has one strong and one cheap candidate, with the
    measured-token (telemetry) row on a DIFFERENT candidate per group so the
    measured/estimated branch is exercised for both chooser extremes."""
    examples = [
        # Group t1: the STRONG row is telemetry (measured tokens).
        _example(0, "t1", "test/strong", 0.9, PROVENANCE_TELEMETRY,
                 est_total_tokens=200.0, actual_total_tokens=4000.0),
        _example(1, "t1", "test/cheap", 0.4, PROVENANCE_BOOTSTRAP,
                 est_total_tokens=100.0),
        # Group t2: the CHEAP row is telemetry (measured tokens).
        _example(2, "t2", "test/strong", 0.9, PROVENANCE_BOOTSTRAP,
                 est_total_tokens=200.0),
        _example(3, "t2", "test/cheap", 0.4, PROVENANCE_TELEMETRY,
                 est_total_tokens=100.0, actual_total_tokens=5000.0),
    ]
    return DatasetV1(name="mixed-tokens", version=1, examples=examples,
                     benchmark_feature_names=(AA,))


# --------------------------------------------------------------------------- #
# GroupOutcome token branch (unit level)
# --------------------------------------------------------------------------- #


def _outcome(example: DatasetExample) -> GroupOutcome:
    return GroupOutcome(chosen=example, oracle_utility=example.label_utility, phase_threshold=0.7)


def test_group_outcome_prefers_measured_tokens() -> None:
    telemetry_row = _example(0, "t1", "test/strong", 0.9, PROVENANCE_TELEMETRY,
                             est_total_tokens=200.0, actual_total_tokens=4000.0)
    outcome = _outcome(telemetry_row)
    assert outcome.has_measured_tokens is True
    assert outcome.tokens == 4000.0  # measured, NOT the 200.0 estimate


def test_group_outcome_falls_back_to_estimated_tokens() -> None:
    bootstrap_row = _example(1, "t1", "test/cheap", 0.4, PROVENANCE_BOOTSTRAP,
                             est_total_tokens=100.0)
    outcome = _outcome(bootstrap_row)
    assert outcome.has_measured_tokens is False
    assert outcome.tokens == 100.0


# --------------------------------------------------------------------------- #
# Results doc: provenance + measured/estimated counters + metric values
# --------------------------------------------------------------------------- #


def test_eval_results_report_true_mixed_provenance() -> None:
    results = evaluate_routers(_mixed_dataset(), CONFIG, session=None)
    assert results["label_provenance"] == "bootstrap_prior+telemetry"
    assert "bootstrap_prior+telemetry" in results["caveat"]


def test_eval_results_count_measured_vs_estimated_rows() -> None:
    results = evaluate_routers(_mixed_dataset(), CONFIG, session=None)
    # 4 rows across the evaluated "test" split: 2 telemetry (measured),
    # 2 bootstrap (estimated).
    assert results["measured_token_rows"] == 2
    assert results["estimated_token_rows"] == 2


def test_token_metrics_use_measured_tokens_when_available() -> None:
    results = evaluate_routers(_mixed_dataset(), CONFIG, session=None)
    split = results["splits"]["test"]

    # benchmark_only picks the highest raw benchmark score (test/strong,
    # utility 0.9) in both groups: t1 measured 4000.0, t2 estimated 200.0
    # -> mean 2100.0 (would be 200.0 if measured tokens were ignored).
    strong = split["benchmark_only"]
    assert strong["tokens_per_task"] == 2100.0
    assert strong["tokens_per_success"] == 2100.0  # 2/2 successes
    assert strong["success_rate"] == 1.0

    # fixed_cheap picks test/cheap in both groups: t1 estimated 100.0,
    # t2 measured 5000.0 -> mean 2550.0.
    cheap = split["fixed_cheap"]
    assert cheap["tokens_per_task"] == 2550.0
    assert cheap["success_rate"] == 0.0  # 0.4 < 0.7 floor in both groups


def test_bootstrap_only_dataset_reports_zero_measured_rows() -> None:
    dataset = DatasetV1(
        name="bootstrap-only",
        version=1,
        examples=[
            _example(0, "t1", "test/strong", 0.9, PROVENANCE_BOOTSTRAP,
                     est_total_tokens=200.0),
            _example(1, "t1", "test/cheap", 0.4, PROVENANCE_BOOTSTRAP,
                     est_total_tokens=100.0),
        ],
        benchmark_feature_names=(AA,),
    )
    results = evaluate_routers(dataset, CONFIG, session=None)
    assert results["label_provenance"] == PROVENANCE_BOOTSTRAP
    assert results["measured_token_rows"] == 0
    assert results["estimated_token_rows"] == 2
