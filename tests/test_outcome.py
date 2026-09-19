"""Outcome rubric tests: per-phase scoring math + idempotent backfill."""

from __future__ import annotations

import pytest

from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.integration.outcome import (
    OutcomeRubricError,
    OutcomeScore,
    apply_outcomes,
    score_execution,
)


def _store(tmp_path):
    store = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    store.init_schema()
    return store


def _execution(execution_id: str, phase: str = "apply", **kw):
    payload = {
        "execution_id": execution_id,
        "phase": phase,
        "model": "test/model-a",
        "effort": "low",
        "total_tokens": 1000,
        "tool_errors": 0,
        "escalation_count": 0,
    }
    payload.update(kw)
    return payload


# --------------------------------------------------------------------------- #
# Rubric math
# --------------------------------------------------------------------------- #


def test_apply_all_tests_pass_is_success_with_full_quality() -> None:
    score = score_execution(
        "apply", tests_passed=10, tests_failed=0, tool_errors=0, escalation_count=0
    )
    assert score == OutcomeScore(task_success=1, quality_score=100.0)


def test_apply_failed_test_is_failure_capped_quality() -> None:
    score = score_execution("apply", tests_passed=9, tests_failed=1, tool_errors=0)
    assert score.task_success == 0
    # Gate failed -> quality capped at 49 even though the ratio is high.
    assert score.quality_score == 49.0


def test_apply_tool_error_gates_success() -> None:
    score = score_execution(
        "apply", tests_passed=5, tests_failed=0, tool_errors=2, escalation_count=0
    )
    assert score.task_success == 0
    # Gate failed -> capped at 49 (raw 100 - 2*10 tool-error deduction = 80).
    assert score.quality_score == 49.0


def test_apply_build_failure_gates_success() -> None:
    score = score_execution(
        "apply", tests_passed=5, tests_failed=0, tool_errors=0, build_green=False
    )
    assert score.task_success == 0
    assert score.quality_score <= 49.0


def test_apply_no_test_evidence_fails_closed() -> None:
    score = score_execution("apply", tool_errors=0)
    assert score.task_success == 0
    assert score.quality_score == 20.0


def test_verify_verification_failed_gates_success() -> None:
    score = score_execution(
        "verify", tests_passed=4, tests_failed=0, verification_passed=False
    )
    assert score.task_success == 0
    assert score.quality_score <= 49.0


def test_verify_passes_with_tests_and_verification() -> None:
    score = score_execution(
        "verify", tests_passed=4, tests_failed=0, verification_passed=True
    )
    assert score.task_success == 1
    assert score.quality_score == 100.0


def test_explore_heuristic_clean_run_is_success() -> None:
    score = score_execution("explore", tool_errors=0, escalation_count=0, latency_ms=10_000)
    assert score.task_success == 1
    assert score.quality_score == 100.0


def test_explore_heuristic_penalties_and_latency() -> None:
    score = score_execution(
        "explore", tool_errors=2, escalation_count=1, latency_ms=120_000
    )
    assert score.task_success == 0  # any tool error fails the heuristic phase
    # 100 - 2*15 (tool errors) - 1*10 (escalation) - 10 (above 90s baseline)
    assert score.quality_score == pytest.approx(50.0)


def test_heuristic_quality_is_clamped_to_0_100() -> None:
    score = score_execution(
        "design", tool_errors=99, escalation_count=99, latency_ms=999_999_999
    )
    assert score.quality_score == 0.0
    score = score_execution("design", quality_score=250.0)
    assert score.quality_score == 100.0


def test_caller_task_success_is_respected() -> None:
    # Rubric would fail this (failed tests); the caller's 1 wins.
    score = score_execution("apply", tests_passed=0, tests_failed=3, task_success=1)
    assert score.task_success == 1


def test_caller_quality_score_is_respected_and_clamped() -> None:
    score = score_execution("apply", tests_passed=10, tests_failed=0, quality_score=88.5)
    assert score.quality_score == 88.5
    score = score_execution("apply", tests_passed=10, tests_failed=0, quality_score=-5.0)
    assert score.quality_score == 0.0


def test_invalid_task_success_and_phase_fail_closed() -> None:
    with pytest.raises(OutcomeRubricError):
        score_execution("apply", task_success=2)
    with pytest.raises(OutcomeRubricError):
        score_execution("bogus-phase", tool_errors=0)
    with pytest.raises(OutcomeRubricError):
        score_execution("apply", tool_errors=-1)


def test_sdd_prefixed_phase_accepted() -> None:
    score = score_execution("sdd-explore", tool_errors=0)
    assert score.task_success == 1


# --------------------------------------------------------------------------- #
# apply_outcomes backfill
# --------------------------------------------------------------------------- #


def test_apply_outcomes_populates_null_rows_and_respects_caller_values(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        store.record_execution(
            session, _execution("e1", "apply", tests_passed=8, tests_failed=0)
        )
        store.record_execution(
            session, _execution("e2", "apply", tests_passed=0, tests_failed=2)
        )
        # quality_score already caller-populated; task_success still NULL ->
        # the row IS scored, and the caller's quality wins over the rubric.
        store.record_execution(
            session,
            _execution("e3", "explore", tool_errors=0, quality_score=77.0),
        )
        counts = apply_outcomes(session)
    assert counts == {"scored": 3}

    with store.session() as session:
        rows = {
            row.execution_id: row
            for row in session.query(telemetry_shim.ExecutionRecord).all()
        }
        assert rows["e1"].task_success == 1
        assert rows["e1"].quality_score == 100.0
        assert rows["e2"].task_success == 0
        assert rows["e3"].task_success == 1
        assert rows["e3"].quality_score == 77.0  # caller value untouched


def test_apply_outcomes_is_idempotent(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        store.record_execution(session, _execution("e1", "apply", tests_passed=3))
        store.record_execution(session, _execution("e2", "explore", tool_errors=1))
        first = apply_outcomes(session)
    assert first == {"scored": 2}
    with store.session() as session:
        second = apply_outcomes(session)
    assert second == {"scored": 0}  # nothing re-scored


def test_apply_outcomes_phase_filter(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        store.record_execution(session, _execution("e1", "apply", tests_passed=3))
        store.record_execution(session, _execution("e2", "verify", tests_passed=3))
        counts = apply_outcomes(session, phase="apply")
    assert counts == {"scored": 1}
    with store.session() as session:
        rows = {
            row.execution_id: row.task_success
            for row in session.query(telemetry_shim.ExecutionRecord).all()
        }
        assert rows == {"e1": 1, "e2": None}
