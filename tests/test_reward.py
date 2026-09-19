"""Reward computation tests: decision<->execution joins, aggregates, win rates."""

from __future__ import annotations

import pytest

from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.router.reward import (
    RewardError,
    aggregate_rewards,
    compute_rewards,
)


def _store(tmp_path):
    store = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    store.init_schema()
    return store


def _decision(store, session, decision_id: str, phase: str, alternatives: list):
    return store.record_decision(
        session,
        decision_id=decision_id,
        phase=phase,
        selected={"model": "test/model-a", "effort": "low"},
        alternatives=alternatives,
        reason_codes=["cheapest_of_meeting"],
        estimated_tokens=14_000.0,
        estimated_cost=0.05,
        policy_version="pol-test",
    )


def _execution(execution_id: str, decision_id: str, *, success: int, tokens: int,
               latency_ms: int | None = None, phase: str = "apply",
               model: str = "test/model-a", effort: str = "low"):
    return {
        "execution_id": execution_id,
        "phase": phase,
        "model": model,
        "effort": effort,
        "total_tokens": tokens,
        "latency_ms": latency_ms,
        "task_success": success,
        "decision_id": decision_id,
    }


def test_join_and_reward_math(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _decision(store, session, "d1", "apply", [])
        store.record_execution(
            session, _execution("e1", "d1", success=1, tokens=200_000, latency_ms=60_000)
        )
        store.record_execution(
            session, _execution("e2", "d1", success=0, tokens=100_000, latency_ms=None)
        )
        rewards = compute_rewards(session)

    assert [r.execution_id for r in rewards] == ["e1", "e2"]  # deterministic order
    # reward = success - tokens/1M - latency/60k
    assert rewards[0].reward == pytest.approx(1.0 - 0.2 - 1.0)
    assert rewards[1].reward == pytest.approx(0.0 - 0.1 - 0.0)
    assert rewards[0].phase == "apply"
    assert rewards[0].model == "test/model-a"
    assert rewards[0].effort == "low"


def test_success_is_dominant_for_small_executions(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _decision(store, session, "d1", "apply", [])
        store.record_execution(
            session, _execution("e1", "d1", success=1, tokens=50_000)
        )
        store.record_execution(
            session, _execution("e2", "d1", success=0, tokens=1_000)
        )
        rewards = compute_rewards(session)
    success_reward = next(r for r in rewards if r.execution_id == "e1")
    failure_reward = next(r for r in rewards if r.execution_id == "e2")
    assert success_reward.reward > failure_reward.reward


def test_unscored_and_unlinked_executions_are_skipped(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _decision(store, session, "d1", "apply", [])
        store.record_execution(
            session, _execution("e1", "d1", success=1, tokens=1000)
        )
        # No decision_id: not attributable -> skipped, not an error.
        store.record_execution(
            session, {"execution_id": "e2", "phase": "apply", "total_tokens": 500,
                      "task_success": 1}
        )
        # decision_id set but task_success NULL: not scorable -> skipped.
        store.record_execution(
            session, {"execution_id": "e3", "phase": "apply", "total_tokens": 500,
                      "decision_id": "d1"}
        )
        rewards = compute_rewards(session)
    assert [r.execution_id for r in rewards] == ["e1"]


def test_dangling_decision_join_fails_closed(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        store.record_execution(
            session, _execution("e1", "d-missing", success=1, tokens=1000)
        )
        with pytest.raises(RewardError, match="dangling decision join"):
            compute_rewards(session)


def test_invalid_alternatives_payload_fails_closed(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _decision(store, session, "d1", "apply", alternatives="not-a-list")  # type: ignore[arg-type]
        store.record_execution(
            session, _execution("e1", "d1", success=1, tokens=1000)
        )
        with pytest.raises(RewardError, match="alternatives payload"):
            compute_rewards(session)


def test_aggregates_group_by_phase_model_effort(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _decision(store, session, "d1", "apply", [])
        store.record_execution(session, _execution("e1", "d1", success=1, tokens=1000))
        store.record_execution(session, _execution("e2", "d1", success=0, tokens=3000))
        store.record_execution(
            session,
            _execution("e3", "d1", success=1, tokens=500, model="test/model-b"),
        )
        aggregates = aggregate_rewards(compute_rewards(session))

    keys = [(a.phase, a.model, a.effort) for a in aggregates]
    assert keys == [("apply", "test/model-a", "low"), ("apply", "test/model-b", "low")]
    arm_a, arm_b = aggregates
    assert arm_a.executions == 2
    assert arm_a.success_rate == 0.5
    assert arm_a.mean_total_tokens == 2000.0
    assert arm_a.tokens_per_success == 4000.0 / 1  # (1000+3000) / 1 success
    assert arm_b.success_rate == 1.0
    assert arm_b.tokens_per_success == 500.0


def test_tokens_per_success_none_when_never_successful(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _decision(store, session, "d1", "apply", [])
        store.record_execution(session, _execution("e1", "d1", success=0, tokens=1000))
        (agg,) = aggregate_rewards(compute_rewards(session))
    assert agg.tokens_per_success is None
    assert agg.win_rate is None  # decision recorded no alternatives


def test_win_rate_vs_alternatives(tmp_path) -> None:
    """Arm A always beats alternative B (higher mean reward) -> win_rate 1.0."""
    store = _store(tmp_path)
    alternatives = [{"model": "test/model-b", "effort": "low"}]
    with store.session() as session:
        _decision(store, session, "d1", "apply", alternatives)
        # Arm A: 2 successes at low token cost -> mean reward ~0.999.
        store.record_execution(session, _execution("e1", "d1", success=1, tokens=100))
        store.record_execution(session, _execution("e2", "d1", success=1, tokens=200))
        # Arm B (observed via another decision where it was selected): all failures.
        _decision(store, session, "d2", "apply", [])
        store.record_execution(
            session, _execution("e3", "d2", success=0, tokens=100, model="test/model-b")
        )
        aggregates = aggregate_rewards(compute_rewards(session))

    by_arm = {(a.model): a for a in aggregates}
    assert by_arm["test/model-a"].win_rate == 1.0  # beats model-b's mean reward
    assert by_arm["test/model-b"].win_rate is None  # d2 recorded no alternatives


def test_min_executions_filters_arms(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _decision(store, session, "d1", "apply", [])
        store.record_execution(session, _execution("e1", "d1", success=1, tokens=100))
        store.record_execution(
            session, _execution("e2", "d1", success=1, tokens=100, model="test/model-b")
        )
        rewards = compute_rewards(session)
    assert aggregate_rewards(rewards, min_executions=2) == []
    assert len(aggregate_rewards(rewards, min_executions=1)) == 2
