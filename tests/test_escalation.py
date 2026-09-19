"""Escalation ladder tests: pure function, four paths per spec."""

from __future__ import annotations

from gentle_ai_model_router.router.decision import Alternative, Decision
from gentle_ai_model_router.router.escalation import (
    EscalationPolicy,
    EscalationState,
    FailureSignal,
    next_candidate,
)

FAIL = FailureSignal(code="quality_floor_missed")


def _decision(
    model: str = "test/a",
    effort: str = "low",
    alternatives: tuple[Alternative, ...] = (),
) -> Decision:
    return Decision(
        phase="design",
        model=model,
        provider="test",
        deployment="default",
        effort=effort,
        score=0.9,
        quality=0.9,
        alternatives=alternatives,
        reason_codes=(),
        estimated_tokens=10_000,
        estimated_cost=0.01,
        policy_version="pol-test",
    )


def _alt(model: str, effort: str) -> Alternative:
    return Alternative(
        model=model,
        provider="test",
        deployment="default",
        effort=effort,
        score=0.8,
        quality=0.85,
        estimated_tokens=20_000,
    )


def test_pass_path_no_escalation() -> None:
    decision = _decision()
    state = EscalationState.from_decision(decision)
    result = next_candidate(state, None, decision, EscalationPolicy())
    assert result.escalated is False
    assert result.exhausted is False
    assert result.model == "test/a"
    assert result.effort == "low"
    assert result.escalation_count == 0
    assert result.initial_model == "test/a"
    assert result.initial_effort == "low"


def test_fail_escalates_effort_first_within_model() -> None:
    decision = _decision()
    state = EscalationState.from_decision(decision)
    efforts = {"test/a": ["low", "medium", "high"], "test/b": ["medium"]}
    result = next_candidate(state, FAIL, decision, EscalationPolicy(), efforts)
    assert result.escalated is True
    assert result.model == "test/a"  # same model ...
    assert result.effort == "medium"  # ... one effort up
    assert result.escalation_count == 1
    assert result.initial_model == "test/a"  # provenance keeps the origin
    assert result.initial_effort == "low"
    assert any(code.startswith("effort_up:low->medium") for code in result.reason_codes)


def test_fail_twice_changes_model_when_effort_ladder_exhausted() -> None:
    decision = _decision(alternatives=(_alt("test/b", "medium"), _alt("test/c", "low")))
    state = EscalationState.from_decision(decision)
    efforts = {"test/a": ["low", "medium"], "test/b": ["medium", "high"]}

    first = next_candidate(state, FAIL, decision, EscalationPolicy(), efforts)
    assert (first.model, first.effort) == ("test/a", "medium")

    second_state = EscalationState(
        current_model=first.model,
        current_effort=first.effort,
        escalation_count=first.escalation_count,
        initial_model=first.initial_model,
        initial_effort=first.initial_effort,
    )
    second = next_candidate(second_state, FAIL, decision, EscalationPolicy(), efforts)
    assert second.escalated is True
    assert second.model == "test/b"  # model change after effort ladder tops out
    assert second.effort == "medium"
    assert second.escalation_count == 2
    assert second.initial_model == "test/a"
    assert second.initial_effort == "low"


def test_cap_respected_and_exhausted() -> None:
    decision = _decision(alternatives=(_alt("test/b", "medium"),))
    state = EscalationState.from_decision(decision)
    policy = EscalationPolicy(max_escalations=1)
    efforts = {"test/a": ["low", "medium"], "test/b": ["medium"]}

    first = next_candidate(state, FAIL, decision, policy, efforts)
    assert first.escalated and first.escalation_count == 1

    second_state = EscalationState(
        current_model=first.model,
        current_effort=first.effort,
        escalation_count=first.escalation_count,
        initial_model=first.initial_model,
        initial_effort=first.initial_effort,
    )
    second = next_candidate(second_state, FAIL, decision, policy, efforts)
    assert second.escalated is False
    assert second.exhausted is True
    assert second.escalation_count == 1  # cap: no further escalation
    assert (second.model, second.effort) == ("test/a", "medium")
    assert "exhausted:max_escalations" in second.reason_codes


def test_alternatives_first_mode() -> None:
    """escalate_effort_first=False climbs alternatives before effort."""
    decision = _decision(alternatives=(_alt("test/b", "medium"),))
    state = EscalationState.from_decision(decision)
    policy = EscalationPolicy(escalate_effort_first=False)
    efforts = {"test/a": ["low", "high"], "test/b": ["medium"]}
    result = next_candidate(state, FAIL, decision, policy, efforts)
    assert (result.model, result.effort) == ("test/b", "medium")
    assert any(code.startswith("model_change:") for code in result.reason_codes)


def test_unsupported_effort_from_variants_cache_skips_alternative() -> None:
    """An alternative whose effort the variants cache does not support is
    skipped, and the ladder reports exhaustion instead of guessing."""
    decision = _decision(alternatives=(_alt("test/b", "high"),))
    state = EscalationState.from_decision(decision)
    efforts = {"test/a": ["low"], "test/b": ["medium"]}  # b/high unsupported
    result = next_candidate(state, FAIL, decision, EscalationPolicy(), efforts)
    assert result.escalated is False
    assert result.exhausted is True
    assert "skip_unsupported:test/b#high" in result.reason_codes
