"""Controlled escalation: climb the candidate ladder when quality is at risk.

Design (docs/architecture.md decision #4): start cheap, escalate only on a
failure signal, never downgrade below the phase floor. The policy is a PURE
FUNCTION: given the current routing state, a failure signal, and the effort
levels the variants cache actually supports, it produces the next candidate
or reports the ladder as exhausted. No I/O, no randomness — receipts first.

Ladder order: effort-up within the same model first (cheapest climb), then
the decision's alternatives (already quality-vetted runners-up). Whether
effort-up happens first is configurable (``escalate_effort_first``) so the
ladder can be A/B-tested against alternatives-first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from gentle_ai_model_router.registry.normalize import Effort
from gentle_ai_model_router.router.decision import Decision

_EFFORT_ORDER: tuple[str, ...] = tuple(level.value for level in Effort)


class EscalationPolicy(BaseModel):
    """Escalation ladder parameters (mirror of router.yaml ``escalation:``)."""

    max_escalations: int = 2
    escalate_effort_first: bool = True


@dataclass(frozen=True)
class FailureSignal:
    """Why the current candidate is being replaced."""

    code: str  # e.g. "quality_floor_missed", "tool_error", "timeout"
    detail: str = ""


@dataclass(frozen=True)
class EscalationState:
    """Where the ladder currently stands for one phase run."""

    current_model: str
    current_effort: str
    escalation_count: int = 0
    initial_model: str | None = None
    initial_effort: str | None = None

    @classmethod
    def from_decision(cls, decision: Decision) -> EscalationState:
        return cls(
            current_model=decision.model,
            current_effort=decision.effort,
            escalation_count=0,
            initial_model=decision.model,
            initial_effort=decision.effort,
        )


@dataclass(frozen=True)
class EscalationResult:
    """Outcome of one escalation step (provenance included)."""

    model: str
    effort: str
    initial_model: str
    initial_effort: str
    escalation_count: int
    escalated: bool
    exhausted: bool
    reason_codes: tuple[str, ...] = field(default_factory=tuple)


def _next_effort_up(current: str, available: list[str]) -> str | None:
    """Lowest available effort strictly above ``current`` (internal taxonomy)."""
    try:
        current_idx = _EFFORT_ORDER.index(current)
    except ValueError:
        return None
    for level in _EFFORT_ORDER[current_idx + 1 :]:
        if level in available:
            return level
    return None


def next_candidate(
    state: EscalationState,
    failure: FailureSignal | None,
    decision: Decision,
    policy: EscalationPolicy,
    available_efforts: dict[str, list[str]] | None = None,
) -> EscalationResult:
    """Pure escalation step.

    - No failure → the state passes through unchanged.
    - Failure with headroom → climb: effort-up within the same model when the
      variants cache supports a higher level (only if
      ``escalate_effort_first``), otherwise the first not-yet-tried
      alternative of the decision.
    - Headroom exhausted (cap reached, no higher effort, no alternatives) →
      ``exhausted=True`` and the current candidate is kept (fail closed).
    """
    initial_model = state.initial_model or state.current_model
    initial_effort = state.initial_effort or state.current_effort
    base = EscalationResult(
        model=state.current_model,
        effort=state.current_effort,
        initial_model=initial_model,
        initial_effort=initial_effort,
        escalation_count=state.escalation_count,
        escalated=False,
        exhausted=False,
        reason_codes=(f"pass:{failure.code}" if failure else "pass:no_failure",),
    )
    if failure is None:
        return base
    if state.escalation_count >= policy.max_escalations:
        return EscalationResult(
            model=state.current_model,
            effort=state.current_effort,
            initial_model=initial_model,
            initial_effort=initial_effort,
            escalation_count=state.escalation_count,
            escalated=False,
            exhausted=True,
            reason_codes=("exhausted:max_escalations", f"failure:{failure.code}"),
        )

    reasons = [f"escalate:{failure.code}"]
    tried: set[tuple[str, str]] = {
        (initial_model, initial_effort),
        (state.current_model, state.current_effort),
    }

    def _step(model: str, effort: str, reason: str) -> EscalationResult:
        return EscalationResult(
            model=model,
            effort=effort,
            initial_model=initial_model,
            initial_effort=initial_effort,
            escalation_count=state.escalation_count + 1,
            escalated=True,
            exhausted=False,
            reason_codes=tuple(reasons + [reason]),
        )

    if policy.escalate_effort_first:
        higher = _next_effort_up(
            state.current_effort, (available_efforts or {}).get(state.current_model, [])
        )
        if higher is not None and (state.current_model, higher) not in tried:
            return _step(state.current_model, higher, f"effort_up:{state.current_effort}->{higher}")

    for alt in decision.alternatives:
        if (alt.model, alt.effort) in tried:
            continue
        supported = (available_efforts or {}).get(alt.model, [alt.effort])
        if available_efforts is not None and alt.effort not in supported:
            reasons.append(f"skip_unsupported:{alt.model}#{alt.effort}")
            continue
        return _step(alt.model, alt.effort, f"model_change:{state.current_model}->{alt.model}")

    if not policy.escalate_effort_first:
        higher = _next_effort_up(
            state.current_effort, (available_efforts or {}).get(state.current_model, [])
        )
        if higher is not None and (state.current_model, higher) not in tried:
            return _step(state.current_model, higher, f"effort_up:{state.current_effort}->{higher}")

    return EscalationResult(
        model=state.current_model,
        effort=state.current_effort,
        initial_model=initial_model,
        initial_effort=initial_effort,
        escalation_count=state.escalation_count,
        escalated=False,
        exhausted=True,
        reason_codes=tuple(reasons + ["exhausted:no_more_candidates"]),
    )
