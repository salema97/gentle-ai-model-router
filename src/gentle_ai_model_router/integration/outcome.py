"""Per-phase outcome rubric: raw execution signals -> task_success / quality_score.

This module implements the scoring rubric that ``telemetry_shim.PHASE_SIGNALS``
documented as DESIGN/pending. It closes the first half of the bandit loop:
executions arrive with raw signals (tests, tool errors, escalations, latency)
and leave with a binary ``task_success`` (0/1) and a ``quality_score`` (0-100)
that :mod:`gentle_ai_model_router.router.reward` turns into arm rewards.

Design rules:

- PURE functions only: :func:`score_execution` has no I/O and no randomness.
- Fail closed on missing signals: the apply/verify phases hard-gate success on
  test/build signals; an execution with no test evidence for a test-gated
  phase scores ``task_success=0``, never a guessed 1.
- Caller-populated outcomes win: when the ingest payload already carries
  ``task_success`` (0/1) and/or ``quality_score``, the rubric respects them.
- :func:`apply_outcomes` backfills only NULL outcome columns, so re-running it
  is idempotent (second pass scores 0 rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.integration.telemetry_shim import ExecutionRecord
from gentle_ai_model_router.router.decision import CANONICAL_PHASES

# Latency reference points per phase (ms). Above the baseline an execution is
# "slow" and the heuristic rubric deducts quality; the table is a documented
# prior, not a measured distribution.
PHASE_LATENCY_BASELINE_MS: dict[str, float] = {
    "init": 30_000.0,
    "explore": 90_000.0,
    "research": 60_000.0,
    "propose": 60_000.0,
    "spec": 90_000.0,
    "design": 120_000.0,
    "tasks": 60_000.0,
    "apply": 180_000.0,
    "verify": 120_000.0,
    "archive": 30_000.0,
    "onboard": 60_000.0,
}

# Phases whose success is hard-gated on test/build signals.
_TEST_GATED_PHASES = {"apply", "verify"}

# Quality of an execution in a test-gated phase with no test evidence at all.
_NO_TEST_EVIDENCE_QUALITY = 20.0

# Cap for quality when a hard gate failed: a failed execution can never look
# "good" on the 0-100 scale.
_FAILED_GATE_QUALITY_CAP = 49.0

_TOOL_ERROR_QUALITY_STEP = 15.0  # heuristic phases: -15 per tool error
_TOOL_ERROR_QUALITY_CAP = 60.0
_ESCALATION_QUALITY_STEP = 10.0
_ESCALATION_QUALITY_CAP = 30.0
_SLOW_LATENCY_QUALITY_PENALTY = 10.0


class OutcomeRubricError(Exception):
    """Invalid rubric input (unknown phase, non-finite numbers). Fails closed."""


@dataclass(frozen=True)
class OutcomeScore:
    """Scored outcome for one execution."""

    task_success: int  # 0/1
    quality_score: float  # 0-100


def _clamp_quality(value: float) -> float:
    return min(100.0, max(0.0, value))


def _check_phase(phase: str) -> str:
    name = phase.removeprefix("sdd-")
    if name not in CANONICAL_PHASES:
        raise OutcomeRubricError(
            f"unknown phase '{phase}' (expected one of: {', '.join(CANONICAL_PHASES)})"
        )
    return name


def _validate_numbers(**signals: float | int | None) -> None:
    for key, value in signals.items():
        if value is None:
            continue
        if isinstance(value, float) and not isfinite(value):
            raise OutcomeRubricError(f"signal '{key}' must be finite, got {value!r}")
        if value < 0:
            raise OutcomeRubricError(f"signal '{key}' must be >= 0, got {value!r}")


def _test_gated_score(
    *,
    tests_passed: int | None,
    tests_failed: int | None,
    tool_errors: int,
    escalation_count: int,
    build_green: bool | None,
    verification_passed: bool | None,
) -> OutcomeScore:
    """apply/verify rubric: tests dominate, tool errors and build gate success."""
    passed = tests_passed or 0
    failed = tests_failed or 0
    total = passed + failed
    ratio = passed / total if total > 0 else None

    gate_failed = (
        (build_green is False)
        or (verification_passed is False)
        or (failed > 0)
        or (tool_errors > 0)
        or (total == 0)  # no test evidence in a test-gated phase -> fail closed
    )
    success = 0 if gate_failed else 1

    if ratio is None:
        quality = _NO_TEST_EVIDENCE_QUALITY
    else:
        quality = 100.0 * ratio
    quality -= 10.0 * min(tool_errors, 5)
    quality -= 5.0 * escalation_count
    if gate_failed:
        quality = min(quality, _FAILED_GATE_QUALITY_CAP)
    return OutcomeScore(task_success=success, quality_score=_clamp_quality(quality))


def _heuristic_score(
    *,
    phase: str,
    tool_errors: int,
    escalation_count: int,
    latency_ms: float | None,
) -> OutcomeScore:
    """Rubric for signal-poor phases (explore/spec/design/...): penalty model."""
    quality = 100.0
    quality -= _TOOL_ERROR_QUALITY_STEP * min(tool_errors, 4)
    quality -= _ESCALATION_QUALITY_STEP * min(escalation_count, 3)
    baseline = PHASE_LATENCY_BASELINE_MS.get(phase)
    if latency_ms is not None and baseline is not None and latency_ms > baseline:
        quality -= _SLOW_LATENCY_QUALITY_PENALTY
    success = 0 if tool_errors > 0 else 1
    return OutcomeScore(task_success=success, quality_score=_clamp_quality(quality))


def score_execution(
    phase: str,
    *,
    tests_passed: int | None = None,
    tests_failed: int | None = None,
    tool_errors: int = 0,
    escalation_count: int = 0,
    latency_ms: float | None = None,
    task_success: int | None = None,
    quality_score: float | None = None,
    build_green: bool | None = None,
    verification_passed: bool | None = None,
) -> OutcomeScore:
    """Score one execution's raw signals into an :class:`OutcomeScore`.

    Per-phase rubric (documented against ``PHASE_SIGNALS``):

    - ``apply``: ``tests_passed / (passed + failed)`` drives quality; success
      is hard-gated on ``tests_failed == 0``, ``tool_errors == 0``, and
      ``build_green is not False``.
    - ``verify``: the apply rubric plus a ``verification_passed is False`` gate.
    - ``explore`` / ``propose`` / ``spec`` / ``design`` / ``tasks`` (and the
      remaining phases): heuristic — start at 100, deduct per tool error, per
      escalation, and for latency above the phase baseline.

    Caller-provided outcomes always win: a ``task_success`` of 0/1 and/or a
    ``quality_score`` in [0, 100] supplied by the caller is respected and
    only clamped/validated, never recomputed.
    """
    name = _check_phase(phase)
    _validate_numbers(
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        tool_errors=tool_errors,
        escalation_count=escalation_count,
        latency_ms=latency_ms,
    )
    if task_success is not None and task_success not in (0, 1):
        raise OutcomeRubricError(f"task_success must be 0 or 1, got {task_success!r}")

    if name in _TEST_GATED_PHASES:
        scored = _test_gated_score(
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tool_errors=tool_errors,
            escalation_count=escalation_count,
            build_green=build_green,
            verification_passed=verification_passed,
        )
    else:
        scored = _heuristic_score(
            phase=name,
            tool_errors=tool_errors,
            escalation_count=escalation_count,
            latency_ms=latency_ms,
        )

    final_success = scored.task_success if task_success is None else int(task_success)
    final_quality = (
        scored.quality_score if quality_score is None else _clamp_quality(float(quality_score))
    )
    return OutcomeScore(task_success=final_success, quality_score=final_quality)


def apply_outcomes(
    session: Session,
    *,
    phase: str | None = None,
) -> dict[str, int]:
    """Backfill NULL ``task_success`` / ``quality_score`` rows. Idempotent.

    Only rows where the outcome is still unknown (``task_success IS NULL``)
    are scored; rows already populated by the caller (or a previous pass) are
    left untouched. Returns ``{"scored": <n>}``.
    """
    stmt = select(ExecutionRecord).where(ExecutionRecord.task_success.is_(None))
    if phase is not None:
        stmt = stmt.where(ExecutionRecord.phase == phase.removeprefix("sdd-"))
    rows = session.execute(stmt).scalars().all()
    scored = 0
    for row in rows:
        outcome = score_execution(
            row.phase,
            tests_passed=row.tests_passed,
            tests_failed=row.tests_failed,
            tool_errors=row.tool_errors or 0,
            escalation_count=row.escalation_count or 0,
            latency_ms=row.latency_ms,
            task_success=row.task_success,
            quality_score=row.quality_score,
        )
        row.task_success = outcome.task_success
        row.quality_score = outcome.quality_score
        scored += 1
    session.flush()
    return {"scored": scored}
