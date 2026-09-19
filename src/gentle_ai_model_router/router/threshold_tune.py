"""Telemetry-driven threshold tuning: pure proposals over RewardAggregates.

Roadmap phase 5 closes the last static piece of the policy loop: the
per-phase ``threshold_quality`` floors. ``router/calibrate.py`` answers the
same question from REGISTRY benchmark priors and is read-only by design; this
module is its MEASURED complement — proposals computed from shim
:class:`~gentle_ai_model_router.router.reward.RewardAggregate` evidence — and,
like calibrate, the quality model is IMPORTED from router/policy.py
(``effort_quality`` + ``EFFORT_RANK``) so a proposed threshold sits on exactly
the quality scale the routing policy computes. No math is duplicated.

Frontier rule per phase (deterministic, cheapest-first):

    1. Pool the phase's arms per effort level (the Effort taxonomy ladder:
       off < minimal < low < medium < high < xhigh < max).
    2. An effort level is *admissible* when its pooled executions >=
       ``config.bandit.min_executions_before_exploit`` (the same cold-start
       evidence bar the bandit uses — one evidence standard, no drift).
    3. The frontier point is the CHEAPEST admissible effort whose pooled
       success_rate >= ``success_floor``.
    4. The proposed ``threshold_quality`` is the calibrated prior quality of
       that effort level: ``effort_quality(pooled_success_rate, effort,
       policy)`` — the policy's own quality curve applied to the MEASURED
       success rate instead of a benchmark prior.

Decisions: ``upgrade`` / ``downgrade`` (proposed vs current threshold),
``uphold`` (frontier found, threshold already right, or evidence exists but
no effort meets the floor — never changed), and ``insufficient_evidence``
(never changed). Only ``upgrade``/``downgrade`` proposals may be applied to
router.yaml — see the explicit applier in
:mod:`gentle_ai_model_router.integration.router_yaml_adapter`.

This module is PURE: no I/O, no registry/shim access, no clock, no
randomness. Fail closed with typed :class:`ThresholdTuneError` on invalid
input (bad floor, bad evidence, unknown effort vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from gentle_ai_model_router.router.config import RouterConfig
from gentle_ai_model_router.router.policy import EFFORT_RANK, effort_quality
from gentle_ai_model_router.router.reward import RewardAggregate

# Difference below which a proposed threshold counts as "equal" to the
# current one (float-noise guard around the 0.98 effort ceiling).
_EQUAL_EPSILON = 1e-9

ProposalKind = Literal["upgrade", "downgrade", "uphold", "insufficient_evidence"]

PROPOSAL_KINDS: tuple[str, ...] = ("upgrade", "downgrade", "uphold", "insufficient_evidence")


class ThresholdTuneError(Exception):
    """Fatal tuner failure (invalid input, unknown effort vocabulary). Fails closed."""


@dataclass(frozen=True)
class ThresholdProposal:
    """One phase's evidence-backed threshold decision.

    ``kind`` is ``upgrade`` / ``downgrade`` (applyable), ``uphold`` (frontier
    found and the current threshold already matches it, or evidence exists but
    no effort meets the floor), or ``insufficient_evidence`` (below
    ``min_executions`` — the phase is NEVER changed). ``selected_effort`` /
    ``success_rate`` / ``tokens_per_success`` describe the frontier point and
    are ``None`` when no admissible effort exists.
    """

    phase: str
    kind: ProposalKind
    current_threshold: float
    proposed_threshold: float
    selected_effort: str | None
    executions: int  # pooled executions at the selected effort level (0 = none)
    success_rate: float | None  # pooled success rate at the selected effort
    tokens_per_success: float | None  # pooled tokens per success at the selected effort
    success_floor: float
    min_executions: int
    note: str


@dataclass(frozen=True)
class _EffortLevel:
    """Pooled per-effort evidence for one phase."""

    effort: str
    executions: int
    successes: int
    total_tokens: float


def _validate_aggregate(aggregate: RewardAggregate) -> None:
    if aggregate.executions < 1:
        raise ThresholdTuneError(
            f"phase {aggregate.phase!r} arm ({aggregate.model!r}, {aggregate.effort!r}): "
            f"executions must be >= 1, got {aggregate.executions}"
        )
    if not 0.0 <= aggregate.success_rate <= 1.0:
        raise ThresholdTuneError(
            f"phase {aggregate.phase!r} arm ({aggregate.model!r}, {aggregate.effort!r}): "
            f"success_rate must be in [0, 1], got {aggregate.success_rate}"
        )
    if aggregate.effort not in EFFORT_RANK:
        raise ThresholdTuneError(
            f"phase {aggregate.phase!r}: unknown effort {aggregate.effort!r} "
            f"(expected one of: {', '.join(EFFORT_RANK)})"
        )


def _pool_effort_levels(aggregates: list[RewardAggregate]) -> list[_EffortLevel]:
    """Pool arm evidence per effort level, sorted by the Effort ladder.

    Deterministic: levels are ordered by (EFFORT_RANK, effort name); pooled
    counters are plain sums over the arms of that level.
    """
    pooled: dict[str, _EffortLevel] = {}
    for aggregate in aggregates:
        _validate_aggregate(aggregate)
        level = pooled.get(aggregate.effort)
        if level is None:
            level = _EffortLevel(
                effort=aggregate.effort,
                executions=0,
                successes=0,
                total_tokens=0.0,
            )
            pooled[aggregate.effort] = level
        # success_rate == successes / executions by construction of
        # aggregate_rewards; round() restores the integer count exactly.
        successes = round(aggregate.success_rate * aggregate.executions)
        pooled[aggregate.effort] = _EffortLevel(
            effort=level.effort,
            executions=level.executions + aggregate.executions,
            successes=level.successes + successes,
            total_tokens=level.total_tokens + aggregate.mean_total_tokens * aggregate.executions,
        )
    return [pooled[name] for name in sorted(pooled, key=lambda e: (EFFORT_RANK[e], e))]


def _propose_phase(
    phase: str,
    aggregates: list[RewardAggregate],
    *,
    success_floor: float,
    config: RouterConfig,
) -> ThresholdProposal:
    current = config.phase_config(phase).threshold_quality
    min_executions = config.bandit.min_executions_before_exploit
    if min_executions < 1:
        raise ThresholdTuneError(f"min_executions must be >= 1, got {min_executions}")

    levels = _pool_effort_levels(aggregates)
    admissible = [level for level in levels if level.executions >= min_executions]
    if not admissible:
        observed = sum(level.executions for level in levels)
        return ThresholdProposal(
            phase=phase,
            kind="insufficient_evidence",
            current_threshold=current,
            proposed_threshold=current,
            selected_effort=None,
            executions=0,
            success_rate=None,
            tokens_per_success=None,
            success_floor=success_floor,
            min_executions=min_executions,
            note=(
                f"no effort level reaches {min_executions} execution(s) "
                f"({observed} observed) — phase left unchanged"
            ),
        )

    # Cheapest-first frontier: the ladder sort makes the first level meeting
    # the success floor the cheapest one; ties are impossible (strict order).
    frontier = next(
        (level for level in admissible if level.successes / level.executions >= success_floor),
        None,
    )
    if frontier is None:
        best = max(admissible, key=lambda level: level.successes / level.executions)
        return ThresholdProposal(
            phase=phase,
            kind="uphold",
            current_threshold=current,
            proposed_threshold=current,
            selected_effort=None,
            executions=0,
            success_rate=None,
            tokens_per_success=None,
            success_floor=success_floor,
            min_executions=min_executions,
            note=(
                f"no admissible effort meets the {success_floor:.2f} success floor "
                f"(best observed: {best.effort} at "
                f"{best.successes / best.executions:.3f}) — phase left unchanged"
            ),
        )

    pooled_rate = frontier.successes / frontier.executions
    pooled_tps = (
        frontier.total_tokens / frontier.successes if frontier.successes > 0 else None
    )
    # The policy's own quality curve applied to the MEASURED success rate:
    # the proposed floor is what the cheapest sufficient effort demonstrably
    # delivers, expressed on the policy's quality scale.
    proposed = effort_quality(pooled_rate, frontier.effort, config.policy)
    if proposed > current + _EQUAL_EPSILON:
        kind: ProposalKind = "upgrade"
        note = f"{frontier.effort} meets the floor at {pooled_rate:.3f} success"
    elif proposed < current - _EQUAL_EPSILON:
        kind = "downgrade"
        note = f"{frontier.effort} meets the floor at {pooled_rate:.3f} success"
    else:
        kind = "uphold"
        note = f"{frontier.effort} already defines the current threshold"
    return ThresholdProposal(
        phase=phase,
        kind=kind,
        current_threshold=current,
        proposed_threshold=proposed,
        selected_effort=frontier.effort,
        executions=frontier.executions,
        success_rate=pooled_rate,
        tokens_per_success=pooled_tps,
        success_floor=success_floor,
        min_executions=min_executions,
        note=note,
    )


def propose_thresholds(
    aggregates: dict[str, list[RewardAggregate]],
    *,
    success_floor: float = 0.8,
    config: RouterConfig,
) -> list[ThresholdProposal]:
    """Propose per-phase ``threshold_quality`` values from measured rewards.

    ``aggregates`` maps a phase name (``explore`` or ``sdd-explore`` accepted;
    keys are normalized) to its reward aggregates — the bandit's input.
    ``success_floor`` is the minimum pooled success rate an effort level must
    demonstrate; ``config.bandit.min_executions_before_exploit`` is the
    evidence bar (shared with the bandit's cold start).

    Pure and deterministic: proposals are sorted by phase name, and repeated
    calls over identical input return identical output.

    Raises:
        ThresholdTuneError: ``success_floor`` outside (0, 1], negative or
            malformed evidence, or an effort value outside the Effort
            taxonomy (fail closed, never guessed).
    """
    if not 0.0 < success_floor <= 1.0:
        raise ThresholdTuneError(f"success_floor must be in (0, 1], got {success_floor}")

    proposals = [
        _propose_phase(
            phase.removeprefix("sdd-"),
            arms,
            success_floor=success_floor,
            config=config,
        )
        for phase, arms in sorted(aggregates.items())
    ]
    return proposals
