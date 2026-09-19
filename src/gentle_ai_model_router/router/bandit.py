"""Constrained UCB-style bandit over (model, deployment, effort) arms.

Roadmap phase 4 closes the loop: decision -> execution -> reward -> better
decision. The bandit consumes :class:`~gentle_ai_model_router.router.reward.RewardAggregate`
rows and reorders the threshold-meeting candidate set produced by
:func:`~gentle_ai_model_router.router.policy.rank_candidates`. Exploration
happens ONLY inside that constraint-satisfying set: an arm outside the input
ranking can never be selected (the ranking is the feasibility certificate).

Score (deterministic UCB — no randomness is used, so the seed is reserved
for future randomized tie-breaking and versioned into ``bandit_version``):

    ucb = mean_reward + exploration_weight * sqrt(ln(total_executions) / arm_executions)

Arm classes, in output order:

1. Unobserved arms (no reward aggregate) — classic UCB front: they must be
   tried before the bandit can compare them. Kept in the input ranking order.
2. Exploitable arms (``success_rate >= quality_floor``) — sorted by UCB
   score descending, ties broken by the input ranking order.
3. Below-floor arms (``success_rate < quality_floor``) — demoted to the tail
   in the input ranking order. The quality floor is the constraint in
   "minimize tokens_per_success subject to quality floors".

Cold start: fewer than ``min_executions_before_exploit`` total executions
observed for the phase (or no aggregates at all) — the ranking is returned
UNCHANGED (byte-identical: same ``CandidateRanking`` object) with reason
code ``bandit:cold_start_fallback``. Routing without telemetry evidence is
exactly today's deterministic behavior.

Every bandit-influenced decision carries a ``bandit:*`` reason code, and the
effective configuration is fingerprinted by :func:`bandit_version` exactly
like ``policy_version``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, replace

from pydantic import BaseModel, Field

from gentle_ai_model_router.router.policy import CandidateRanking, RankedCandidate
from gentle_ai_model_router.router.reward import RewardAggregate

logger = logging.getLogger(__name__)

REASON_UCB = "bandit:ucb"
REASON_COLD_START = "bandit:cold_start_fallback"


class BanditError(Exception):
    """Fatal bandit failure (invalid config input). Fails closed."""


class BanditConfig(BaseModel):
    """Constrained bandit parameters.

    ``seed`` is currently unused by the deterministic UCB score; it is kept
    in the config (and hashed into ``bandit_version``) so any future
    randomized tie-breaking is seeded and reproducible by construction.
    """

    exploration_weight: float = 1.0
    min_executions_before_exploit: int = 3
    seed: int = 42
    quality_floor: float = 0.5  # minimum success_rate to stay UCB-eligible
    phases: dict[str, float] = Field(
        default_factory=dict
    )  # optional per-phase quality_floor overrides


@dataclass(frozen=True)
class BanditResult:
    """Outcome of one bandit consultation (provenance included)."""

    ranking: CandidateRanking
    applied: bool  # True when UCB reordering happened (reason REASON_UCB)
    reason_code: str  # REASON_UCB | REASON_COLD_START
    bandit_version: str


def bandit_version(config: BanditConfig) -> str:
    """Fingerprint of the effective bandit configuration (deterministic)."""
    canonical = json.dumps(config.model_dump(), sort_keys=True, default=str)
    return "ban-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _quality_floor(config: BanditConfig, phase: str) -> float:
    name = phase.removeprefix("sdd-")
    return config.phases.get(name, config.quality_floor)


def _arm_key(candidate: RankedCandidate) -> tuple[str, str]:
    return (candidate.model.canonical_id, candidate.variant.effort)


def apply_bandit(
    ranking: CandidateRanking,
    aggregates: list[RewardAggregate],
    config: BanditConfig,
) -> BanditResult:
    """Reorder a phase ranking by UCB over observed arm rewards.

    Never introduces an arm outside ``ranking.candidates`` and never drops
    one: constraint satisfaction is preserved by construction (this is a
    pure reordering). Cold start returns the input ranking unchanged.
    """
    version = bandit_version(config)
    per_phase: dict[str, list[RewardAggregate]] = {}
    for agg in aggregates:
        per_phase.setdefault(agg.phase.removeprefix("sdd-"), []).append(agg)
    phase_aggs = per_phase.get(ranking.phase, [])

    total_executions = sum(a.executions for a in phase_aggs)
    if total_executions < config.min_executions_before_exploit:
        return BanditResult(
            ranking=ranking,
            applied=False,
            reason_code=REASON_COLD_START,
            bandit_version=version,
        )

    by_arm = {
        (a.model, a.effort): a
        for a in phase_aggs
        if a.phase.removeprefix("sdd-") == ranking.phase
    }
    floor = _quality_floor(config, ranking.phase)

    unobserved: list[RankedCandidate] = []
    exploitable: list[tuple[float, int, RankedCandidate]] = []
    below_floor: list[RankedCandidate] = []
    for index, candidate in enumerate(ranking.candidates):
        agg = by_arm.get(_arm_key(candidate))
        if agg is None:
            unobserved.append(candidate)
            continue
        if agg.success_rate < floor:
            below_floor.append(candidate)
            continue
        bonus = config.exploration_weight * math.sqrt(
            math.log(total_executions) / agg.executions
        )
        exploitable.append((agg.mean_reward + bonus, -index, candidate))

    exploitable.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered = (
        unobserved
        + [candidate for _, _, candidate in exploitable]
        + below_floor
    )

    reranked = tuple(
        replace(candidate, reason_codes=tuple(candidate.reason_codes) + (REASON_UCB,))
        for candidate in ordered
    )
    new_ranking = CandidateRanking(
        phase=ranking.phase,
        threshold=ranking.threshold,
        candidates=reranked,
        pairs_considered=ranking.pairs_considered,
        raw_count=ranking.raw_count,
        policy_version=f"{ranking.policy_version}+bandit",
    )
    return BanditResult(
        ranking=new_ranking,
        applied=True,
        reason_code=REASON_UCB,
        bandit_version=version,
    )
