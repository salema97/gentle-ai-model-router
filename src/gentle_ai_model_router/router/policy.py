"""Deterministic prior-weighted heuristic policy.

This is the BASELINE the learned ModernBERT router must beat. It is fully
deterministic: no randomness, stable tie-breaking by canonical id, and a
policy version hash that fingerprints the exact configuration used.

Ranking model (documented choice — configurable table over exponential):

- Benchmark prior: per benchmark key (``<benchmark>`` or ``<benchmark>:<category>``),
  scores are min-max normalized across the candidate set and combined with the
  phase-configured weights. Missing benchmark data falls back to a flat prior.
- Effort quality: diminishing returns via a configurable gain table
  ``quality(effort) = prior + (ceiling - prior) * gain[effort]``. A gain table
  is used instead of ``1 - e^{-k·level}`` because it is auditable per level and
  needs no calibration constant; the shape is the same (concave, ceiling-bound).
- Effort cost: superlinear token multiplier table (reasoning tokens grow
  faster than quality).
- Score = quality − λ_price·estimated_cost − λ_latency·latency_penalty.
  The registry has no per-deployment tps column yet, so the latency term reads
  an optional benchmark key (``policy.speed_benchmark``) and is 0 when absent.
- Minimum-sufficient-effort: per (model, deployment) pick the LOWEST effort
  whose quality meets the phase threshold; among candidates meeting the
  threshold minimize estimated tokens, break ties by quality.

Hard filters (``available_models`` / ``available_efforts``) restrict the
candidate set BEFORE prior normalization: the ranking is recomputed over the
filtered set so the response reflects only what the caller can actually run.
Both the API server and the inspection CLI share :func:`rank_candidates`;
:func:`select_candidate` is the thin Decision-building wrapper on top.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Collection
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.registry.models import (
    Deployment,
    Model,
    ModelBenchmark,
    ModelPrice,
    ModelVariant,
    Provider,
)
from gentle_ai_model_router.registry.normalize import Effort
from gentle_ai_model_router.router.config import PhaseConfig, PolicyConfig, RouterConfig
from gentle_ai_model_router.router.decision import (
    CANONICAL_PHASES,
    Alternative,
    Decision,
    TaskContext,
)

logger = logging.getLogger(__name__)

EFFORT_RANK: dict[str, int] = {level.value: idx for idx, level in enumerate(Effort)}


def effort_quality(prior: float, effort: str, policy: PolicyConfig) -> float:
    """Quality estimate for (benchmark prior, effort) — THE shared quality model.

    Used by the deterministic policy AND by the dataset builder's bootstrap
    utility labels, so training labels and the baseline ranker can never drift
    apart: quality = prior + (ceiling - prior) * gain[effort].
    """
    gain = policy.effort_quality_gain.get(effort, 0.0)
    return prior + (policy.effort_ceiling - prior) * gain

# Hard capability requirements per phase. spec/design *prefer* structured
# output (weighted bonus, not a hard filter) — only apply/verify hard-require
# tool calling.
_TOOL_CALLING_REQUIRED_PHASES = {"apply", "verify"}


class PolicyError(Exception):
    """Fatal policy failure (e.g. empty registry). Fails closed, never guesses."""


@dataclass(frozen=True)
class _Candidate:
    model: Model
    provider: Provider
    deployment: Deployment
    variant: ModelVariant
    prior: float
    prior_missing: bool
    quality: float
    estimated_tokens: float
    estimated_cost: float
    latency_penalty: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class RankedCandidate:
    """One threshold-meeting candidate with its policy score (public view).

    Exposes the ORM rows read-only so inspection surfaces (API ``/policy``,
    ``router explain``) can render model/deployment/effort details without a
    second query. ``score`` is the policy objective value (NOT the ranking
    key — ranking minimizes estimated tokens, tie-broken by quality).
    """

    model: Model
    provider: Provider
    deployment: Deployment
    variant: ModelVariant
    prior: float
    prior_missing: bool
    quality: float
    estimated_tokens: float
    estimated_cost: float
    score: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class CandidateRanking:
    """Full deterministic ranking for one phase invocation.

    ``candidates`` is sorted winner-first with the policy's ordering
    (min tokens, then quality, then canonical id). ``pairs_considered`` counts
    (model, deployment) groups that survived the hard filters;
    ``raw_count`` is the unfiltered group count.
    """

    phase: str
    threshold: float
    candidates: tuple[RankedCandidate, ...]
    pairs_considered: int
    raw_count: int
    policy_version: str


def normalize_phase(phase: str) -> str:
    """Accept ``explore``, ``sdd-explore``, or common runtime aliases; reject anything else."""
    alias_map = {
        "gentle-orchestrator": "explore",
        "orchestrator": "explore",
        "primary": "explore",
        "chat": "explore",
        "general": "explore",
    }
    if phase in alias_map:
        return alias_map[phase]
    name = phase.removeprefix("sdd-")
    if name in alias_map:
        return alias_map[name]
    if name not in CANONICAL_PHASES:
        raise PolicyError(
            f"unknown phase '{phase}' (expected one of: {', '.join(CANONICAL_PHASES)})"
        )
    return name


def policy_version(phase_cfg: PhaseConfig, config: RouterConfig) -> str:
    """Fingerprint of the effective policy configuration (deterministic)."""
    payload = {
        "phase": phase_cfg.model_dump(),
        "policy": config.policy.model_dump(),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return "pol-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def full_policy_version(config: RouterConfig) -> str:
    """Fingerprint of the policy across ALL canonical phases (deterministic).

    Used by surfaces that version the whole policy at once (``/health``,
    ``router export``) instead of a single phase's effective config.
    """
    payload = {
        "policy": config.policy.model_dump(),
        "phases": {
            phase: config.phase_config(phase).model_dump() for phase in CANONICAL_PHASES
        },
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return "pol-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _load_candidates(session: Session) -> list[tuple[Model, Provider, Deployment, ModelVariant]]:
    stmt = (
        select(Model, Provider, Deployment, ModelVariant)
        .join(Deployment, Deployment.model_id == Model.id)
        .join(Provider, Provider.id == Deployment.provider_id)
        .join(ModelVariant, ModelVariant.deployment_id == Deployment.id)
        .order_by(Model.canonical_id, Deployment.deployment_ref, ModelVariant.effort)
    )
    return list(session.execute(stmt).all())


def _benchmark_priors(
    session: Session, models: list[Model], weights: dict[str, float], flat_prior: float
) -> tuple[dict[int, float], set[int]]:
    """Min-max normalized weighted benchmark prior per model id.

    Returns (prior_by_model_id, model_ids_missing_all_data).
    """
    model_ids = [m.id for m in models]
    rows = session.execute(
        select(ModelBenchmark).where(ModelBenchmark.model_id.in_(model_ids))
    ).scalars().all()
    by_key: dict[str, dict[int, float]] = {}
    for row in rows:
        key = row.benchmark if row.category is None else f"{row.benchmark}:{row.category}"
        by_key.setdefault(key, {})[row.model_id] = row.score

    prior: dict[int, float] = {mid: 0.0 for mid in model_ids}
    weight_used = 0.0
    covered: set[int] = set()
    for key, weight in sorted(weights.items()):
        scores = by_key.get(key)
        if not scores:
            continue
        covered.update(scores)
        lo, hi = min(scores.values()), max(scores.values())
        span = hi - lo
        for mid in model_ids:
            if mid in scores:
                norm = 0.5 if span == 0 else (scores[mid] - lo) / span
                prior[mid] += weight * norm
        weight_used += weight

    if weight_used == 0:
        return {mid: flat_prior for mid in model_ids}, set(model_ids)
    missing: set[int] = set()
    for mid in model_ids:
        if mid in covered:
            prior[mid] /= weight_used
        else:
            # Model contributed to no weighted benchmark: flat prior, flagged.
            prior[mid] = flat_prior
            missing.add(mid)
    return prior, missing


def _latest_price(session: Session, deployment_id: int) -> ModelPrice | None:
    return session.execute(
        select(ModelPrice)
        .where(ModelPrice.deployment_id == deployment_id)
        .order_by(ModelPrice.effective_date.desc(), ModelPrice.id.desc())
        .limit(1)
    ).scalars().first()


# Public wrappers — the dataset builder reuses the SAME prior/quality/cost
# model so bootstrap labels cannot drift from the baseline policy.
def benchmark_priors(
    session: Session, models: list[Model], weights: dict[str, float], flat_prior: float
) -> tuple[dict[int, float], set[int]]:
    """Public alias of the weighted min-max benchmark prior computation."""
    return _benchmark_priors(session, models, weights, flat_prior)


def latest_price(session: Session, deployment_id: int) -> ModelPrice | None:
    """Public alias of the latest-price lookup."""
    return _latest_price(session, deployment_id)


def estimate_cost(
    policy: PolicyConfig, input_price: float, output_price: float, tokens: float
) -> float:
    """Cost in USD: blended price per token × token count (1M-token units)."""
    blended = (1 - policy.output_fraction) * input_price + policy.output_fraction * output_price
    return tokens * blended / 1_000_000


def _passes_hard_filters(
    model: Model, phase: str, context_tokens: int | None, reason_codes: list[str]
) -> bool:
    if context_tokens is not None and model.context_window is not None:
        if model.context_window < context_tokens:
            return False
    if phase in _TOOL_CALLING_REQUIRED_PHASES and model.tool_calling is not True:
        reason_codes.append("filtered:tool_calling_required")
        return False
    return True


def select_candidate(
    session: Session,
    phase: str,
    config: RouterConfig,
    context: TaskContext | None = None,
) -> Decision:
    """Select (model, deployment, effort) for a phase. Deterministic.

    Thin wrapper over :func:`rank_candidates` that keeps the Decision
    contract unchanged. Raises :class:`PolicyError` on unknown phase or an
    empty registry — the policy fails closed and never hallucinates candidates.
    """
    ranking = rank_candidates(session, phase, config, context)
    policy = config.policy
    winner = ranking.candidates[0]

    alternatives = tuple(
        Alternative(
            model=c.model.canonical_id,
            provider=c.provider.registry_key,
            deployment=c.deployment.deployment_ref,
            effort=c.variant.effort,
            score=c.score,
            quality=c.quality,
            estimated_tokens=c.estimated_tokens,
        )
        for c in ranking.candidates[1 : policy.top_k]
    )
    reasons = list(winner.reason_codes) + ["cheapest_of_meeting"]
    from gentle_ai_model_router.router.system_one import evaluate_system_one

    sys1 = evaluate_system_one(ranking, context)

    return Decision(
        phase=ranking.phase,
        model=winner.model.canonical_id,
        provider=winner.provider.registry_key,
        deployment=winner.deployment.deployment_ref,
        effort=winner.variant.effort,
        score=winner.score,
        quality=winner.quality,
        alternatives=alternatives,
        reason_codes=tuple(reasons),
        estimated_tokens=winner.estimated_tokens,
        estimated_cost=round(winner.estimated_cost, 6),
        policy_version=ranking.policy_version,
        confidence=sys1.confidence,
        probabilities=sys1.probabilities,
        system_one=sys1.to_dict(),
    )


def rank_candidates(
    session: Session,
    phase: str,
    config: RouterConfig,
    context: TaskContext | None = None,
    allowed_models: Collection[str] | None = None,
    allowed_efforts: Collection[str] | None = None,
) -> CandidateRanking:
    """Rank all threshold-meeting candidates for a phase. Deterministic.

    Hard filters (applied before benchmark prior normalization, so the
    ranking reflects exactly the runnable set):

    - ``allowed_models``: restrict to these canonical model ids.
    - ``allowed_efforts``: restrict variants to these internal effort levels.

    Raises :class:`PolicyError` on unknown phase, an empty registry, or when
    no candidate survives the filters and meets the phase threshold.
    """
    phase_name = normalize_phase(phase)
    phase_cfg = config.phase_config(phase_name)
    policy = config.policy
    context = context or TaskContext()

    raw = _load_candidates(session)
    if not raw:
        raise PolicyError(
            "registry is empty: no (model, deployment, variant) candidates. "
            "Run 'router collect' and 'router normalize' first."
        )

    per_model_dep: dict[
        tuple[int, int], list[tuple[Model, Provider, Deployment, ModelVariant]]
    ] = {}
    for model, provider, deployment, variant in raw:
        if allowed_models is not None and model.canonical_id not in allowed_models:
            continue
        if allowed_efforts is not None and variant.effort not in allowed_efforts:
            continue
        per_model_dep.setdefault((model.id, deployment.id), []).append(
            (model, provider, deployment, variant)
        )
    if not per_model_dep:
        raise PolicyError(
            f"no candidates left after hard filters "
            f"(models={sorted(allowed_models) if allowed_models is not None else 'any'}, "
            f"efforts={sorted(allowed_efforts) if allowed_efforts is not None else 'any'})"
        )
    raw_count = len({(m.id, d.id) for m, _, d, _ in raw})

    models = {m.id: m for m, _, _, _ in raw}
    priors, missing_prior = _benchmark_priors(
        session, list(models.values()), phase_cfg.weights, policy.flat_prior
    )

    # Optional latency proxy: inverse-normalized speed benchmark.
    speed_rows: dict[int, float] = {}
    if policy.speed_benchmark:
        stmt = select(ModelBenchmark).where(
            ModelBenchmark.benchmark == policy.speed_benchmark,
            ModelBenchmark.model_id.in_(models.keys()),
        )
        for row in session.execute(stmt).scalars().all():
            speed_rows[row.model_id] = row.score

    threshold = phase_cfg.threshold_quality
    meeting: list[_Candidate] = []
    for variants in per_model_dep.values():
        variants.sort(key=lambda t: EFFORT_RANK.get(t[3].effort, 0))
        model, provider, deployment, _ = variants[0]
        local_reasons: list[str] = []
        if not _passes_hard_filters(model, phase_name, context.context_tokens, local_reasons):
            continue
        prior = priors[model.id]
        prior_missing = model.id in missing_prior
        if prior_missing:
            local_reasons.append("missing_benchmark_data:using_prior")

        chosen: _Candidate | None = None
        for _, _, _, variant in variants:
            quality = effort_quality(prior, variant.effort, policy)
            if quality >= threshold:
                multiplier = policy.effort_token_multiplier.get(variant.effort, 1.0)
                base = max(policy.base_tokens, context.context_tokens or 0)
                estimated_tokens = base * multiplier
                price = _latest_price(session, deployment.id)
                has_price = price is not None and (
                    price.input_price is not None or price.output_price is not None
                )
                if has_price:
                    in_p = (
                        price.input_price
                        if price.input_price is not None
                        else policy.default_input_price
                    )
                    out_p = (
                        price.output_price
                        if price.output_price is not None
                        else policy.default_output_price
                    )
                else:
                    in_p, out_p = policy.default_input_price, policy.default_output_price
                    local_reasons.append("missing_price_data:using_default")
                estimated_cost = estimate_cost(policy, in_p, out_p, estimated_tokens)
                if speed_rows and model.id in speed_rows:
                    top = max(speed_rows.values())
                    latency_penalty = (top - speed_rows[model.id]) / top if top else 0.0
                else:
                    latency_penalty = 0.0
                reasons = list(local_reasons) + [f"meets_threshold:{variant.effort}"]
                chosen = _Candidate(
                    model=model,
                    provider=provider,
                    deployment=deployment,
                    variant=variant,
                    prior=prior,
                    prior_missing=prior_missing,
                    quality=quality,
                    estimated_tokens=estimated_tokens,
                    estimated_cost=estimated_cost,
                    latency_penalty=latency_penalty,
                    reason_codes=tuple(reasons),
                )
                break
        if chosen is not None:
            meeting.append(chosen)

    if not meeting:
        raise PolicyError(
            f"no candidate meets threshold_quality={threshold} for phase "
            f"'{phase_name}'; lower the threshold or enrich the registry"
        )

    def score_of(c: _Candidate) -> float:
        return (
            c.quality
            - policy.lambda_price * c.estimated_cost
            - policy.lambda_latency * c.latency_penalty
        )

    # Minimum-sufficient-effort policy: minimize tokens, break ties by quality,
    # then by canonical id for determinism.
    meeting.sort(
        key=lambda c: (c.estimated_tokens, -c.quality, c.model.canonical_id, c.variant.effort)
    )
    ranked = tuple(
        RankedCandidate(
            model=c.model,
            provider=c.provider,
            deployment=c.deployment,
            variant=c.variant,
            prior=c.prior,
            prior_missing=c.prior_missing,
            quality=round(c.quality, 6),
            estimated_tokens=c.estimated_tokens,
            estimated_cost=round(c.estimated_cost, 6),
            score=round(score_of(c), 6),
            reason_codes=c.reason_codes,
        )
        for c in meeting
    )
    return CandidateRanking(
        phase=phase_name,
        threshold=threshold,
        candidates=ranked,
        pairs_considered=len(per_model_dep),
        raw_count=raw_count,
        policy_version=policy_version(phase_cfg, config),
    )
