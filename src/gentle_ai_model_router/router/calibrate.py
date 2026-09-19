"""Threshold calibration: decision support for per-phase quality floors.

READ-ONLY by design: this module never writes router.yaml or the registry.
It answers one question per phase: "given the benchmark priors currently in
the registry, what quality is actually achievable, and does the configured
``threshold_quality`` sit in a sane place?"

The quality model is IMPORTED from router/policy.py (``effort_quality`` +
``benchmark_priors``) so the distribution shown here is exactly the quality
the routing policy computes — no duplicated math, no drift.

⚠ BOOTSTRAP-QUALITY INPUT: the priors are min-max-normalized external
benchmark scores (plus a flat prior for uncovered models), NOT measured task
success. Treat the printed percentiles as "where the threshold sits relative
to the achievable-prior frontier", not as production quality estimates.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.registry.models import Deployment, Model, ModelVariant
from gentle_ai_model_router.router.config import RouterConfig
from gentle_ai_model_router.router.decision import CANONICAL_PHASES
from gentle_ai_model_router.router.policy import (
    EFFORT_RANK,
    benchmark_priors,
    effort_quality,
    normalize_phase,
)


@dataclass(frozen=True)
class CalibrationRow:
    """One phase's achievable-quality distribution vs its configured floor."""

    phase: str
    threshold: float  # current configured threshold_quality
    p25: float  # percentiles of the pooled per-(candidate, effort) quality
    p50: float  # distribution, using the policy's own quality model
    p75: float
    fraction_meeting: float  # candidates whose BEST effort meets threshold
    suggested: float  # min(max(threshold, p50), p90) — see calibrate_thresholds


def calibrate_thresholds(
    session: Session, config: RouterConfig, phase: str | None = None
) -> list[CalibrationRow]:
    """Compute calibration rows for one phase (normalized) or all canonical ones.

    Raises:
        PolicyError: unknown phase name (fail fast; the CLI prints it).
    """
    phases = [normalize_phase(phase)] if phase else list(CANONICAL_PHASES)
    return [_calibrate_phase(session, config, ph) for ph in phases]


def _calibrate_phase(session: Session, config: RouterConfig, phase: str) -> CalibrationRow:
    phase_cfg = config.phase_config(phase)
    policy = config.policy
    threshold = phase_cfg.threshold_quality

    stmt = (
        select(Model, ModelVariant.effort)
        .join(Deployment, Deployment.model_id == Model.id)
        .join(ModelVariant, ModelVariant.deployment_id == Deployment.id)
        .order_by(Model.canonical_id)
    )
    rows = session.execute(stmt).all()
    models: list[Model] = []
    seen: set[int] = set()
    for model, _effort in rows:
        if model.id not in seen:
            seen.add(model.id)
            models.append(model)

    if not models:
        return CalibrationRow(
            phase=phase,
            threshold=threshold,
            p25=0.0,
            p50=0.0,
            p75=0.0,
            fraction_meeting=0.0,
            suggested=threshold,
        )

    priors, _missing = benchmark_priors(
        session, models, phase_cfg.weights, policy.flat_prior
    )
    # The full effort table (not just per-deployment variants) — "achievable"
    # means the policy's quality model at every configured effort level.
    efforts = sorted(policy.effort_quality_gain, key=lambda e: EFFORT_RANK.get(e, 0))

    qualities: list[float] = []
    per_candidate_best: list[float] = []
    for model in models:
        candidate_qualities = [
            effort_quality(priors[model.id], effort, policy) for effort in efforts
        ]
        qualities.extend(candidate_qualities)
        per_candidate_best.append(max(candidate_qualities))

    p25, p50, p75 = statistics.quantiles(qualities, n=4)
    p90 = statistics.quantiles(qualities, n=10)[8]
    fraction_meeting = sum(1 for q in per_candidate_best if q >= threshold) / len(
        per_candidate_best
    )
    # Heuristic (documented, defensible):
    #   suggested = min(max(current, p50), p90)
    # Never suggest going DOWN below the current floor, and never suggest
    # going ABOVE the p90 of what is achievable — a threshold over p90 fails
    # closed for almost every candidate, which is an operations bug, not a
    # quality bar. Rationale: the floor should sit at or above the median
    # achievable quality (otherwise it filters nothing) but below the point
    # where it starves the phase of candidates.
    suggested = min(max(threshold, p50), p90)
    return CalibrationRow(
        phase=phase,
        threshold=threshold,
        p25=p25,
        p50=p50,
        p75=p75,
        fraction_meeting=fraction_meeting,
        suggested=suggested,
    )
