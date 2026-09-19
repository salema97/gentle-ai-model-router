"""Neural reranking pipeline for gentle-ai-model-router.

Uses trained OnnxRanker to rerank candidate models per task context.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.registry.models import ModelBenchmark, ModelPrice
from gentle_ai_model_router.router.config import RouterConfig
from gentle_ai_model_router.router.policy import CandidateRanking, RankedCandidate

DEFAULT_BENCHMARK_NAMES: tuple[str, ...] = (
    "artificial_analysis_intelligence_index",
    "lmarena_elo:text",
    "lmarena_elo:webdev",
    "telemetry_success_rate:explore",
    "telemetry_success_rate:sdd-apply",
    "telemetry_success_rate:sdd-archive",
    "telemetry_success_rate:sdd-design",
    "telemetry_success_rate:sdd-explore",
    "telemetry_success_rate:sdd-init",
    "telemetry_success_rate:sdd-propose",
    "telemetry_success_rate:sdd-research",
    "telemetry_success_rate:sdd-spec",
    "telemetry_success_rate:sdd-tasks",
    "telemetry_success_rate:sdd-verify",
    "telemetry_success_rate:verify",
)


def get_configured_benchmark_names(config: RouterConfig) -> list[str]:
    """Collect benchmark names across all configured phases in alphabetical order (15 keys)."""
    keys: set[str] = set()
    for phase_cfg in config.phases.values():
        keys.update(phase_cfg.weights.keys())
    if len(keys) == 15:
        return sorted(keys)
    return list(DEFAULT_BENCHMARK_NAMES)


def build_candidate_features(
    session: Session,
    candidates: list[RankedCandidate],
    config: RouterConfig,
) -> list[list[float]]:
    """Build 22-dimensional numeric feature vector for each candidate.

    Features per candidate (22):
      7 model features:
        - log10(context_window)
        - log10(max_output)
        - input_price
        - output_price
        - 1.0 if tool_calling else 0.0
        - 1.0 if structured_output else 0.0
        - latency_penalty
      15 benchmark features:
        - normalized or raw score for each benchmark key in alphabetical order (0.0 if missing).
    """
    if not candidates:
        return []

    benchmark_names = get_configured_benchmark_names(config)

    model_ids = {c.model.id for c in candidates}
    deployment_ids = {c.deployment.id for c in candidates}

    bench_rows = session.execute(
        select(ModelBenchmark).where(ModelBenchmark.model_id.in_(model_ids))
    ).scalars().all()

    price_rows = session.execute(
        select(ModelPrice)
        .where(ModelPrice.deployment_id.in_(deployment_ids))
        .order_by(ModelPrice.effective_date.desc(), ModelPrice.id.desc())
    ).scalars().all()

    scores: dict[tuple[int, str], float] = {}
    speed_rows: dict[int, float] = {}
    for row in bench_rows:
        key = row.benchmark if row.category is None else f"{row.benchmark}:{row.category}"
        scores[(row.model_id, key)] = float(row.score) if row.score is not None else 0.0
        if config.policy.speed_benchmark and row.benchmark == config.policy.speed_benchmark:
            if row.score is not None:
                speed_rows[row.model_id] = float(row.score)

    top_speed = max(speed_rows.values()) if speed_rows else 0.0

    price_by_deployment: dict[int, ModelPrice] = {}
    for row in price_rows:
        if row.deployment_id not in price_by_deployment:
            price_by_deployment[row.deployment_id] = row

    features: list[list[float]] = []
    for c in candidates:
        log_cw = (
            math.log10(c.model.context_window)
            if c.model.context_window and c.model.context_window > 0
            else 0.0
        )
        log_mo = (
            math.log10(c.model.max_output)
            if c.model.max_output and c.model.max_output > 0
            else 0.0
        )

        price = price_by_deployment.get(c.deployment.id)
        in_p = (
            float(price.input_price)
            if price and price.input_price is not None
            else float(config.policy.default_input_price)
        )
        out_p = (
            float(price.output_price)
            if price and price.output_price is not None
            else float(config.policy.default_output_price)
        )
        tool_calling = 1.0 if c.model.tool_calling else 0.0
        structured_output = 1.0 if c.model.structured_output else 0.0

        if top_speed > 0.0 and c.model.id in speed_rows:
            latency_penalty = (top_speed - speed_rows[c.model.id]) / top_speed
        else:
            latency_penalty = 0.0

        model_features = [
            log_cw,
            log_mo,
            in_p,
            out_p,
            tool_calling,
            structured_output,
            latency_penalty,
        ]
        bench_features = [scores.get((c.model.id, k), 0.0) for k in benchmark_names]
        features.append(model_features + bench_features)

    return features


def neural_rerank(
    session: Session,
    ranking: CandidateRanking,
    ranker: Any,
    task: str,
    config: RouterConfig,
    max_candidates: int = 50,
) -> CandidateRanking:
    """Rerank candidates using a neural ranker model.

    If not ranking.candidates or not task: returns ranking unchanged.
    Takes candidates_to_score = list(ranking.candidates[:max_candidates]).
    Builds numeric features with build_candidate_features.
    Builds texts with format:
      [phase] {phase}
      [task] {task}
      [candidate] {canonical_id} effort={effort}
    Scores with scores = ranker.score(texts, features).
    Sorts candidates_to_score by neural score descending (tie-broken by quality).
    Creates new RankedCandidate for winner with neural score and reason codes
    updated to include "neural_ranker:onnx".
    Reassembles full candidate list (candidates_to_score + ranking.candidates[max_candidates:]).
    Returns new CandidateRanking with policy_version=f"{ranking.policy_version}+neural".
    """
    if not ranking.candidates or not task:
        return ranking

    candidates_to_score = list(ranking.candidates[:max_candidates])
    features = build_candidate_features(session, candidates_to_score, config)
    texts = [
        (
            f"[phase] {ranking.phase}\n"
            f"[task] {task}\n"
            f"[candidate] {c.model.canonical_id} effort={c.variant.effort}"
        )
        for c in candidates_to_score
    ]
    raw_scores = ranker.score(texts, features)
    scores = [float(s) for s in raw_scores]

    scored = list(zip(candidates_to_score, scores, strict=True))
    scored.sort(key=lambda item: (item[1], item[0].quality), reverse=True)

    winner_orig, winner_score = scored[0]
    winner_reasons = tuple(winner_orig.reason_codes) + ("neural_ranker:onnx",)
    new_winner = dataclasses.replace(
        winner_orig,
        score=round(float(winner_score), 6),
        reason_codes=winner_reasons,
    )
    reranked_subset = [new_winner] + [c for c, _ in scored[1:]]
    full_candidates = tuple(reranked_subset + list(ranking.candidates[max_candidates:]))

    return CandidateRanking(
        phase=ranking.phase,
        threshold=ranking.threshold,
        candidates=full_candidates,
        pairs_considered=ranking.pairs_considered,
        raw_count=ranking.raw_count,
        policy_version=f"{ranking.policy_version}+neural",
    )
