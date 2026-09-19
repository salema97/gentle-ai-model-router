"""System One non-autoregressive calibrated decision router.

Transforms heuristic or neural ranker scores into calibrated decision
primitives (Choice, Score, Noul) following TypeSafe Jev principles:
- Non-autoregressive: single-forward-pass evaluation.
- Calibrated probabilities: temperature-scaled arm probabilities summing to 1.0.
- Rubric expectation: Score = E[effort] = sum(k * p_k) across ordered rubric bins.
- Escalation gate: Noul = P(fast_success without escalation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gentle_ai_model_router.router.decision import TaskContext
    from gentle_ai_model_router.router.policy import CandidateRanking

# Canonical 4-bin effort rubric (low=0, medium=1, high=2, max=3)
RUBRIC_EFFORT_BINS: tuple[str, ...] = ("low", "medium", "high", "max")
RUBRIC_EFFORT_INDEX: dict[str, int] = {name: idx for idx, name in enumerate(RUBRIC_EFFORT_BINS)}

# Mapping from internal Effort values to the 4 rubric bins
_EFFORT_TO_RUBRIC: dict[str, int] = {
    "off": 0,
    "minimal": 0,
    "low": 0,
    "medium": 1,
    "high": 2,
    "xhigh": 3,
    "max": 3,
}


@dataclass(frozen=True)
class SystemOneDecision:
    """Calibrated decision primitives emitted by the System One engine."""

    probabilities: dict[str, float]
    confidence: float
    effort_score: float
    effort_probabilities: dict[str, float]
    noul_fast_success: float
    calibrated: bool = True

    @property
    def noul_fast_success_probability(self) -> float:
        """Alias for noul_fast_success."""
        return self.noul_fast_success

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata fields for embedding in Decision and RouteResponse."""
        return {
            "effort_score": self.effort_score,
            "effort_probabilities": self.effort_probabilities,
            "noul_fast_success": self.noul_fast_success,
            "calibrated": self.calibrated,
        }


def evaluate_system_one(
    ranking: CandidateRanking,
    context: TaskContext | None = None,
    ranker: Any | None = None,
    *,
    temperature: float = 1.0,
    calibrated: bool = True,
    **kwargs: Any,
) -> SystemOneDecision:
    """Evaluate candidate ranking into calibrated System One decision primitives.

    Args:
        ranking: Threshold-meeting candidates ordered winner-first.
        context: Optional task context (tokens, repo features, etc.).
        ranker: Optional neural ranker or SystemOneModernBERT instance.
        temperature: Softmax temperature parameter (must be > 0).
        calibrated: True if probabilities are calibrated.
        kwargs: Additional arguments (e.g. input tensors for neural forward pass).

    Returns:
        SystemOneDecision containing Choice probabilities, confidence,
        Score rubric expectation, and Noul fast success viability.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    if not ranking.candidates:
        return SystemOneDecision(
            probabilities={},
            confidence=0.0,
            effort_score=0.0,
            effort_probabilities={bin_name: 0.25 for bin_name in RUBRIC_EFFORT_BINS},
            noul_fast_success=0.0,
            calibrated=calibrated,
        )

    canonical_ids = [c.model.canonical_id for c in ranking.candidates]
    if len(canonical_ids) == len(set(canonical_ids)):
        arm_keys = canonical_ids
    else:
        arm_keys = [
            f"{c.model.canonical_id}:{c.deployment.deployment_ref}"
            for c in ranking.candidates
        ]

    scores = [c.score for c in ranking.candidates]
    scaled_scores = [s / temperature for s in scores]
    max_scaled = max(scaled_scores)
    exp_scores = [math.exp(s - max_scaled) for s in scaled_scores]
    total_exp = sum(exp_scores)
    raw_probs = [val / total_exp for val in exp_scores]

    probabilities = {
        key: float(p) for key, p in zip(arm_keys, raw_probs, strict=True)
    }
    confidence = max(probabilities.values())

    model = ranker or kwargs.get("model")
    if model is not None and hasattr(model, "score_head") and hasattr(model, "noul_head"):
        input_ids = kwargs.get("input_ids")
        if input_ids is not None:
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                import torch

                attention_mask = torch.ones_like(input_ids)
            numeric_features = kwargs.get("numeric_features")
            import torch

            with torch.no_grad():
                out = model(input_ids, attention_mask, numeric_features)
            probs_tensor = out.score_probs[0]
            effort_probs_list = [float(probs_tensor[i].item()) for i in range(4)]
            effort_probabilities = {
                RUBRIC_EFFORT_BINS[i]: effort_probs_list[i] for i in range(4)
            }
            effort_score = float(out.expected_score[0].item())
            noul_fast_success = float(out.noul_prob[0].item())
            return SystemOneDecision(
                probabilities=probabilities,
                confidence=confidence,
                effort_score=effort_score,
                effort_probabilities=effort_probabilities,
                noul_fast_success=noul_fast_success,
                calibrated=calibrated,
            )

    winner = ranking.candidates[0]
    winner_effort = winner.variant.effort
    k_star = _EFFORT_TO_RUBRIC.get(winner_effort, 0)

    effort_logits = [-1.5 * abs(k - k_star) for k in range(4)]
    max_el = max(effort_logits)
    exp_el = [math.exp(el - max_el) for el in effort_logits]
    sum_el = sum(exp_el)
    effort_probs_list = [val / sum_el for val in exp_el]
    effort_probabilities = {
        RUBRIC_EFFORT_BINS[k]: effort_probs_list[k] for k in range(4)
    }
    effort_score = sum(k * effort_probs_list[k] for k in range(4))

    winner_quality = winner.quality
    noul_fast_success = float(min(max(winner_quality, 0.0), 1.0))

    return SystemOneDecision(
        probabilities=probabilities,
        confidence=confidence,
        effort_score=effort_score,
        effort_probabilities=effort_probabilities,
        noul_fast_success=noul_fast_success,
        calibrated=calibrated,
    )
