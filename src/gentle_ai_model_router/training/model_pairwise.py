"""Pairwise ranker: shared encoder scores (A, B); BCE on the score difference.

Preference pairs come from the dataset builder with a margin filter, so the
pairwise objective sees only pairs the bootstrap utility model is confident
about. ``BCEWithLogitsLoss`` on ``s_a - s_b`` against label 1.0 (A beats B)
is the standard robust choice for noisy preference labels: it only asks the
model to get the ORDER right, not to reproduce the exact utility gap — the
right inductive bias while labels are priors, not ground truth.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import torch

from gentle_ai_model_router.training.model import _require_train_extra, build_model

# Re-export so tests/dev can build either head from one module namespace.
__all__ = ["build_pairwise_model", "collate_pairs", "pairwise_loss"]


def build_pairwise_model(
    model_name: str = "microsoft/deberta-v3-base",
    numeric_dim: int = 0,
    tiny_config: Any = None,
) -> torch.nn.Module:
    """Wraps the pointwise ranker; forward returns (s_a, s_b, diff)."""

    _require_train_extra()
    import torch

    pointwise = build_model(model_name=model_name, numeric_dim=numeric_dim, tiny_config=tiny_config)

    class _PairwiseRanker(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scorer = pointwise

        def forward(
            self,
            input_ids_a: torch.Tensor,
            attention_mask_a: torch.Tensor,
            numeric_a: torch.Tensor,
            input_ids_b: torch.Tensor,
            attention_mask_b: torch.Tensor,
            numeric_b: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            s_a = self.scorer(input_ids_a, attention_mask_a, numeric_a)
            s_b = self.scorer(input_ids_b, attention_mask_b, numeric_b)
            return s_a, s_b, s_a - s_b

    return _PairwiseRanker()


def pairwise_loss(diff: torch.Tensor) -> torch.Tensor:
    """BCE-with-logits on the score difference; label is always 1 (A beats B)."""
    _require_train_extra()
    import torch

    target = torch.ones_like(diff)
    return torch.nn.functional.binary_cross_entropy_with_logits(diff, target)


def collate_pairs(
    pairs: list[Any],
    examples_by_key: dict[str, Any],
    tokenizer: Any,
    model_feature_names: tuple[str, ...],
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """Tokenize preference pairs into a model batch."""
    _require_train_extra()
    import torch

    from gentle_ai_model_router.dataset.builder import render_features_text
    from gentle_ai_model_router.training.model import encode_pair_text

    def _text(example: Any) -> str:
        return encode_pair_text(
            example.task_text,
            example.candidate.model,
            example.candidate.deployment,
            example.candidate.effort,
            render_features_text(
                list(model_feature_names) + list(example.benchmark_feature_names),
                list(example.model_features) + list(example.benchmark_features),
            ),
        )

    def _numeric(example: Any) -> list[float]:
        return list(example.model_features) + list(example.benchmark_features)

    sides: dict[str, dict[str, torch.Tensor]] = {}
    for side, attr in (("a", "candidate_a"), ("b", "candidate_b")):
        refs = [getattr(p, attr) for p in pairs]
        examples = [examples_by_key[r.key] for r in refs]
        enc = tokenizer(
            [_text(e) for e in examples],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        sides[side] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "numeric": torch.tensor([_numeric(e) for e in examples], dtype=torch.float32),
        }
    return {
        "input_ids_a": sides["a"]["input_ids"],
        "attention_mask_a": sides["a"]["attention_mask"],
        "numeric_a": sides["a"]["numeric"],
        "input_ids_b": sides["b"]["input_ids"],
        "attention_mask_b": sides["b"]["attention_mask"],
        "numeric_b": sides["b"]["numeric"],
    }
