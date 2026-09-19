"""Offline evaluation: ranking metrics + business metrics per split.

Splits evaluated: ``test`` (date-held-out) and ``temporal_test`` (candidate
models first seen after the train cutoff) — the two splits that answer
"does the router generalize across TIME and across UNSEEN MODELS".

⚠ BOOTSTRAP-LABEL CAVEAT (docs/evaluation.md): every metric here is computed
against ``label_utility`` values that are PRIORS from external benchmarks,
not measured task success. These numbers measure internal consistency with
the prior model — useful to catch regressions and rank routers RELATIVE to
each other, meaningless as absolute production-quality estimates.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gentle_ai_model_router.dataset.builder import load_dataset
from gentle_ai_model_router.dataset.schema import DatasetExample, DatasetV1
from gentle_ai_model_router.router.config import RouterConfig
from gentle_ai_model_router.training.baselines import (
    Chooser,
    benchmark_only_chooser,
    fixed_cheap_chooser,
    fixed_strong_chooser,
    group_examples,
    policy_chooser,
)

logger = logging.getLogger(__name__)

SPLITS_TO_EVALUATE = ("test", "temporal_test")
EVALUATED_SPLITS = ("validation", "test", "temporal_test")

METRIC_KEYS = (
    "groups",
    "top1_accuracy",
    "top3_recall",
    "mrr",
    "ndcg@3",
    "ndcg@5",
    "tokens_per_task",
    "tokens_per_success",
    "success_rate",
    "mean_quality",
    "routing_regret",
)


@dataclass
class GroupOutcome:
    """One (group, chosen example) pair with derived quantities."""

    chosen: DatasetExample
    oracle_utility: float
    phase_threshold: float

    @property
    def success(self) -> bool:
        return self.chosen.label_utility >= self.phase_threshold

    @property
    def tokens(self) -> float:
        return self.chosen.cost_features["est_total_tokens"]

    @property
    def regret(self) -> float:
        # >= 0 by construction (oracle = group max utility).
        return self.oracle_utility - self.chosen.label_utility


# --------------------------------------------------------------------------- #
# Ranking metrics (labels = bootstrap utility; see caveat above)
# --------------------------------------------------------------------------- #


def _dcg(gains: list[float], k: int) -> float:
    return sum(g / math.log2(idx + 2) for idx, g in enumerate(gains[:k]))


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #


def evaluate_routers(
    dataset: DatasetV1,
    config: RouterConfig,
    session: Any = None,
    ranker: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate reference routers (and an optional ranker) on all splits.

    ``ranker`` = (model, tokenizer) from training/checkpoints; requires the
    [train] extra. When ``session`` is given, the deterministic policy is
    added as a reference chooser.
    """
    choosers: dict[str, Chooser] = {
        "fixed_strong": fixed_strong_chooser,
        "fixed_cheap": fixed_cheap_chooser,
        "benchmark_only": benchmark_only_chooser,
    }
    if session is not None:
        choosers["baseline_policy"] = policy_chooser(session, config)
    if ranker is not None:
        choosers["learned_ranker"] = _ranker_chooser(dataset, *ranker)

    results: dict[str, Any] = {
        "splits": {},
        "label_provenance": "bootstrap_prior",
        "caveat": (
            "Metrics are computed against bootstrap_prior labels (bootstrap "
            "PRIORS from external benchmarks), not ground truth telemetry. "
            "Compare routers relative to each other only."
        ),
    }
    for split in EVALUATED_SPLITS:
        examples = [e for e in dataset.examples if e.split == split]
        if not examples:
            continue
        groups = group_examples(examples)
        split_result: dict[str, Any] = {}
        for name, choose in choosers.items():
            split_result[name] = _evaluate_chooser(groups, choose, config)
        results["splits"][split] = split_result
    return results


def _evaluate_chooser(
    groups: dict[str, list[DatasetExample]], choose: Chooser, config: RouterConfig
) -> dict[str, float]:
    outcomes: list[GroupOutcome] = []
    top1_hits = 0
    top3_hits = 0
    rr_sum = 0.0
    ndcg3_sum = 0.0
    ndcg5_sum = 0.0

    for group in groups.values():
        chosen = choose(group)
        utilities = {e.candidate.key: e.label_utility for e in group}
        # Ideal ordering: bootstrap utility desc, candidate key asc (ties).
        ranked = sorted(group, key=lambda e: (-e.label_utility, e.candidate.key))
        oracle = ranked[0]
        oracle_key = oracle.candidate.key
        outcomes.append(
            GroupOutcome(
                chosen=chosen,
                oracle_utility=oracle.label_utility,
                phase_threshold=config.phase_config(chosen.phase).threshold_quality,
            )
        )
        if chosen.candidate.key == oracle_key:
            top1_hits += 1
        # Single-pick choosers: "top3 recall" = chosen candidate is among the
        # 3 best-by-utility candidates of the group (near-oracle picks).
        if chosen.candidate.key in {e.candidate.key for e in ranked[:3]}:
            top3_hits += 1
        for idx, e in enumerate(ranked):
            if e.candidate.key == chosen.candidate.key:
                rr_sum += 1.0 / (idx + 1)
                break
        # Predicted ordering for NDCG: chosen first, then ideal order.
        gains_ideal = [e.label_utility for e in ranked]
        gains_chosen = [utilities[chosen.candidate.key]] + [
            e.label_utility for e in ranked if e.candidate.key != chosen.candidate.key
        ]
        ideal3, ideal5 = _dcg(gains_ideal, 3), _dcg(gains_ideal, 5)
        ndcg3_sum += _dcg(gains_chosen, 3) / ideal3 if ideal3 else 0.0
        ndcg5_sum += _dcg(gains_chosen, 5) / ideal5 if ideal5 else 0.0

    n = len(outcomes)
    total_tokens = sum(o.tokens for o in outcomes)
    successes = sum(1 for o in outcomes if o.success)
    return {
        "groups": float(n),
        "top1_accuracy": top1_hits / n if n else 0.0,
        "top3_recall": top3_hits / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        "ndcg@3": ndcg3_sum / n if n else 0.0,
        "ndcg@5": ndcg5_sum / n if n else 0.0,
        "tokens_per_task": total_tokens / n if n else 0.0,
        "tokens_per_success": total_tokens / successes if successes else 0.0,
        "success_rate": successes / n if n else 0.0,
        "mean_quality": sum(o.chosen.label_utility for o in outcomes) / n if n else 0.0,
        "routing_regret": sum(o.regret for o in outcomes) / n if n else 0.0,
    }


def _ranker_chooser(dataset: DatasetV1, model: Any, tokenizer: Any) -> Chooser:
    """Chooser from a trained checkpoint: argmax ranker score within a group."""
    import torch  # lazy: [train] extra

    from gentle_ai_model_router.dataset.builder import render_features_text
    from gentle_ai_model_router.training.model import encode_pair_text

    model.eval()

    def score(e: DatasetExample) -> float:
        text = encode_pair_text(
            e.task_text,
            e.candidate.model,
            e.candidate.deployment,
            e.candidate.effort,
            render_features_text(
                list(dataset.model_feature_names) + list(e.benchmark_feature_names),
                list(e.model_features) + list(e.benchmark_features),
            ),
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        numeric = torch.tensor(
            [e.model_features + e.benchmark_features], dtype=torch.float32
        )
        with torch.no_grad():
            return float(model(enc["input_ids"], enc["attention_mask"], numeric)[0])

    def choose(group: list[DatasetExample]) -> DatasetExample:
        return min(group, key=lambda e: (-score(e), e.candidate.key))

    return choose


# --------------------------------------------------------------------------- #
# Entry point (used by the CLI)
# --------------------------------------------------------------------------- #


def evaluate_dataset(
    dataset_path: str | Path,
    config: RouterConfig,
    session: Any = None,
    checkpoint: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a dataset (+ optional checkpoint) and run the full evaluation."""
    dataset = load_dataset(dataset_path)
    ranker = None
    if checkpoint is not None:
        ranker = _load_checkpoint(checkpoint)
    results = evaluate_routers(dataset, config, session=session, ranker=ranker)
    results["dataset"] = {"name": dataset.name, "version": dataset.version}
    if checkpoint is not None:
        results["checkpoint"] = str(checkpoint)
    if output_path is not None:
        Path(output_path).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def _load_checkpoint(checkpoint: str | Path) -> tuple[Any, Any]:
    """Load (model, tokenizer) from a training checkpoint dir. Lazy [train]."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "evaluating a checkpoint requires the [train] extra: "
            "pip install 'gentle-ai-model-router[train]'"
        ) from exc
    path = Path(checkpoint)
    model = AutoModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
