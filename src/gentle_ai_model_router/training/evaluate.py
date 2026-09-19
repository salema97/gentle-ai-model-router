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
        # Sufficiency is defined on QUALITY (the "minimum sufficient effort"
        # contract: pick the cheapest candidate whose quality meets the phase
        # threshold). Utility (quality minus cost penalties) drives ranking
        # and regret, not the success bit — otherwise the evaluator would
        # systematically disagree with the policy's own selection criterion.
        return self.chosen.label_quality_estimate >= self.phase_threshold

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
    batch_size: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    """Evaluate reference routers (and an optional ranker) on all splits.

    ``ranker`` = (model, tokenizer) from training/checkpoints; requires the
    [train] extra. When ``session`` is given, the deterministic policy is
    added as a reference chooser. ``batch_size``/``device`` only affect the
    learned-ranker chooser (batched scoring); baselines-only runs never
    import torch.
    """
    choosers: dict[str, Chooser] = {
        "fixed_strong": fixed_strong_chooser,
        "fixed_cheap": fixed_cheap_chooser,
        "benchmark_only": benchmark_only_chooser,
    }
    if session is not None:
        choosers["baseline_policy"] = policy_chooser(session, config)
    if ranker is not None:
        choosers["learned_ranker"] = _ranker_chooser(
            dataset, *ranker, batch_size=batch_size, device=device
        )

    results: dict[str, Any] = {
        "splits": {},
        "chooser_errors": [],
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
            metrics = _evaluate_chooser_split(
                groups,
                choose,
                config,
                split=split,
                chooser=name,
                chooser_errors=results["chooser_errors"],
            )
            if metrics is not None:
                split_result[name] = metrics
        results["splits"][split] = split_result
    return results


def _evaluate_chooser_split(
    groups: dict[str, list[DatasetExample]],
    choose: Chooser,
    config: RouterConfig,
    split: str,
    chooser: str,
    chooser_errors: list[dict[str, str]],
) -> dict[str, float] | None:
    """Evaluate one chooser on one split, PER PHASE, never dropping it whole.

    A reference chooser that raises for a phase group (e.g. the deterministic
    policy failing closed on "no candidate meets threshold") must NOT silently
    remove the chooser from the whole split: the error is recorded in
    ``chooser_errors`` as {split, phase, chooser, error}, that phase is
    skipped for this chooser only, and the remaining phases still contribute
    metrics (group-count-weighted merge). Returns None only when the chooser
    failed for EVERY phase in the split.
    """
    by_phase: dict[str, dict[str, list[DatasetExample]]] = {}
    for task_id, group in groups.items():
        phase = group[0].phase
        by_phase.setdefault(phase, {})[task_id] = group

    merged: list[dict[str, float]] = []
    for phase, phase_groups in by_phase.items():
        try:
            merged.append(_evaluate_chooser(phase_groups, choose, config))
        except Exception as exc:
            chooser_errors.append(
                {"split": split, "phase": phase, "chooser": chooser, "error": str(exc)}
            )
            logger.warning(
                "evaluation chooser_error split=%s phase=%s chooser=%s error=%s",
                split,
                phase,
                chooser,
                exc,
            )
    if not merged:
        return None
    return _merge_phase_metrics(merged)


# Metrics that are ratios over groups / successes / tokens are re-derived by
# _merge_phase_metrics from the private counters below; everything else is a
# group-count-weighted mean across phases.
_DERIVED_KEYS = ("groups", "tokens_per_task", "tokens_per_success", "success_rate")


def _merge_phase_metrics(per_phase: list[dict[str, float]]) -> dict[str, float]:
    """Merge per-phase metric dicts, weighting by group counts.

    ``_evaluate_chooser`` stashes ``_groups`` / ``_successes`` /
    ``_total_tokens`` so tokens_per_task / tokens_per_success / success_rate
    remain exact under merging instead of becoming averages of averages.
    """
    groups = int(sum(m["_groups"] for m in per_phase))
    successes = int(sum(m["_successes"] for m in per_phase))
    total_tokens = sum(m["_total_tokens"] for m in per_phase)
    out: dict[str, float] = {"groups": float(groups)}
    for key in METRIC_KEYS:
        if key in _DERIVED_KEYS:
            continue
        out[key] = (
            sum(m[key] * m["_groups"] for m in per_phase) / groups if groups else 0.0
        )
    out["tokens_per_task"] = total_tokens / groups if groups else 0.0
    out["tokens_per_success"] = total_tokens / successes if successes else 0.0
    out["success_rate"] = successes / groups if groups else 0.0
    return out


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
    metrics: dict[str, float] = {
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
    # Private merge counters (consumed by _merge_phase_metrics, stripped there).
    metrics["_groups"] = float(n)
    metrics["_successes"] = float(successes)
    metrics["_total_tokens"] = total_tokens
    return metrics


def _example_text(dataset: DatasetV1, e: DatasetExample) -> str:
    """Encoder input text for one example (kept in one place for reuse)."""
    from gentle_ai_model_router.dataset.builder import render_features_text
    from gentle_ai_model_router.training.model import encode_pair_text

    return encode_pair_text(
        e.task_text,
        e.candidate.model,
        e.candidate.deployment,
        e.candidate.effort,
        render_features_text(
            list(dataset.model_feature_names) + list(e.benchmark_feature_names),
            list(e.model_features) + list(e.benchmark_features),
        ),
    )


def _score_group(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    numeric_rows: list[list[float]],
    *,
    batch_size: int,
    device: str,
) -> list[float]:
    """Score one group's candidates in batches on the resolved device.

    Batched (padded) tokenization + one forward pass per batch instead of one
    forward per example: the old single-example loop took >30 min on CPU for
    a full validation split. Called under torch.no_grad with model.eval() by
    the chooser; scores map back to examples by position.
    """
    import torch  # lazy: [train] extra

    from gentle_ai_model_router.training.device import resolve_device

    resolved = resolve_device(device, torch.cuda.is_available())
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        numeric = torch.tensor(
            numeric_rows[start : start + batch_size], dtype=torch.float32
        )
        with torch.no_grad():
            out = model(
                enc["input_ids"].to(resolved),
                enc["attention_mask"].to(resolved),
                numeric.to(resolved),
            )
        scores.extend(float(v) for v in out.tolist())
    return scores


def _ranker_chooser(
    dataset: DatasetV1,
    model: Any,
    tokenizer: Any,
    batch_size: int = 64,
    device: str = "auto",
) -> Chooser:
    """Chooser from a trained checkpoint: argmax ranker score within a group.

    Scores are computed in BATCHES on the resolved device (cuda when
    available). The chooser contract (group in -> chosen example out) and the
    tie-break (lower candidate key wins equal scores) are identical to the
    old single-example version.
    """
    import torch  # lazy: [train] extra

    from gentle_ai_model_router.training.device import resolve_device

    resolved = resolve_device(device, torch.cuda.is_available())
    model.to(resolved)
    model.eval()
    logger.info("evaluation device=%s batch_size=%d", resolved, batch_size)

    def choose(group: list[DatasetExample]) -> DatasetExample:
        texts = [_example_text(dataset, e) for e in group]
        numeric_rows = [list(e.model_features) + list(e.benchmark_features) for e in group]
        scores = _score_group(
            model,
            tokenizer,
            texts,
            numeric_rows,
            batch_size=batch_size,
            device=device,
        )
        # min() is stable: on equal (-score, key) the earliest candidate wins,
        # exactly like the old per-example key function.
        best = min(
            range(len(group)),
            key=lambda i: (-scores[i], group[i].candidate.key),
        )
        return group[best]

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
    batch_size: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load a dataset (+ optional checkpoint) and run the full evaluation.

    ``batch_size``/``device`` override ``config.evaluation`` for the learned
    ranker only; both are ignored (and torch never imported) when no
    checkpoint is given.
    """
    dataset = load_dataset(dataset_path)
    ranker = None
    if checkpoint is not None:
        ranker = _load_checkpoint(checkpoint)
    eval_batch_size = batch_size if batch_size is not None else config.evaluation.batch_size
    eval_device = device if device is not None else config.evaluation.device
    results = evaluate_routers(
        dataset,
        config,
        session=session,
        ranker=ranker,
        batch_size=eval_batch_size,
        device=eval_device,
    )
    if ranker is not None:
        results["ranker_eval"] = {"device": eval_device, "batch_size": eval_batch_size}
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
    if (path / "head.pt").exists():
        # New format: full ranker wrapper (encoder + MLP head) — required by
        # the chooser's forward(input_ids, attention_mask, numeric) contract.
        from gentle_ai_model_router.training.model import load_ranker

        return load_ranker(str(path))
    # Legacy format: raw encoder only (no head.pt); kept for backward compat.
    model = AutoModel.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer
