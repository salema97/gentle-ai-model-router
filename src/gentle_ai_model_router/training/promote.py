"""Model promotion: candidate checkpoint -> evaluate -> compare -> promote.

NEVER auto-replace the active router (docs/training.md "Promotion workflow").
A checkpoint is promoted only when its *evaluation* metrics beat the currently
promoted checkpoint on the business-metric set:

- primary: ``tokens_per_success`` (LOWER is better) on the newest available
  evaluated split (``temporal_test`` > ``test`` > ``validation``);
- guardrails: ``success_rate`` and ``mean_quality`` must NOT regress beyond
  ``epsilon`` (absolute, default 0.02);
- ranking metrics (``ndcg@5``, ``mrr``) are informational in the report.

Promotion record layout (``models/`` is gitignored; dirs are created as
needed):

- ``models/promoted/promoted.json`` — the promotion record:
  ``{promoted_checkpoint, metrics_path, promotion_reason, promoted_at,
  git_commit, promoted_by: "manual"}``;
- ``models/promoted/metrics.json`` — a full COPY of the promoted checkpoint's
  ``metrics.json`` (design choice: a copy, not a pointer — the record stays
  self-contained even if the checkpoint dir is later deleted or moved).

Both files are written atomically (tmp file + ``os.replace``). Train-loss-only
checkpoints are NEVER eligible: without ``--dataset`` the checkpoint must
already carry an ``eval`` (or ``evaluation``) section with per-split metrics
in the same shape ``training/evaluate.py`` produces.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gentle_ai_model_router.training.evaluate import EVALUATED_SPLITS
from gentle_ai_model_router.training.train import _git_commit

PRIMARY_METRIC = "tokens_per_success"
GUARDRAIL_METRICS = ("success_rate", "mean_quality")
INFO_METRICS = ("ndcg@5", "mrr")
DEFAULT_EPSILON = 0.02
PROMOTED_SUBDIR = "promoted"
RECORD_FILENAME = "promoted.json"
METRICS_FILENAME = "metrics.json"

# Provenance fields a checkpoint must carry before it can be promoted.
PROVENANCE_FIELDS = ("dataset.version", "source_snapshot_ids", "label_provenance")


class PromotionError(Exception):
    """Usage/validation error (candidate missing, bad provenance, no eval)."""


@dataclass
class GuardrailResult:
    metric: str
    candidate: float
    promoted: float
    delta: float  # candidate - promoted (negative = regression)
    ok: bool  # delta >= -epsilon


@dataclass
class Comparison:
    """Pure decision output: same inputs -> same decision (determinism)."""

    split: str
    metric: str
    candidate_values: dict[str, float]
    promoted_values: dict[str, float] | None
    guardrails: list[GuardrailResult]
    decision: str  # "promote" | "keep"
    reasons: list[str]


@dataclass
class PromotionOutcome:
    comparison: Comparison
    record: dict[str, Any] | None  # promotion record actually written
    wrote: bool
    dry_run: bool
    record_path: Path | None


# --------------------------------------------------------------------------- #
# Metrics access
# --------------------------------------------------------------------------- #


def missing_provenance(metrics: dict[str, Any]) -> list[str]:
    """Provenance fields required before a checkpoint may be promoted."""
    missing: list[str] = []
    dataset = metrics.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("version") is None:
        missing.append("dataset.version")
    if not metrics.get("source_snapshot_ids"):
        missing.append("source_snapshot_ids")
    if not metrics.get("label_provenance"):
        missing.append("label_provenance")
    return missing


def extract_eval(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Pull the evaluation section out of a metrics dict (or a bare eval doc).

    Accepts a checkpoint ``metrics.json`` with an ``eval``/``evaluation``
    section, or an evaluation document itself (has ``splits`` at top level).
    """
    for key in ("eval", "evaluation"):
        value = metrics.get(key)
        if isinstance(value, dict) and value.get("splits"):
            return value
    if metrics.get("splits"):
        return metrics
    return None


def chooser_metrics(split_entry: dict[str, Any]) -> dict[str, float]:
    """The learned-ranker metric row for one split.

    Eval docs from ``evaluate.py`` key choosers by name (``learned_ranker``);
    hand-written/synthetic eval sections may be the metric row directly.
    """
    if PRIMARY_METRIC in split_entry:
        return split_entry
    ranker = split_entry.get("learned_ranker")
    if isinstance(ranker, dict):
        return ranker
    raise PromotionError(
        "eval metrics contain no 'learned_ranker' entry for the split; "
        "expected the per-split shape produced by training/evaluate.py"
    )


def newest_evaluated_split(eval_results: dict[str, Any]) -> str:
    """Newest split present in the eval doc, per EVALUATED_SPLITS order."""
    splits = eval_results.get("splits", {})
    for split in reversed(EVALUATED_SPLITS):
        if split in splits:
            return split
    raise PromotionError(
        f"eval metrics contain none of the evaluated splits {list(EVALUATED_SPLITS)}"
    )


# --------------------------------------------------------------------------- #
# Comparison (pure)
# --------------------------------------------------------------------------- #


def compare(
    candidate_eval: dict[str, Any],
    promoted_eval: dict[str, Any] | None = None,
    *,
    metric: str = PRIMARY_METRIC,
    epsilon: float = DEFAULT_EPSILON,
) -> Comparison:
    """Decide promote vs keep. Pure: no I/O, no clocks, fully deterministic."""
    if metric != PRIMARY_METRIC:
        raise PromotionError(
            f"unsupported primary metric '{metric}' (only '{PRIMARY_METRIC}' is defined)"
        )
    if epsilon < 0:
        raise PromotionError(f"epsilon must be >= 0, got {epsilon}")

    split = newest_evaluated_split(candidate_eval)
    candidate_values = chooser_metrics(candidate_eval["splits"][split])

    if promoted_eval is None:
        reasons = [
            "first promotion: no currently promoted checkpoint",
            f"eval metrics present on split '{split}'",
        ]
        return Comparison(split, metric, candidate_values, None, [], "promote", reasons)

    promoted_split = newest_evaluated_split(promoted_eval)
    promoted_values = chooser_metrics(promoted_eval["splits"][promoted_split])
    for key in (metric, *GUARDRAIL_METRICS):
        if key not in candidate_values or key not in promoted_values:
            raise PromotionError(f"eval metrics missing required metric '{key}'")

    reasons: list[str] = []
    if promoted_split != split:
        reasons.append(
            f"note: promoted metrics are from split '{promoted_split}', "
            f"candidate from '{split}'"
        )

    guardrails = [
        GuardrailResult(
            metric=name,
            candidate=candidate_values[name],
            promoted=promoted_values[name],
            delta=candidate_values[name] - promoted_values[name],
            ok=candidate_values[name] - promoted_values[name] >= -epsilon,
        )
        for name in GUARDRAIL_METRICS
    ]
    for g in guardrails:
        if g.ok:
            reasons.append(
                f"guardrail {g.metric}: {g.candidate:.4f} vs {g.promoted:.4f} "
                f"(delta {g.delta:+.4f}, within epsilon {epsilon})"
            )
        else:
            reasons.append(
                f"guardrail {g.metric} REGRESSED: {g.candidate:.4f} vs "
                f"{g.promoted:.4f} (delta {g.delta:+.4f} beyond epsilon {epsilon})"
            )

    primary_delta = candidate_values[metric] - promoted_values[metric]
    improved = primary_delta < 0
    reasons.append(
        f"primary {metric}: {candidate_values[metric]:.1f} vs "
        f"{promoted_values[metric]:.1f} (delta {primary_delta:+.1f}, lower is better) "
        + ("improved" if improved else "not improved")
    )
    if not improved:
        reasons.append(f"primary {metric} did not improve — keep current router")

    decision = "promote" if (improved and all(g.ok for g in guardrails)) else "keep"
    return Comparison(
        split, metric, candidate_values, promoted_values, guardrails, decision, reasons
    )


# --------------------------------------------------------------------------- #
# Promotion record I/O
# --------------------------------------------------------------------------- #


def promoted_dir(models_dir: str | Path) -> Path:
    return Path(models_dir) / PROMOTED_SUBDIR


def load_promoted(models_dir: str | Path) -> dict[str, Any] | None:
    """Current promotion record + metrics copy, or None if never promoted."""
    record_path = promoted_dir(models_dir) / RECORD_FILENAME
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    metrics_path = promoted_dir(models_dir) / METRICS_FILENAME
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    )
    return {"record": record, "metrics": metrics}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _evaluate_candidate(
    candidate: Path, dataset: str | Path, config: Any
) -> dict[str, Any]:
    """Fresh eval of the candidate checkpoint. Reuses evaluate.py as-is.

    Loading the checkpoint requires the [train] extra — the import stays
    lazy inside evaluate.py so this path is the only one tests without the
    extra cannot exercise.
    """
    from gentle_ai_model_router.training.evaluate import evaluate_dataset

    return evaluate_dataset(dataset, config, checkpoint=candidate)


def promote_checkpoint(
    candidate: str | Path,
    *,
    models_dir: str | Path = "models",
    dataset: str | Path | None = None,
    config: Any = None,
    metric: str = PRIMARY_METRIC,
    epsilon: float = DEFAULT_EPSILON,
    dry_run: bool = False,
) -> PromotionOutcome:
    """Run the full promotion workflow for one candidate checkpoint.

    Raises PromotionError on any validation failure (caller maps it to exit 2).
    """
    candidate = Path(candidate)
    if not candidate.is_dir():
        raise PromotionError(f"candidate checkpoint not found: {candidate}")
    metrics_path = candidate / METRICS_FILENAME
    if not metrics_path.is_file():
        raise PromotionError(f"candidate has no metrics.json: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    missing = missing_provenance(metrics)
    if missing:
        raise PromotionError(
            "candidate checkpoint is missing provenance field(s): "
            + ", ".join(missing)
            + " — refusing to promote a checkpoint we cannot trace"
        )

    if dataset is not None:
        if config is None:
            raise PromotionError("a RouterConfig is required when --dataset is given")
        candidate_eval = _evaluate_candidate(candidate, dataset, config)
    else:
        candidate_eval = extract_eval(metrics)
        if candidate_eval is None:
            raise PromotionError(
                f"checkpoint {candidate} carries train metrics only; re-run with "
                "--dataset to evaluate it, or embed an 'eval' section in metrics.json. "
                "Never promote on train loss alone."
            )

    current = load_promoted(models_dir)
    promoted_eval = extract_eval(current["metrics"]) if current is not None else None
    comparison = compare(candidate_eval, promoted_eval, metric=metric, epsilon=epsilon)

    if comparison.decision == "keep":
        return PromotionOutcome(comparison, None, wrote=False, dry_run=dry_run, record_path=None)
    if dry_run:
        return PromotionOutcome(comparison, None, wrote=False, dry_run=True, record_path=None)

    out_dir = promoted_dir(models_dir)
    record = {
        "promoted_checkpoint": str(candidate),
        "metrics_path": str(out_dir / METRICS_FILENAME),
        "promotion_reason": "; ".join(comparison.reasons),
        "promoted_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "promoted_by": "manual",
    }
    record_path = out_dir / RECORD_FILENAME
    _atomic_write_json(record_path, record)
    _atomic_write_json(out_dir / METRICS_FILENAME, metrics)
    return PromotionOutcome(comparison, record, wrote=True, dry_run=False, record_path=record_path)
