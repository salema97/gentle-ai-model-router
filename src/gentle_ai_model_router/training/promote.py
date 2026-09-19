"""Model promotion: candidate checkpoint -> evaluate -> compare -> promote.

NEVER auto-replace the active router (docs/training.md "Promotion workflow").
A checkpoint is promoted only when its *evaluation* metrics beat the currently
promoted checkpoint on the business-metric set:

- primary: ``tokens_per_success`` (LOWER is better) on the newest available
  evaluated split (``temporal_test`` > ``test`` > ``validation``);
- guardrails: ``success_rate`` and ``mean_quality`` must NOT regress beyond
  ``epsilon`` (absolute, default 0.02);
- ranking metrics (``ndcg@5``, ``mrr``) are informational in the report.

The promotion record is generalized over artifact kinds
(``artifact_kind`` field, ``"checkpoint"`` | ``"thresholds"``):

- ``artifact_kind: "checkpoint"`` (today's behavior, the default) — the
  record points at a candidate checkpoint dir:
  ``{artifact_kind, promoted_checkpoint, metrics_path, promotion_reason,
  promoted_at, git_commit, promoted_by: "manual"}``;
- ``artifact_kind: "thresholds"`` — the record carries a promoted per-phase
  ``threshold_quality`` payload ``{phase: value}`` plus filtered evidence
  (only ``upgrade``/``downgrade`` threshold-tune proposals, with
  executions/success_rate per phase) and a ``source`` reference (the
  proposals JSON path or a ``bandit_version`` label). The checkpoint
  guardrail comparison does NOT apply to thresholds: ``compare()`` is
  checkpoint-only (tokens_per_success primary + guardrails); threshold
  promotions record evidence but never run a metric comparison.

There is exactly ONE active record: promoting a thresholds payload replaces
a checkpoint record and vice versa (NEVER auto-replaced — every promotion is
an explicit manual action, dry-run first).

Checkpoint records additionally keep
``models/promoted/metrics.json`` — a full COPY of the promoted checkpoint's
``metrics.json`` (design choice: a copy, not a pointer — the record stays
self-contained even if the checkpoint dir is later deleted or moved). A
thresholds promotion removes that copy: the metrics copy belongs to the
checkpoint record only, and a stale one must never feed a later comparison.

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

# Artifact kinds the promotion record can carry. "checkpoint" is today's
# behavior and the DEFAULT: records written before artifact kinds existed
# have no artifact_kind field and are treated as checkpoints.
ARTIFACT_KIND_CHECKPOINT = "checkpoint"
ARTIFACT_KIND_THRESHOLDS = "thresholds"
ARTIFACT_KINDS = (ARTIFACT_KIND_CHECKPOINT, ARTIFACT_KIND_THRESHOLDS)

# Threshold-tune proposal kinds that may be promoted; uphold /
# insufficient_evidence proposals never change a threshold and are dropped
# from the promoted evidence (same rule the router.yaml applier uses).
APPLYABLE_PROPOSAL_KINDS = ("upgrade", "downgrade")

# ONNX file names resolved from a promoted checkpoint dir, most preferred
# first (INT8-quantized before fp32 fallback).
PREFERRED_ONNX_FILES = ("model.quant.onnx", "model.onnx")

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
    comparison: Comparison | None  # None for thresholds (compare is checkpoint-only)
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
    """Decide promote vs keep for CHECKPOINTS. Pure: no I/O, no clocks.

    Checkpoint-only by design: the tokens_per_success primary metric and
    success_rate/mean_quality guardrails compare two ranker evaluations.
    Threshold promotions carry evidence instead and never pass through here.
    """
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


def artifact_kind_of(record: dict[str, Any]) -> str:
    """Artifact kind of a promotion record (backward compatible).

    Records written before artifact kinds existed have no ``artifact_kind``
    field; they are treated as ``"checkpoint"``. Unknown kinds fail closed —
    a record we do not understand must never silently drive serving.
    """
    kind = record.get("artifact_kind", ARTIFACT_KIND_CHECKPOINT)
    if kind not in ARTIFACT_KINDS:
        raise PromotionError(f"unknown artifact_kind in promotion record: {kind!r}")
    return kind


# --------------------------------------------------------------------------- #
# Thresholds artifact payloads
# --------------------------------------------------------------------------- #


def thresholds_payload_from_proposals(
    proposals: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Build a ``{phase: threshold_quality}`` payload + evidence from proposals.

    ``proposals`` are threshold-tune proposal dicts (the JSON shape
    ``router thresholds propose`` prints: ``dataclasses.asdict`` of
    :class:`~gentle_ai_model_router.router.threshold_tune.ThresholdProposal`).
    Only ``upgrade``/``downgrade`` kinds are kept — ``uphold`` and
    ``insufficient_evidence`` proposals never change a threshold. Evidence
    retains the measured executions / success_rate per promoted phase.

    Deterministic: phases are processed in sorted order.
    """
    payload: dict[str, float] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for proposal in sorted(proposals, key=lambda p: str(p.get("phase", ""))):
        if proposal.get("kind") not in APPLYABLE_PROPOSAL_KINDS:
            continue
        phase = proposal.get("phase")
        if not isinstance(phase, str) or not phase:
            raise PromotionError(
                f"threshold proposal carries no phase name: {proposal!r}"
            )
        proposed = proposal.get("proposed_threshold")
        if isinstance(proposed, bool) or not isinstance(proposed, (int, float)):
            raise PromotionError(
                f"threshold proposal for phase {phase!r} carries a non-numeric "
                f"proposed_threshold: {proposed!r}"
            )
        payload[phase] = float(proposed)
        evidence[phase] = {
            "kind": proposal.get("kind"),
            "selected_effort": proposal.get("selected_effort"),
            "executions": proposal.get("executions", 0),
            "success_rate": proposal.get("success_rate"),
            "tokens_per_success": proposal.get("tokens_per_success"),
        }
    return payload, evidence


def validate_thresholds_payload(thresholds: dict[str, Any]) -> dict[str, float]:
    """Validate + normalize a ``{phase: threshold_quality}`` payload.

    Values must be plain numbers in [0, 1] (threshold_quality is a quality
    floor on the policy's 0..1 scale). Fails closed on any deviation.
    """
    if not isinstance(thresholds, dict) or not thresholds:
        raise PromotionError("thresholds payload must be a non-empty {phase: value} mapping")
    normalized: dict[str, float] = {}
    for phase, value in thresholds.items():
        if not isinstance(phase, str) or not phase:
            raise PromotionError(f"threshold phase names must be non-empty strings: {phase!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PromotionError(
                f"threshold for phase {phase!r} must be a number, got {value!r}"
            )
        if not 0.0 <= float(value) <= 1.0:
            raise PromotionError(
                f"threshold for phase {phase!r} must be in [0, 1], got {value!r}"
            )
        normalized[phase] = float(value)
    return dict(sorted(normalized.items()))


def resolve_promoted_onnx(models_dir: str | Path = "models") -> Path | None:
    """ONNX file of the promoted checkpoint, per the promotion record.

    Resolution: active record (``artifact_kind=checkpoint``) → its checkpoint
    dir → ``model.quant.onnx`` preferred, ``model.onnx`` fallback.

    Returns None when no checkpoint is promoted (no record at all, or the
    active record is a thresholds promotion). Raises PromotionError when the
    record points to a MISSING artifact (checkpoint dir gone, or no ONNX
    file inside it) — serving must fail closed, never guess a path.
    """
    current = load_promoted(models_dir)
    if current is None:
        return None
    if artifact_kind_of(current["record"]) != ARTIFACT_KIND_CHECKPOINT:
        return None
    checkpoint = Path(current["record"]["promoted_checkpoint"])
    if not checkpoint.is_dir():
        raise PromotionError(
            f"promotion record points to a missing checkpoint dir: {checkpoint}"
        )
    for name in PREFERRED_ONNX_FILES:
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise PromotionError(
        f"promoted checkpoint {checkpoint} contains none of "
        f"{', '.join(PREFERRED_ONNX_FILES)} — refusing to serve a missing artifact"
    )


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
    candidate: str | Path | None = None,
    *,
    models_dir: str | Path = "models",
    dataset: str | Path | None = None,
    config: Any = None,
    metric: str = PRIMARY_METRIC,
    epsilon: float = DEFAULT_EPSILON,
    dry_run: bool = False,
    thresholds: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    source: str | None = None,
) -> PromotionOutcome:
    """Run the full promotion workflow for one artifact.

    Two artifact kinds (mutually exclusive):

    - ``candidate`` (default): a checkpoint dir, promoted only when its eval
      beats the active checkpoint via :func:`compare` (guardrails unchanged).
    - ``thresholds``: a ``{phase: threshold_quality}`` payload promoted with
      measured evidence and a ``source`` reference. No metric comparison —
      evidence is recorded as-is. ``evidence`` accepts threshold-tune
      proposal dicts; only upgrade/downgrade kinds are retained.

    Same safety rules for both kinds: dry-run first, NEVER auto-replace,
    ``promoted_by: "manual"``, atomic write.

    Raises PromotionError on any validation failure (caller maps it to exit 2).
    """
    if thresholds is not None:
        return _promote_thresholds(
            thresholds,
            evidence=evidence,
            source=source,
            models_dir=models_dir,
            dry_run=dry_run,
        )
    if candidate is None:
        raise PromotionError("either --candidate or --thresholds is required")
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
    # Only a checkpoint record can be compared against: a thresholds record
    # promotes floors, not a ranker, and any stale metrics copy it may have
    # inherited must never feed the guardrail comparison.
    promoted_eval = None
    if current is not None and artifact_kind_of(current["record"]) == ARTIFACT_KIND_CHECKPOINT:
        promoted_eval = extract_eval(current["metrics"])
    comparison = compare(candidate_eval, promoted_eval, metric=metric, epsilon=epsilon)

    if comparison.decision == "keep":
        return PromotionOutcome(comparison, None, wrote=False, dry_run=dry_run, record_path=None)
    if dry_run:
        return PromotionOutcome(comparison, None, wrote=False, dry_run=True, record_path=None)

    out_dir = promoted_dir(models_dir)
    record = {
        "artifact_kind": ARTIFACT_KIND_CHECKPOINT,
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


def _promote_thresholds(
    thresholds: dict[str, Any],
    *,
    evidence: list[dict[str, Any]] | None,
    source: str | None,
    models_dir: str | Path,
    dry_run: bool,
) -> PromotionOutcome:
    """Thresholds promotion: evidence-backed payload, no metric comparison.

    Records the payload with per-phase measured evidence (executions /
    success_rate) and a ``source`` reference so the promotion stays
    traceable. NEVER auto-replaces: same record slot as checkpoints, but the
    caller must still pass ``dry_run=True`` first by convention.
    """
    payload = validate_thresholds_payload(thresholds)
    if not isinstance(source, str) or not source.strip():
        raise PromotionError(
            "a thresholds promotion requires a source reference "
            "(proposals JSON path or bandit_version) — refusing an untraceable promotion"
        )
    filtered: dict[str, dict[str, Any]] = {}
    if evidence is not None:
        _payload, filtered = thresholds_payload_from_proposals(evidence)
        # Evidence documents the payload: every promoted phase must carry it.
        unknown = sorted(set(payload) - set(filtered))
        if unknown:
            raise PromotionError(
                "thresholds payload has no matching proposal evidence for phase(s): "
                + ", ".join(unknown)
            )

    if dry_run:
        return PromotionOutcome(None, None, wrote=False, dry_run=True, record_path=None)

    out_dir = promoted_dir(models_dir)
    record = {
        "artifact_kind": ARTIFACT_KIND_THRESHOLDS,
        "thresholds": payload,
        "evidence": filtered,
        "source": source,
        "promotion_reason": (
            f"thresholds promotion from {source}: "
            + ", ".join(f"{phase}={value:.3f}" for phase, value in payload.items())
        ),
        "promoted_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "promoted_by": "manual",
    }
    record_path = out_dir / RECORD_FILENAME
    _atomic_write_json(record_path, record)
    # The metrics copy belongs to a checkpoint record only; remove any stale
    # copy so the promoted dir always matches the active record kind.
    stale_metrics = out_dir / METRICS_FILENAME
    if stale_metrics.is_file():
        stale_metrics.unlink()
    return PromotionOutcome(None, record, wrote=True, dry_run=False, record_path=record_path)
