"""Dataset builder: registry priors → DatasetV1 examples + preference pairs.

One example per (phase, task_context, candidate), where a candidate is the
project's unit: ``(model, deployment, effort)``.

Labels are **bootstrap priors**, never ground truth (see
:mod:`gentle_ai_model_router.dataset.schema` and docs/training.md): utility =
quality_estimate − token_cost_penalty, where quality_estimate reuses the
EXACT quality model of router/policy.py (`effort_quality`,
`benchmark_priors`, `estimate_cost`) so labels and the baseline policy cannot
drift apart.

Anti-leakage (hard rules, build errors — not warnings):
- ``as_of`` (default: newest snapshot date) is the knowledge cutoff: every
  benchmark/price row used must come from a snapshot dated <= as_of.
- Every benchmark/price row must reference a known snapshot id (else it
  cannot be dated → build error).
- Splits are temporal (never random): train < train_end <= validation <
  val_end <= test, plus ``temporal_test`` for candidate models whose FIRST
  registry appearance is after train_end ("new model" simulation).
- No train example may share a task_id with validation/test/temporal_test;
  overlapping train rows are dropped (counted in the manifest).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.collector.routing_benchmarks import map_task_to_phase
from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.dataset.schema import (
    GROUND_TRUTH_PROVENANCE_ADDENDUM,
    MODEL_FEATURE_NAMES,
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_EMPIRICAL,
    PROVENANCE_GROUND_TRUTH,
    PROVENANCE_STATEMENT,
    PROVENANCE_TELEMETRY,
    TELEMETRY_PROVENANCE_ADDENDUM,
    CandidateRef,
    DatasetExample,
    DatasetV1,
    PreferencePair,
)
from gentle_ai_model_router.dataset.telemetry_bridge import (
    TelemetryExecution,
    TelemetryReadError,
    read_telemetry_executions,
)
from gentle_ai_model_router.registry.models import (
    Deployment,
    Model,
    ModelBenchmark,
    ModelPrice,
    ModelSnapshot,
    ModelVariant,
    Provider,
)
from gentle_ai_model_router.router.config import RouterConfig
from gentle_ai_model_router.router.decision import CANONICAL_PHASES, TaskContext
from gentle_ai_model_router.router.policy import (
    benchmark_priors,
    effort_quality,
    estimate_cost,
)

logger = logging.getLogger(__name__)


class DatasetBuildError(Exception):
    """Fatal dataset build failure (leakage, undatable rows, empty registry)."""


class DatasetLeakageError(DatasetBuildError):
    """Anti-leakage violation — a feature row is newer than the as-of cutoff."""


# --------------------------------------------------------------------------- #
# Builder configuration
# --------------------------------------------------------------------------- #


class DatasetBuilderConfig(BaseModel):
    """Configuration for one dataset build (deterministic; seed fixed)."""

    name: str = "router-priors"
    phases: list[str] = Field(default_factory=lambda: list(CANONICAL_PHASES))
    task_types: list[str] = Field(default_factory=lambda: ["feature", "bugfix", "docs"])
    context_sizes: list[int] = Field(default_factory=lambda: [10_000, 60_000])
    repo_features: dict[str, str] = Field(
        default_factory=lambda: {"primary_language": "typescript", "test_framework": "vitest"}
    )
    # Pairwise generation.
    pair_margin: float = 0.05
    max_pairs_per_group: int = 50
    # Token cost penalty weight for the bootstrap utility label:
    # penalty = cost_weight * est_tokens / max_est_tokens(group).
    cost_weight: float = 0.5
    # Quality threshold conditioning:
    threshold_penalty: float = 0.0
    hard_threshold: bool = False
    # Empirical benchmark tasks
    include_empirical_benchmarks: bool = True
    empirical_snapshot_source: str = "routing-benchmarks"
    max_empirical_tasks_per_phase: int = 30
    # Ground-truth traces (RouterBench, RouteLLM, SWE-Traces)
    include_ground_truth_traces: bool = True
    ground_truth_snapshot_sources: list[str] = Field(
        default_factory=lambda: ["routerbench", "routellm", "swe-traces"]
    )
    max_ground_truth_tasks_per_phase: int = 30
    # Temporal split (required — forcing an explicit decision is deliberate).
    train_end: date
    val_end: date | None = None
    # Knowledge cutoff; rows from newer snapshots are a build error.
    as_of: date | None = None
    seed: int = 13


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _utility_label(
    quality: float,
    total_tokens: float,
    max_group_tokens: float,
    *,
    cost_weight: float,
    threshold: float,
    threshold_penalty: float,
    hard_threshold: bool,
) -> float:
    """Threshold-conditioned utility label shared by ALL provenances.

    Bootstrap rows pass ESTIMATED tokens, telemetry rows pass ACTUAL tokens;
    the formula is identical so labels stay comparable across provenances:
    utility = clip(quality - cost_weight * tokens / max_group_tokens
                    - (threshold_penalty if below floor), 0, 1), or 0.0
    outright under a hard threshold.
    """
    penalty = cost_weight * total_tokens / max_group_tokens if max_group_tokens else 0.0
    below_threshold = quality < threshold
    if below_threshold and hard_threshold:
        return 0.0
    fail_penalty = threshold_penalty if below_threshold else 0.0
    return max(0.0, min(1.0, quality - penalty - fail_penalty))


def _temporal_split(
    example_date: date,
    first_seen_date: date,
    train_end: date,
    val_end: date | None,
) -> str:
    """Temporal split assignment (never random), shared by all provenances."""
    if first_seen_date > train_end:
        return "temporal_test"
    if example_date < train_end:
        return "train"
    if val_end is not None and example_date < val_end:
        return "validation"
    return "test"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        commit = out.stdout.strip()
        return commit or "unknown"
    except Exception:  # pragma: no cover - not a git checkout
        return "unknown"


def _task_id(phase: str, task_type: str, context_tokens: int, repo: dict[str, str]) -> str:
    canonical = json.dumps(
        {"phase": phase, "task_type": task_type, "context_tokens": context_tokens,
         "repo": sorted(repo.items())},
        sort_keys=True,
    )
    return "task-" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def render_task_text(
    phase: str, task_type: str, context_tokens: int, repo_features: dict[str, str]
) -> str:
    """Human/encoder-readable rendering of the task context."""
    repo = "; ".join(f"{k}={v}" for k, v in sorted(repo_features.items()))
    return (
        f"[phase] {phase}\n"
        f"[task_type] {task_type}\n"
        f"[context] approximately {context_tokens} tokens of repository context\n"
        f"[repo_features] {repo}"
    )


def render_features_text(names: list[str] | tuple[str, ...], values: list[float]) -> str:
    """Serialize a numeric feature vector as key=value text for the encoder.

    Standard hybrid trick: the text branch lets the encoder attend over
    features semantically, while the numeric MLP branch (training/model.py)
    gives exact numeric gradients. We do BOTH on purpose (docs/training.md).
    """
    return "; ".join(f"{name}={value:.6g}" for name, value in zip(names, values, strict=True))


def _snapshot_dates(session: Session) -> dict[str, date]:
    rows = session.execute(select(ModelSnapshot)).scalars().all()
    return {row.snapshot_id: row.fetched_at.date() for row in rows}


def _check_row_date(snapshot_dates: dict[str, date], source_snapshot_id: str, as_of: date) -> date:
    """Date one provenance row; enforce the as-of cutoff (build error)."""
    row_date = snapshot_dates.get(source_snapshot_id)
    if row_date is None:
        raise DatasetBuildError(
            f"row references unknown snapshot_id '{source_snapshot_id}'; "
            "every benchmark/price row must be datable to enforce anti-leakage"
        )
    if row_date > as_of:
        raise DatasetLeakageError(
            f"anti-leakage violation: row from snapshot '{source_snapshot_id}' "
            f"dated {row_date} is newer than the as-of cutoff {as_of}"
        )
    return row_date


def _model_feature_vector(
    model: Model,
    price: ModelPrice | None,
    policy_prices: tuple[float, float],
    tps_proxy: float = 0.0,
) -> list[float]:
    """Fixed-width numeric model feature vector (MODEL_FEATURE_NAMES order)."""
    in_p = price.input_price if price and price.input_price is not None else policy_prices[0]
    out_p = price.output_price if price and price.output_price is not None else policy_prices[1]
    return [
        math.log10(model.context_window) if model.context_window else 0.0,
        math.log10(model.max_output) if model.max_output else 0.0,
        in_p,
        out_p,
        1.0 if model.tool_calling else 0.0,
        1.0 if model.structured_output else 0.0,
        tps_proxy,
    ]


def _extract_empirical_samples(data: Any) -> list[dict[str, Any]]:
    """Extract normalized samples across dars, xroutebench, and compendium."""
    if not isinstance(data, dict):
        if isinstance(data, list):
            res = []
            for item in data:
                if isinstance(item, dict):
                    task_name = str(
                        item.get("task_name")
                        or item.get("task")
                        or item.get("dataset")
                        or "benchmark"
                    )
                    prompt = str(
                        item.get("prompt")
                        or item.get("input_question")
                        or item.get("question")
                        or item.get("query")
                        or ""
                    )
                    phase = str(
                        item.get("mapped_phase")
                        or item.get("phase")
                        or map_task_to_phase(task_name)
                    )
                    model = str(item.get("model") or item.get("model_name") or "")
                    res.append(
                        {
                            "task_name": task_name,
                            "prompt": prompt,
                            "model": model,
                            "model_name": model,
                            "quality": item.get(
                                "quality", item.get("performance", item.get("score", 0.5))
                            ),
                            "cost": item.get("cost", 0.0),
                            "input_tokens": item.get("input_tokens")
                            or item.get("prompt_tokens")
                            or 1000,
                            "phase": phase,
                            "mapped_phase": phase,
                        }
                    )
            return res
        return []

    samples: list[dict[str, Any]] = []

    # 1. dars
    dars = data.get("dars")
    if isinstance(dars, list):
        for item in dars:
            if isinstance(item, dict):
                task_name = str(item.get("task_name") or item.get("task") or "drop-800")
                prompt = str(
                    item.get("prompt")
                    or item.get("input_question")
                    or item.get("question")
                    or ""
                )
                phase = str(
                    item.get("mapped_phase")
                    or item.get("phase")
                    or map_task_to_phase(task_name)
                )
                model = str(item.get("model") or item.get("model_name") or "")
                quality = (
                    item.get("quality")
                    if item.get("quality") is not None
                    else item.get("score", 0.5)
                )
                input_tokens = item.get("input_tokens") or item.get("prompt_tokens") or 1000
                cost = item.get("cost", 0.0)
                samples.append(
                    {
                        "task_name": task_name,
                        "prompt": prompt,
                        "model": model,
                        "model_name": model,
                        "quality": quality,
                        "cost": cost,
                        "input_tokens": input_tokens,
                        "phase": phase,
                        "mapped_phase": phase,
                    }
                )

    # 2. xroutebench
    xrb = data.get("xroutebench")
    if isinstance(xrb, list):
        for item in xrb:
            if isinstance(item, dict):
                task_name = str(item.get("task_name") or item.get("task") or "benchmark")
                prompt = str(item.get("prompt") or item.get("query") or "")
                phase = str(
                    item.get("mapped_phase")
                    or item.get("phase")
                    or map_task_to_phase(task_name)
                )
                model = str(item.get("model_name") or item.get("model") or "")
                quality = (
                    item.get("performance")
                    if item.get("performance") is not None
                    else item.get("score", 0.5)
                )
                input_tokens = item.get("input_tokens") or item.get("prompt_tokens") or 1000
                cost = item.get("cost", 0.0)
                samples.append(
                    {
                        "task_name": task_name,
                        "prompt": prompt,
                        "model": model,
                        "model_name": model,
                        "quality": quality,
                        "cost": cost,
                        "input_tokens": input_tokens,
                        "phase": phase,
                        "mapped_phase": phase,
                    }
                )

    # 3. compendium
    comp = data.get("compendium")
    if isinstance(comp, list):
        for item in comp:
            if isinstance(item, dict):
                task_name = str(
                    item.get("dataset")
                    or item.get("task_name")
                    or item.get("task")
                    or "benchmark"
                )
                prompt = str(item.get("prompt") or item.get("query") or "")
                phase = str(
                    item.get("mapped_phase")
                    or item.get("phase")
                    or map_task_to_phase(task_name)
                )
                models_name = item.get("models_name") or item.get("models")
                models_perf = item.get("models_performance") or item.get("performance")
                input_tokens = item.get("input_tokens") or item.get("prompt_tokens") or 1000
                cost = item.get("cost", 0.0)
                if isinstance(models_name, (list, tuple)) and isinstance(
                    models_perf, (list, tuple)
                ):
                    for m, p in zip(models_name, models_perf, strict=False):
                        samples.append(
                            {
                                "task_name": task_name,
                                "prompt": prompt,
                                "model": str(m),
                                "model_name": str(m),
                                "quality": p,
                                "cost": cost,
                                "input_tokens": input_tokens,
                                "phase": phase,
                                "mapped_phase": phase,
                            }
                        )
                else:
                    model = str(item.get("model") or item.get("model_name") or "")
                    quality = item.get(
                        "quality", item.get("performance", item.get("score", 0.5))
                    )
                    samples.append(
                        {
                            "task_name": task_name,
                            "prompt": prompt,
                            "model": model,
                            "model_name": model,
                            "quality": quality,
                            "cost": cost,
                            "input_tokens": input_tokens,
                            "phase": phase,
                            "mapped_phase": phase,
                        }
                    )

    # 4. Direct samples fallback
    samples_list = data.get("samples")
    if isinstance(samples_list, list):
        for item in samples_list:
            if isinstance(item, dict):
                task_name = str(
                    item.get("task_name")
                    or item.get("task")
                    or item.get("dataset")
                    or "benchmark"
                )
                prompt = str(
                    item.get("prompt")
                    or item.get("input_question")
                    or item.get("question")
                    or item.get("query")
                    or ""
                )
                phase = str(
                    item.get("mapped_phase")
                    or item.get("phase")
                    or map_task_to_phase(task_name)
                )
                model = str(item.get("model") or item.get("model_name") or "")
                quality = item.get(
                    "quality", item.get("performance", item.get("score", 0.5))
                )
                input_tokens = item.get("input_tokens") or item.get("prompt_tokens") or 1000
                cost = item.get("cost", 0.0)
                samples.append(
                    {
                        "task_name": task_name,
                        "prompt": prompt,
                        "model": model,
                        "model_name": model,
                        "quality": quality,
                        "cost": cost,
                        "input_tokens": input_tokens,
                        "phase": phase,
                        "mapped_phase": phase,
                    }
                )

    return samples


def _extract_ground_truth_samples(source_name: str, payload: Any) -> list[dict[str, Any]]:
    """Extract normalized ground-truth evaluation samples from snapshot payload."""
    data = (
        payload.get("data")
        if isinstance(payload, dict) and "data" in payload
        else payload
    )
    if not isinstance(data, dict):
        if isinstance(data, list):
            rows = data
        else:
            return []
    else:
        rows = data.get("samples") or data.get("trajectories") or []
        if not isinstance(rows, list):
            return []

    samples: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        if source_name == "swe-traces" or "diff" in item:
            inst_id = str(item.get("instance_id") or item.get("task_id") or "trace")
            prompt = str(
                item.get("task_description")
                or item.get("problem_statement")
                or item.get("prompt")
                or ""
            )
            task_name = str(item.get("repo") or inst_id)
            phase = str(item.get("mapped_phase") or "apply")
            model = str(item.get("model_name") or item.get("model") or "")
            quality = 1.0 if item.get("test_passed") else 0.0
            cost = float(item.get("cost") or 0.0)
            in_tokens = int(item.get("input_tokens") or item.get("prompt_tokens") or 1500)
            samples.append({
                "task_name": task_name,
                "prompt": prompt,
                "model": model,
                "model_name": model,
                "quality": quality,
                "cost": cost,
                "input_tokens": in_tokens,
                "phase": phase,
                "mapped_phase": phase,
            })
        elif source_name == "routellm" or "model_a" in item:
            prompt = str(item.get("prompt") or item.get("instruction") or "")
            task_name = str(item.get("task_name") or item.get("task") or "routellm")
            phase = str(item.get("mapped_phase") or "explore")
            in_tokens = int(item.get("input_tokens") or 1000)
            m_a = str(item.get("model_a") or "")
            if m_a:
                samples.append({
                    "task_name": task_name,
                    "prompt": prompt,
                    "model": m_a,
                    "model_name": m_a,
                    "quality": float(
                        item.get("score_a") if item.get("score_a") is not None else 0.5
                    ),
                    "cost": float(item.get("cost_a") or 0.0),
                    "input_tokens": in_tokens,
                    "phase": phase,
                    "mapped_phase": phase,
                })
            m_b = str(item.get("model_b") or "")
            if m_b:
                samples.append({
                    "task_name": task_name,
                    "prompt": prompt,
                    "model": m_b,
                    "model_name": m_b,
                    "quality": float(
                        item.get("score_b") if item.get("score_b") is not None else 0.5
                    ),
                    "cost": float(item.get("cost_b") or 0.0),
                    "input_tokens": in_tokens,
                    "phase": phase,
                    "mapped_phase": phase,
                })
        else:
            prompt = str(item.get("prompt") or item.get("query") or "")
            task_name = str(item.get("task_name") or item.get("task") or "routerbench")
            phase = str(item.get("mapped_phase") or "apply")
            model = str(item.get("model_name") or item.get("model") or "")
            quality = float(
                item.get("correctness")
                if item.get("correctness") is not None
                else item.get("score", 0.5)
            )
            cost = float(item.get("cost") or 0.0)
            in_tokens = int(item.get("input_tokens") or item.get("prompt_tokens") or 1000)
            samples.append({
                "task_name": task_name,
                "prompt": prompt,
                "model": model,
                "model_name": model,
                "quality": quality,
                "cost": cost,
                "input_tokens": in_tokens,
                "phase": phase,
                "mapped_phase": phase,
            })
    return samples


def _match_candidate_model(
    model: Model, task_group: dict[str, Any]
) -> tuple[bool, float, float | None]:
    """Check if candidate model matches empirical sample; return (matched, quality, cost)."""
    canonical = model.canonical_id.strip().lower()
    name = (model.name or "").strip().lower()
    short = canonical.split("/")[-1] if "/" in canonical else canonical
    candidates = {canonical, name, short}

    # Check task_group["models"] dictionary if grouped
    models_dict = task_group.get("models")
    if isinstance(models_dict, dict):
        for raw_m, eval_data in models_dict.items():
            m_str = str(raw_m).strip().lower()
            m_short = m_str.split("/")[-1] if "/" in m_str else m_str
            if m_str in candidates or m_short in candidates:
                raw_q = eval_data.get("quality")
                if raw_q is None:
                    raw_q = eval_data.get("performance")
                if raw_q is None:
                    raw_q = eval_data.get("score")
                if raw_q is None:
                    raw_q = 0.5
                try:
                    q = float(raw_q)
                except (ValueError, TypeError):
                    q = 0.5
                c = eval_data.get("cost")
                try:
                    cost_val = float(c) if c is not None else None
                except (ValueError, TypeError):
                    cost_val = None
                return True, q, cost_val

    # Check task_group.get("model") or task_group.get("model_name")
    raw_m = task_group.get("model") or task_group.get("model_name")
    if raw_m:
        m_str = str(raw_m).strip().lower()
        m_short = m_str.split("/")[-1] if "/" in m_str else m_str
        if m_str in candidates or m_short in candidates:
            raw_q = task_group.get("quality")
            if raw_q is None:
                raw_q = task_group.get("performance")
            if raw_q is None:
                raw_q = task_group.get("score")
            if raw_q is None:
                raw_q = 0.5
            try:
                q = float(raw_q)
            except (ValueError, TypeError):
                q = 0.5
            c = task_group.get("cost")
            try:
                cost_val = float(c) if c is not None else None
            except (ValueError, TypeError):
                cost_val = None
            return True, q, cost_val

    return False, 0.5, None


def _resolve_telemetry_candidate(
    raw: list[tuple[Model, Provider, Deployment, ModelVariant]],
    model_name: str,
    deployment_ref: str | None,
    effort: str | None,
) -> tuple[Model, Provider, Deployment, ModelVariant] | None:
    """Map a telemetry execution's free-text candidate to exactly one registry row.

    Strict canonical-id match first, case-insensitive name fallback; an
    explicitly recorded deployment/effort must match exactly (otherwise the
    row is ambiguous → None → skipped with a counted warning). Missing
    deployment/effort resolve to the first registry row in the build's
    deterministic (canonical, deployment, effort) ordering.
    """
    matches = [row for row in raw if row[0].canonical_id == model_name]
    if not matches:
        low = model_name.strip().lower()
        matches = [
            row
            for row in raw
            if row[0].canonical_id.strip().lower() == low
            or (row[0].name or "").strip().lower() == low
        ]
    if not matches:
        return None
    if deployment_ref:
        selected = [row for row in matches if row[2].deployment_ref == deployment_ref]
        if not selected:
            low = deployment_ref.strip().lower()
            selected = [
                row for row in matches if row[2].deployment_ref.strip().lower() == low
            ]
        if not selected:
            return None
        matches = selected
    if effort:
        selected = [row for row in matches if row[3].effort == effort]
        if not selected:
            return None
        matches = selected
    return matches[0]


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_examples(
    session: Session,
    config: RouterConfig,
    builder: DatasetBuilderConfig,
    store: SnapshotStore | None = None,
    telemetry_db: str | None = None,
) -> DatasetV1:
    """Generate DatasetV1 from registry priors. Deterministic; no randomness.

    ``telemetry_db`` enables the optional telemetry bridge (path to the shim
    SQLite DB, or a falsy string to use ``config.telemetry_url``). Off by
    default: without it the output is byte-identical to a bootstrap-only
    build. When on, scored shim executions are added as
    ``label_provenance='telemetry'`` rows (see dataset/telemetry_bridge.py);
    invalid rows are skipped with a counted warning, an unreadable DB is a
    typed DatasetBuildError, and an absent DB yields zero telemetry rows.
    """
    if store is None:
        store = SnapshotStore(config.data_dir / "snapshots")
    snapshot_dates = _snapshot_dates(session)
    if not snapshot_dates:
        raise DatasetBuildError(
            "registry has no model_snapshots rows; cannot date features or "
            "enforce anti-leakage. Run 'router collect' + 'router normalize' first."
        )
    as_of = builder.as_of or max(snapshot_dates.values())
    policy = config.policy

    # --- optional telemetry bridge (read early; fail closed before staging) - #
    telemetry_rows: list[TelemetryExecution] = []
    if telemetry_db is not None:
        url = telemetry_db or config.telemetry_url
        if "://" not in url:
            url = f"sqlite:///{url}"
        try:
            telemetry_rows = read_telemetry_executions(url).rows
        except TelemetryReadError as exc:
            raise DatasetBuildError(str(exc)) from exc
        logger.info("telemetry bridge: %d scored shim row(s) read", len(telemetry_rows))

    # --- candidates ------------------------------------------------------- #
    stmt = (
        select(Model, Provider, Deployment, ModelVariant)
        .join(Deployment, Deployment.model_id == Model.id)
        .join(Provider, Provider.id == Deployment.provider_id)
        .join(ModelVariant, ModelVariant.deployment_id == Deployment.id)
        .order_by(Model.canonical_id, Deployment.deployment_ref, ModelVariant.effort)
    )
    raw = list(session.execute(stmt).all())
    if not raw:
        raise DatasetBuildError(
            "registry is empty: no (model, deployment, variant) candidates."
        )
    models = {m.id: m for m, _, _, _ in raw}
    deployments = {d.id: d for _, _, d, _ in raw}

    # --- benchmark rows (dated + leakage-checked) ------------------------- #
    bench_rows = session.execute(
        select(ModelBenchmark).where(ModelBenchmark.model_id.in_(models.keys()))
    ).scalars().all()
    scores: dict[tuple[int, str], float] = {}
    model_source_dates: dict[int, list[date]] = {mid: [] for mid in models}
    for row in bench_rows:
        key = row.benchmark if row.category is None else f"{row.benchmark}:{row.category}"
        scores[(row.model_id, key)] = row.score
        row_date = _check_row_date(snapshot_dates, row.source_snapshot_id, as_of)
        model_source_dates[row.model_id].append(row_date)

    # --- prices (dated + leakage-checked; ModelPrice maps via deployment) --- #
    price_rows = session.execute(
        select(ModelPrice).where(ModelPrice.deployment_id.in_(deployments.keys()))
    ).scalars().all()
    price_by_deployment: dict[int, ModelPrice] = {}
    for row in price_rows:
        row_date = _check_row_date(snapshot_dates, row.source_snapshot_id, as_of)
        deployment = deployments[row.deployment_id]
        model_source_dates[deployment.model_id].append(row_date)
        price_by_deployment[row.deployment_id] = row

    # --- first-appearance bookkeeping -------------------------------------- #
    newest_snapshot_date = max(snapshot_dates.values())
    first_seen: dict[int, date] = {}
    for mid in models:
        dates = model_source_dates.get(mid) or []
        if dates:
            first_seen[mid] = min(dates)
        else:
            # Fallback (documented): the model carries no dated provenance
            # rows, so its first appearance is not trackable. Assign the
            # NEWEST snapshot bucket — i.e. treat it as a new model. This is
            # conservative for temporal evaluation and loud in the manifest.
            first_seen[mid] = newest_snapshot_date

    # --- benchmark feature keys (union of phase weight keys) --------------- #
    weight_keys: set[str] = set()
    for phase in builder.phases:
        weight_keys.update(config.phase_config(phase).weights.keys())
    bench_names = tuple(sorted(weight_keys))

    # --- per-phase priors (SAME model as router/policy.py) ----------------- #
    priors_by_phase: dict[str, dict[int, float]] = {}
    for phase in builder.phases:
        phase_cfg = config.phase_config(phase)
        priors, _missing = benchmark_priors(
            session, list(models.values()), phase_cfg.weights, policy.flat_prior
        )
        priors_by_phase[phase] = priors

    # --- optional tps proxy (same source as the policy latency term) ------- #
    speed_rows: dict[int, float] = {}
    if policy.speed_benchmark:
        stmt = select(ModelBenchmark).where(
            ModelBenchmark.benchmark == policy.speed_benchmark,
            ModelBenchmark.model_id.in_(models.keys()),
        )
        for row in session.execute(stmt).scalars().all():
            row_date = _check_row_date(snapshot_dates, row.source_snapshot_id, as_of)
            speed_rows[row.model_id] = row.score
            model_source_dates[row.model_id].append(row_date)
    speed_top = max(speed_rows.values()) if speed_rows else 0.0

    dataset = DatasetV1(
        name=builder.name,
        version=0,  # assigned by write_dataset
        source_snapshot_ids=sorted(snapshot_dates.keys()),
        label_provenance_statement=PROVENANCE_STATEMENT,
        git_commit=_git_commit(),
        created_at=datetime.now(UTC).isoformat(),
        threshold_penalty=builder.threshold_penalty,
        hard_threshold=builder.hard_threshold,
    )

    # --- examples ----------------------------------------------------------- #
    # Group token maxima per (phase, task_id) for the cost penalty normalization.
    group_tokens: dict[tuple[str, str], float] = {}
    staged: list[tuple[DatasetExample, int]] = []  # (example, model_id)
    for phase in builder.phases:
        phase_cfg = config.phase_config(phase)
        priors = priors_by_phase[phase]
        for task_type in builder.task_types:
            for context_tokens in builder.context_sizes:
                task_id = _task_id(phase, task_type, context_tokens, builder.repo_features)
                task_text = render_task_text(
                    phase, task_type, context_tokens, builder.repo_features
                )
                context = TaskContext(
                    task_type=task_type, context_tokens=context_tokens,
                    repo_features=dict(builder.repo_features),
                )
                for model, _provider, deployment, variant in raw:
                    base = max(policy.base_tokens, context.context_tokens or 0)
                    multiplier = policy.effort_token_multiplier.get(variant.effort, 1.0)
                    est_tokens = base * multiplier
                    group_key = (phase, task_id)
                    group_tokens[group_key] = max(group_tokens.get(group_key, 0.0), est_tokens)

                    price = price_by_deployment.get(deployment.id)
                    in_p = (
                        price.input_price
                        if price and price.input_price is not None
                        else policy.default_input_price
                    )
                    out_p = (
                        price.output_price
                        if price and price.output_price is not None
                        else policy.default_output_price
                    )
                    est_cost = estimate_cost(policy, in_p, out_p, est_tokens)
                    quality = effort_quality(priors[model.id], variant.effort, policy)
                    dates = model_source_dates[model.id]
                    example_date = max(dates) if dates else newest_snapshot_date

                    example = DatasetExample(
                        example_id="",
                        task_id=task_id,
                        phase=phase,
                        task_type=task_type,
                        context_tokens=context_tokens,
                        task_text=task_text,
                        candidate=CandidateRef(
                            model=model.canonical_id,
                            deployment=deployment.deployment_ref,
                            effort=variant.effort,
                        ),
                        model_features=_model_feature_vector(
                            model,
                            price,
                            (in_p, out_p),
                            tps_proxy=(
                                speed_rows[model.id] / speed_top
                                if speed_rows and speed_top and model.id in speed_rows
                                else 0.0
                            ),
                        ),
                        benchmark_features=[scores.get((model.id, k), 0.0) for k in bench_names],
                        benchmark_feature_names=bench_names,
                        cost_features={
                            "est_input_tokens": est_tokens * (1 - policy.output_fraction),
                            "est_output_tokens": est_tokens * policy.output_fraction,
                            "est_total_tokens": est_tokens,
                            "est_cost": est_cost,
                        },
                        label_utility=0.0,  # filled after group maxima are known
                        label_quality_estimate=quality,
                        label_provenance=PROVENANCE_BOOTSTRAP,
                        snapshot_date=example_date.isoformat(),
                    )
                    staged.append((example, model.id))

    # --- empirical benchmark tasks ------------------------------------------ #
    if builder.include_empirical_benchmarks:
        latest_snap = store.latest(builder.empirical_snapshot_source)
        if latest_snap:
            snap_id = latest_snap.get("snapshot_id")
            if snap_id and snap_id not in dataset.source_snapshot_ids:
                dataset.source_snapshot_ids.append(snap_id)
            payload = (
                latest_snap.get("data")
                if isinstance(latest_snap, dict) and "data" in latest_snap
                else latest_snap
            )
            raw_samples = _extract_empirical_samples(payload)
            for phase in builder.phases:
                priors = priors_by_phase[phase]
                phase_samples = [
                    s
                    for s in raw_samples
                    if (s.get("phase") or s.get("mapped_phase")) == phase
                ]
                # Group samples by (task_name, prompt) to allow multiple model
                # evals on the same task.
                task_groups: dict[tuple[str, str], dict[str, Any]] = {}
                for s in phase_samples:
                    t_name = str(s.get("task_name") or "benchmark")
                    p_text = str(s.get("prompt") or "")
                    key = (t_name, p_text)
                    if key not in task_groups:
                        if len(task_groups) >= builder.max_empirical_tasks_per_phase:
                            continue
                        task_groups[key] = {
                            "task_name": t_name,
                            "prompt": p_text,
                            "phase": phase,
                            "input_tokens": s.get("input_tokens") or 1000,
                            "models": {},
                        }
                    m = s.get("model") or s.get("model_name")
                    if m:
                        task_groups[key]["models"][m] = {
                            "quality": s.get(
                                "quality",
                                s.get("performance", s.get("score", 0.5)),
                            ),
                            "cost": s.get("cost", 0.0),
                        }
                        if "model" not in task_groups[key]:
                            task_groups[key]["model"] = m
                            task_groups[key]["quality"] = s.get(
                                "quality",
                                s.get("performance", s.get("score", 0.5)),
                            )
                            task_groups[key]["cost"] = s.get("cost", 0.0)

                for task_group in task_groups.values():
                    t_name = task_group.get("task_name", "")
                    p_text = task_group.get("prompt", "")
                    task_id = "task-emp-" + hashlib.sha256(
                        f"{phase}:{t_name}:{p_text}".encode()
                    ).hexdigest()[:16]
                    task_text = (
                        f"[phase] {phase}\n"
                        f"[task_type] {task_group.get('task_name', 'benchmark')}\n"
                        f"[task] {str(p_text)[:400].strip()}"
                    )
                    input_tokens = int(task_group.get("input_tokens") or 1000)

                    for model, _provider, deployment, variant in raw:
                        base = max(policy.base_tokens, input_tokens)
                        multiplier = policy.effort_token_multiplier.get(variant.effort, 1.0)
                        est_tokens = base * multiplier
                        group_key = (phase, task_id)
                        group_tokens[group_key] = max(
                            group_tokens.get(group_key, 0.0), est_tokens
                        )

                        price = price_by_deployment.get(deployment.id)
                        in_p = (
                            price.input_price
                            if price and price.input_price is not None
                            else policy.default_input_price
                        )
                        out_p = (
                            price.output_price
                            if price and price.output_price is not None
                            else policy.default_output_price
                        )
                        est_cost = estimate_cost(policy, in_p, out_p, est_tokens)

                        matched, match_q, match_cost = _match_candidate_model(
                            model, task_group
                        )
                        if matched:
                            quality = match_q
                            if match_cost is not None and match_cost > 0.0:
                                est_cost = match_cost
                            label_provenance = PROVENANCE_EMPIRICAL
                        else:
                            quality = effort_quality(priors[model.id], variant.effort, policy)
                            label_provenance = PROVENANCE_BOOTSTRAP

                        dates = model_source_dates[model.id]
                        example_date = max(dates) if dates else newest_snapshot_date

                        example = DatasetExample(
                            example_id="",
                            task_id=task_id,
                            phase=phase,
                            task_type=task_group.get("task_name", "benchmark"),
                            context_tokens=input_tokens,
                            task_text=task_text,
                            candidate=CandidateRef(
                                model=model.canonical_id,
                                deployment=deployment.deployment_ref,
                                effort=variant.effort,
                            ),
                            model_features=_model_feature_vector(
                                model,
                                price,
                                (in_p, out_p),
                                tps_proxy=(
                                    speed_rows[model.id] / speed_top
                                    if speed_rows and speed_top and model.id in speed_rows
                                    else 0.0
                                ),
                            ),
                            benchmark_features=[
                                scores.get((model.id, k), 0.0) for k in bench_names
                            ],
                            benchmark_feature_names=bench_names,
                            cost_features={
                                "est_input_tokens": est_tokens * (1 - policy.output_fraction),
                                "est_output_tokens": est_tokens * policy.output_fraction,
                                "est_total_tokens": est_tokens,
                                "est_cost": est_cost,
                            },
                            label_utility=0.0,
                            label_quality_estimate=quality,
                            label_provenance=label_provenance,
                            snapshot_date=example_date.isoformat(),
                        )
                        staged.append((example, model.id))

    # --- ground-truth traces (RouterBench, RouteLLM, SWE-Traces) ------------- #
    if builder.include_ground_truth_traces and store is not None:
        for source_name in builder.ground_truth_snapshot_sources:
            latest_snap = store.latest(source_name)
            if not latest_snap:
                continue
            snap_id = latest_snap.get("snapshot_id")
            if snap_id and snap_id not in dataset.source_snapshot_ids:
                dataset.source_snapshot_ids.append(snap_id)
            gt_samples = _extract_ground_truth_samples(source_name, latest_snap)
            for phase in builder.phases:
                priors = priors_by_phase[phase]
                phase_samples = [
                    s
                    for s in gt_samples
                    if (s.get("phase") or s.get("mapped_phase")) == phase
                ]
                task_groups: dict[tuple[str, str], dict[str, Any]] = {}
                for s in phase_samples:
                    t_name = str(s.get("task_name") or "ground_truth")
                    p_text = str(s.get("prompt") or "")
                    key = (t_name, p_text)
                    if key not in task_groups:
                        if len(task_groups) >= builder.max_ground_truth_tasks_per_phase:
                            continue
                        task_groups[key] = {
                            "task_name": t_name,
                            "prompt": p_text,
                            "phase": phase,
                            "input_tokens": s.get("input_tokens") or 1000,
                            "models": {},
                        }
                    m = s.get("model") or s.get("model_name")
                    if m:
                        task_groups[key]["models"][m] = {
                            "quality": s.get("quality", 0.5),
                            "cost": s.get("cost", 0.0),
                        }
                        if "model" not in task_groups[key]:
                            task_groups[key]["model"] = m
                            task_groups[key]["quality"] = s.get("quality", 0.5)
                            task_groups[key]["cost"] = s.get("cost", 0.0)

                for task_group in task_groups.values():
                    t_name = task_group.get("task_name", "")
                    p_text = task_group.get("prompt", "")
                    task_id = "task-gt-" + hashlib.sha256(
                        f"{source_name}:{phase}:{t_name}:{p_text}".encode()
                    ).hexdigest()[:16]
                    task_text = (
                        f"[phase] {phase}\n"
                        f"[source] {source_name}\n"
                        f"[task_type] {t_name}\n"
                        f"[task] {str(p_text)[:400].strip()}"
                    )
                    input_tokens = int(task_group.get("input_tokens") or 1000)

                    for model, _provider, deployment, variant in raw:
                        base = max(policy.base_tokens, input_tokens)
                        multiplier = policy.effort_token_multiplier.get(variant.effort, 1.0)
                        est_tokens = base * multiplier
                        group_key = (phase, task_id)
                        group_tokens[group_key] = max(
                            group_tokens.get(group_key, 0.0), est_tokens
                        )

                        price = price_by_deployment.get(deployment.id)
                        in_p = (
                            price.input_price
                            if price and price.input_price is not None
                            else policy.default_input_price
                        )
                        out_p = (
                            price.output_price
                            if price and price.output_price is not None
                            else policy.default_output_price
                        )
                        est_cost = estimate_cost(policy, in_p, out_p, est_tokens)

                        matched, match_q, match_cost = _match_candidate_model(
                            model, task_group
                        )
                        if matched:
                            quality = match_q
                            if match_cost is not None and match_cost > 0.0:
                                est_cost = match_cost
                            label_provenance = PROVENANCE_GROUND_TRUTH
                        else:
                            quality = effort_quality(priors[model.id], variant.effort, policy)
                            label_provenance = PROVENANCE_BOOTSTRAP

                        dates = model_source_dates[model.id]
                        example_date = max(dates) if dates else newest_snapshot_date

                        example = DatasetExample(
                            example_id="",
                            task_id=task_id,
                            phase=phase,
                            task_type=task_group.get("task_name", "ground_truth"),
                            context_tokens=input_tokens,
                            task_text=task_text,
                            candidate=CandidateRef(
                                model=model.canonical_id,
                                deployment=deployment.deployment_ref,
                                effort=variant.effort,
                            ),
                            model_features=_model_feature_vector(
                                model,
                                price,
                                (in_p, out_p),
                                tps_proxy=(
                                    speed_rows[model.id] / speed_top
                                    if speed_rows and speed_top and model.id in speed_rows
                                    else 0.0
                                ),
                            ),
                            benchmark_features=[
                                scores.get((model.id, k), 0.0) for k in bench_names
                            ],
                            benchmark_feature_names=bench_names,
                            cost_features={
                                "est_input_tokens": est_tokens * (1 - policy.output_fraction),
                                "est_output_tokens": est_tokens * policy.output_fraction,
                                "est_total_tokens": est_tokens,
                                "est_cost": est_cost,
                            },
                            label_utility=0.0,
                            label_quality_estimate=quality,
                            label_provenance=label_provenance,
                            snapshot_date=example_date.isoformat(),
                        )
                        staged.append((example, model.id))

    # --- telemetry executions (additive; utility filled after group maxima) - #
    staged_tel: list[tuple[DatasetExample, int]] = []  # (example, model_id)
    telemetry_group_tokens: dict[tuple[str, str], float] = {}
    telemetry_skipped = 0
    if telemetry_db is not None:
        for trow in telemetry_rows:
            skip_reason: str | None = None
            resolved: tuple[Model, Provider, Deployment, ModelVariant] | None = None
            quality = 0.0
            if trow.phase not in builder.phases:
                skip_reason = f"phase '{trow.phase}' not in build phases"
            elif (
                trow.input_tokens < 0
                or trow.output_tokens < 0
                or trow.total_tokens < 0
            ):
                skip_reason = "negative token counts"
            elif trow.model.strip() == "":
                skip_reason = "empty model id"
            else:
                resolved = _resolve_telemetry_candidate(
                    raw, trow.model, trow.deployment, trow.effort
                )
                if resolved is None:
                    skip_reason = (
                        f"candidate ({trow.model}, {trow.deployment}, {trow.effort}) "
                        "not resolvable in the registry"
                    )
            if skip_reason is None:
                quality = (
                    trow.quality_score
                    if trow.quality_score is not None
                    else float(trow.task_success)  # quality_score None ⇒ success set
                )
                if not 0.0 <= quality <= 1.0:
                    skip_reason = f"outcome {quality} outside [0, 1]"
            if skip_reason is not None:
                telemetry_skipped += 1
                logger.warning(
                    "telemetry bridge: skipping execution %s: %s",
                    trow.execution_id,
                    skip_reason,
                )
                continue

            model, _provider, deployment, variant = resolved
            actual_total = float(
                trow.total_tokens or (trow.input_tokens + trow.output_tokens)
            )
            price = price_by_deployment.get(deployment.id)
            in_p = (
                price.input_price
                if price and price.input_price is not None
                else policy.default_input_price
            )
            out_p = (
                price.output_price
                if price and price.output_price is not None
                else policy.default_output_price
            )
            # est_* keeps the decision's pre-execution estimate where present
            # (0 means "not estimated" on legacy rows → fall back to actuals so
            # the v1 keys stay meaningful on telemetry rows too).
            est_total = trow.estimated_tokens if trow.estimated_tokens > 0 else actual_total
            est_cost = (
                trow.estimated_cost
                if trow.estimated_cost > 0
                else estimate_cost(policy, in_p, out_p, est_total)
            )
            actual_cost = estimate_cost(policy, in_p, out_p, actual_total)

            task_type = trow.task_type or "telemetry"
            task_id = "task-tel-" + hashlib.sha256(
                f"{trow.phase}|{task_type}|{trow.event_date.isoformat()}".encode()
            ).hexdigest()[:16]
            repo = (
                trow.repo_features
                if isinstance(trow.repo_features, dict)
                else dict(builder.repo_features)
            )
            task_text = render_task_text(trow.phase, task_type, trow.input_tokens, repo)
            group_key = (trow.phase, task_id)
            telemetry_group_tokens[group_key] = max(
                telemetry_group_tokens.get(group_key, 0.0), actual_total
            )

            example = DatasetExample(
                example_id="ex-tel-" + hashlib.sha256(
                    f"{task_id}|{model.canonical_id}#{deployment.deployment_ref}"
                    f"#{variant.effort}|{trow.execution_id}".encode()
                ).hexdigest()[:16],
                task_id=task_id,
                phase=trow.phase,
                task_type=task_type,
                context_tokens=trow.input_tokens,
                task_text=task_text,
                candidate=CandidateRef(
                    model=model.canonical_id,
                    deployment=deployment.deployment_ref,
                    effort=variant.effort,
                ),
                model_features=_model_feature_vector(
                    model,
                    price,
                    (in_p, out_p),
                    tps_proxy=(
                        speed_rows[model.id] / speed_top
                        if speed_rows and speed_top and model.id in speed_rows
                        else 0.0
                    ),
                ),
                benchmark_features=[scores.get((model.id, k), 0.0) for k in bench_names],
                benchmark_feature_names=bench_names,
                cost_features={
                    "est_input_tokens": est_total * (1 - policy.output_fraction),
                    "est_output_tokens": est_total * policy.output_fraction,
                    "est_total_tokens": est_total,
                    "est_cost": est_cost,
                    "actual_input_tokens": float(trow.input_tokens),
                    "actual_output_tokens": float(trow.output_tokens),
                    "actual_total_tokens": actual_total,
                    "actual_cost": actual_cost,
                    "latency_ms": (
                        float(trow.latency_ms) if trow.latency_ms is not None else None
                    ),
                },
                label_utility=0.0,  # filled after telemetry group maxima are known
                label_quality_estimate=quality,
                label_provenance=PROVENANCE_TELEMETRY,
                snapshot_date=trow.event_date.isoformat(),
                split=_temporal_split(
                    trow.event_date,
                    first_seen[model.id],
                    builder.train_end,
                    builder.val_end,
                ),
            )
            staged_tel.append((example, model.id))

    for example, mid in staged:
        example.label_utility = _utility_label(
            example.label_quality_estimate,
            example.cost_features["est_total_tokens"],
            group_tokens[(example.phase, example.task_id)],
            cost_weight=builder.cost_weight,
            threshold=config.phase_config(example.phase).threshold_quality,
            threshold_penalty=builder.threshold_penalty,
            hard_threshold=builder.hard_threshold,
        )
        example.example_id = "ex-" + hashlib.sha256(
            f"{example.task_id}|{example.candidate.key}".encode()
        ).hexdigest()[:16]
        # --- temporal split ------------------------------------------------ #
        example.split = _temporal_split(
            date.fromisoformat(example.snapshot_date),
            first_seen[mid],
            builder.train_end,
            builder.val_end,
        )

    # Telemetry utilities use ACTUAL total tokens against the telemetry group
    # maxima (same formula as bootstrap, so provenances stay comparable).
    for example, _mid in staged_tel:
        example.label_utility = _utility_label(
            example.label_quality_estimate,
            example.cost_features["actual_total_tokens"],
            telemetry_group_tokens[(example.phase, example.task_id)],
            cost_weight=builder.cost_weight,
            threshold=config.phase_config(example.phase).threshold_quality,
            threshold_penalty=builder.threshold_penalty,
            hard_threshold=builder.hard_threshold,
        )

    dataset.examples = [example for example, _ in staged] + [
        example for example, _ in staged_tel
    ]
    if telemetry_db is not None:
        # Uniform schema-v2 cost keys across EVERY row: pyarrow infers the
        # struct type from the data, so rows without measurements must carry
        # explicit None instead of omitting the key.
        for example in dataset.examples:
            for key in (
                "actual_input_tokens",
                "actual_output_tokens",
                "actual_total_tokens",
                "actual_cost",
                "latency_ms",
            ):
                example.cost_features.setdefault(key, None)
        dataset.telemetry_stats = {
            "read": len(telemetry_rows),
            "emitted": len(staged_tel),
            "skipped": telemetry_skipped,
        }
        if staged_tel:
            dataset.label_provenance_statement = (
                PROVENANCE_STATEMENT + "\n\n" + TELEMETRY_PROVENANCE_ADDENDUM
            )
    dataset.benchmark_feature_names = bench_names
    dataset.model_feature_names = MODEL_FEATURE_NAMES

    # --- anti-leakage: task_id overlap -------------------------------------- #
    non_train_tasks = {e.task_id for e in dataset.examples if e.split != "train"}
    kept: list[DatasetExample] = []
    for example in dataset.examples:
        if example.split == "train" and example.task_id in non_train_tasks:
            dataset.dropped_train_task_overlap += 1
            continue
        kept.append(example)
    dataset.examples = kept

    # --- pairwise preference pairs ------------------------------------------ #
    by_group: dict[str, list[DatasetExample]] = {}
    for example in dataset.examples:
        by_group.setdefault(example.task_id, []).append(example)
    temporal_models = {mid for mid, d in first_seen.items() if d > builder.train_end}
    canonical_to_id = {m.canonical_id: mid for mid, m in models.items()}

    def _pair_split(pair_date: str, mid_a: int, mid_b: int) -> str:
        if mid_a in temporal_models or mid_b in temporal_models:
            return "temporal_test"
        parsed = date.fromisoformat(pair_date)
        if parsed < builder.train_end:
            return "train"
        if builder.val_end is not None and parsed < builder.val_end:
            return "validation"
        return "test"

    for task_id in sorted(by_group):
        if task_id.startswith("task-tel-"):
            continue  # telemetry rows are pointwise labels; pairs stay prior-derived
        group = sorted(by_group[task_id], key=lambda e: e.candidate.key)
        phase = group[0].phase
        pair_count = 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if pair_count >= builder.max_pairs_per_group:
                    break
                a, b = group[i], group[j]
                margin = a.label_utility - b.label_utility
                if margin < builder.pair_margin:
                    continue
                pair_date = max(a.snapshot_date, b.snapshot_date)
                mid_a = canonical_to_id[a.candidate.model]
                mid_b = canonical_to_id[b.candidate.model]
                if (
                    a.label_provenance == PROVENANCE_GROUND_TRUTH
                    or b.label_provenance == PROVENANCE_GROUND_TRUTH
                ):
                    pair_provenance = PROVENANCE_GROUND_TRUTH
                elif (
                    a.label_provenance == PROVENANCE_EMPIRICAL
                    or b.label_provenance == PROVENANCE_EMPIRICAL
                ):
                    pair_provenance = PROVENANCE_EMPIRICAL
                else:
                    pair_provenance = PROVENANCE_BOOTSTRAP
                dataset.pairs.append(
                    PreferencePair(
                        pair_id="pr-" + hashlib.sha256(
                            f"{task_id}|{a.candidate.key}|{b.candidate.key}".encode()
                        ).hexdigest()[:16],
                        task_id=task_id,
                        phase=phase,
                        candidate_a=a.candidate,
                        candidate_b=b.candidate,
                        score_a=a.label_utility,
                        score_b=b.label_utility,
                        margin=margin,
                        label_provenance=pair_provenance,
                        snapshot_date=pair_date,
                        split=_pair_split(pair_date, mid_a, mid_b),
                    )
                )
                pair_count += 1

    if any(e.label_provenance == PROVENANCE_GROUND_TRUTH for e in dataset.examples):
        dataset.label_provenance_statement = (
            dataset.label_provenance_statement + "\n\n" + GROUND_TRUTH_PROVENANCE_ADDENDUM
        )

    return dataset


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _example_to_row(example: DatasetExample) -> dict:
    row = asdict(example)
    row["candidate_model"] = example.candidate.model
    row["candidate_deployment"] = example.candidate.deployment
    row["candidate_effort"] = example.candidate.effort
    row.pop("candidate")
    row["model_features"] = list(example.model_features)
    row["benchmark_features"] = list(example.benchmark_features)
    row["benchmark_feature_names"] = list(example.benchmark_feature_names)
    return row


def _pair_to_row(pair: PreferencePair) -> dict:
    row = asdict(pair)
    for side in ("a", "b"):
        ref: CandidateRef = getattr(pair, f"candidate_{side}")
        row[f"candidate_{side}_model"] = ref.model
        row[f"candidate_{side}_deployment"] = ref.deployment
        row[f"candidate_{side}_effort"] = ref.effort
    row.pop("candidate_a")
    row.pop("candidate_b")
    return row


def write_dataset(
    dataset: DatasetV1,
    data_dir: str | Path,
    builder: DatasetBuilderConfig | None = None,
) -> Path:
    """Write examples + pairs + manifest under data_dir/datasets/<name>/v<N>/.

    Parquet when pyarrow is available, JSONL fallback otherwise.
    """
    root = Path(data_dir) / "datasets" / dataset.name
    existing = [p for p in root.glob("v*") if p.is_dir()]
    version = 1 + max((int(p.name[1:]) for p in existing if p.name[1:].isdigit()), default=0)
    dataset.version = version
    out_dir = root / f"v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pa_table = pa.Table.from_pylist([_example_to_row(e) for e in dataset.examples])
        pq.write_table(pa_table, out_dir / "examples.parquet")
        pq.write_table(
            pa.Table.from_pylist([_pair_to_row(p) for p in dataset.pairs]),
            out_dir / "pairs.parquet",
        )
        format_used = "parquet"
    except ImportError:  # pragma: no cover - pyarrow is a hard dependency
        format_used = "jsonl"
        with (out_dir / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for example in dataset.examples:
                fh.write(json.dumps(_example_to_row(example)) + "\n")
        with (out_dir / "pairs.jsonl").open("w", encoding="utf-8") as fh:
            for pair in dataset.pairs:
                fh.write(json.dumps(_pair_to_row(pair)) + "\n")

    manifest = dataset.manifest()
    manifest["format"] = format_used
    if builder is not None:
        manifest["threshold_penalty"] = builder.threshold_penalty
        manifest["hard_threshold"] = builder.hard_threshold
    else:
        manifest["threshold_penalty"] = getattr(dataset, "threshold_penalty", 0.0)
        manifest["hard_threshold"] = getattr(dataset, "hard_threshold", False)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("dataset written: %s (%s)", out_dir, format_used)
    return out_dir


def load_dataset(path: str | Path) -> DatasetV1:
    """Reload a written dataset (parquet or jsonl) into DatasetV1."""
    import pyarrow.parquet as pq

    path = Path(path)
    dataset = DatasetV1(name=path.parent.name, version=int(path.name[1:]))
    examples_file = path / "examples.parquet"
    rows = (
        pq.read_table(examples_file).to_pylist()
        if examples_file.exists()
        else [json.loads(line) for line in (path / "examples.jsonl").read_text().splitlines()]
    )
    for row in rows:
        dataset.examples.append(
            DatasetExample(
                example_id=row["example_id"],
                task_id=row["task_id"],
                phase=row["phase"],
                task_type=row["task_type"],
                context_tokens=int(row.get("context_tokens") or 0),
                task_text=row["task_text"],
                candidate=CandidateRef(
                    model=row["candidate_model"],
                    deployment=row["candidate_deployment"],
                    effort=row["candidate_effort"],
                ),
                model_features=list(row["model_features"]),
                benchmark_features=list(row["benchmark_features"]),
                benchmark_feature_names=tuple(row["benchmark_feature_names"]),
                cost_features={
                    k: float(v) for k, v in row["cost_features"].items() if v is not None
                },
                label_utility=float(row["label_utility"]),
                label_quality_estimate=float(row["label_quality_estimate"]),
                label_provenance=row["label_provenance"],
                snapshot_date=row["snapshot_date"],
                split=row["split"],
            )
        )
    pairs_file = path / "pairs.parquet"
    pair_rows = (
        pq.read_table(pairs_file).to_pylist()
        if pairs_file.exists()
        else [json.loads(line) for line in (path / "pairs.jsonl").read_text().splitlines()]
    )
    for row in pair_rows:
        dataset.pairs.append(
            PreferencePair(
                pair_id=row["pair_id"],
                task_id=row["task_id"],
                phase=row["phase"],
                candidate_a=CandidateRef(
                    model=row["candidate_a_model"],
                    deployment=row["candidate_a_deployment"],
                    effort=row["candidate_a_effort"],
                ),
                candidate_b=CandidateRef(
                    model=row["candidate_b_model"],
                    deployment=row["candidate_b_deployment"],
                    effort=row["candidate_b_effort"],
                ),
                score_a=float(row["score_a"]),
                score_b=float(row["score_b"]),
                margin=float(row["margin"]),
                label_provenance=row["label_provenance"],
                snapshot_date=row["snapshot_date"],
                split=row["split"],
            )
        )
    manifest_file = path / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        dataset.benchmark_feature_names = tuple(manifest.get("benchmark_feature_names", ()))
        dataset.model_feature_names = tuple(manifest.get("model_feature_names", ()))
        dataset.source_snapshot_ids = list(manifest.get("source_snapshot_ids", []))
        dataset.label_provenance_statement = manifest.get("label_provenance_statement", "")
        dataset.git_commit = manifest.get("git_commit", "unknown")
        dataset.created_at = manifest.get("created_at", "")
        dataset.threshold_penalty = float(manifest.get("threshold_penalty", 0.0))
        dataset.hard_threshold = bool(manifest.get("hard_threshold", False))
    return dataset
