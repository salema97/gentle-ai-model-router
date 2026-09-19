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

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.dataset.schema import (
    FEATURE_SCHEMA_VERSION,
    MODEL_FEATURE_NAMES,
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_STATEMENT,
    CandidateRef,
    DatasetExample,
    DatasetV1,
    PreferencePair,
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
    # Temporal split (required — forcing an explicit decision is deliberate).
    train_end: date
    val_end: date | None = None
    # Knowledge cutoff; rows from newer snapshots are a build error.
    as_of: date | None = None
    seed: int = 13


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_examples(
    session: Session,
    config: RouterConfig,
    builder: DatasetBuilderConfig,
) -> DatasetV1:
    """Generate DatasetV1 from registry priors. Deterministic; no randomness."""
    snapshot_dates = _snapshot_dates(session)
    if not snapshot_dates:
        raise DatasetBuildError(
            "registry has no model_snapshots rows; cannot date features or "
            "enforce anti-leakage. Run 'router collect' + 'router normalize' first."
        )
    as_of = builder.as_of or max(snapshot_dates.values())
    policy = config.policy

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

    for example, mid in staged:
        max_tokens = group_tokens[(example.phase, example.task_id)]
        penalty = (
            builder.cost_weight * example.cost_features["est_total_tokens"] / max_tokens
            if max_tokens
            else 0.0
        )
        example.label_utility = max(
            0.0, min(1.0, example.label_quality_estimate - penalty)
        )
        example.example_id = "ex-" + hashlib.sha256(
            f"{example.task_id}|{example.candidate.key}".encode()
        ).hexdigest()[:16]
        # --- temporal split ------------------------------------------------ #
        if first_seen[mid] > builder.train_end:
            example.split = "temporal_test"
        elif date.fromisoformat(example.snapshot_date) < builder.train_end:
            example.split = "train"
        elif (
            builder.val_end is not None
            and date.fromisoformat(example.snapshot_date) < builder.val_end
        ):
            example.split = "validation"
        else:
            example.split = "test"
    dataset.examples = [example for example, _ in staged]
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
                        label_provenance=PROVENANCE_BOOTSTRAP,
                        snapshot_date=pair_date,
                        split=_pair_split(pair_date, mid_a, mid_b),
                    )
                )
                pair_count += 1

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


def write_dataset(dataset: DatasetV1, data_dir: str | Path) -> Path:
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
    manifest["feature_schema_version"] = FEATURE_SCHEMA_VERSION
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
                cost_features={k: float(v) for k, v in row["cost_features"].items()},
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
    return dataset
