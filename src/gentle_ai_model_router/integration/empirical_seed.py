"""Seed the telemetry shim from empirical benchmark observations.

Phase 5 retraining needs MEASURED outcomes, and ``data/telemetry.sqlite``
starts out empty: zero real executions exist anywhere. This module is the
bootstrap path: it converts the empirical-provenance examples already present
in written datasets (``label_provenance='empirical_benchmark'`` — measured
benchmark outcomes collected by the routing-benchmarks collector, see
:mod:`gentle_ai_model_router.dataset.builder`) into scored shim execution
records, so the telemetry bridge (:mod:`gentle_ai_model_router.dataset.
telemetry_bridge`) and downstream retraining have something real to learn from.

Honesty rules (loud, not silent):

- Every seeded row is stamped ``router_version='empirical-bootstrap'`` so
  bootstrap-derived executions can NEVER be mistaken for real shim
  executions from the instrumented path.
- Token counts are ESTIMATES: empirical dataset rows only carry
  ``cost_features['est_total_tokens']`` (never measured actuals), so the
  seeded ``total_tokens``/``input_tokens``/``output_tokens`` are the
  builder's estimates, not measurements. Real telemetry rows carry
  ``actual_*`` keys; bootstrap-seeded rows never will.
- Outcomes are MAPPED, not fabricated: the empirical label IS a measured
  benchmark quality on the dataset's 0..1 scale, and the shim store keeps
  that same 0..1 scale (the telemetry bridge validates ``0.0 <= quality <=
  1.0`` and existing shim rows store e.g. 0.9). So the seeded
  ``quality_score`` is the label unchanged, and binary success derives at
  the 0.5 midpoint (``task_success = 1 if quality_score >= 0.5 else 0`` —
  the 50 point on the rubric's internal 0-100 scale). The mapping goes
  through :func:`gentle_ai_model_router.integration.outcome.score_execution`
  with caller-provided outcomes (the rubric's "caller-populated outcomes
  win" path), which validates instead of recomputing — the rubric is reused
  as the validation gate, never duplicated. (The rubric's internal 0-100
  scale is a scoring detail of unscored rows; seeded rows arrive
  pre-scored, like caller-populated ingest payloads.)

Decision rows: :func:`gentle_ai_model_router.dataset.telemetry_bridge.
read_telemetry_executions` selects ``executions INNER JOIN decisions`` —
executions-only seeding would produce ZERO telemetry dataset rows (verified
by reading the bridge). So each execution also gets one minimal decision
row with a deterministic ``decision_id``, ``selected`` = the candidate,
empty ``alternatives`` (no runner-ups were recorded — win rates stay
honestly ``None``), and ``policy_version='empirical-bootstrap'``. The
execution is LINKED to that decision. (Unlinked rows would also be safe for
rewards: :mod:`gentle_ai_model_router.router.reward` skips executions
without a ``decision_id`` or with ``task_success IS NULL`` — verified in
``compute_rewards`` — but the INNER JOIN makes linking mandatory.)

Determinism: execution/decision ids derive from the example id, rows are
processed in sorted ``example_id`` order, timestamps come from the
example's ``snapshot_date`` (not wall clock), and re-seeding upserts on the
same ids — repeated runs are idempotent no-ops.

Fail closed: an unreadable dataset, a dataset with zero empirical rows, or
an empirical row that fails rubric validation raises the typed
:class:`EmpiricalSeedError` — never a silent partial seed.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.dataset.builder import load_dataset
from gentle_ai_model_router.dataset.schema import PROVENANCE_EMPIRICAL
from gentle_ai_model_router.integration.outcome import OutcomeRubricError, score_execution
from gentle_ai_model_router.integration.telemetry_shim import (
    DecisionRecord,
    ShimStore,
)

logger = logging.getLogger(__name__)

ROUTER_VERSION_BOOTSTRAP = "empirical-bootstrap"
POLICY_VERSION_BOOTSTRAP = "empirical-bootstrap"

# Binary success midpoint on the shim store's 0..1 quality scale
# (the 50 point on the rubric's internal 0-100 scale).
_QUALITY_SUCCESS_MIDPOINT = 0.5


class EmpiricalSeedError(Exception):
    """Fatal seed failure (unreadable dataset, no empirical rows, invalid row)."""


def _example_finished_at(snapshot_date: str, example_id: str) -> datetime:
    """Deterministic execution timestamp: the example's snapshot date at UTC midnight."""
    try:
        day = date.fromisoformat(snapshot_date)
    except ValueError as exc:
        raise EmpiricalSeedError(
            f"example {example_id}: invalid snapshot_date {snapshot_date!r}"
        ) from exc
    return datetime.combine(day, time.min, tzinfo=UTC)


def _seed_one(store: ShimStore, session: Session, example: Any) -> None:
    """Upsert the (decision, execution) pair for one empirical example."""
    execution_id = f"empirical-bootstrap-{example.example_id}"
    decision_id = f"empirical-bootstrap-decision-{example.example_id}"
    candidate = example.candidate

    cost = example.cost_features
    est_total = cost.get("est_total_tokens")
    if est_total is None:
        raise EmpiricalSeedError(
            f"example {example.example_id}: empirical row has no "
            "cost_features['est_total_tokens']"
        )
    # ESTIMATES, not measurements — empirical rows never carry actuals. The
    # input/output split falls back to (total, 0) when the v1 keys are absent.
    est_in = cost.get("est_input_tokens")
    est_out = cost.get("est_output_tokens")
    total_tokens = int(round(est_total))
    input_tokens = int(round(est_in)) if est_in is not None else total_tokens
    output_tokens = int(round(est_out)) if est_out is not None else 0

    # The empirical label is a measured benchmark quality on the dataset's
    # 0..1 scale; the shim store keeps the same scale, so the quality_score is
    # the label unchanged and binary success derives at the 0.5 midpoint.
    # score_execution validates caller-provided outcomes (it never recomputes
    # them) — the rubric is the validation gate.
    quality_score = float(example.label_quality_estimate)
    task_success = 1 if quality_score >= _QUALITY_SUCCESS_MIDPOINT else 0
    try:
        outcome = score_execution(
            example.phase,
            task_success=task_success,
            quality_score=quality_score,
        )
    except OutcomeRubricError as exc:
        raise EmpiricalSeedError(
            f"example {example.example_id}: outcome mapping failed: {exc}"
        ) from exc

    finished_at = _example_finished_at(example.snapshot_date, example.example_id)
    selected = {
        "model": candidate.model,
        "deployment": candidate.deployment,
        "effort": candidate.effort,
    }

    # Decision row (upsert on deterministic decision_id — idempotent re-seeds).
    decision_fields: dict[str, Any] = {
        "decision_id": decision_id,
        "phase": example.phase,
        "selected": selected,
        "alternatives": [],
        "reason_codes": ["empirical_bootstrap_seed"],
        "estimated_tokens": float(est_total),
        "estimated_cost": float(cost.get("est_cost") or 0.0),
        "policy_version": POLICY_VERSION_BOOTSTRAP,
        "created_at": finished_at,
    }
    existing = session.scalar(
        select(DecisionRecord).where(DecisionRecord.decision_id == decision_id)
    )
    if existing is None:
        store.record_decision(session, **decision_fields)
    else:
        for key, value in decision_fields.items():
            setattr(existing, key, value)

    # Execution row (record_execution upserts on execution_id).
    store.record_execution(
        session,
        {
            "execution_id": execution_id,
            "phase": example.phase,
            "task_type": example.task_type,
            "model": candidate.model,
            "deployment": candidate.deployment,
            "effort": candidate.effort,
            "started_at": finished_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "task_success": outcome.task_success,
            "quality_score": outcome.quality_score,
            "router_version": ROUTER_VERSION_BOOTSTRAP,
            "decision_id": decision_id,
        },
    )


def seed_empirical_dataset(dataset_path: str | Path, db_url: str) -> dict[str, int]:
    """Seed the shim store with executions derived from empirical dataset rows.

    Reads ``dataset_path`` (a written dataset directory, parquet or jsonl)
    via :func:`load_dataset`, converts every example whose
    ``label_provenance`` contains ``PROVENANCE_EMPIRICAL`` into one linked
    (decision, execution) pair, and upserts them into the shim DB at
    ``db_url``. Idempotent: deterministic ids make re-runs no-ops.

    Returns ``{"seeded": <empirical rows written>, "skipped": <non-empirical
    rows seen>, "total": <examples read>}``.

    Fail closed: unreadable dataset, zero empirical rows, or a row failing
    rubric validation raises :class:`EmpiricalSeedError`.
    """
    try:
        dataset = load_dataset(dataset_path)
    except EmpiricalSeedError:
        raise
    except Exception as exc:
        raise EmpiricalSeedError(
            f"unreadable dataset at '{dataset_path}': {exc}"
        ) from exc

    empirical = sorted(
        (e for e in dataset.examples if PROVENANCE_EMPIRICAL in e.label_provenance),
        key=lambda e: e.example_id,
    )
    if not empirical:
        raise EmpiricalSeedError(
            f"dataset at '{dataset_path}' contains no empirical-provenance rows "
            f"({PROVENANCE_EMPIRICAL!r} not found in any label_provenance); "
            "nothing to seed"
        )

    url = db_url if "://" in db_url else f"sqlite:///{db_url}"  # plain path → URL
    store = ShimStore(url)
    store.init_schema()
    with store.session() as session:
        for example in empirical:
            _seed_one(store, session, example)

    summary = {
        "seeded": len(empirical),
        "skipped": len(dataset.examples) - len(empirical),
        "total": len(dataset.examples),
    }
    logger.info(
        "empirical seed: %d execution(s) seeded from %s (%d non-empirical skipped)",
        summary["seeded"], dataset_path, summary["skipped"],
    )
    return summary
