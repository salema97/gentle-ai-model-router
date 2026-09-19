"""Telemetry → dataset bridge: read scored shim executions for the builder.

This is the read side of the bandit feedback loop for dataset construction:
executions recorded by the telemetry shim (``integration/telemetry_shim.py``)
and scored by the outcome rubric (``integration/outcome.py``) become
``DatasetExample`` rows with ``label_provenance='telemetry'`` (see
:mod:`gentle_ai_model_router.dataset.builder`).

This module only READS the shim store and coerces rows into plain dataclasses.
All registry-dependent conversion (features, prices, splits, utility labels)
lives in the builder, which owns those lookups.

Fail-closed contract:

- An ABSENT database file yields zero rows — there is simply nothing to
  learn from yet, and the build continues with bootstrap rows only.
- A PRESENT but unreadable database (corrupt file, missing tables, not a
  SQLite URL we can open) raises :class:`TelemetryReadError`; the builder
  surfaces it as a typed ``DatasetBuildError``. We never silently half-read
  a store we could not fully query.

Row selection: ``executions`` INNER JOIN ``decisions`` on ``decision_id``
(only attributable executions are usable for learning), restricted to rows
with a non-NULL ``task_success`` or ``quality_score`` (unscored executions
are skipped by the rubric, not by us), ordered by ``execution_id`` for
determinism.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from gentle_ai_model_router.integration.telemetry_shim import DecisionRecord, ExecutionRecord

logger = logging.getLogger(__name__)


class TelemetryReadError(Exception):
    """The shim DB exists but cannot be read. Fails closed, never half-read."""


@dataclass(frozen=True)
class TelemetryExecution:
    """One scored, attributable shim execution coerced for the builder."""

    execution_id: str
    phase: str
    task_type: str | None
    model: str
    deployment: str | None
    effort: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int | None
    task_success: int | None  # 0/1; None when only quality_score is known
    quality_score: float | None
    event_date: date  # finished_at date, falling back to the decision date
    estimated_tokens: float  # decision's pre-execution estimate (0 = unknown)
    estimated_cost: float
    repo_features: dict[str, Any] | None


@dataclass
class TelemetryReadResult:
    """Coerced scored rows read from the shim store."""

    rows: list[TelemetryExecution]


def _sqlite_file_path(db_url: str) -> Path | None:
    """Map a SQLite URL (or plain path) to a file path; None otherwise."""
    if db_url.startswith("sqlite:///"):
        return Path(db_url.removeprefix("sqlite:///")).expanduser()
    if "://" not in db_url:
        return Path(db_url).expanduser()
    return None


def read_telemetry_executions(db_url: str) -> TelemetryReadResult:
    """Read scored executions joined to their decisions from the shim store.

    Absent DB → zero rows (logged). Unreadable DB → TelemetryReadError.
    """
    db_file = _sqlite_file_path(db_url)
    if db_file is not None and not db_file.exists():
        logger.info("telemetry bridge: no shim DB at %s — zero telemetry rows", db_file)
        return TelemetryReadResult(rows=[])

    engine = create_engine(
        db_url if "://" in db_url else f"sqlite:///{db_url}",
        connect_args={"check_same_thread": False},
        future=True,
    )
    try:
        with Session(engine) as session:
            all_scored = session.execute(
                select(ExecutionRecord, DecisionRecord)
                .join(DecisionRecord, ExecutionRecord.decision_id == DecisionRecord.decision_id)
                .where(
                    (ExecutionRecord.task_success.is_not(None))
                    | (ExecutionRecord.quality_score.is_not(None))
                )
                .order_by(ExecutionRecord.execution_id)
            ).all()
    except Exception as exc:
        raise TelemetryReadError(
            f"unreadable telemetry shim DB '{db_url}': {exc}"
        ) from exc
    finally:
        engine.dispose()

    rows = [
        TelemetryExecution(
            execution_id=str(execution.execution_id),
            phase=execution.phase,
            task_type=execution.task_type,
            model=str(execution.model or ""),
            deployment=execution.deployment,
            effort=execution.effort,
            input_tokens=int(execution.input_tokens),
            output_tokens=int(execution.output_tokens),
            total_tokens=int(execution.total_tokens),
            latency_ms=int(execution.latency_ms) if execution.latency_ms is not None else None,
            task_success=(
                int(execution.task_success) if execution.task_success is not None else None
            ),
            quality_score=(
                float(execution.quality_score)
                if execution.quality_score is not None
                else None
            ),
            event_date=(execution.finished_at or decision.created_at).date(),
            estimated_tokens=float(decision.estimated_tokens),
            estimated_cost=float(decision.estimated_cost),
            repo_features=(
                execution.repo_features
                if isinstance(execution.repo_features, dict)
                else None
            ),
        )
        for execution, decision in all_scored
    ]
    return TelemetryReadResult(rows=rows)


def extract_ground_truth_executions(
    snapshot_doc: dict[str, Any], default_phase: str = "apply"
) -> list[TelemetryExecution]:
    """Convert ground-truth trace snapshot data into TelemetryExecution records.

    Allows ground-truth traces from RouterBench, RouteLLM, or SWE-Traces to be
    consumed via the telemetry execution data structure with deterministic execution IDs.
    """
    data = (
        snapshot_doc.get("data")
        if isinstance(snapshot_doc, dict) and "data" in snapshot_doc
        else snapshot_doc
    )
    if not isinstance(data, dict):
        return []

    event_date = date.today()
    fetched_at_str = (
        snapshot_doc.get("fetched_at") if isinstance(snapshot_doc, dict) else None
    )
    if fetched_at_str:
        try:
            event_date = datetime.fromisoformat(fetched_at_str).date()
        except (ValueError, TypeError):
            pass

    records: list[TelemetryExecution] = []
    # 1. SWE-traces
    trajectories = data.get("trajectories")
    if isinstance(trajectories, list):
        for idx, item in enumerate(trajectories):
            if isinstance(item, dict):
                inst_id = str(item.get("instance_id") or f"swe-{idx}")
                model = str(item.get("model_name") or item.get("model") or "")
                phase = str(item.get("mapped_phase") or default_phase)
                in_tok = int(item.get("input_tokens") or 0)
                out_tok = int(item.get("output_tokens") or 0)
                tot_tok = int(item.get("total_tokens") or (in_tok + out_tok))
                latency = item.get("latency")
                latency_ms = int(float(latency) * 1000) if latency is not None else None
                success = 1 if item.get("test_passed") else 0
                cost = float(item.get("cost") or 0.0)
                records.append(
                    TelemetryExecution(
                        execution_id=f"gt-swe-{inst_id}",
                        phase=phase,
                        task_type="swe-bench",
                        model=model,
                        deployment=None,
                        effort=None,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        total_tokens=tot_tok,
                        latency_ms=latency_ms,
                        task_success=success,
                        quality_score=float(success),
                        event_date=event_date,
                        estimated_tokens=float(tot_tok),
                        estimated_cost=cost,
                        repo_features={"repo": item.get("repo", "unknown")},
                    )
                )

    # 2. RouterBench & RouteLLM samples
    samples = data.get("samples")
    if isinstance(samples, list):
        source = str(data.get("source") or "")
        for idx, item in enumerate(samples):
            if isinstance(item, dict):
                phase = str(item.get("mapped_phase") or default_phase)
                task_name = str(item.get("task_name") or "benchmark")
                latency = item.get("latency")
                latency_ms = int(float(latency) * 1000) if latency is not None else None
                cost = float(item.get("cost") or 0.0)
                in_tok = int(item.get("input_tokens") or 0)
                out_tok = int(item.get("output_tokens") or 0)
                tot_tok = in_tok + out_tok

                if source == "routellm" or "model_a" in item:
                    score_a = float(
                        item.get("score_a") if item.get("score_a") is not None else 0.5
                    )
                    success_a = 1 if item.get("choice_label") in ("model_a", "tie") else 0
                    records.append(
                        TelemetryExecution(
                            execution_id=f"gt-routellm-{idx}-a",
                            phase=phase,
                            task_type=task_name,
                            model=str(item.get("model_a") or ""),
                            deployment=None,
                            effort=None,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            total_tokens=tot_tok,
                            latency_ms=latency_ms,
                            task_success=success_a,
                            quality_score=score_a,
                            event_date=event_date,
                            estimated_tokens=float(tot_tok),
                            estimated_cost=float(item.get("cost_a") or cost),
                            repo_features=None,
                        )
                    )
                else:
                    model = str(item.get("model_name") or item.get("model") or "")
                    q = float(
                        item.get("correctness")
                        if item.get("correctness") is not None
                        else 0.5
                    )
                    records.append(
                        TelemetryExecution(
                            execution_id=f"gt-routerbench-{idx}",
                            phase=phase,
                            task_type=task_name,
                            model=model,
                            deployment=None,
                            effort=None,
                            input_tokens=in_tok,
                            output_tokens=out_tok,
                            total_tokens=tot_tok,
                            latency_ms=latency_ms,
                            task_success=1 if q >= 0.5 else 0,
                            quality_score=q,
                            event_date=event_date,
                            estimated_tokens=float(tot_tok),
                            estimated_cost=cost,
                            repo_features=None,
                        )
                    )

    return records
