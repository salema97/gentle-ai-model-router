"""Router-owned telemetry shim: our own instrumentation path and store.

Why this exists (docs/telemetry-probe.md §1.5, with file:line evidence into the
Gentle AI reference clone): the Gentle AI collector store cannot be the primary
learning source — it lacks session/task ids and exact model ids (privacy-scrubbed
by design, ``runtime.go:201,220-251``), has only aggregate latency
(``runtime.go:106-112``), always reports ``unavailable`` for OpenCode effort
(``runtime_opencode.go:117``), and has no tool-call/test/outcome fields at all.
Its wire schema rejects unknown fields (``runtime.go:410-428``), so this shim
is a parallel path: same hook surfaces (``message.updated`` / ``SubagentStop``
for OpenCode, ``turn_context`` for Pi) with our own local-only store.

Privacy stance: local SQLite only, no prompt/content payloads — tokens,
counters, durations, and phase outcomes only.

Phase outcome signals (``PHASE_SIGNALS``) are documented expectations; the
per-phase scoring rubric that populates ``task_success`` / ``quality_score``
is DESIGN/pending (docs/telemetry-shim.md).
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger(__name__)


class TelemetryBase(DeclarativeBase):
    pass


# Documented (not yet enforced) per-phase outcome signals. Populating
# task_success/quality_score from these is DESIGN/pending per phase.
PHASE_SIGNALS: dict[str, list[str]] = {
    "init": ["sdd_context_initialized", "registry_bootstrapped"],
    "explore": ["findings_count", "findings_corroborated_ratio", "exploration_coverage"],
    "research": ["sources_verified", "claims_with_citations"],
    "propose": ["proposal_accepted", "scope_clarity_score"],
    "spec": ["requirements_count", "scenario_coverage", "spec_delta_applied"],
    "design": ["design_review_passed", "tasks_derived_count"],
    "tasks": ["tasks_completed_ratio", "task_granularity_score"],
    "apply": ["tests_passed", "tests_failed", "build_green", "diff_review_approved"],
    "verify": ["verification_passed", "regressions_found"],
    "archive": ["delta_specs_synced", "artifacts_preserved"],
    "onboard": ["onboarding_completed", "docs_updated"],
}


class DecisionRecord(TelemetryBase):
    __tablename__ = "decisions"

    decision_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    phase: Mapped[str] = mapped_column(String(64), nullable=False)
    selected: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    alternatives: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    reason_codes: Mapped[list[Any]] = mapped_column(JSON, nullable=False)
    estimated_tokens: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    estimated_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=lambda: datetime.now(UTC))


class ExecutionRecord(TelemetryBase):
    __tablename__ = "executions"

    execution_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    project_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    phase: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deployment: Mapped[str | None] = mapped_column(String(128), nullable=True)
    effort: Mapped[str | None] = mapped_column(String(16), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_errors: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tests_passed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tests_failed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    task_success: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 0/1, None = unknown
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    escalation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repo_features: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    router_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("decisions.decision_id"), nullable=True
    )


def _get_engine(database_url: str) -> Engine:
    if database_url.startswith("sqlite"):
        Path(database_url.removeprefix("sqlite:///")).expanduser().parent.mkdir(
            parents=True, exist_ok=True
        )
        return create_engine(database_url, connect_args={"check_same_thread": False}, future=True)
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
    connect_args: dict[str, Any] = {}
    if "postgres" in database_url:
        connect_args["connect_timeout"] = 2
    return create_engine(database_url, future=True, connect_args=connect_args)


class ShimStore:
    """Context-managed access to the telemetry SQLite store."""

    def __init__(self, database_url: str) -> None:
        self.engine = _get_engine(database_url)

    def init_schema(self) -> None:
        TelemetryBase.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with Session(self.engine) as session:
            yield session
            session.commit()

    # ------------------------------------------------------------------ #
    # Insert helpers
    # ------------------------------------------------------------------ #

    def record_decision(self, session: Session, **fields: Any) -> DecisionRecord:
        fields.setdefault("decision_id", uuid.uuid4().hex)
        record = DecisionRecord(**fields)
        session.add(record)
        session.flush()
        return record

    def record_execution(
        self, session: Session, payload: dict[str, Any]
    ) -> tuple[ExecutionRecord, bool]:
        """Upsert one execution row keyed by ``execution_id``.

        Returns (record, created); a duplicate ``execution_id`` updates the
        existing row instead of failing (idempotent re-ingest).
        """
        execution_id = str(payload.get("execution_id") or "")
        if not execution_id:
            raise ValueError("execution payload requires 'execution_id'")
        record = session.scalar(
            select(ExecutionRecord).where(ExecutionRecord.execution_id == execution_id)
        )
        fields = self._execution_fields(payload)
        if record is None:
            record = ExecutionRecord(execution_id=execution_id, **fields)
            session.add(record)
            created = True
        else:
            for key, value in fields.items():
                setattr(record, key, value)
            created = False
        session.flush()
        return record, created

    @staticmethod
    def _execution_fields(payload: dict[str, Any]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for key in (
            "session_id", "project_id", "phase", "task_type", "model", "deployment",
            "effort", "latency_ms", "tool_calls", "tool_errors", "tests_passed",
            "tests_failed", "task_success", "quality_score", "escalation_count",
            "router_version", "decision_id",
        ):
            if key in payload:
                fields[key] = payload[key]
        token_keys = (
            "input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "cached_tokens",
            "total_tokens",
        )
        for key in token_keys:
            value = payload.get(key)
            fields[key] = int(value) if isinstance(value, (int, float)) else 0
        for key in ("started_at", "finished_at"):
            value = payload.get(key)
            if isinstance(value, datetime):
                fields[key] = value
            elif isinstance(value, str):
                try:
                    fields[key] = datetime.fromisoformat(value)
                except ValueError:
                    fields[key] = None
        repo_features = payload.get("repo_features")
        if isinstance(repo_features, dict):
            fields["repo_features"] = repo_features
        return fields

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #

    def tokens_per_success(
        self, session: Session, phase: str | None = None
    ) -> list[dict[str, Any]]:
        """Total tokens divided by successful executions, per phase.

        Successes of 0 yield None (undefined), not infinity.
        """
        total = func.coalesce(func.sum(ExecutionRecord.total_tokens), 0)
        successes = func.coalesce(
            func.sum(func.coalesce(ExecutionRecord.task_success, 0)), 0
        )
        stmt = select(
            ExecutionRecord.phase,
            total.label("total_tokens"),
            successes.label("successes"),
        ).group_by(ExecutionRecord.phase)
        if phase is not None:
            stmt = stmt.where(ExecutionRecord.phase == phase)
        rows = session.execute(stmt).all()
        return [
            {
                "phase": row.phase,
                "total_tokens": int(row.total_tokens),
                "successes": int(row.successes),
                "tokens_per_success": (
                    int(row.total_tokens) / int(row.successes)
                    if int(row.successes) > 0
                    else None
                ),
            }
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # JSON-lines ingestion (stdin hook surface)
    # ------------------------------------------------------------------ #

    def ingest_jsonl(self, session: Session, lines: TextIO) -> dict[str, int]:
        """Ingest JSON-lines execution payloads. Returns insert/update/skip counts."""
        counts = {"inserted": 0, "updated": 0, "skipped": 0}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                counts["skipped"] += 1
                continue
            if not isinstance(payload, dict) or "execution_id" not in payload:
                counts["skipped"] += 1
                continue
            _, created = self.record_execution(session, payload)
            counts["inserted" if created else "updated"] += 1
        return counts
