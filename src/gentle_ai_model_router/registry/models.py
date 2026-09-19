"""Registry schema (SQLAlchemy 2.0, Postgres-compatible, SQLite fallback).

Design constraints:
- Explicit integer PKs everywhere; no SQLite-only types (``JSON`` columns are
  portable across SQLite/Postgres).
- Natural unique constraints drive idempotent upserts.
- Every benchmark and price row carries ``source_snapshot_id`` — provenance
  is mandatory, per the RDD culture of the reference project.
- ``phase_policies`` is defined now; population is a later phase.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    registry_key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)


class Model(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    org: Mapped[str | None] = mapped_column(String(128), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    context_window: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_output: Mapped[int | None] = mapped_column(Integer, nullable=True)
    modalities: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    tool_calling: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    structured_output: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    reasoning_support: Mapped[bool | None] = mapped_column(Boolean, nullable=True)


class Deployment(Base):
    __tablename__ = "deployments"
    __table_args__ = (UniqueConstraint("model_id", "provider_id", "deployment_ref"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"), nullable=False)
    deployment_ref: Mapped[str] = mapped_column(String(128), nullable=False, default="default")


class ModelVariant(Base):
    __tablename__ = "model_variants"
    __table_args__ = (UniqueConstraint("deployment_id", "effort"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id"), nullable=False)
    effort: Mapped[str] = mapped_column(String(16), nullable=False)
    provider_value: Mapped[str] = mapped_column(String(64), nullable=False)


class ModelCapability(Base):
    __tablename__ = "model_capabilities"
    __table_args__ = (UniqueConstraint("model_id", "capability"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    capability: Mapped[str] = mapped_column(String(64), nullable=False)
    supported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"
    __table_args__ = (UniqueConstraint("model_id", "benchmark", "category"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"), nullable=False)
    benchmark: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)


class ModelPrice(Base):
    __tablename__ = "model_prices"
    __table_args__ = (UniqueConstraint("deployment_id", "source_snapshot_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    deployment_id: Mapped[int] = mapped_column(ForeignKey("deployments.id"), nullable=False)
    input_price: Mapped[float | None] = mapped_column(Float, nullable=True)  # per 1M tokens
    output_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    cached_input_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    effective_date: Mapped[str] = mapped_column(String(16), nullable=False, default="")
    source_snapshot_id: Mapped[str] = mapped_column(String(128), nullable=False)


class ModelSnapshot(Base):
    __tablename__ = "model_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    raw_path: Mapped[str] = mapped_column(String(512), nullable=False, default="")


class PhasePolicy(Base):
    """Phase policy table — schema defined now, populated in a later phase."""

    __tablename__ = "phase_policies"
    __table_args__ = (UniqueConstraint("phase", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    phase: Mapped[str] = mapped_column(String(64), nullable=False)
    task_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    threshold_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    weights: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
