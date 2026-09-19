"""Registry database access: engine, schema init, idempotent upserts.

All writes are keyed by natural unique constraints, so re-applying the same
snapshot is a no-op (idempotent upserts). Every benchmark/price row carries
the snapshot id it came from — provenance is mandatory.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from gentle_ai_model_router.collector.logging_conf import get_logger
from gentle_ai_model_router.registry.models import (
    Base,
    Deployment,
    Model,
    ModelBenchmark,
    ModelPrice,
    ModelSnapshot,
    ModelVariant,
    Provider,
)
from gentle_ai_model_router.registry.normalize import (
    Effort,
    NormalizedAAModel,
    NormalizedArenaRecord,
    NormalizedTelemetryRecord,
    internal_effort,
    normalize_aa_models,
    normalize_arena_records,
    normalize_telemetry_records,
)

logger = get_logger(__name__)


def _map_raw_effort(raw: str) -> Effort | None:
    """Map a provider-native variant string from discovery to the internal taxonomy.

    OpenCode speaks the internal vocabulary directly; ``none`` (seen in real
    variants caches) maps to OFF. Unknown values return None (skipped, counted).
    """
    effort = internal_effort("opencode", raw)
    if effort is not None:
        return effort
    if raw == "none":
        return Effort.OFF
    try:
        return Effort(raw)
    except ValueError:
        return None


def apply_local_candidates(session: Session, candidates: Iterable[Any]) -> dict[str, int]:
    """Upsert locally discovered (model, provider, efforts) into the registry.

    Candidates are duck-typed (``model``, ``provider``, ``efforts`` attributes)
    to avoid a collector dependency. Each (model, provider) becomes a
    deployment with one variant row per mapped effort level. Idempotent.
    """
    counts = {"providers": 0, "models": 0, "deployments": 0, "variants": 0, "skipped": 0}
    seen_providers: set[int] = set()
    seen_models: set[int] = set()
    seen_deployments: set[int] = set()
    for cand in candidates:
        if not getattr(cand, "model", None) or not getattr(cand, "provider", None):
            counts["skipped"] += 1
            continue
        provider = get_or_create_provider(session, str(cand.provider))
        model = upsert_model(session, canonical_id=str(cand.model))
        deployment = get_or_create_deployment(session, model, provider, None)
        seen_providers.add(provider.id)
        seen_models.add(model.id)
        seen_deployments.add(deployment.id)
        for raw in getattr(cand, "efforts", None) or []:
            internal = _map_raw_effort(str(raw))
            if internal is None:
                counts["skipped"] += 1
                continue
            upsert_variant(session, deployment, internal.value, str(raw))
            counts["variants"] += 1
    counts["providers"] = len(seen_providers)
    counts["models"] = len(seen_models)
    counts["deployments"] = len(seen_deployments)
    return counts


def get_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine (SQLite file or Postgres URL)."""
    if database_url.startswith("sqlite"):
        # SQLite cannot create the db file inside a missing directory.
        Path(database_url.removeprefix("sqlite:///")).expanduser().parent.mkdir(
            parents=True, exist_ok=True
        )
        return create_engine(
            database_url, connect_args={"check_same_thread": False}, future=True
        )
    return create_engine(database_url, future=True)


def get_engine_with_fallback(database_url: str, sqlite_fallback_url: str) -> tuple[Engine, str]:
    """Create an engine, degrading to SQLite when Postgres is unavailable.

    The configured Postgres URL wins (yaml ``registry.database_url`` >
    ``DATABASE_URL`` env), but a dev machine without Postgres — or without the
    ``psycopg`` driver — must still work. Degradation is logged, never silent.
    """
    if database_url.startswith("sqlite"):
        return get_engine(database_url), database_url
    try:
        engine = get_engine(database_url)
        with engine.connect():
            pass
        return engine, database_url
    except Exception as exc:  # missing driver, refused connection, etc.
        logger.warning(
            "registry postgres_unavailable fallback=sqlite error=%s", exc
        )
        return get_engine(sqlite_fallback_url), sqlite_fallback_url


def init_schema(engine: Engine) -> None:
    """Create all registry tables (idempotent)."""
    Base.metadata.create_all(engine)


# --------------------------------------------------------------------------- #
# Upsert helpers (select-then-insert; portable across SQLite and Postgres)
# --------------------------------------------------------------------------- #


def get_or_create_provider(
    session: Session, registry_key: str, name: str | None = None
) -> Provider:
    provider = session.scalar(select(Provider).where(Provider.registry_key == registry_key))
    if provider is None:
        provider = Provider(registry_key=registry_key, name=name or registry_key)
        session.add(provider)
        session.flush()
    return provider


def upsert_model(
    session: Session,
    canonical_id: str,
    org: str | None = None,
    name: str | None = None,
    context_window: int | None = None,
    max_output: int | None = None,
    modalities: list[str] | None = None,
    tool_calling: bool | None = None,
    structured_output: bool | None = None,
    reasoning_support: bool | None = None,
) -> Model:
    model = session.scalar(select(Model).where(Model.canonical_id == canonical_id))
    if model is None:
        model = Model(
            canonical_id=canonical_id,
            org=org,
            name=name or canonical_id.split("/")[-1],
            context_window=context_window,
            max_output=max_output,
            modalities=modalities,
            tool_calling=tool_calling,
            structured_output=structured_output,
            reasoning_support=reasoning_support,
        )
        session.add(model)
    else:
        # Fill in fields we previously did not know; never blank out data.
        if org and not model.org:
            model.org = org
        if context_window and not model.context_window:
            model.context_window = context_window
        if max_output and not model.max_output:
            model.max_output = max_output
        if modalities and not model.modalities:
            model.modalities = modalities
        if tool_calling is not None and model.tool_calling is None:
            model.tool_calling = tool_calling
        if structured_output is not None and model.structured_output is None:
            model.structured_output = structured_output
        if reasoning_support is not None and model.reasoning_support is None:
            model.reasoning_support = reasoning_support
    session.flush()
    return model


def get_or_create_deployment(
    session: Session, model: Model, provider: Provider, deployment_ref: str | None
) -> Deployment:
    ref = deployment_ref or "default"
    deployment = session.scalar(
        select(Deployment).where(
            Deployment.model_id == model.id,
            Deployment.provider_id == provider.id,
            Deployment.deployment_ref == ref,
        )
    )
    if deployment is None:
        deployment = Deployment(model_id=model.id, provider_id=provider.id, deployment_ref=ref)
        session.add(deployment)
        session.flush()
    return deployment


def upsert_variant(
    session: Session, deployment: Deployment, effort: str, provider_value: str
) -> ModelVariant:
    variant = session.scalar(
        select(ModelVariant).where(
            ModelVariant.deployment_id == deployment.id, ModelVariant.effort == effort
        )
    )
    if variant is None:
        variant = ModelVariant(
            deployment_id=deployment.id, effort=effort, provider_value=provider_value
        )
        session.add(variant)
        session.flush()
    else:
        variant.provider_value = provider_value
    return variant


def upsert_benchmark(
    session: Session,
    model: Model,
    benchmark: str,
    score: float,
    category: str | None,
    source_snapshot_id: str,
) -> ModelBenchmark:
    row = session.scalar(
        select(ModelBenchmark).where(
            ModelBenchmark.model_id == model.id,
            ModelBenchmark.benchmark == benchmark,
            ModelBenchmark.category == category,
        )
    )
    if row is None:
        row = ModelBenchmark(
            model_id=model.id,
            benchmark=benchmark,
            score=score,
            category=category,
            source_snapshot_id=source_snapshot_id,
        )
        session.add(row)
    else:
        row.score = score
        # Keep the freshest provenance: update to the newer snapshot id.
        row.source_snapshot_id = source_snapshot_id
    session.flush()
    return row


def upsert_price(
    session: Session,
    deployment: Deployment,
    source_snapshot_id: str,
    input_price: float | None,
    output_price: float | None,
    cached_input_price: float | None,
    currency: str = "USD",
    effective_date: str = "",
) -> ModelPrice:
    row = session.scalar(
        select(ModelPrice).where(
            ModelPrice.deployment_id == deployment.id,
            ModelPrice.source_snapshot_id == source_snapshot_id,
        )
    )
    if row is None:
        row = ModelPrice(
            deployment_id=deployment.id,
            input_price=input_price,
            output_price=output_price,
            cached_input_price=cached_input_price,
            currency=currency,
            effective_date=effective_date,
            source_snapshot_id=source_snapshot_id,
        )
        session.add(row)
    else:
        row.input_price = input_price
        row.output_price = output_price
        row.cached_input_price = cached_input_price
        row.currency = currency
        row.effective_date = effective_date
    session.flush()
    return row


def record_snapshot(
    session: Session,
    source: str,
    snapshot_id: str,
    fetched_at: datetime,
    record_count: int,
    raw_path: str = "",
) -> ModelSnapshot:
    row = session.scalar(select(ModelSnapshot).where(ModelSnapshot.snapshot_id == snapshot_id))
    if row is None:
        row = ModelSnapshot(
            source=source,
            snapshot_id=snapshot_id,
            fetched_at=fetched_at,
            record_count=record_count,
            raw_path=raw_path,
        )
        session.add(row)
        session.flush()
    return row


# --------------------------------------------------------------------------- #
# Snapshot application
# --------------------------------------------------------------------------- #


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=None)
    return dt.astimezone().replace(tzinfo=None)


def apply_aa_snapshot(session: Session, snapshot_doc: dict[str, Any]) -> dict[str, int]:
    """Apply one Artificial Analysis snapshot document to the registry."""
    snapshot_id = str(snapshot_doc.get("snapshot_id") or "unknown")
    payload = snapshot_doc.get("data")
    records: list[NormalizedAAModel] = normalize_aa_models(payload)
    counts = {"models": 0, "prices": 0, "benchmarks": 0}
    for rec in records:
        provider = get_or_create_provider(session, rec.provider_key)
        model = upsert_model(
            session,
            canonical_id=rec.canonical_id,
            org=rec.org,
            name=rec.name,
            context_window=rec.context_window,
            max_output=rec.max_output,
            modalities=rec.modalities or None,
            tool_calling=rec.tool_calling,
            structured_output=rec.structured_output,
            reasoning_support=rec.reasoning_support,
        )
        deployment = get_or_create_deployment(session, model, provider, rec.deployment_ref)
        counts["models"] += 1
        if (
            rec.input_price is not None
            or rec.output_price is not None
            or rec.cached_input_price is not None
        ):
            upsert_price(
                session,
                deployment,
                source_snapshot_id=snapshot_id,
                input_price=rec.input_price,
                output_price=rec.output_price,
                cached_input_price=rec.cached_input_price,
            )
            counts["prices"] += 1
        # rec.benchmarks carries every mapped index (intelligence/coding/
        # agentic + informational tps/ttft/cost rows); each becomes its own
        # registry benchmark row with mandatory snapshot provenance.
        # rec.cached_write_price is NOT persisted: ModelPrice has no column.
        for benchmark, score in rec.benchmarks.items():
            upsert_benchmark(
                session,
                model,
                benchmark=benchmark,
                score=score,
                category=None,
                source_snapshot_id=snapshot_id,
            )
            counts["benchmarks"] += 1
    fetched_at = datetime.fromisoformat(str(snapshot_doc.get("fetched_at")))
    meta = snapshot_doc.get("meta") or {}
    record_snapshot(
        session,
        source="artificial-analysis",
        snapshot_id=snapshot_id,
        fetched_at=_to_utc(fetched_at),
        record_count=int(meta.get("record_count") or len(records)),
        raw_path="",
    )
    logger.info(
        "apply_snapshot source=artificial-analysis snapshot_id=%s counts=%s",
        snapshot_id,
        counts,
    )
    return counts


def apply_arena_snapshot(session: Session, snapshot_doc: dict[str, Any]) -> dict[str, int]:
    """Apply one LMArena snapshot document to the registry."""
    snapshot_id = str(snapshot_doc.get("snapshot_id") or "unknown")
    data = snapshot_doc.get("data") or {}
    categories = data.get("categories") if isinstance(data, dict) else {}
    if not isinstance(categories, dict):
        categories = {}
    counts = {"models": 0, "benchmarks": 0}
    for category, rows in categories.items():
        normalized: list[NormalizedArenaRecord] = normalize_arena_records(
            [r for r in rows if isinstance(r, dict)], str(category)
        )
        for rec in normalized:
            provider_key = rec.organization or "unknown"
            provider = get_or_create_provider(session, provider_key)
            # Keep the model name as-is when it is already provider-qualified.
            canonical = (
                rec.model_name
                if "/" in rec.model_name or ":" in rec.model_name
                else f"{provider_key}/{rec.model_name}"
            )
            model = upsert_model(
                session,
                canonical_id=canonical,
                org=rec.organization,
                name=rec.model_name,
            )
            get_or_create_deployment(session, model, provider, None)
            counts["models"] += 1
            if rec.rating is not None:
                upsert_benchmark(
                    session,
                    model,
                    benchmark="lmarena_elo",
                    score=rec.rating,
                    category=rec.category,
                    source_snapshot_id=snapshot_id,
                )
                counts["benchmarks"] += 1
    fetched_at = datetime.fromisoformat(str(snapshot_doc.get("fetched_at")))
    meta = snapshot_doc.get("meta") or {}
    record_snapshot(
        session,
        source="lmarena",
        snapshot_id=snapshot_id,
        fetched_at=_to_utc(fetched_at),
        record_count=int(meta.get("record_count") or counts["benchmarks"]),
        raw_path="",
    )
    logger.info("apply_snapshot source=lmarena snapshot_id=%s counts=%s", snapshot_id, counts)
    return counts


def apply_telemetry_snapshot(session: Session, snapshot_doc: dict[str, Any]) -> dict[str, int]:
    """Apply one Gentle Telemetry snapshot document to the registry."""
    snapshot_id = str(snapshot_doc.get("snapshot_id") or "unknown")
    records: list[NormalizedTelemetryRecord] = normalize_telemetry_records(snapshot_doc)
    counts = {"models": 0, "benchmarks": 0}
    for rec in records:
        tool_calling = (
            True
            if rec.agent_class in {"sdd-apply", "sdd-verify", "apply", "verify", "worker"}
            else None
        )
        model = upsert_model(
            session,
            canonical_id=rec.canonical_id,
            org=rec.org,
            name=rec.name,
            tool_calling=tool_calling,
        )
        provider = get_or_create_provider(session, rec.org or "unknown")
        deployment = get_or_create_deployment(session, model, provider, "default")
        for eff in ("off", "low", "medium", "high"):
            upsert_variant(session, deployment, eff, eff)
        counts["models"] += 1
        upsert_benchmark(
            session,
            model,
            benchmark="telemetry_success_rate",
            score=rec.success_rate,
            category=rec.agent_class,
            source_snapshot_id=snapshot_id,
        )
        counts["benchmarks"] += 1
        if rec.tokens_per_success > 0:
            upsert_benchmark(
                session,
                model,
                benchmark="telemetry_tokens_per_success",
                score=rec.tokens_per_success,
                category=rec.agent_class,
                source_snapshot_id=snapshot_id,
            )
            counts["benchmarks"] += 1

    raw_fetched = snapshot_doc.get("fetched_at")
    if isinstance(raw_fetched, datetime):
        fetched_at = raw_fetched
    elif raw_fetched:
        fetched_at = datetime.fromisoformat(str(raw_fetched))
    else:
        fetched_at = datetime.now(tz=UTC)

    meta = snapshot_doc.get("meta") or {}
    record_snapshot(
        session,
        source="gentle-telemetry",
        snapshot_id=snapshot_id,
        fetched_at=_to_utc(fetched_at),
        record_count=int(meta.get("record_count") or len(records)),
        raw_path="",
    )
    logger.info(
        "apply_snapshot source=gentle-telemetry snapshot_id=%s counts=%s",
        snapshot_id,
        counts,
    )
    return counts


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


def registry_stats(engine: Engine) -> dict[str, int]:
    """Row counts per registry table (for ``router registry stats``)."""
    tables = {
        "providers": Provider,
        "models": Model,
        "deployments": Deployment,
        "model_variants": ModelVariant,
        "model_benchmarks": ModelBenchmark,
        "model_prices": ModelPrice,
        "model_snapshots": ModelSnapshot,
    }
    stats: dict[str, int] = {}
    with Session(engine) as session:
        for name, table in tables.items():
            stats[name] = int(session.scalar(select(func.count()).select_from(table)) or 0)
    return stats
