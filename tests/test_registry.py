"""Registry: full collect -> normalize -> query flow on a temp SQLite db."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import respx
from sqlalchemy.orm import Session

from gentle_ai_model_router.collector.artificial_analysis import (
    ArtificialAnalysisCollector,
    QuotaTracker,
)
from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.models import (
    Deployment,
    Model,
    ModelBenchmark,
    ModelPrice,
    ModelSnapshot,
    ModelVariant,
)
from gentle_ai_model_router.registry.normalize import Effort, internal_effort, provider_effort_value
from gentle_ai_model_router.router.config import ArtificialAnalysisConfig, LMArenaConfig
from tests.test_arena import TEXT_ROWS, FakeDataset

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)
BASE = "https://artificialanalysis.ai"


def _engine(tmp_path):
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    return engine


def _collect_aa(store: SnapshotStore, data_dir, payload) -> None:
    route = respx.get(f"{BASE}/api/v2/language/models/free").mock(
        return_value=httpx.Response(200, json=payload)
    )
    quota = QuotaTracker(data_dir / "aa-quota.json", budget=100)
    collector = ArtificialAnalysisCollector(
        ArtificialAnalysisConfig(),
        store,
        api_key=None,
        client=httpx.Client(base_url=BASE),
        quota=quota,
    )
    collector.collect(now=NOW)
    assert route.called


def _apply_latest(engine, store: SnapshotStore, source: str) -> None:
    doc = store.latest(source)
    assert doc is not None
    with Session(engine) as session:
        if source == "artificial-analysis":
            registry_db.apply_aa_snapshot(session, doc)
        else:
            registry_db.apply_arena_snapshot(session, doc)
        session.commit()


@respx.mock
def test_full_flow_and_provenance(tmp_path, store, data_dir, aa_payload) -> None:
    engine = _engine(tmp_path)
    _collect_aa(store, data_dir, aa_payload)
    _apply_latest(engine, store, "artificial-analysis")

    with Session(engine) as session:
        model = session.query(Model).filter_by(canonical_id="anthropic/claude-sonnet-4").one()
        assert model.context_window == 200000
        assert model.tool_calling is True

        dep = session.query(Deployment).filter_by(model_id=model.id).one()
        price = session.query(ModelPrice).filter_by(deployment_id=dep.id).one()
        assert price.input_price == 3.0
        assert price.output_price == 15.0
        assert price.cached_input_price == 0.3

        bench = (
            session.query(ModelBenchmark)
            .join(Model, ModelBenchmark.model_id == Model.id)
            .filter(
                Model.canonical_id == "anthropic/claude-sonnet-4",
                ModelBenchmark.benchmark == "artificial_analysis_intelligence_index",
            )
            .one()
        )
        assert bench.score == 42.5
        snapshot_id = bench.source_snapshot_id
        assert snapshot_id  # provenance is mandatory
        assert snapshot_id == price.source_snapshot_id

        snap = session.query(ModelSnapshot).filter_by(snapshot_id=snapshot_id).one()
        assert snap.source == "artificial-analysis"
        assert snap.record_count == 2


@respx.mock
def test_idempotent_upserts(tmp_path, store, data_dir, aa_payload) -> None:
    engine = _engine(tmp_path)
    _collect_aa(store, data_dir, aa_payload)
    _apply_latest(engine, store, "artificial-analysis")
    _apply_latest(engine, store, "artificial-analysis")  # second run, same snapshot

    stats = registry_db.registry_stats(engine)
    assert stats["models"] == 2
    assert stats["providers"] == 2
    assert stats["deployments"] == 2
    assert stats["model_prices"] == 2
    assert stats["model_benchmarks"] == 2
    assert stats["model_snapshots"] == 1  # same snapshot id recorded once

def test_apply_arena_snapshot(tmp_path, store):
    engine = _engine(tmp_path)
    from gentle_ai_model_router.collector.arena import LMArenaCollector

    collector = LMArenaCollector(
        LMArenaConfig(), store, load_fn=lambda d, c, split="latest": FakeDataset(TEXT_ROWS)
    )
    collector.collect(categories=["text"], now=NOW)
    _apply_latest(engine, store, "lmarena")

    with Session(engine) as session:
        bench = (
            session.query(ModelBenchmark)
            .join(Model, ModelBenchmark.model_id == Model.id)
            .filter(
                Model.canonical_id == "anthropic/claude-sonnet-4",
                ModelBenchmark.benchmark == "lmarena_elo",
                ModelBenchmark.category == "text",
            )
            .one()
        )
        assert bench.score == 1385.2
        assert bench.source_snapshot_id
        model = session.get(Model, bench.model_id)
        assert model is not None
        assert model.canonical_id == "anthropic/claude-sonnet-4"


def test_effort_adapters() -> None:
    # Pi speaks the full internal vocabulary (passthrough).
    assert provider_effort_value("pi", Effort.XHIGH) == "xhigh"
    # Codex has no off/minimal/max.
    assert provider_effort_value("codex", Effort.HIGH) == "high"
    assert provider_effort_value("codex", Effort.OFF) is None
    # Unknown provider keys fall back to generic passthrough.
    assert provider_effort_value("some-new-provider", Effort.MEDIUM) == "medium"
    # Reverse mapping.
    assert internal_effort("codex", "xhigh") is Effort.XHIGH
    assert internal_effort("codex", "max") is None


def test_variant_upsert(tmp_path):
    engine = _engine(tmp_path)
    with Session(engine) as session:
        provider = registry_db.get_or_create_provider(session, "anthropic")
        model = registry_db.upsert_model(session, "anthropic/claude-sonnet-4")
        deployment = registry_db.get_or_create_deployment(session, model, provider, None)
        registry_db.upsert_variant(session, deployment, Effort.HIGH.value, "high")
        registry_db.upsert_variant(session, deployment, Effort.HIGH.value, "high")  # idempotent
        registry_db.upsert_variant(session, deployment, Effort.MAX.value, "max")
        session.commit()
        variants = session.query(ModelVariant).all()
        assert len(variants) == 2
