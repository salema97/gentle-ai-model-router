"""Empirical seed tests: dataset empirical rows -> bootstrap-stamped shim rows.

Fixtures mirror tests/test_dataset_telemetry.py (dated design-phase registry)
plus a seeded routing-benchmarks snapshot (pattern from
tests/test_dataset_builder.py) so the built dataset carries real
PROVENANCE_EMPIRICAL examples.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.dataset.builder import (
    DatasetBuilderConfig,
    build_examples,
    load_dataset,
    write_dataset,
)
from gentle_ai_model_router.dataset.schema import (
    PROVENANCE_EMPIRICAL,
    PROVENANCE_TELEMETRY,
)
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.integration.empirical_seed import (
    EmpiricalSeedError,
    seed_empirical_dataset,
)
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import PhaseConfig, RouterConfig, load_config

runner = CliRunner()

AA = "artificial_analysis_intelligence_index"
ARENA_TEXT = "lmarena_elo:text"

SNAP_OLD = "snap-2026-01"
SNAP_MID = "snap-2026-02"
SNAP_NEW = "snap-2026-03"
DATES = {
    SNAP_OLD: datetime(2026, 1, 1),
    SNAP_MID: datetime(2026, 2, 1),
    SNAP_NEW: datetime(2026, 3, 1),
}

TRAIN_END = date(2026, 2, 15)
VAL_END = date(2026, 2, 20)


def _engine(tmp_path):
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    return engine


def _snap(session: Session, snapshot_id: str) -> None:
    registry_db.record_snapshot(session, "test", snapshot_id, DATES[snapshot_id], 1)


def _add_model(
    session: Session,
    canonical: str,
    benchmark_snap: str,
    price_snap: str | None,
    benchmarks: dict[str, float],
    price: tuple[float, float] = (2.0, 8.0),
    variants: tuple[str, ...] = ("low", "medium", "high"),
) -> None:
    provider = registry_db.get_or_create_provider(session, canonical.split("/")[0])
    model = registry_db.upsert_model(session, canonical_id=canonical, tool_calling=True)
    deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
    for effort in variants:
        registry_db.upsert_variant(session, deployment, effort, effort)
    for key, score in benchmarks.items():
        name, _, category = key.partition(":")
        registry_db.upsert_benchmark(session, model, name, score, category or None, benchmark_snap)
    if price_snap is not None:
        registry_db.upsert_price(session, deployment, price_snap, price[0], price[1], None)


@pytest.fixture
def builder_config() -> DatasetBuilderConfig:
    return DatasetBuilderConfig(
        name="test-empirical-seed",
        phases=["design"],
        task_types=["feature"],
        context_sizes=[10_000],
        train_end=TRAIN_END,
        val_end=VAL_END,
    )


@pytest.fixture
def dated_session(tmp_path):
    engine = _engine(tmp_path)
    with registry_db.Session(engine) as session:
        for snap in DATES:
            _snap(session, snap)
        _add_model(session, "test/old-model", SNAP_OLD, SNAP_OLD, {AA: 50.0, ARENA_TEXT: 90.0})
        # Price dated SNAP_OLD keeps every example pre-train_end (no temporal
        # splits, no train-overlap drops) so all empirical rows survive.
        _add_model(session, "test/mid-model", SNAP_MID, SNAP_OLD, {AA: 40.0, ARENA_TEXT: 95.0})
        session.commit()
        yield session


def _design_config(tmp_path) -> RouterConfig:
    return load_config(
        config_path="/nonexistent/router.yaml",
        data_dir=tmp_path / "data",
    ).model_copy(
        update={
            "phases": {
                "design": PhaseConfig(
                    threshold_quality=0.85, weights={AA: 1.0, ARENA_TEXT: 1.0}
                )
            }
        }
    )


def _empirical_store(tmp_path) -> SnapshotStore:
    """One routing-benchmarks snapshot; old-model scores 0.95, mid-model 0.35."""
    store = SnapshotStore(tmp_path / "data" / "snapshots")
    store.save(
        "routing-benchmarks",
        data={
            "source": "huggingface",
            "dars": [
                {
                    "query_id": "d1",
                    "prompt": "Optimize a distributed raft consensus cluster.",
                    "model": "test/old-model",
                    "quality": 0.95,
                    "cost": 0.001,
                    "input_tokens": 1200,
                    "task_name": "raft-optimization",
                    "mapped_phase": "design",
                },
                {
                    "query_id": "d1",
                    "prompt": "Optimize a distributed raft consensus cluster.",
                    "model": "test/mid-model",
                    "quality": 0.35,
                    "cost": 0.002,
                    "input_tokens": 1200,
                    "task_name": "raft-optimization",
                    "mapped_phase": "design",
                },
            ],
        },
        fetched_at=datetime(2026, 1, 15, tzinfo=UTC),
    )
    return store


def _build_empirical_dataset(dated_session, builder_config, tmp_path) -> Path:
    """Build + write a dataset with PROVENANCE_EMPIRICAL rows; return its dir."""
    dataset = build_examples(
        dated_session,
        _design_config(tmp_path),
        builder_config,
        store=_empirical_store(tmp_path),
    )
    empirical = [e for e in dataset.examples if e.label_provenance == PROVENANCE_EMPIRICAL]
    assert len(empirical) == 6  # 2 matched models x 3 efforts
    return write_dataset(dataset, tmp_path / "data", builder=builder_config)


def _shim_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'telemetry.sqlite'}"


def _counts(store: telemetry_shim.ShimStore) -> tuple[int, int]:
    with store.session() as session:
        executions = session.scalar(select(func.count(telemetry_shim.ExecutionRecord.id)))
        decisions = session.scalar(select(func.count(telemetry_shim.DecisionRecord.decision_id)))
    return int(executions or 0), int(decisions or 0)


# --------------------------------------------------------------------------- #
# Seeding: rows, mapping, determinism, idempotency
# --------------------------------------------------------------------------- #


def test_seed_writes_linked_execution_and_decision_rows(
    dated_session, builder_config, tmp_path
) -> None:
    out_dir = _build_empirical_dataset(dated_session, builder_config, tmp_path)
    summary = seed_empirical_dataset(out_dir, _shim_url(tmp_path))
    assert summary == {"seeded": 6, "skipped": 6, "total": 12}

    store = telemetry_shim.ShimStore(_shim_url(tmp_path))
    with store.session() as session:
        rows = session.execute(
            select(telemetry_shim.ExecutionRecord).order_by(
                telemetry_shim.ExecutionRecord.execution_id
            )
        ).scalars().all()
        assert len(rows) == 6
        for row in rows:
            assert row.execution_id.startswith("empirical-bootstrap-ex-")
            assert row.router_version == "empirical-bootstrap"
            assert row.phase == "design"
            assert row.model in {"test/old-model", "test/mid-model"}
            # decision_id is set and points at an existing decision row.
            assert row.decision_id is not None
            decision = session.scalar(
                select(telemetry_shim.DecisionRecord).where(
                    telemetry_shim.DecisionRecord.decision_id == row.decision_id
                )
            )
            assert decision is not None
            assert decision.policy_version == "empirical-bootstrap"
            assert decision.selected["model"] == row.model
            assert decision.alternatives == []
            # ESTIMATED tokens (empirical rows carry no actuals).
            assert row.total_tokens > 0
            assert row.total_tokens == int(
                round(decision.estimated_tokens)
            )

        # Outcome mapping: quality_estimate stays on the shim's 0..1 scale;
        # task_success = 1 iff quality >= 0.5.
        by_model: dict[str, telemetry_shim.ExecutionRecord] = {}
        for row in rows:
            if row.effort == "low":
                by_model[row.model] = row
        old = by_model["test/old-model"]
        assert old.quality_score == pytest.approx(0.95)
        assert old.task_success == 1
        mid = by_model["test/mid-model"]
        assert mid.quality_score == pytest.approx(0.35)
        assert mid.task_success == 0


def test_seed_is_idempotent(dated_session, builder_config, tmp_path) -> None:
    out_dir = _build_empirical_dataset(dated_session, builder_config, tmp_path)
    first = seed_empirical_dataset(out_dir, _shim_url(tmp_path))
    store = telemetry_shim.ShimStore(_shim_url(tmp_path))
    assert _counts(store) == (6, 6)

    second = seed_empirical_dataset(out_dir, _shim_url(tmp_path))
    assert second == first
    assert _counts(store) == (6, 6)  # upserts, no duplicates


def test_seed_is_deterministic_across_databases(
    dated_session, builder_config, tmp_path
) -> None:
    out_dir = _build_empirical_dataset(dated_session, builder_config, tmp_path)
    seed_empirical_dataset(out_dir, f"sqlite:///{tmp_path / 'a.sqlite'}")
    seed_empirical_dataset(out_dir, f"sqlite:///{tmp_path / 'b.sqlite'}")

    def fingerprint(url: str):
        store = telemetry_shim.ShimStore(url)
        with store.session() as session:
            rows = session.execute(
                select(telemetry_shim.ExecutionRecord).order_by(
                    telemetry_shim.ExecutionRecord.execution_id
                )
            ).scalars().all()
            return [
                (
                    r.execution_id, r.phase, r.model, r.deployment, r.effort,
                    r.total_tokens, r.task_success, r.quality_score,
                    r.router_version, r.decision_id,
                )
                for r in rows
            ]

    assert fingerprint(f"sqlite:///{tmp_path / 'a.sqlite'}") == fingerprint(
        f"sqlite:///{tmp_path / 'b.sqlite'}"
    )


def test_summary_counts_non_empirical_rows_as_skipped(
    dated_session, builder_config, tmp_path
) -> None:
    out_dir = _build_empirical_dataset(dated_session, builder_config, tmp_path)
    reloaded = load_dataset(out_dir)
    empirical = sum(
        1 for e in reloaded.examples if PROVENANCE_EMPIRICAL in e.label_provenance
    )
    summary = seed_empirical_dataset(out_dir, _shim_url(tmp_path))
    assert summary["seeded"] == empirical
    assert summary["total"] == len(reloaded.examples)
    assert summary["skipped"] == len(reloaded.examples) - empirical


# --------------------------------------------------------------------------- #
# Fail-closed behavior
# --------------------------------------------------------------------------- #


def test_no_empirical_rows_fails_closed(dated_session, builder_config, tmp_path) -> None:
    dataset = build_examples(
        dated_session,
        _design_config(tmp_path),
        builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
    )
    out_dir = write_dataset(dataset, tmp_path / "data", builder=builder_config)
    with pytest.raises(EmpiricalSeedError, match="no empirical-provenance rows"):
        seed_empirical_dataset(out_dir, _shim_url(tmp_path))


def test_unreadable_dataset_fails_closed(tmp_path) -> None:
    with pytest.raises(EmpiricalSeedError, match="unreadable dataset"):
        seed_empirical_dataset(tmp_path / "does-not-exist", _shim_url(tmp_path))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def test_cli_shim_seed(dated_session, builder_config, tmp_path) -> None:
    from gentle_ai_model_router.cli.main import app

    out_dir = _build_empirical_dataset(dated_session, builder_config, tmp_path)
    result = runner.invoke(
        app,
        [
            "shim", "seed",
            "--dataset", str(out_dir),
            "--db", str(tmp_path / "telemetry.sqlite"),
            "--data-dir", str(tmp_path / "data"),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary == {"seeded": 6, "skipped": 6, "total": 12}


def test_cli_shim_seed_fails_closed_with_exit_2(
    dated_session, builder_config, tmp_path
) -> None:
    from gentle_ai_model_router.cli.main import app

    dataset = build_examples(
        dated_session,
        _design_config(tmp_path),
        builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
    )
    out_dir = write_dataset(dataset, tmp_path / "data", builder=builder_config)
    result = runner.invoke(
        app,
        ["shim", "seed", "--dataset", str(out_dir), "--data-dir", str(tmp_path / "data")],
    )
    assert result.exit_code == 2
    assert "no empirical-provenance rows" in result.output


# --------------------------------------------------------------------------- #
# End-to-end: seeded shim -> build-dataset --telemetry-db -> telemetry rows
# --------------------------------------------------------------------------- #


def test_seeded_shim_feeds_dataset_build_end_to_end(
    dated_session, builder_config, tmp_path
) -> None:
    """The actual blocker: seeded shim rows must become telemetry dataset rows."""
    out_dir = _build_empirical_dataset(dated_session, builder_config, tmp_path)
    seed_empirical_dataset(out_dir, _shim_url(tmp_path))

    dataset = build_examples(
        dated_session,
        _design_config(tmp_path),
        builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(tmp_path / "telemetry.sqlite"),
    )
    tel = [e for e in dataset.examples if e.label_provenance == PROVENANCE_TELEMETRY]
    assert len(tel) == 6
    assert dataset.telemetry_stats == {"read": 6, "emitted": 6, "skipped": 0}
    # Telemetry rows carry MEASURED-style actuals taken from the seeded rows
    # (which are estimates by provenance — never mixed silently: the shim row
    # keeps router_version='empirical-bootstrap').
    by_model = {(e.candidate.model, e.candidate.effort): e for e in tel}
    old = by_model[("test/old-model", "low")]
    assert old.label_quality_estimate == pytest.approx(0.95)
    assert old.cost_features["actual_total_tokens"] == pytest.approx(
        old.cost_features["est_total_tokens"]
    )

    out_dir2 = write_dataset(dataset, tmp_path / "data2", builder=builder_config)
    manifest = json.loads((out_dir2 / "manifest.json").read_text())
    assert manifest["telemetry"] == {"read": 6, "emitted": 6, "skipped": 0}
    assert PROVENANCE_TELEMETRY in manifest["label_provenance"]
    assert manifest["feature_schema_version"] == "2"
