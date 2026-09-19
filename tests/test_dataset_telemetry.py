"""Telemetry bridge tests: synthetic shim executions -> telemetry DatasetExample rows.

Fixtures mirror tests/test_dataset_builder.py (dated design-phase registry)
plus a seeded telemetry shim (pattern from tests/test_bandit_cli.py).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.dataset.builder import (
    DatasetBuilderConfig,
    DatasetBuildError,
    build_examples,
    load_dataset,
    write_dataset,
)
from gentle_ai_model_router.dataset.schema import (
    PROVENANCE_BOOTSTRAP,
    PROVENANCE_TELEMETRY,
    TELEMETRY_PROVENANCE_ADDENDUM,
)
from gentle_ai_model_router.integration import telemetry_shim
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
        name="test-telemetry",
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
        _add_model(
            session,
            "test/mid-model",
            SNAP_MID,
            SNAP_NEW,
            {AA: 40.0, ARENA_TEXT: 95.0},
        )
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


def _db_path(tmp_path: Path) -> Path:
    return tmp_path / "telemetry.sqlite"


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{_db_path(tmp_path)}"


def _seed_shim(tmp_path: Path) -> telemetry_shim.ShimStore:
    """One decision plus scored/mixed executions covering every branch."""
    store = telemetry_shim.ShimStore(_db_url(tmp_path))
    store.init_schema()
    with store.session() as session:
        decision = store.record_decision(
            session,
            phase="design",
            selected={"model": "test/old-model", "effort": "low"},
            alternatives=[{"model": "test/mid-model", "effort": "low"}],
            reason_codes=["cheapest_of_meeting"],
            estimated_tokens=14_000.0,
            estimated_cost=0.05,
            policy_version="pol-test",
        )
        payloads = [
            # Same (phase, task_type, date) -> same telemetry task group, so
            # the token penalty normalization is exercised for real.
            {
                "execution_id": "e1",
                "phase": "design",
                "task_type": "feature",
                "model": "test/old-model",
                "deployment": "default",
                "effort": "low",
                "input_tokens": 4000,
                "output_tokens": 1000,
                "total_tokens": 5000,
                "latency_ms": 1200,
                "quality_score": 0.9,
                "finished_at": "2026-01-10T12:00:00+00:00",
                "decision_id": decision.decision_id,
            },
            {
                "execution_id": "e2",
                "phase": "design",
                "task_type": "feature",
                "model": "test/mid-model",
                "deployment": "default",
                "effort": "low",
                "input_tokens": 8000,
                "output_tokens": 1000,
                "total_tokens": 9000,
                "latency_ms": 2400,
                "quality_score": 0.4,
                "finished_at": "2026-01-10T13:00:00+00:00",
                "decision_id": decision.decision_id,
            },
            {
                # task_success only (no quality_score) -> quality coerced 1.0
                "execution_id": "e7",
                "phase": "design",
                "task_type": "feature",
                "model": "test/old-model",
                "deployment": "default",
                "effort": "high",
                "input_tokens": 1500,
                "output_tokens": 500,
                "total_tokens": 2000,
                "task_success": 1,
                "finished_at": "2026-01-10T14:00:00+00:00",
                "decision_id": decision.decision_id,
            },
            # --- unscored / invalid: read but never emitted ---------------- #
            {
                "execution_id": "e3",  # NULL outcomes -> not even read
                "phase": "design",
                "model": "test/old-model",
                "total_tokens": 1000,
                "finished_at": "2026-01-10T15:00:00+00:00",
                "decision_id": decision.decision_id,
            },
            {
                "execution_id": "e4",  # unknown model -> skipped
                "phase": "design",
                "model": "test/ghost-model",
                "total_tokens": 1000,
                "quality_score": 0.8,
                "finished_at": "2026-01-10T15:00:00+00:00",
                "decision_id": decision.decision_id,
            },
            {
                "execution_id": "e5",  # phase not in build -> skipped
                "phase": "verify",
                "model": "test/old-model",
                "total_tokens": 1000,
                "quality_score": 0.8,
                "finished_at": "2026-01-10T15:00:00+00:00",
                "decision_id": decision.decision_id,
            },
            {
                "execution_id": "e6",  # negative tokens -> skipped
                "phase": "design",
                "model": "test/old-model",
                "total_tokens": -5,
                "quality_score": 0.8,
                "finished_at": "2026-01-10T15:00:00+00:00",
                "decision_id": decision.decision_id,
            },
            {
                "execution_id": "e8",  # outcome outside [0,1] -> skipped
                "phase": "design",
                "model": "test/old-model",
                "total_tokens": 1000,
                "quality_score": 1.5,
                "finished_at": "2026-01-10T15:00:00+00:00",
                "decision_id": decision.decision_id,
            },
        ]
        for payload in payloads:
            store.record_execution(session, payload)
    return store


def _telemetry_examples(dataset):
    return [e for e in dataset.examples if e.label_provenance == PROVENANCE_TELEMETRY]


# --------------------------------------------------------------------------- #
# Emission, provenance, labels
# --------------------------------------------------------------------------- #


def test_telemetry_rows_emitted_with_provenance_and_actuals(
    dated_session, builder_config, tmp_path
) -> None:
    _seed_shim(tmp_path)
    dataset = build_examples(
        dated_session, _design_config(tmp_path), builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(_db_path(tmp_path)),
    )
    tel = _telemetry_examples(dataset)
    assert len(tel) == 3  # e1, e2, e7

    by_key = {(e.candidate.model, e.candidate.effort): e for e in tel}
    e1 = by_key[("test/old-model", "low")]
    assert e1.task_id.startswith("task-tel-")
    assert e1.label_provenance == PROVENANCE_TELEMETRY
    # ACTUAL (measured) token values, not estimates.
    assert e1.cost_features["actual_total_tokens"] == 5000.0
    assert e1.cost_features["actual_input_tokens"] == 4000.0
    assert e1.cost_features["actual_output_tokens"] == 1000.0
    assert e1.cost_features["latency_ms"] == 1200.0
    # est_* still present (decision's pre-execution estimate) for comparability.
    assert e1.cost_features["est_total_tokens"] == 14_000.0
    assert e1.label_quality_estimate == pytest.approx(0.9)

    e2 = by_key[("test/mid-model", "low")]
    assert e2.cost_features["actual_total_tokens"] == 9000.0
    assert e2.label_quality_estimate == pytest.approx(0.4)

    e7 = by_key[("test/old-model", "high")]
    assert e7.label_quality_estimate == pytest.approx(1.0)  # task_success coerced

    # Stats surfaced for the build summary.
    assert dataset.telemetry_stats == {"read": 7, "emitted": 3, "skipped": 4}
    # Provenance statement gains the telemetry addendum.
    assert TELEMETRY_PROVENANCE_ADDENDUM in dataset.label_provenance_statement


def test_utility_label_matches_bootstrap_formula_with_actual_tokens(
    dated_session, builder_config, tmp_path
) -> None:
    """utility = clip(quality - cost_weight * ACTUAL_total / max_group_actual)."""
    _seed_shim(tmp_path)
    dataset = build_examples(
        dated_session, _design_config(tmp_path), builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(_db_path(tmp_path)),
    )
    tel = _telemetry_examples(dataset)
    group = {(e.candidate.model, e.candidate.effort): e for e in tel}
    max_tokens = max(e.cost_features["actual_total_tokens"] for e in tel)
    assert max_tokens == 9000.0

    e1 = group[("test/old-model", "low")]
    expected = max(0.0, min(1.0, 0.9 - 0.5 * 5000.0 / 9000.0))
    assert e1.label_utility == pytest.approx(expected, abs=1e-9)

    e2 = group[("test/mid-model", "low")]
    expected2 = max(0.0, min(1.0, 0.4 - 0.5 * 9000.0 / 9000.0))
    assert e2.label_utility == pytest.approx(expected2, abs=1e-9)


def test_threshold_conditioning_applies_to_telemetry(
    dated_session, builder_config, tmp_path
) -> None:
    """Same threshold rules as bootstrap: e2 quality 0.4 < 0.85 floor."""
    _seed_shim(tmp_path)
    config = _design_config(tmp_path)
    builder_hard = builder_config.model_copy(update={"hard_threshold": True})
    dataset = build_examples(
        dated_session, config, builder_hard,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(_db_path(tmp_path)),
    )
    tel = _telemetry_examples(dataset)
    e2 = next(e for e in tel if e.candidate.model == "test/mid-model")
    assert e2.label_quality_estimate < config.phase_config("design").threshold_quality
    assert e2.label_utility == 0.0

# --------------------------------------------------------------------------- #
# Determinism & split assignment
# --------------------------------------------------------------------------- #


def test_split_assignment_deterministic_across_builds(
    dated_session, builder_config, tmp_path
) -> None:
    _seed_shim(tmp_path)
    kwargs = dict(
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(_db_path(tmp_path)),
    )
    first = build_examples(dated_session, _design_config(tmp_path), builder_config, **kwargs)
    second = build_examples(dated_session, _design_config(tmp_path), builder_config, **kwargs)

    def fingerprint(d):
        return [
            (e.example_id, e.split, e.label_utility, e.label_provenance)
            for e in d.examples
        ]

    assert fingerprint(first) == fingerprint(second)
    # All telemetry rows are train-dated (2026-01-10 < train_end).
    tel = _telemetry_examples(first)
    assert {e.split for e in tel} == {"train"}


def test_split_follows_execution_date(dated_session, builder_config, tmp_path) -> None:
    store = telemetry_shim.ShimStore(_db_url(tmp_path))
    store.init_schema()
    with store.session() as session:
        decision = store.record_decision(
            session,
            phase="design",
            selected={"model": "test/old-model", "effort": "low"},
            alternatives=[],
            reason_codes=[],
        )
        for execution_id, finished in (
            ("t-train", "2026-01-10T00:00:00+00:00"),
            ("t-val", "2026-02-18T00:00:00+00:00"),
            ("t-test", "2026-05-01T00:00:00+00:00"),
        ):
            store.record_execution(
                session,
                {
                    "execution_id": execution_id,
                    "phase": "design",
                    "model": "test/old-model",
                    "total_tokens": 1000,
                    "quality_score": 0.9,
                    "finished_at": finished,
                    "decision_id": decision.decision_id,
                },
            )
    dataset = build_examples(
        dated_session, _design_config(tmp_path), builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(_db_path(tmp_path)),
    )
    splits = {e.snapshot_date: e.split for e in _telemetry_examples(dataset)}
    assert splits == {"2026-01-10": "train", "2026-02-18": "validation", "2026-05-01": "test"}


# --------------------------------------------------------------------------- #
# Fail-closed behavior
# --------------------------------------------------------------------------- #


def test_absent_db_yields_zero_rows_and_unchanged_bootstrap(
    dated_session, builder_config, tmp_path
) -> None:
    missing = tmp_path / "does-not-exist.sqlite"
    off = build_examples(
        dated_session, _design_config(tmp_path), builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
    )
    on = build_examples(
        dated_session, _design_config(tmp_path), builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(missing),
    )
    assert _telemetry_examples(on) == []
    assert on.telemetry_stats == {"read": 0, "emitted": 0, "skipped": 0}
    assert [(e.example_id, e.label_utility, e.split) for e in off.examples] == [
        (e.example_id, e.label_utility, e.split) for e in on.examples
    ]


def test_empty_db_yields_zero_telemetry_rows(
    dated_session, builder_config, tmp_path
) -> None:
    store = telemetry_shim.ShimStore(_db_url(tmp_path))
    store.init_schema()  # schema present, zero rows
    dataset = build_examples(
        dated_session, _design_config(tmp_path), builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(_db_path(tmp_path)),
    )
    assert _telemetry_examples(dataset) == []
    assert dataset.telemetry_stats["read"] == 0


def test_unreadable_db_is_typed_build_error(dated_session, builder_config, tmp_path) -> None:
    corrupt = tmp_path / "corrupt.sqlite"
    corrupt.write_text("this is not a sqlite database")
    with pytest.raises(DatasetBuildError, match="unreadable telemetry shim DB"):
        build_examples(
            dated_session, _design_config(tmp_path), builder_config,
            store=SnapshotStore(tmp_path / "data" / "snapshots"),
            telemetry_db=str(corrupt),
        )


def test_off_by_default_is_byte_identical(dated_session, builder_config, tmp_path) -> None:
    _seed_shim(tmp_path)
    kwargs = dict(store=SnapshotStore(tmp_path / "data" / "snapshots"))
    a = build_examples(dated_session, _design_config(tmp_path), builder_config, **kwargs)
    b = build_examples(dated_session, _design_config(tmp_path), builder_config, **kwargs)
    assert a.telemetry_stats == {}
    assert "telemetry" not in a.manifest()
    assert [(e.example_id, e.label_utility) for e in a.examples] == [
        (e.example_id, e.label_utility) for e in b.examples
    ]
    # No v2 actual_* keys leak into the default cost_features.
    assert all(
        set(e.cost_features) == {"est_input_tokens", "est_output_tokens",
                                 "est_total_tokens", "est_cost"}
        for e in a.examples
    )


# --------------------------------------------------------------------------- #
# Persistence round-trip (schema v2)
# --------------------------------------------------------------------------- #


def test_write_load_roundtrip_preserves_actuals(dated_session, builder_config, tmp_path) -> None:
    _seed_shim(tmp_path)
    dataset = build_examples(
        dated_session, _design_config(tmp_path), builder_config,
        store=SnapshotStore(tmp_path / "data" / "snapshots"),
        telemetry_db=str(_db_path(tmp_path)),
    )
    out_dir = write_dataset(dataset, tmp_path / "data", builder=builder_config)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["feature_schema_version"] == "2"
    assert manifest["telemetry"] == {"read": 7, "emitted": 3, "skipped": 4}

    reloaded = load_dataset(out_dir)
    tel = _telemetry_examples(reloaded)
    assert len(tel) == 3
    e1 = next(e for e in tel if e.candidate.model == "test/old-model")
    assert e1.cost_features["actual_total_tokens"] == 5000.0
    assert e1.cost_features["latency_ms"] == 1200.0
    # Bootstrap rows keep only the v1 keys (None actuals are dropped on load).
    boot = next(e for e in reloaded.examples if e.label_provenance == PROVENANCE_BOOTSTRAP)
    assert set(boot.cost_features) == {"est_input_tokens", "est_output_tokens",
                                       "est_total_tokens", "est_cost"}


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #


def _seed_registry(data_dir) -> None:
    engine = registry_db.get_engine(f"sqlite:///{data_dir / 'router.db'}")
    registry_db.init_schema(engine)
    with registry_db.Session(engine) as session:
        registry_db.record_snapshot(session, "test", SNAP_OLD, datetime(2026, 1, 1, tzinfo=UTC), 1)
        _add_model(session, "test/cheap-1", SNAP_OLD, SNAP_OLD, {AA: 50.0, ARENA_TEXT: 90.0})
        session.commit()


def test_cli_build_dataset_with_telemetry_db(tmp_path) -> None:
    from gentle_ai_model_router.cli.main import app

    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    shim_path = tmp_path / "telemetry.sqlite"
    store = telemetry_shim.ShimStore(f"sqlite:///{shim_path}")
    store.init_schema()
    with store.session() as session:
        decision = store.record_decision(
            session,
            phase="design",
            selected={"model": "test/cheap-1", "effort": "low"},
            alternatives=[],
            reason_codes=[],
            estimated_tokens=10_000.0,
            estimated_cost=0.04,
        )
        store.record_execution(
            session,
            {
                "execution_id": "cli-e1",
                "phase": "design",
                "model": "test/cheap-1",
                "effort": "low",
                "total_tokens": 6000,
                "input_tokens": 5000,
                "output_tokens": 1000,
                "latency_ms": 900,
                "quality_score": 0.95,
                "finished_at": "2026-01-05T10:00:00+00:00",
                "decision_id": decision.decision_id,
            },
        )
    result = runner.invoke(
        app,
        [
            "build-dataset",
            "--name", "tel_smoke",
            "--train-end", "2026-02-15",
            "--phase", "design",
            "--telemetry-db", str(shim_path),
            "--data-dir", str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "telemetry" in result.output
    out_dir = data_dir / "datasets" / "tel_smoke" / "v1"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["telemetry"]["emitted"] == 1
    assert manifest["telemetry"]["skipped"] == 0
