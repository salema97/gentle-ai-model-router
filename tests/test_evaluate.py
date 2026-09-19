"""Offline evaluation tests: baselines-only mode on a tiny synthetic dataset."""

from __future__ import annotations

from datetime import date, datetime

from gentle_ai_model_router.dataset.builder import (
    DatasetBuilderConfig,
    build_examples,
    write_dataset,
)
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import PhaseConfig, load_config
from gentle_ai_model_router.training.evaluate import (
    EVALUATED_SPLITS,
    METRIC_KEYS,
    evaluate_dataset,
)

AA = "artificial_analysis_intelligence_index"


def _build(tmp_path, with_policy_session: bool):
    """Single-group dataset: 2 cheap-weak + 1 strong-expensive candidates.

    Split layout (train_end=2026-02-15, val_end=2026-02-20):
    - cheap models appear at snap-2026-01 but carry a LATER price row
      (2026-03-01) -> snapshot_date after val_end -> split "test".
    - strong model appears at snap-2026-03 -> first seen after train_end
      -> split "temporal_test".
    """
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    snaps = {
        "snap-01": datetime(2026, 1, 1),
        "snap-03": datetime(2026, 3, 1),
    }
    with registry_db.Session(engine) as session:
        for snap_id, fetched in snaps.items():
            registry_db.record_snapshot(session, "test", snap_id, fetched, 1)
        rows = [
            ("test/cheap-1", 30.0, 1.0, 2.0),
            ("test/cheap-2", 35.0, 1.2, 3.0),
            ("test/strong", 90.0, 8.0, 30.0),
        ]
        for canonical, aa, in_p, out_p in rows:
            provider = registry_db.get_or_create_provider(session, "test")
            model = registry_db.upsert_model(session, canonical_id=canonical, tool_calling=True)
            deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
            registry_db.upsert_variant(session, deployment, "low", "low")
            registry_db.upsert_variant(session, deployment, "high", "high")
            bench_snap = "snap-01" if canonical.startswith("test/cheap") else "snap-03"
            registry_db.upsert_benchmark(session, model, AA, aa, None, bench_snap)
            registry_db.upsert_price(session, deployment, "snap-03", in_p, out_p, None)
        session.commit()

    config = load_config(
        config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data"
    ).model_copy(
        update={"phases": {"design": PhaseConfig(threshold_quality=0.7, weights={AA: 1.0})}}
    )
    builder = DatasetBuilderConfig(
        name="eval-fixture",
        phases=["design"],
        task_types=["feature"],
        context_sizes=[10_000],
        train_end=date(2026, 2, 15),
        val_end=date(2026, 2, 20),
        pair_margin=0.0,
    )
    with registry_db.Session(engine) as session:
        dataset = build_examples(session, config, builder)
        out_dir = write_dataset(dataset, tmp_path / "data")
    session_for_policy = registry_db.Session(engine) if with_policy_session else None
    return config, out_dir, session_for_policy


def test_baselines_only_runs_without_checkpoint(tmp_path) -> None:
    config, out_dir, _ = _build(tmp_path, with_policy_session=False)
    results = evaluate_dataset(out_dir, config, session=None, checkpoint=None)

    assert set(results["splits"]) == {"test", "temporal_test"}
    for split in EVALUATED_SPLITS:
        if split not in results["splits"]:
            continue
        routers = results["splits"][split]
        assert {"fixed_strong", "fixed_cheap", "benchmark_only"} <= set(routers)
        for metrics in routers.values():
            assert set(METRIC_KEYS) <= set(metrics)
            assert metrics["routing_regret"] >= 0.0  # oracle dominates by construction
            assert 0.0 <= metrics["success_rate"] <= 1.0
            assert metrics["tokens_per_task"] > 0
    assert "bootstrap_prior" in results["caveat"]


def test_business_metrics_distinguish_routers(tmp_path) -> None:
    config, out_dir, session = _build(tmp_path, with_policy_session=True)
    results = evaluate_dataset(out_dir, config, session=session, checkpoint=None)

    test_split = results["splits"]["test"]
    assert "baseline_policy" in test_split  # policy reference replayed on registry
    # fixed_cheap must never spend more than fixed_strong per task.
    assert (
        test_split["fixed_cheap"]["tokens_per_task"]
        <= test_split["fixed_strong"]["tokens_per_task"]
    )
    # Regret ordering sanity: cheap picks can be far from the oracle.
    assert (
        test_split["fixed_cheap"]["routing_regret"]
        >= test_split["fixed_strong"]["routing_regret"] - 1e-9
    )


def test_metrics_json_written(tmp_path) -> None:
    config, out_dir, _ = _build(tmp_path, with_policy_session=False)
    output = tmp_path / "metrics.json"
    evaluate_dataset(out_dir, config, output_path=output)
    import json

    payload = json.loads(output.read_text())
    assert "splits" in payload and "caveat" in payload
