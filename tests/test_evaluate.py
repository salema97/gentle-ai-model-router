"""Offline evaluation tests: baselines-only mode on a tiny synthetic dataset."""

from __future__ import annotations

from datetime import date, datetime

from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.dataset.builder import (
    DatasetBuilderConfig,
    build_examples,
    write_dataset,
)
from gentle_ai_model_router.dataset.schema import CandidateRef, DatasetExample, DatasetV1
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import PhaseConfig, load_config
from gentle_ai_model_router.training.evaluate import (
    EVALUATED_SPLITS,
    METRIC_KEYS,
    evaluate_dataset,
    evaluate_routers,
)

runner = CliRunner()

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


# --------------------------------------------------------------------------- #
# Fix: a failing reference chooser for one phase must not be dropped silently
# --------------------------------------------------------------------------- #


def _mk_example(idx: int, phase: str, task_id: str, model: str, utility: float) -> DatasetExample:
    return DatasetExample(
        example_id=f"ex-{idx}",
        task_id=task_id,
        phase=phase,
        task_type="feature",
        context_tokens=10_000,
        task_text="synthetic task",
        candidate=CandidateRef(model=model, deployment="default", effort="high"),
        model_features=[0.0] * 7,
        benchmark_features=[utility],
        benchmark_feature_names=(AA,),
        cost_features={
            "est_input_tokens": 100.0,
            "est_output_tokens": 100.0,
            "est_total_tokens": 200.0,
            "est_cost": 1.0,
        },
        label_utility=utility,
        label_quality_estimate=utility,
        label_provenance="bootstrap_prior",
        snapshot_date="2026-01-01",
        split="test",
    )


def _synthetic_dataset() -> DatasetV1:
    examples: list[DatasetExample] = []
    idx = 0
    for phase in ("design", "spec"):
        for task_id in ("t1", "t2"):
            for model, utility in (("test/strong", 0.9), ("test/cheap-1", 0.4)):
                examples.append(_mk_example(idx, phase, f"{phase}-{task_id}", model, utility))
                idx += 1
    return DatasetV1(
        name="chooser-errors",
        version=1,
        examples=examples,
        benchmark_feature_names=(AA,),
    )


def _seed_registry(engine) -> None:
    with registry_db.Session(engine) as session:
        registry_db.record_snapshot(session, "test", "snap-01", datetime(2026, 1, 1), 2)
        registry_db.record_snapshot(session, "test", "snap-03", datetime(2026, 3, 1), 2)
        for canonical, aa in [("test/cheap-1", 30.0), ("test/strong", 90.0)]:
            provider = registry_db.get_or_create_provider(session, "test")
            model = registry_db.upsert_model(session, canonical_id=canonical)
            deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
            for effort in ("low", "high"):
                registry_db.upsert_variant(session, deployment, effort, effort)
            registry_db.upsert_benchmark(session, model, AA, aa, None, "snap-01")
            # Price at snap-03 (after val_end) -> examples split "test".
            registry_db.upsert_price(session, deployment, "snap-03", 1.0, 2.0, None)
        session.commit()


def _unmeetable_config(tmp_path):
    """spec threshold 0.999 > max achievable quality (~0.992): the policy can
    NEVER meet it (a normalized prior of 1.0 exceeds effort_ceiling 0.98, so
    even a 0.99 threshold would be meetable — 0.999 is safely above)."""
    return load_config(
        config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data"
    ).model_copy(
        update={
            "phases": {
                "design": PhaseConfig(threshold_quality=0.7, weights={AA: 1.0}),
                "spec": PhaseConfig(threshold_quality=0.999, weights={AA: 1.0}),
            }
        }
    )


def test_chooser_error_recorded_per_split_phase_and_others_unaffected(tmp_path) -> None:
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    _seed_registry(engine)
    config = _unmeetable_config(tmp_path)
    dataset = _synthetic_dataset()

    with registry_db.Session(engine) as session:
        results = evaluate_routers(dataset, config, session=session)

    errors = results["chooser_errors"]
    assert len(errors) == 1
    entry = errors[0]
    assert entry["split"] == "test"
    assert entry["phase"] == "spec"
    assert entry["chooser"] == "baseline_policy"
    assert "no candidate meets threshold" in entry["error"]

    split = results["splits"]["test"]
    # The failing chooser still has metrics — from the phases that worked.
    assert "baseline_policy" in split
    assert split["baseline_policy"]["groups"] == 2.0  # design t1 + t2 only
    # Other choosers evaluated ALL groups (both phases unaffected).
    for name in ("fixed_strong", "fixed_cheap", "benchmark_only"):
        assert split[name]["groups"] == 4.0
    # Merge counters must not leak into the public metrics.
    for metrics in split.values():
        assert set(metrics) == set(METRIC_KEYS)


def test_chooser_errors_written_to_metrics_json(tmp_path) -> None:
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(engine)
    _seed_registry(engine)
    config = _unmeetable_config(tmp_path)
    dataset = _synthetic_dataset()
    output = tmp_path / "metrics.json"

    with registry_db.Session(engine) as session:
        results = evaluate_routers(dataset, config, session=session)
    import json

    output.write_text(json.dumps(results) + "\n", encoding="utf-8")
    payload = json.loads(output.read_text())
    assert payload["chooser_errors"][0]["phase"] == "spec"


def test_evaluate_cli_prints_chooser_error_note(tmp_path) -> None:
    data_dir = tmp_path / "data"
    engine = registry_db.get_engine(f"sqlite:///{data_dir / 'router.db'}")
    registry_db.init_schema(engine)
    _seed_registry(engine)
    # The CLI resolves config from cwd (router.yaml.example) unless told
    # otherwise — pass an explicit config with the unmeetable spec floor.
    config_path = tmp_path / "router.yaml"
    config_path.write_text(
        "phases:\n"
        "  design:\n"
        "    threshold_quality: 0.7\n"
        f'    weights: {{"{AA}": 1.0}}\n'
        "  spec:\n"
        "    threshold_quality: 0.999\n"
        f'    weights: {{"{AA}": 1.0}}\n',
        encoding="utf-8",
    )
    config = _unmeetable_config(tmp_path)

    builder = DatasetBuilderConfig(
        name="chooser-errors-cli",
        phases=["design", "spec"],
        task_types=["feature"],
        context_sizes=[10_000],
        train_end=date(2026, 2, 15),
        val_end=date(2026, 2, 20),
        pair_margin=0.0,
    )
    with registry_db.Session(engine) as session:
        dataset = build_examples(session, config, builder)
        out_dir = write_dataset(dataset, data_dir)

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--dataset",
            str(out_dir),
            "--output",
            str(tmp_path / "metrics.json"),
            "--config",
            str(config_path),
            "--data-dir",
            str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    # spec fails closed in the populated test split.
    assert "1 chooser error(s): see metrics json" in result.output
