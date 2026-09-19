"""Phase 2 CLI smoke tests: build-dataset + evaluate --baselines-only.

No network, no [train] extra: a synthetic dated registry is built directly
in the CLI's data dir, then both commands run end-to-end.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.registry import db as registry_db

runner = CliRunner()

AA = "artificial_analysis_intelligence_index"


def _seed_registry(data_dir) -> None:
    engine = registry_db.get_engine(f"sqlite:///{data_dir / 'router.db'}")
    registry_db.init_schema(engine)
    with registry_db.Session(engine) as session:
        registry_db.record_snapshot(session, "test", "snap-01", datetime(2026, 1, 1), 2)
        registry_db.record_snapshot(session, "test", "snap-03", datetime(2026, 3, 1), 1)
        rows = [
            ("test/cheap-1", 30.0, "snap-01", 1.0, 2.0),
            ("test/cheap-2", 35.0, "snap-01", 1.2, 3.0),
            ("test/strong", 90.0, "snap-03", 8.0, 30.0),
        ]
        for canonical, aa, bench_snap, in_p, out_p in rows:
            provider = registry_db.get_or_create_provider(session, "test")
            model = registry_db.upsert_model(session, canonical_id=canonical, tool_calling=True)
            deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
            for effort in ("low", "high"):
                registry_db.upsert_variant(session, deployment, effort, effort)
            registry_db.upsert_benchmark(session, model, AA, aa, None, bench_snap)
            registry_db.upsert_price(session, deployment, "snap-03", in_p, out_p, None)
        session.commit()


def test_cli_build_dataset_smoke(tmp_path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(
        app,
        [
            "build-dataset",
            "--name", "smoke",
            "--train-end", "2026-02-15",
            "--val-end", "2026-02-20",
            "--phase", "design",
            "--data-dir", str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bootstrap priors" in result.output
    out_dir = data_dir / "datasets" / "smoke" / "v1"
    assert (out_dir / "examples.parquet").exists()
    assert (out_dir / "pairs.parquet").exists()
    assert (out_dir / "manifest.json").exists()


def test_cli_evaluate_baselines_only_smoke(tmp_path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    build = runner.invoke(
        app,
        [
            "build-dataset",
            "--name", "smoke",
            "--train-end", "2026-02-15",
            "--val-end", "2026-02-20",
            "--phase", "design",
            "--data-dir", str(data_dir),
        ],
    )
    assert build.exit_code == 0, build.output
    dataset_dir = data_dir / "datasets" / "smoke" / "v1"
    output = tmp_path / "metrics.json"

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--dataset", str(dataset_dir),
            "--evaluate-baselines-only",
            "--output", str(output),
            "--data-dir", str(data_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "split: test" in result.output  # rich table rendered (narrow-terminal safe)
    assert output.exists()
    import json

    payload = json.loads(output.read_text())
    assert "splits" in payload
    assert {"test", "temporal_test"} <= set(payload["splits"])
    for metrics in payload["splits"]["test"].values():
        assert metrics["routing_regret"] >= 0.0


def test_cli_train_requires_train_extra_or_runs(tmp_path) -> None:
    """`router train` fails gracefully (exit 2) without the [train] extra."""
    import importlib.util

    has_train = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    runner.invoke(
        app,
        [
            "build-dataset",
            "--name", "smoke",
            "--train-end", "2026-02-15",
            "--phase", "design",
            "--data-dir", str(data_dir),
        ],
    )
    dataset_dir = data_dir / "datasets" / "smoke" / "v1"
    result = runner.invoke(
        app,
        [
            "train",
            "--dataset", str(dataset_dir),
            "--objective", "pointwise",
            "--epochs", "1",
            "--data-dir", str(data_dir),
        ],
    )
    if has_train:  # pragma: no cover - depends on local env
        # A real `router train` smoke run would need the deberta-v3-base
        # tokenizer from Hugging Face — downloads are forbidden in tests.
        pytest.skip("full train smoke run requires an HF download")
    assert result.exit_code == 2
    assert "[train] extra" in result.output
