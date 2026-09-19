"""Phase 3a CLI tests: policy / explain / export inspection commands."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.registry import db as registry_db

runner = CliRunner()

AA = "artificial_analysis_intelligence_index"


def _seed_registry(data_dir: Path) -> None:
    engine = registry_db.get_engine(f"sqlite:///{data_dir / 'router.db'}")
    registry_db.init_schema(engine)
    with registry_db.Session(engine) as session:
        registry_db.record_snapshot(session, "test", "snap-01", datetime(2026, 1, 1), 2)
        for canonical, aa, (in_p, out_p) in [
            ("test/cheap-1", 30.0, (1.0, 2.0)),
            ("test/strong", 90.0, (8.0, 30.0)),
        ]:
            provider = registry_db.get_or_create_provider(session, "test")
            model = registry_db.upsert_model(
                session, canonical_id=canonical, tool_calling=True
            )
            deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
            for effort in ("low", "high"):
                registry_db.upsert_variant(session, deployment, effort, effort)
            registry_db.upsert_benchmark(session, model, AA, aa, None, "snap-01")
            registry_db.upsert_price(session, deployment, "snap-01", in_p, out_p, None)
        session.commit()


def test_policy_one_phase(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(
        app, ["policy", "--phase", "explore", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "policy: explore" in result.output
    assert "test/strong" in result.output
    assert "test/cheap-1" in result.output  # top-3 alternatives included
    assert "est cost" in result.output


def test_policy_all_phases(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(app, ["policy", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    for phase in ("init", "explore", "research", "propose", "spec", "design",
                  "tasks", "apply", "verify", "archive", "onboard"):
        assert f"policy: {phase}" in result.output


def test_policy_empty_registry_exits_2(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    result = runner.invoke(app, ["policy", "--phase", "explore", "--data-dir", str(data_dir)])
    assert result.exit_code == 2
    assert "registry is empty" in result.output


def test_explain_ranking_table(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(
        app,
        ["explain", "--phase", "sdd-explore", "--task", "map the module",
         "--data-dir", str(data_dir)],
    )
    assert result.exit_code == 0, result.output
    assert "phase (canonical)" in result.output
    assert "candidate ranking" in result.output
    assert "selected configuration" in result.output
    assert "cheapest_of_meeting" in result.output
    assert "test/strong" in result.output


def test_explain_json_emits_decision(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(
        app, ["explain", "--phase", "design", "--json", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    decision = json.loads(result.output)
    assert decision["phase"] == "design"
    assert decision["model"] == "test/strong"
    assert decision["reason_codes"]
    assert decision["policy_version"].startswith("pol-")


def test_explain_unknown_phase_exits_2(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(
        app, ["explain", "--phase", "bogus", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 2
    assert "unknown phase" in result.output


def test_export_writes_artifact(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    out = tmp_path / "policy.json"
    result = runner.invoke(
        app, ["export", "--output", str(out), "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out.read_text())
    assert payload["policy_version"].startswith("pol-")
    assert payload["registry_hash"].startswith("reg-")
    assert "exported_at" in payload
    assert set(payload["phases"]) == {
        "init", "explore", "research", "propose", "spec", "design",
        "tasks", "apply", "verify", "archive", "onboard",
    }
    explore = payload["phases"]["explore"]
    assert explore["selected"]["model"] == "test/strong"
    assert explore["thresholds"]["threshold_quality"] == 0.6
    assert explore["weights"]
    assert explore["alternatives"]


def test_calibrate_thresholds_one_phase(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(
        app, ["calibrate-thresholds", "--phase", "explore", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0, result.output
    assert "threshold calibration" in result.output
    assert "explore" in result.output
    assert "current" in result.output and "suggested" in result.output
    assert "min(max(current, p50), p90)" in result.output


def test_calibrate_thresholds_all_phases(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(app, ["calibrate-thresholds", "--data-dir", str(data_dir)])
    assert result.exit_code == 0, result.output
    for phase in ("init", "explore", "research", "propose", "spec", "design",
                  "tasks", "apply", "verify", "archive", "onboard"):
        assert phase in result.output
    # With two priors (0 and 1 normalized) the suggested floor sits at or
    # above the current threshold: never suggests going down.
    assert "meet" in result.output


def test_calibrate_thresholds_unknown_phase_exits_0(tmp_path: Path) -> None:
    """Informational command: even a usage error exits 0."""
    data_dir = tmp_path / "data"
    _seed_registry(data_dir)
    result = runner.invoke(
        app, ["calibrate-thresholds", "--phase", "bogus", "--data-dir", str(data_dir)]
    )
    assert result.exit_code == 0
    assert "unknown phase" in result.output
