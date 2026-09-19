"""Phase 4 CLI tests: `router bandit report|update` and `router feedback`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration import telemetry_shim

runner = CliRunner()


def _db_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'telemetry.sqlite'}"


def _seed_shim(tmp_path: Path) -> telemetry_shim.ShimStore:
    """A decision for test/model-a plus one successful + one failed execution."""
    store = telemetry_shim.ShimStore(_db_url(tmp_path))
    store.init_schema()
    with store.session() as session:
        decision = store.record_decision(
            session,
            phase="apply",
            selected={"model": "test/model-a", "effort": "low"},
            alternatives=[{"model": "test/model-b", "effort": "low"}],
            reason_codes=["cheapest_of_meeting"],
            estimated_tokens=14_000.0,
            estimated_cost=0.05,
            policy_version="pol-test",
        )
        for execution_id, success, tokens in (("e1", 1, 1000), ("e2", 0, 2000)):
            store.record_execution(
                session,
                {
                    "execution_id": execution_id,
                    "phase": "apply",
                    "model": "test/model-a",
                    "effort": "low",
                    "total_tokens": tokens,
                    "task_success": success,
                    "decision_id": decision.decision_id,
                },
            )
    return store


def test_bandit_report_empty_store(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["bandit", "report", "--db", _db_url(tmp_path), "--data-dir", str(tmp_path / "data")],
    )
    assert result.exit_code == 0, result.output
    assert "no reward data" in result.output
    assert "cold start" in result.output


def test_bandit_report_shows_aggregates(tmp_path: Path) -> None:
    _seed_shim(tmp_path)
    result = runner.invoke(
        app,
        ["bandit", "report", "--phase", "apply", "--db", _db_url(tmp_path),
         "--data-dir", str(tmp_path / "data")],
    )
    assert result.exit_code == 0, result.output
    assert "bandit reward aggregates" in result.output
    assert "test/model-a" in result.output
    assert "tokens/success" in result.output
    assert "ban-" in result.output


def test_bandit_report_phase_filter_excludes_other_phases(tmp_path: Path) -> None:
    _seed_shim(tmp_path)
    result = runner.invoke(
        app,
        ["bandit", "report", "--phase", "verify", "--db", _db_url(tmp_path),
         "--data-dir", str(tmp_path / "data")],
    )
    assert result.exit_code == 0, result.output
    assert "no reward data" in result.output


def test_bandit_update_prints_version_and_scores_outcomes(tmp_path: Path) -> None:
    store = telemetry_shim.ShimStore(_db_url(tmp_path))
    store.init_schema()
    with store.session() as session:
        store.record_execution(
            session,
            {
                "execution_id": "e1",
                "phase": "apply",
                "model": "test/model-a",
                "effort": "low",
                "total_tokens": 1000,
                "tests_passed": 4,
                "tests_failed": 0,
            },
        )
    result = runner.invoke(
        app,
        ["bandit", "update", "--db", _db_url(tmp_path), "--data-dir", str(tmp_path / "data")],
    )
    assert result.exit_code == 0, result.output
    assert "bandit_version" in result.output
    assert "ban-" in result.output
    assert "outcomes_scored" in result.output

    # The NULL-outcome row was scored by the rubric.
    with store.session() as session:
        (row,) = session.query(telemetry_shim.ExecutionRecord).all()
        assert row.task_success == 1
        assert row.quality_score == 100.0

    # Idempotent: second update scores nothing.
    again = runner.invoke(
        app,
        ["bandit", "update", "--db", _db_url(tmp_path), "--data-dir", str(tmp_path / "data")],
    )
    assert again.exit_code == 0, again.output


def test_feedback_ingests_and_scores_jsonl(tmp_path: Path) -> None:
    lines = "\n".join(
        [
            json.dumps(
                {
                    "execution_id": "e1",
                    "phase": "explore",
                    "model": "test/model-a",
                    "effort": "low",
                    "total_tokens": 500,
                    "tool_errors": 0,
                    "escalation_count": 0,
                }
            ),
            json.dumps(
                {
                    "execution_id": "e2",
                    "phase": "apply",
                    "model": "test/model-b",
                    "effort": "low",
                    "total_tokens": 900,
                    "tests_passed": 2,
                    "tests_failed": 1,
                }
            ),
            "not json\n",
        ]
    )
    result = runner.invoke(
        app,
        ["feedback", "--db", _db_url(tmp_path), "--data-dir", str(tmp_path / "data")],
        input=lines,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["ingested"] == {"inserted": 2, "updated": 0, "skipped": 1}
    assert payload["outcomes"] == {"scored": 2}
    assert payload["bandit_version"].startswith("ban-")

    store = telemetry_shim.ShimStore(_db_url(tmp_path))
    with store.session() as session:
        rows = {
            row.execution_id: row
            for row in session.query(telemetry_shim.ExecutionRecord).all()
        }
        assert rows["e1"].task_success == 1  # clean heuristic phase
        assert rows["e2"].task_success == 0  # one failed test gates apply
        # Raw quality 2/3*100 = 66.6, capped at 49 by the failed gate.
        assert rows["e2"].quality_score == 49.0
