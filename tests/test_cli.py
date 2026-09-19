"""CLI smoke tests: exit codes and deterministic output via CliRunner."""

from __future__ import annotations

import httpx
import respx
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app

runner = CliRunner()
BASE = "https://artificialanalysis.ai"


def test_registry_stats_fresh_db(tmp_path) -> None:
    result = runner.invoke(app, ["registry", "stats", "--data-dir", str(tmp_path / "data")])
    assert result.exit_code == 0
    assert "providers" in result.output
    assert "models" in result.output


def test_collect_local_runs_without_crashing(tmp_path) -> None:
    result = runner.invoke(
        app,
        ["collect", "--source", "local", "--data-dir", str(tmp_path / "data"),
         "--config", str(tmp_path / "router.yaml")],
    )
    assert result.exit_code == 0
    assert "local candidates" in result.output


def test_collect_unknown_source_exit_2(tmp_path) -> None:
    result = runner.invoke(app, ["collect", "--source", "bogus"])
    assert result.exit_code == 2


def test_snapshots_list_empty(tmp_path) -> None:
    result = runner.invoke(
        app, ["snapshots", "list", "--source", "aa", "--data-dir", str(tmp_path / "data")]
    )
    assert result.exit_code == 0
    assert "no snapshots" in result.output


@respx.mock
def test_collect_then_normalize_then_stats(tmp_path) -> None:
    payload = [
        {
            "model": "claude-sonnet-4",
            "provider": "anthropic",
            "intelligence_index": 42.5,
            "pricing": {"input": 3.0, "output": 15.0},
        }
    ]
    respx.get(f"{BASE}/api/v2/language/models/free").mock(
        return_value=httpx.Response(200, json=payload)
    )
    data_dir = str(tmp_path / "data")
    r1 = runner.invoke(app, ["collect", "--source", "aa", "--data-dir", data_dir])
    assert r1.exit_code == 0

    r2 = runner.invoke(app, ["normalize", "--data-dir", data_dir])
    assert r2.exit_code == 0
    assert "artificial-analysis" in r2.output

    r3 = runner.invoke(app, ["registry", "stats", "--data-dir", data_dir])
    assert r3.exit_code == 0
    assert "models" in r3.output
