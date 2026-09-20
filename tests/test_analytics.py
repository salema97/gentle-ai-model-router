"""Tests for analytics engine, ROI metrics calculation, /dashboard, and CLI roi command."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from gentle_ai_model_router.api.server import create_app
from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration.telemetry_shim import ShimStore
from gentle_ai_model_router.router.analytics import compute_roi
from gentle_ai_model_router.router.config import RouterConfig

runner = CliRunner()


def test_compute_roi_with_mock_executions(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry_test.sqlite"
    store = ShimStore(f"sqlite:///{db_path}")
    store.init_schema()

    with store.session() as session:
        # Record 1: explore with Gemini Flash (cheap, fast, success)
        store.record_execution(
            session,
            {
                "execution_id": "exec-1",
                "phase": "explore",
                "model": "google/gemini-2.5-flash",
                "input_tokens": 10000,
                "output_tokens": 2000,
                "latency_ms": 450,
                "task_success": 1,
                "quality_score": 1.0,
            },
        )
        # Record 2: apply with DeepSeek Coder (cheap, success)
        store.record_execution(
            session,
            {
                "execution_id": "exec-2",
                "phase": "apply",
                "model": "deepseek/deepseek-coder-v2",
                "input_tokens": 25000,
                "output_tokens": 5000,
                "latency_ms": 800,
                "task_success": 1,
                "quality_score": 0.9,
            },
        )
        # Record 3: verify with Sonnet 3.5 (frontier, success)
        store.record_execution(
            session,
            {
                "execution_id": "exec-3",
                "phase": "verify",
                "model": "anthropic/claude-3-5-sonnet",
                "input_tokens": 15000,
                "output_tokens": 1500,
                "latency_ms": 1900,
                "task_success": 1,
                "quality_score": 1.0,
            },
        )

    with store.session() as session:
        summary = compute_roi(session)

    assert summary.total_executions == 3
    assert summary.total_input_tokens == 50000
    assert summary.total_output_tokens == 8500
    assert summary.total_tokens == 58500
    assert summary.total_saved_usd > 0.0
    assert summary.savings_percentage > 0.0
    assert summary.overall_success_rate == 1.0
    assert len(summary.phases) == 3
    assert len(summary.top_models) == 3


def test_api_analytics_roi_and_dashboard_endpoints(tmp_path: Path) -> None:
    db_path = tmp_path / "telemetry_api.sqlite"
    store = ShimStore(f"sqlite:///{db_path}")
    store.init_schema()

    config = RouterConfig(data_dir=tmp_path)
    from gentle_ai_model_router.registry import db as registry_db
    engine, _ = registry_db.get_engine_with_fallback(
        config.database_url, config.sqlite_fallback_url
    )
    registry_db.init_schema(engine)

    app_instance = create_app(engine=engine, config=config, shim_store=store)
    client = TestClient(app_instance)

    # 1. GET /analytics/roi
    res_roi = client.get("/analytics/roi")
    assert res_roi.status_code == 200
    data = res_roi.json()
    assert "total_saved_usd" in data
    assert "savings_percentage" in data
    assert "phases" in data

    # 2. GET /dashboard
    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "text/html" in res_dash.headers["content-type"]
    assert "Gentle AI Model Router" in res_dash.text
    assert "fetchRoiData" in res_dash.text


def test_cli_roi_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["roi", "--data-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Business ROI Summary" in result.stdout
    assert "Total Executions Tracked" in result.stdout
