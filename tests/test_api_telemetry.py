"""Tests for the telemetry shim HTTP ingestion endpoints (/shim/*).

Validates:
- POST /shim/execution (single ingest, rubric auto-scoring, idempotency)
- POST /shim/executions (batch ingestion in single transaction)
- POST /shim/feedback (forced outcome re-evaluation)
- Error paths: 503 when shim is unconfigured, 422 on invalid schemas/signals.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from gentle_ai_model_router.api.server import create_app
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import RouterConfig, load_config


@pytest.fixture
def config(tmp_path: Path) -> RouterConfig:
    return load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")


@pytest.fixture
def engine(tmp_path: Path):
    eng = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(eng)
    return eng


@pytest.fixture
def shim_store(tmp_path: Path) -> telemetry_shim.ShimStore:
    store = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    store.init_schema()
    return store


@pytest.fixture
def client(config: RouterConfig, engine, shim_store: telemetry_shim.ShimStore) -> TestClient:
    app = create_app(config, engine, shim_store=shim_store)
    return TestClient(app)


def test_shim_execution_single_creates_and_scores(
    client: TestClient, shim_store: telemetry_shim.ShimStore
) -> None:
    payload = {
        "execution_id": "exec-001",
        "phase": "apply",
        "model": "anthropic/claude-3-5-sonnet",
        "deployment": "default",
        "effort": "high",
        "total_tokens": 1500,
        "tests_passed": 10,
        "tests_failed": 0,
        "tool_errors": 0,
    }
    response = client.post("/shim/execution", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["execution_id"] == "exec-001"
    assert data["created"] is True
    # Rubric auto-scored apply phase with 10 passed, 0 failed -> success=1, quality=100.0
    assert data["task_success"] == 1
    assert data["quality_score"] == 100.0

    # Verify persisted in SQLite
    with shim_store.session() as session:
        record = session.scalar(
            select(telemetry_shim.ExecutionRecord).where(
                telemetry_shim.ExecutionRecord.execution_id == "exec-001"
            )
        )
        assert record is not None
        assert record.phase == "apply"
        assert record.total_tokens == 1500
        assert record.task_success == 1
        assert record.quality_score == 100.0


def test_shim_execution_idempotent_update(
    client: TestClient, shim_store: telemetry_shim.ShimStore
) -> None:
    payload = {
        "execution_id": "exec-dup",
        "phase": "explore",
        "total_tokens": 500,
        "tool_errors": 0,
    }
    r1 = client.post("/shim/execution", json=payload)
    assert r1.status_code == 200
    assert r1.json()["created"] is True

    # Same execution_id with updated token count
    payload["total_tokens"] = 1200
    r2 = client.post("/shim/execution", json=payload)
    assert r2.status_code == 200
    assert r2.json()["created"] is False

    with shim_store.session() as session:
        record = session.scalar(
            select(telemetry_shim.ExecutionRecord).where(
                telemetry_shim.ExecutionRecord.execution_id == "exec-dup"
            )
        )
        assert record is not None
        assert record.total_tokens == 1200


def test_shim_execution_explicit_outcomes_respected(client: TestClient) -> None:
    payload = {
        "execution_id": "exec-explicit",
        "phase": "apply",
        "tests_passed": 1,
        "tests_failed": 10,  # rubric would score this as 0
        "task_success": 1,  # explicit human override
        "quality_score": 85.0,
    }
    response = client.post("/shim/execution", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task_success"] == 1
    assert data["quality_score"] == 85.0


def test_shim_execution_no_rubric(client: TestClient) -> None:
    payload = {
        "execution_id": "exec-no-rubric",
        "phase": "apply",
        "tests_passed": 5,
        "tests_failed": 0,
    }
    response = client.post("/shim/execution?apply_rubric=false", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task_success"] is None
    assert data["quality_score"] is None


def test_shim_executions_batch(client: TestClient, shim_store: telemetry_shim.ShimStore) -> None:
    payloads = [
        {
            "execution_id": "batch-1",
            "phase": "explore",
            "total_tokens": 300,
        },
        {
            "execution_id": "batch-2",
            "phase": "spec",
            "total_tokens": 400,
        },
        {
            "execution_id": "batch-3",
            "phase": "apply",
            "total_tokens": 500,
            "tests_passed": 5,
            "tests_failed": 0,
        },
    ]
    response = client.post("/shim/executions", json=payloads)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total"] == 3
    assert data["created"] == 3
    assert data["updated"] == 0
    assert data["execution_ids"] == ["batch-1", "batch-2", "batch-3"]

    # Re-post batch with 1 existing updated and 1 new
    payloads2 = [
        {"execution_id": "batch-1", "phase": "explore", "total_tokens": 600},
        {"execution_id": "batch-4", "phase": "verify", "total_tokens": 700},
    ]
    r2 = client.post("/shim/executions", json=payloads2)
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["total"] == 2
    assert d2["created"] == 1
    assert d2["updated"] == 1


def test_shim_feedback_overrides_outcomes(client: TestClient) -> None:
    payload = {
        "execution_id": "feedback-1",
        "phase": "apply",
        "tests_passed": 0,
        "tests_failed": 5,
        "tool_errors": 2,
        "task_success": 1,  # Previous stale/incorrect value
        "quality_score": 90.0,
    }
    response = client.post("/shim/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    # Feedback forces rubric re-evaluation: failed tests + tool errors -> 0
    assert data["task_success"] == 0
    assert data["quality_score"] < 50.0


def test_shim_unconfigured_returns_503(config: RouterConfig, engine) -> None:
    app = create_app(config, engine, shim_store=None)
    no_shim_client = TestClient(app)

    payload = {"execution_id": "e1", "phase": "apply"}
    r1 = no_shim_client.post("/shim/execution", json=payload)
    assert r1.status_code == 503
    assert "telemetry shim store is not configured" in r1.json()["detail"]

    r2 = no_shim_client.post("/shim/executions", json=[payload])
    assert r2.status_code == 503

    r3 = no_shim_client.post("/shim/feedback", json=payload)
    assert r3.status_code == 503


def test_shim_invalid_payload_returns_422(client: TestClient) -> None:
    # Empty execution_id
    r1 = client.post("/shim/execution", json={"execution_id": "", "phase": "apply"})
    assert r1.status_code == 422

    # Unknown phase
    r2 = client.post("/shim/execution", json={"execution_id": "e1", "phase": "nonexistent"})
    assert r2.status_code == 422

    # Negative latency
    r3 = client.post(
        "/shim/execution",
        json={"execution_id": "e2", "phase": "apply", "latency_ms": -100},
    )
    assert r3.status_code == 422
