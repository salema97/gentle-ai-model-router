"""Phase 3a API tests: FastAPI server over a synthetic registry.

No network: a tmp SQLite registry is seeded through the real registry db
helpers, and the app is exercised through fastapi's TestClient.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gentle_ai_model_router.api.server import create_app
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import RouterConfig, load_config
from gentle_ai_model_router.router.decision import CANONICAL_PHASES

AA = "artificial_analysis_intelligence_index"


def _seed(session: Session) -> None:
    """Two models with distinct priors/prices and full effort ladders."""
    rows = [
        # canonical, aa score, (in, out) price, context, tool_calling
        ("test/cheap-1", 30.0, (1.0, 2.0), 128_000, True),
        ("test/strong", 90.0, (8.0, 30.0), 256_000, True),
        ("test/no-tools", 50.0, (2.0, 6.0), 128_000, False),
    ]
    for canonical, aa, (in_p, out_p), ctx, tools in rows:
        provider = registry_db.get_or_create_provider(session, "test")
        model = registry_db.upsert_model(
            session,
            canonical_id=canonical,
            context_window=ctx,
            tool_calling=tools,
        )
        deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
        for effort in ("low", "medium", "high"):
            registry_db.upsert_variant(session, deployment, effort, effort)
        registry_db.upsert_benchmark(session, model, AA, aa, None, "snap-test")
        registry_db.upsert_price(session, deployment, "snap-test", in_p, out_p, None)


@pytest.fixture
def config(tmp_path: Path) -> RouterConfig:
    return load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")


@pytest.fixture
def engine(tmp_path: Path):
    eng = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(eng)
    with registry_db.Session(eng) as session:
        _seed(session)
        session.commit()
    return eng


@pytest.fixture
def client(config: RouterConfig, engine, tmp_path: Path) -> TestClient:
    shim = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    shim.init_schema()
    return TestClient(create_app(config, engine, shim))


def _post_route(client: TestClient, **overrides):
    body = {"task": "summarize this module", "phase": "explore"}
    body.update(overrides)
    return client.post("/route", json=body)


def test_route_happy_path(client: TestClient) -> None:
    response = _post_route(client)
    assert response.status_code == 200, response.text
    data = response.json()
    # All models meet the explore floor at "low" (equal tokens) — the
    # minimum-sufficient-effort policy then tie-breaks by quality, so the
    # strongest prior wins at the cheapest sufficient effort level.
    assert data["model"] == "test/strong"
    assert data["deployment"] == "default"
    assert data["effort"] == "low"
    assert data["alternatives"], "expected runner-up alternatives"
    assert {a["model"] for a in data["alternatives"]} == {"test/cheap-1", "test/no-tools"}
    assert data["reason_codes"], "reason_codes must answer 'why this model?'"
    assert any(r.startswith("meets_threshold:") for r in data["reason_codes"])
    assert "cheapest_of_meeting" in data["reason_codes"]
    assert data["estimated_tokens"] > 0
    assert data["estimated_cost"] > 0
    assert data["policy_version"].startswith("pol-")
    assert data["registry_hash"].startswith("reg-")


def test_route_is_deterministic(client: TestClient) -> None:
    """Identical request body + same registry + same config ⇒ identical bytes."""
    first = _post_route(client, context_tokens=42_000)
    second = _post_route(client, context_tokens=42_000)
    assert first.status_code == 200 and second.status_code == 200
    assert first.content == second.content


def test_route_hard_filter_available_models(client: TestClient) -> None:
    response = _post_route(client, available_models=["test/strong"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model"] == "test/strong"
    assert all(a["model"] == "test/strong" for a in data["alternatives"])


def test_route_hard_filter_available_efforts(client: TestClient) -> None:
    response = _post_route(client, available_efforts=["high"])
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["effort"] == "high"
    assert all(a["effort"] == "high" for a in data["alternatives"])


def test_route_invalid_effort_is_422(client: TestClient) -> None:
    response = _post_route(client, available_efforts=["turbo"])
    assert response.status_code == 422
    assert "turbo" in response.text


def test_route_unknown_phase_is_422(client: TestClient) -> None:
    response = _post_route(client, phase="sdd-bogus")
    assert response.status_code == 422
    assert "unknown phase" in response.text


def test_route_empty_registry_is_503(config: RouterConfig, tmp_path: Path) -> None:
    empty = registry_db.get_engine(f"sqlite:///{tmp_path / 'empty.db'}")
    registry_db.init_schema(empty)
    client = TestClient(create_app(config, empty))
    response = _post_route(client)
    assert response.status_code == 503
    assert "registry is empty" in response.text


def test_route_filter_excluding_everything_is_503(client: TestClient) -> None:
    response = _post_route(client, available_models=["test/does-not-exist"])
    assert response.status_code == 503


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["registry_hash"].startswith("reg-")
    assert data["policy_version"].startswith("pol-")


def test_policy_covers_all_canonical_phases(client: TestClient) -> None:
    response = client.get("/policy")
    assert response.status_code == 200
    data = response.json()
    assert set(data["phases"]) == set(CANONICAL_PHASES)
    explore = data["phases"]["explore"]
    assert explore["selected"]["model"] == "test/strong"
    assert explore["alternatives"]
    # apply/verify hard-require tool calling: the no-tools model is excluded.
    apply_phase = data["phases"]["apply"]
    assert apply_phase["selected"]["model"] != "test/no-tools"


def test_policy_cache_invalidates_on_registry_change(
    config: RouterConfig, engine, tmp_path: Path
) -> None:
    client = TestClient(create_app(config, engine))
    before = client.get("/policy").json()["registry_hash"]
    with registry_db.Session(engine) as session:
        provider = registry_db.get_or_create_provider(session, "test")
        model = registry_db.upsert_model(
            session, canonical_id="test/brand-new", tool_calling=True
        )
        deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
        registry_db.upsert_variant(session, deployment, "low", "low")
        registry_db.upsert_benchmark(session, model, AA, 99.0, None, "snap-test")
        session.commit()
    after = client.get("/policy").json()
    assert after["registry_hash"] != before
    # The new model has the best benchmark prior and ties on tokens at "low",
    # so it wins the quality tie-break and becomes the selection.
    assert after["phases"]["explore"]["selected"]["model"] == "test/brand-new"


def test_decision_logged_to_shim(
    config: RouterConfig, engine, tmp_path: Path
) -> None:
    shim_path = tmp_path / "telemetry.sqlite"
    shim = telemetry_shim.ShimStore(f"sqlite:///{shim_path}")
    shim.init_schema()
    client = TestClient(create_app(config, engine, shim))
    _post_route(client)
    _post_route(client, phase="design")
    with shim.session() as session:
        rows = session.query(telemetry_shim.DecisionRecord).all()
        assert len(rows) == 2
        by_phase = {row.phase: row for row in rows}
        assert by_phase["explore"].selected["model"] == "test/strong"
        assert by_phase["explore"].reason_codes
        assert by_phase["design"].estimated_tokens > 0


def test_route_without_shim_logs_only(config: RouterConfig, engine) -> None:
    client = TestClient(create_app(config, engine))  # no shim_store
    response = _post_route(client)
    assert response.status_code == 200
