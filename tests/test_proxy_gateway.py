"""Tests for OpenAI-Compatible Gateway Proxy."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from gentle_ai_model_router.api.server import create_app
from gentle_ai_model_router.gateway.proxy import (
    extract_task_and_phase,
    list_gateway_models,
    resolve_upstream,
)
from gentle_ai_model_router.gateway.schemas import ChatMessage
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.integration.telemetry_shim import DecisionRecord, ExecutionRecord
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import RouterConfig, load_config
from gentle_ai_model_router.router.decision import Alternative, Decision

AA = "artificial_analysis_intelligence_index"


def _seed(session: Session) -> None:
    """Seed test models with distinct priors and effort ladders."""
    rows = [
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
    cfg = load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")
    cfg.gateway.upstream_base_url = "https://api.kimi.ai/coding/v1"
    cfg.gateway.upstream_api_key = "test-gateway-key"
    return cfg


@pytest.fixture
def engine(tmp_path: Path):
    eng = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(eng)
    with registry_db.Session(eng) as session:
        _seed(session)
        session.commit()
    return eng


@pytest.fixture
def shim_store(tmp_path: Path) -> telemetry_shim.ShimStore:
    shim = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    shim.init_schema()
    return shim


@pytest.fixture
def client(config: RouterConfig, engine, shim_store: telemetry_shim.ShimStore) -> TestClient:
    return TestClient(create_app(config, engine, shim_store))


def _make_dummy_decision(model: str = "test/strong", effort: str = "medium") -> Decision:
    return Decision(
        phase="explore",
        model=model,
        provider="test",
        deployment="default",
        effort=effort,
        score=0.85,
        quality=0.88,
        alternatives=(
            Alternative(
                model="test/cheap-1",
                provider="test",
                deployment="default",
                effort="low",
                score=0.75,
                quality=0.7,
                estimated_tokens=1000.0,
            ),
        ),
        reason_codes=("cheapest_of_meeting",),
        estimated_tokens=1000.0,
        estimated_cost=0.005,
        policy_version="1.0.0",
        confidence=0.9,
        probabilities={"test/strong": 0.9},
        system_one=None,
    )


# ====================================================================== #
# 1. Phase extraction tests
# ====================================================================== #


def test_phase_extraction_default() -> None:
    msgs = [ChatMessage(role="user", content="Hello, how does this work?")]
    task, phase, tokens = extract_task_and_phase(msgs)
    assert task == "Hello, how does this work?"
    assert phase == "explore"
    assert tokens > 0


def test_phase_extraction_sdd_markers() -> None:
    cases = [
        ("Run sdd-apply on module A", "apply"),
        ("Execute sdd-spec step", "spec"),
        ("Let's do sdd-design for this component", "design"),
        ("Now sdd-tasks breakdown", "tasks"),
        ("Run sdd-verify on tests", "verify"),
        ("Let's do sdd-propose", "propose"),
        ("Just explore around sdd-explore", "explore"),
    ]
    for prompt, expected_phase in cases:
        msgs = [ChatMessage(role="user", content=prompt)]
        task, phase, _ = extract_task_and_phase(msgs)
        assert phase == expected_phase
        assert task == prompt


def test_phase_extraction_named_message_and_multimodal() -> None:
    # Named message
    msgs1 = [ChatMessage(role="user", name="sdd-apply", content="Fix the function")]
    _, phase1, _ = extract_task_and_phase(msgs1)
    assert phase1 == "apply"

    # Multimodal / content list
    msgs2 = [
        ChatMessage(
            role="user",
            content=[{"type": "text", "text": "Check this sdd-verify requirement"}],
        )
    ]
    task2, phase2, tokens2 = extract_task_and_phase(msgs2)
    assert phase2 == "verify"
    assert "sdd-verify" in task2
    assert tokens2 > 0


# ====================================================================== #
# 2. Upstream resolution tests
# ====================================================================== #


def test_resolve_upstream_kimi(config: RouterConfig) -> None:
    config.gateway.upstream_base_url = "https://api.kimi.ai/coding/v1"
    config.gateway.upstream_api_key = "kimi-secret"

    # low effort -> kimi-for-coding-highspeed
    dec_low = _make_dummy_decision(effort="low")
    target_url, target_model, api_key = resolve_upstream(dec_low, "auto", config)
    assert target_url == "https://api.kimi.ai/coding/v1/chat/completions"
    assert target_model == "kimi-for-coding-highspeed"
    assert api_key == "kimi-secret"

    # high effort -> kimi-for-coding
    dec_high = _make_dummy_decision(effort="high")
    _, target_model_high, _ = resolve_upstream(dec_high, "gentle-router/auto", config)
    assert target_model_high == "kimi-for-coding"


def test_resolve_upstream_openrouter(config: RouterConfig) -> None:
    config.gateway.upstream_base_url = "https://openrouter.ai/api/v1"
    config.gateway.upstream_api_key = "openrouter-secret"

    dec = _make_dummy_decision(model="anthropic/claude-3.5-sonnet")
    target_url, target_model, api_key = resolve_upstream(dec, "auto", config)
    assert target_url == "https://openrouter.ai/api/v1/chat/completions"
    assert target_model == "anthropic/claude-3.5-sonnet"
    assert api_key == "openrouter-secret"


def test_resolve_upstream_specific_model_and_auth_header(config: RouterConfig) -> None:
    config.gateway.upstream_base_url = "https://api.kimi.ai/coding/v1"
    config.gateway.upstream_api_key = None  # test fallback to auth_header

    dec = _make_dummy_decision()
    target_url, target_model, api_key = resolve_upstream(
        dec,
        requested_model="custom/my-fine-tuned-model",
        config=config,
        auth_header="Bearer client-sent-token",
    )
    assert target_model == "custom/my-fine-tuned-model"
    assert api_key == "client-sent-token"


# ====================================================================== #
# 3. Model listing tests
# ====================================================================== #


def test_list_models(engine) -> None:
    models_resp = list_gateway_models(engine)
    ids = [m.id for m in models_resp.data]
    assert "auto" in ids
    assert "gentle-router/auto" in ids
    assert "test/cheap-1" in ids
    assert "test/strong" in ids


def test_get_models_http_endpoints(client: TestClient) -> None:
    for endpoint in ("/v1/models", "/models"):
        res = client.get(endpoint)
        assert res.status_code == 200
        data = res.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "auto" in ids
        assert "gentle-router/auto" in ids
        assert "test/cheap-1" in ids


# ====================================================================== #
# 4. Non-streaming completions tests
# ====================================================================== #


@respx.mock
def test_non_streaming_chat_completion(
    client: TestClient, shim_store: telemetry_shim.ShimStore
) -> None:
    mock_upstream_response = {
        "id": "chatcmpl-test-abc",
        "object": "chat.completion",
        "created": 1726790400,
        "model": "kimi-for-coding",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Here is the code solution."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 58,
            "total_tokens": 100,
            "prompt_tokens_details": {"cached_tokens": 10},
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }

    route = respx.post("https://api.kimi.ai/coding/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=mock_upstream_response)
    )

    req_body = {
        "model": "auto",
        "messages": [
            {"role": "user", "content": "Write quicksort in python sdd-apply"},
        ],
        "stream": False,
        "temperature": 0.2,
    }

    res = client.post("/v1/chat/completions", json=req_body)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["id"] == "chatcmpl-test-abc"
    assert data["choices"][0]["message"]["content"] == "Here is the code solution."
    assert route.called

    # Check forward payload sent to upstream
    sent_request = route.calls.last.request
    import json
    sent_payload = json.loads(sent_request.content)
    assert sent_payload["model"] in ("kimi-for-coding", "kimi-for-coding-highspeed")
    assert sent_payload["temperature"] == 0.2
    assert sent_payload["stream"] is False

    # Verify telemetry in shim_store
    with shim_store.session() as s:
        execs = s.scalars(
            select(ExecutionRecord).where(ExecutionRecord.execution_id == "chatcmpl-test-abc")
        ).all()
        assert len(execs) == 1
        record = execs[0]
        assert record.phase == "apply"
        assert record.input_tokens == 42
        assert record.output_tokens == 58
        assert record.total_tokens == 100
        assert record.cached_tokens == 10
        assert record.reasoning_tokens == 12
        assert record.task_success == 1
        assert record.latency_ms is not None and record.latency_ms >= 0
        assert record.decision_id is not None

        # Verify decision record was also written
        decs = s.scalars(
            select(DecisionRecord).where(DecisionRecord.decision_id == record.decision_id)
        ).all()
        assert len(decs) == 1
        assert decs[0].phase == "apply"


# ====================================================================== #
# 5. Streaming completions tests
# ====================================================================== #


@respx.mock
def test_streaming_chat_completion(
    client: TestClient, shim_store: telemetry_shim.ShimStore
) -> None:
    sse_body = (
        b"data: {\"id\":\"chatcmpl-stream-999\","
        b"\"choices\":[{\"delta\":{\"content\":\"Hello \"}}]}\n\n"
        b"data: {\"choices\":[{\"delta\":{\"content\":\"world!\"}}]}\n\n"
        b"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":18,\"total_tokens\":30}}\n\n"
        b"data: [DONE]\n\n"
    )

    route = respx.post("https://api.kimi.ai/coding/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_body,
        )
    )

    req_body = {
        "model": "gentle-router/auto",
        "messages": [
            {"role": "user", "content": "Stream a greeting sdd-tasks"},
        ],
        "stream": True,
    }

    res = client.post("/chat/completions", json=req_body)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    assert "Hello " in res.text
    assert "world!" in res.text
    assert "[DONE]" in res.text
    assert route.called

    # Verify telemetry in shim_store
    with shim_store.session() as s:
        execs = s.scalars(
            select(ExecutionRecord).where(ExecutionRecord.execution_id == "chatcmpl-stream-999")
        ).all()
        assert len(execs) == 1
        record = execs[0]
        assert record.phase == "tasks"
        assert record.input_tokens == 12
        assert record.output_tokens == 18
        assert record.total_tokens == 30
        assert record.task_success == 1
        assert record.latency_ms is not None and record.latency_ms >= 0


# ====================================================================== #
# 6. Error handling tests
# ====================================================================== #


@respx.mock
def test_upstream_rate_limit_error(
    client: TestClient, shim_store: telemetry_shim.ShimStore
) -> None:
    error_payload = {
        "error": {
            "message": "Rate limit reached for requests",
            "type": "rate_limit_error",
            "code": 429,
        }
    }
    route = respx.post("https://api.kimi.ai/coding/v1/chat/completions").mock(
        return_value=httpx.Response(429, json=error_payload)
    )

    res = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Trigger 429"}]},
    )
    assert res.status_code == 429
    assert "rate_limit_error" in res.text
    assert route.called

    # Telemetry should record failure (task_success = 0)
    with shim_store.session() as s:
        execs = s.scalars(select(ExecutionRecord)).all()
        assert len(execs) == 1
        assert execs[0].task_success == 0


@respx.mock
def test_upstream_500_error(
    client: TestClient, shim_store: telemetry_shim.ShimStore
) -> None:
    route = respx.post("https://api.kimi.ai/coding/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    res = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Trigger 500"}]},
    )
    assert res.status_code == 500
    assert "Internal Server Error" in res.text
    assert route.called

    with shim_store.session() as s:
        execs = s.scalars(select(ExecutionRecord)).all()
        assert len(execs) == 1
        assert execs[0].task_success == 0
