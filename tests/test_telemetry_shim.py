"""Telemetry shim tests: schema, upserts, queries, JSONL ingestion."""

from __future__ import annotations

import io
import json

from sqlalchemy import inspect

from gentle_ai_model_router.integration import telemetry_shim


def _store(tmp_path):
    store = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    store.init_schema()
    return store


def _execution(execution_id: str, phase: str = "apply", success: int = 1, tokens: int = 1000, **kw):
    payload = {
        "execution_id": execution_id,
        "session_id": "sess-1",
        "project_id": "proj-1",
        "phase": phase,
        "model": "anthropic/claude-sonnet-4",
        "deployment": "default",
        "effort": "low",
        "input_tokens": tokens,
        "output_tokens": tokens // 4,
        "total_tokens": tokens + tokens // 4,
        "tool_calls": 3,
        "tool_errors": 0,
        "task_success": success,
        "router_version": "0.0.1",
    }
    payload.update(kw)
    return payload


def test_schema_creation(tmp_path) -> None:
    store = _store(tmp_path)
    tables = set(inspect(store.engine).get_table_names())
    assert {"executions", "decisions"} <= tables


def test_insert_and_tokens_per_success(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        store.record_execution(session, _execution("e1", "apply", success=1, tokens=1000))
        store.record_execution(session, _execution("e2", "apply", success=1, tokens=500))
        store.record_execution(session, _execution("e3", "apply", success=0, tokens=2000))
        store.record_execution(session, _execution("e4", "verify", success=1, tokens=800))
        rows = {row["phase"]: row for row in store.tokens_per_success(session)}
    # total_tokens = tokens + tokens//4 (output share) per _execution.
    assert rows["apply"]["total_tokens"] == 1250 + 625 + 2500
    assert rows["apply"]["successes"] == 2
    assert rows["apply"]["tokens_per_success"] == rows["apply"]["total_tokens"] / 2
    assert rows["verify"]["tokens_per_success"] == 1000
    # Filtered by phase.
    with store.session() as session:
        only = store.tokens_per_success(session, phase="verify")
    assert [row["phase"] for row in only] == ["verify"]


def test_tokens_per_success_zero_successes_is_none(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        store.record_execution(session, _execution("e1", "explore", success=0, tokens=100))
        rows = store.tokens_per_success(session, phase="explore")
    assert rows[0]["successes"] == 0
    assert rows[0]["tokens_per_success"] is None


def test_record_decision_and_link(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        decision = store.record_decision(
            session,
            phase="apply",
            selected={"model": "anthropic/claude-sonnet-4", "effort": "low"},
            alternatives=[],
            reason_codes=["cheapest_of_meeting"],
            estimated_tokens=14000.0,
            estimated_cost=0.05,
            policy_version="pol-deadbeef",
        )
        _, created = store.record_execution(
            session, _execution("e1", decision_id=decision.decision_id)
        )
        assert created
    with store.session() as session:
        rows = store.tokens_per_success(session, phase="apply")
    assert rows[0]["successes"] == 1


def test_jsonl_ingestion_two_lines(tmp_path) -> None:
    store = _store(tmp_path)
    lines = io.StringIO(
        "\n".join(
            [
                json.dumps(_execution("e1", "apply", tokens=100)),
                json.dumps(_execution("e2", "verify", tokens=200)),
            ]
        )
        + "\n"
    )
    with store.session() as session:
        counts = store.ingest_jsonl(session, lines)
    assert counts == {"inserted": 2, "updated": 0, "skipped": 0}
    with store.session() as session:
        rows = {row["phase"]: row for row in store.tokens_per_success(session)}
    assert set(rows) == {"apply", "verify"}


def test_jsonl_skips_malformed_lines(tmp_path) -> None:
    store = _store(tmp_path)
    lines = io.StringIO(
        '{"execution_id": "e1", "phase": "apply", "total_tokens": 10}\nnot json\n{}\n'
    )
    with store.session() as session:
        counts = store.ingest_jsonl(session, lines)
    assert counts == {"inserted": 1, "updated": 0, "skipped": 2}


def test_duplicate_execution_id_upserts(tmp_path) -> None:
    store = _store(tmp_path)
    with store.session() as session:
        _, created = store.record_execution(session, _execution("e1", tokens=100))
        assert created
        _, created = store.record_execution(
            session, _execution("e1", tokens=999, task_success=0)
        )
        assert not created
        rows = store.tokens_per_success(session, phase="apply")
    assert rows[0]["total_tokens"] == 999 + 999 // 4
    assert rows[0]["successes"] == 0

def test_phase_signals_covers_all_phases() -> None:
    from gentle_ai_model_router.router.decision import CANONICAL_PHASES

    assert set(telemetry_shim.PHASE_SIGNALS) == set(CANONICAL_PHASES)
    assert "tests_passed" in telemetry_shim.PHASE_SIGNALS["apply"]
    assert "findings_count" in telemetry_shim.PHASE_SIGNALS["explore"]
