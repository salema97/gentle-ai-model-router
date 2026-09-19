"""Unit tests for the Gentle AI Telemetry collector."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.collector.gentle_telemetry import (
    GentleTelemetryCollector,
)
from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.router.config import GentleTelemetryConfig

BASE_URL = "https://gentlemanprogramming.com/data/datasets"
NOW = datetime(2026, 9, 19, 1, 0, tzinfo=UTC)

SAMPLE_AGENT_MODELS_CSV = """\
agent_class,agent_kind,model,rows,responses,tokens_processed,errored_rows,error_categories
explore,built_in,openai/gpt-5,100,200,50000,10,unknown=10
verify,custom,anthropic/claude-3-7-sonnet,50,50,20000,0,
empty,built_in,meta/llama-3,0,0,0,0,
all_error,built_in,test/model,10,10,5000,10,error=10
"""

SAMPLE_BY_MODEL_CSV = """\
model,rows,responses,tokens_processed
openai/gpt-5,100,200,50000
anthropic/claude-3-7-sonnet,50,50,20000
"""

SAMPLE_BY_EFFORT_CSV = """\
effort,rows,responses,tokens_processed
high,100,200,50000
medium,50,50,20000
"""


@respx.mock
def test_successful_collection(store: SnapshotStore) -> None:
    respx.get(f"{BASE_URL}/runtime-agent-models.csv").mock(
        return_value=httpx.Response(200, text=SAMPLE_AGENT_MODELS_CSV)
    )
    respx.get(f"{BASE_URL}/runtime-by-model.csv").mock(
        return_value=httpx.Response(200, text=SAMPLE_BY_MODEL_CSV)
    )
    respx.get(f"{BASE_URL}/runtime-by-effort.csv").mock(
        return_value=httpx.Response(200, text=SAMPLE_BY_EFFORT_CSV)
    )

    config = GentleTelemetryConfig(base_url=BASE_URL)
    collector = GentleTelemetryCollector(config, store)
    record = collector.collect(now=NOW)
    collector.close()

    assert record.source == "gentle-telemetry"
    # 4 rows from agent-models, 2 from by-model, 2 from by-effort
    assert record.record_count == 8
    assert record.errors == []
    assert record.path.is_file()

    doc = json.loads(record.path.read_text(encoding="utf-8"))
    assert doc["source"] == "gentle-telemetry"
    data = doc["data"]
    assert data["source_url"] == BASE_URL
    assert len(data["agent_models"]) == 4
    assert len(data["by_model"]) == 2
    assert len(data["by_effort"]) == 2

    # Verify agent_models parsing and metric derivations
    row0 = data["agent_models"][0]
    assert row0["agent_class"] == "explore"
    assert row0["model"] == "openai/gpt-5"
    assert row0["rows"] == 100
    assert row0["responses"] == 200
    assert row0["tokens_processed"] == 50000
    assert row0["errored_rows"] == 10
    assert row0["success_rate"] == pytest.approx(0.9)
    assert row0["tokens_per_response"] == pytest.approx(250.0)
    assert row0["tokens_per_success"] == pytest.approx(50000 / 90)

    # Verify edge cases in derivations (rows=0, rows=errored_rows)
    row2 = data["agent_models"][2]
    assert row2["rows"] == 0
    assert row2["success_rate"] == 0.0
    assert row2["tokens_per_response"] == 0.0
    assert row2["tokens_per_success"] == 0.0

    row3 = data["agent_models"][3]
    assert row3["rows"] == 10
    assert row3["errored_rows"] == 10
    assert row3["success_rate"] == 0.0
    assert row3["tokens_per_response"] == pytest.approx(500.0)
    assert row3["tokens_per_success"] == 0.0

    # Verify store latest lookup
    latest_doc = store.latest("gentle-telemetry")
    assert latest_doc is not None
    assert latest_doc["snapshot_id"] == record.snapshot_id


@respx.mock
def test_partial_download_handles_network_errors(store: SnapshotStore) -> None:
    respx.get(f"{BASE_URL}/runtime-agent-models.csv").mock(
        return_value=httpx.Response(200, text=SAMPLE_AGENT_MODELS_CSV)
    )
    respx.get(f"{BASE_URL}/runtime-by-model.csv").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    respx.get(f"{BASE_URL}/runtime-by-effort.csv").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    config = GentleTelemetryConfig(base_url=BASE_URL)
    with GentleTelemetryCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    # Only agent-models succeeded (4 rows)
    assert record.record_count == 4
    assert len(record.errors) == 2
    assert any("runtime-by-model.csv" in err for err in record.errors)
    assert any("runtime-by-effort.csv" in err for err in record.errors)

    doc = json.loads(record.path.read_text(encoding="utf-8"))
    data = doc["data"]
    assert len(data["agent_models"]) == 4
    assert data["by_model"] == []
    assert data["by_effort"] == []


@respx.mock
def test_all_endpoints_fail_creates_snapshot_with_errors(store: SnapshotStore) -> None:
    respx.get(f"{BASE_URL}/runtime-agent-models.csv").mock(
        return_value=httpx.Response(404, text="Not Found")
    )
    respx.get(f"{BASE_URL}/runtime-by-model.csv").mock(
        return_value=httpx.Response(404, text="Not Found")
    )
    respx.get(f"{BASE_URL}/runtime-by-effort.csv").mock(
        return_value=httpx.Response(404, text="Not Found")
    )

    config = GentleTelemetryConfig(base_url=BASE_URL)
    with GentleTelemetryCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.record_count == 0
    assert len(record.errors) == 3
    assert record.path.is_file()

    doc = json.loads(record.path.read_text(encoding="utf-8"))
    assert doc["meta"]["record_count"] == 0
    assert len(doc["meta"]["errors"]) == 3


@respx.mock
def test_cli_collect_telemetry(tmp_path: Path) -> None:
    respx.get(f"{BASE_URL}/runtime-agent-models.csv").mock(
        return_value=httpx.Response(200, text=SAMPLE_AGENT_MODELS_CSV)
    )
    respx.get(f"{BASE_URL}/runtime-by-model.csv").mock(
        return_value=httpx.Response(200, text=SAMPLE_BY_MODEL_CSV)
    )
    respx.get(f"{BASE_URL}/runtime-by-effort.csv").mock(
        return_value=httpx.Response(200, text=SAMPLE_BY_EFFORT_CSV)
    )

    runner = CliRunner()
    data_dir = str(tmp_path / "data")
    result = runner.invoke(app, ["collect", "--source", "telemetry", "--data-dir", data_dir])

    assert result.exit_code == 0
    assert "gentle-telemetry" in result.output


@respx.mock
def test_defensive_parsing_non_numeric_and_empty(store: SnapshotStore) -> None:
    bad_csv = """\
agent_class,agent_kind,model,rows,responses,tokens_processed,errored_rows
explore,built_in,test/model,not_an_int,,invalid,bad
"""
    respx.get(f"{BASE_URL}/runtime-agent-models.csv").mock(
        return_value=httpx.Response(200, text=bad_csv)
    )
    respx.get(f"{BASE_URL}/runtime-by-model.csv").mock(
        return_value=httpx.Response(200, text="")
    )
    respx.get(f"{BASE_URL}/runtime-by-effort.csv").mock(
        return_value=httpx.Response(200, text="")
    )

    config = GentleTelemetryConfig(base_url=BASE_URL)
    with GentleTelemetryCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.record_count == 1
    doc = json.loads(record.path.read_text(encoding="utf-8"))
    row = doc["data"]["agent_models"][0]
    assert row["rows"] == 0
    assert row["responses"] == 0
    assert row["tokens_processed"] == 0
    assert row["errored_rows"] == 0
    assert row["success_rate"] == 0.0
    assert row["tokens_per_response"] == 0.0
    assert row["tokens_per_success"] == 0.0

