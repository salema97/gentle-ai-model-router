"""Unit tests for the empirical routing benchmarks collector and registry normalization."""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import respx
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.collector.routing_benchmarks import (
    TASK_TO_PHASE,
    RoutingBenchmarksCollector,
    map_task_to_phase,
)
from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.models import Model, ModelBenchmark, ModelSnapshot
from gentle_ai_model_router.registry.normalize import (
    NormalizedBenchmarkSample,
    canonicalize_model_id,
    normalize_routing_benchmarks,
)
from gentle_ai_model_router.router.config import RoutingBenchmarksConfig

NOW = datetime(2026, 9, 19, 2, 0, tzinfo=UTC)

DARS_URL = (
    "https://huggingface.co/datasets/AIGNLAI/DARS/resolve/main/drop-800/"
    "train_scored_generations.jsonl"
)
XRB_URL = (
    "https://huggingface.co/datasets/ulab-ai/xRouteBench/resolve/main/"
    "llmrouter_generic/train.parquet"
)
COMP_URL = (
    "https://huggingface.co/datasets/Wikit/RoutingCompendium-perf/resolve/main/"
    "data/RouterBench-00000-of-00001.parquet"
)

DARS_JSONL = (
    json.dumps({
        "query_id": "q1",
        "input_question": "What happened in 1999?",
        "model": "openai/gpt-4",
        "score": 0.9,
        "cost": 0.005,
        "prompt_tokens": 120,
        "completion_tokens": 40,
        "task_name": "drop-800",
    })
    + "\n"
    + json.dumps({
        "query_id": "q2",
        "input_question": "Who scored the goal?",
        "model": "anthropic/claude-3-opus",
        "score": 0.85,
        "cost": 0.008,
        "prompt_tokens": 150,
        "completion_tokens": 50,
        "task_name": "drop-800",
    })
    + "\n"
)

XRB_DICT = {
    "task_name": ["mbpp", "gsm8k"],
    "query": ["def is_even(n):", "Calculate 15 * 3"],
    "model_name": ["openai/gpt-4", "meta-llama/Llama-2-7b-chat-hf"],
    "performance": [0.92, 0.88],
    "input_tokens": [100, 60],
    "output_tokens": [50, 25],
    "response_time": [1.2, 0.6],
}

COMP_DICT = {
    "prompt": ["Solve x^2 = 4", "Write a binary search function"],
    "dataset": ["math", "humaneval"],
    "models_name": [
        ["openai/gpt-4", "meta/llama-3"],
        ["openai/gpt-4", "anthropic/claude-3-opus"],
    ],
    "models_performance": [
        [0.89, 0.75],
        [0.91, 0.95],
    ],
}


def _make_parquet_bytes(data: dict) -> bytes:
    table = pa.Table.from_pydict(data)
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@respx.mock
def test_routing_benchmarks_collector_success(store: SnapshotStore) -> None:
    respx.get(DARS_URL).mock(return_value=httpx.Response(200, text=DARS_JSONL))
    respx.get(XRB_URL).mock(
        return_value=httpx.Response(200, content=_make_parquet_bytes(XRB_DICT))
    )
    respx.get(COMP_URL).mock(
        return_value=httpx.Response(200, content=_make_parquet_bytes(COMP_DICT))
    )

    config = RoutingBenchmarksConfig()
    with RoutingBenchmarksCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.source == "routing-benchmarks"
    # 2 DARS + 2 xRouteBench + 2 Compendium prompts (record_count = 6)
    assert record.record_count == 6
    assert record.errors == []
    assert record.path.is_file()

    doc = json.loads(record.path.read_text(encoding="utf-8"))
    assert doc["source"] == "routing-benchmarks"
    data = doc["data"]
    assert data["source"] == "huggingface"

    # Verify DARS extraction
    assert len(data["dars"]) == 2
    d0 = data["dars"][0]
    assert d0["query_id"] == "q1"
    assert d0["prompt"] == "What happened in 1999?"
    assert d0["model"] == "openai/gpt-4"
    assert d0["quality"] == pytest.approx(0.9)
    assert d0["cost"] == pytest.approx(0.005)
    assert d0["prompt_tokens"] == 120
    assert d0["completion_tokens"] == 40
    assert d0["mapped_phase"] == "explore"

    # Verify xRouteBench extraction
    assert len(data["xroutebench"]) == 2
    x0 = data["xroutebench"][0]
    assert x0["task_name"] == "mbpp"
    assert x0["model_name"] == "openai/gpt-4"
    assert x0["performance"] == pytest.approx(0.92)
    assert x0["input_tokens"] == 100
    assert x0["output_tokens"] == 50
    assert x0["response_time"] == pytest.approx(1.2)
    assert x0["mapped_phase"] == "apply"

    # Verify Compendium extraction
    assert len(data["compendium"]) == 2
    c0 = data["compendium"][0]
    assert c0["dataset"] == "math"
    assert c0["models_name"] == ["openai/gpt-4", "meta/llama-3"]
    assert c0["models_performance"] == [pytest.approx(0.89), pytest.approx(0.75)]
    assert c0["mapped_phase"] == "design"


@respx.mock
def test_routing_benchmarks_partial_and_total_failure(store: SnapshotStore) -> None:
    # Partial failure: DARS 500, xRouteBench network error, Compendium succeeds
    respx.get(DARS_URL).mock(return_value=httpx.Response(500, text="Internal Server Error"))
    respx.get(XRB_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
    respx.get(COMP_URL).mock(
        return_value=httpx.Response(200, content=_make_parquet_bytes(COMP_DICT))
    )

    config = RoutingBenchmarksConfig()
    with RoutingBenchmarksCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.record_count == 2
    assert len(record.errors) == 2
    assert any("DARS" in err for err in record.errors)
    assert any("xRouteBench" in err for err in record.errors)

    doc = json.loads(record.path.read_text(encoding="utf-8"))
    assert doc["data"]["dars"] == []
    assert doc["data"]["xroutebench"] == []
    assert len(doc["data"]["compendium"]) == 2

    # Total failure: all fail
    respx.get(COMP_URL).mock(return_value=httpx.Response(404, text="Not Found"))
    with RoutingBenchmarksCollector(config, store) as collector:
        record_failed = collector.collect(now=NOW)

    assert record_failed.record_count == 0
    assert len(record_failed.errors) == 3


def test_task_to_phase_mapping() -> None:
    expected = {
        "mbpp": "apply",
        "humaneval": "apply",
        "code_eval": "apply",
        "gsm8k": "design",
        "math": "design",
        "gpqa": "design",
        "drop": "explore",
        "drop-800": "explore",
        "reading_comprehension": "explore",
        "abstract2title": "propose",
        "r2bench": "spec",
        "sprout": "tasks",
        "fusionbench": "verify",
    }
    for task, phase in expected.items():
        assert TASK_TO_PHASE[task] == phase
        assert map_task_to_phase(task) == phase
        assert map_task_to_phase(task.upper()) == phase

    # Substring match and fallback
    assert map_task_to_phase("gsm8k_cot") == "design"
    assert map_task_to_phase("nonexistent_benchmark") == "explore"


def test_canonicalize_model_id() -> None:
    assert canonicalize_model_id("openai/gpt-4") == "openai/gpt-4"
    assert canonicalize_model_id("openai//gpt-4") == "openai/gpt-4"
    assert canonicalize_model_id(" / openai / gpt-4 / ") == "openai/gpt-4"
    assert canonicalize_model_id("meta-llama--Llama-2-7b") == "meta-llama/Llama-2-7b"
    assert canonicalize_model_id("gpt-4o") == "openai/gpt-4o"
    assert canonicalize_model_id("claude-3-5-sonnet") == "anthropic/claude-3-5-sonnet"
    assert canonicalize_model_id("custom-model") == "unknown/custom-model"
    assert canonicalize_model_id("") == "unknown/unknown"


def test_normalize_routing_benchmarks_metrics() -> None:
    payload = {
        "dars": [
            {
                "model": "openai/gpt-4",
                "task_name": "drop-800",
                "score": 0.88,
                "cost": 0.004,
                "prompt_tokens": 100,
                "completion_tokens": 40,
            }
        ],
        "xroutebench": [
            {
                "model_name": "meta-llama/Llama-3-8b",
                "task_name": "gsm8k",
                "performance": 0.82,
                "response_time": 0.45,
                "input_tokens": 80,
                "output_tokens": 30,
            }
        ],
        "compendium": [
            {
                "dataset": "humaneval",
                "models_name": ["gpt-4", "claude-3-opus"],
                "models_performance": [0.90, 0.85],
            }
        ],
    }

    samples = normalize_routing_benchmarks(payload)
    # 1 DARS + 1 xRouteBench + 2 Compendium models = 4 normalized samples
    assert len(samples) == 4

    # Sample 0: DARS
    s0 = samples[0]
    assert isinstance(s0, NormalizedBenchmarkSample)
    assert s0.canonical_id == "openai/gpt-4"
    assert s0.task_name == "drop-800"
    assert s0.mapped_phase == "explore"
    assert s0.quality == pytest.approx(0.88)
    assert s0.cost == pytest.approx(0.004)
    assert s0.input_tokens == 100
    assert s0.output_tokens == 40
    assert s0.provenance == "dars"

    # Sample 1: xRouteBench
    s1 = samples[1]
    assert s1.canonical_id == "meta-llama/Llama-3-8b"
    assert s1.task_name == "gsm8k"
    assert s1.mapped_phase == "design"
    assert s1.quality == pytest.approx(0.82)
    assert s1.latency == pytest.approx(0.45)
    assert s1.input_tokens == 80
    assert s1.output_tokens == 30
    assert s1.provenance == "xroutebench"

    # Sample 2 & 3: Compendium unrolled models
    s2 = samples[2]
    assert s2.canonical_id == "openai/gpt-4"
    assert s2.task_name == "humaneval"
    assert s2.mapped_phase == "apply"
    assert s2.quality == pytest.approx(0.90)
    assert s2.provenance == "compendium"

    s3 = samples[3]
    assert s3.canonical_id == "anthropic/claude-3-opus"
    assert s3.task_name == "humaneval"
    assert s3.mapped_phase == "apply"
    assert s3.quality == pytest.approx(0.85)
    assert s3.provenance == "compendium"


def test_apply_routing_benchmarks_snapshot(tmp_path: Path) -> None:
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'benchmarks_test.db'}")
    registry_db.init_schema(engine)

    snapshot_doc = {
        "snapshot_id": "2026-09-19-routing-benchmarks",
        "fetched_at": "2026-09-19T02:00:00+00:00",
        "data": {
            "dars": [
                {
                    "model": "openai/gpt-4",
                    "task_name": "drop-800",
                    "score": 0.80,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
                {
                    # Another drop sample for gpt-4 to test quality averaging
                    "model": "openai/gpt-4",
                    "task_name": "drop-800",
                    "score": 0.90,
                    "prompt_tokens": 120,
                    "completion_tokens": 60,
                },
            ],
            "xroutebench": [
                {
                    "model_name": "openai/gpt-4",
                    "task_name": "gsm8k",
                    "performance": 0.95,
                    "response_time": 1.2,
                },
                {
                    "model_name": "meta-llama/Llama-3-8b",
                    "task_name": "mbpp",
                    "performance": 0.70,
                    "response_time": 0.5,
                },
            ],
            "compendium": [
                {
                    "dataset": "drop-800",
                    "models_name": ["openai/gpt-4"],
                    "models_performance": [0.85],
                }
            ],
        },
        "meta": {"record_count": 5},
    }

    with Session(engine) as session:
        counts = registry_db.apply_routing_benchmarks_snapshot(session, snapshot_doc)
        session.commit()

    # Two models: openai/gpt-4 and meta-llama/Llama-3-8b
    assert counts["models"] == 2
    # Benchmarks:
    # 1. gpt-4 in explore (quality avg: (0.80 + 0.90 + 0.85)/3 = 0.85) -> 1
    # 2. gpt-4 in design (quality 0.95 + latency 1.2) -> 2
    # 3. Llama-3-8b in apply (quality 0.70 + latency 0.5) -> 2
    # Total benchmarks = 5
    assert counts["benchmarks"] == 5

    with Session(engine) as session:
        # Check models
        gpt4 = session.query(Model).filter_by(canonical_id="openai/gpt-4").one()
        assert gpt4.org == "openai"
        assert gpt4.name == "gpt-4"

        llama = session.query(Model).filter_by(canonical_id="meta-llama/Llama-3-8b").one()
        assert llama.org == "meta-llama"
        assert llama.name == "Llama-3-8b"

        # Check empirical_quality for gpt-4 in explore
        b_explore = (
            session.query(ModelBenchmark)
            .filter_by(
                model_id=gpt4.id,
                benchmark="empirical_quality",
                category="explore",
            )
            .one()
        )
        assert b_explore.score == pytest.approx(0.85)

        # Check empirical_latency for gpt-4 in design
        b_lat = (
            session.query(ModelBenchmark)
            .filter_by(
                model_id=gpt4.id,
                benchmark="empirical_latency",
                category="design",
            )
            .one()
        )
        assert b_lat.score == pytest.approx(1.2)

        # Check snapshot record
        snap = (
            session.query(ModelSnapshot)
            .filter_by(snapshot_id="2026-09-19-routing-benchmarks")
            .one()
        )
        assert snap.source == "routing-benchmarks"


@respx.mock
def test_cli_collect_and_normalize_benchmarks(tmp_path: Path) -> None:
    respx.get(DARS_URL).mock(return_value=httpx.Response(200, text=DARS_JSONL))
    respx.get(XRB_URL).mock(
        return_value=httpx.Response(200, content=_make_parquet_bytes(XRB_DICT))
    )
    respx.get(COMP_URL).mock(
        return_value=httpx.Response(200, content=_make_parquet_bytes(COMP_DICT))
    )

    runner = CliRunner()
    data_dir = str(tmp_path / "data")

    # 1. Collect
    result_collect = runner.invoke(
        app, ["collect", "--source", "benchmarks", "--data-dir", data_dir]
    )
    assert result_collect.exit_code == 0
    assert "routing-benchmarks" in result_collect.output

    # 2. Normalize
    result_norm = runner.invoke(
        app, ["normalize", "--source", "benchmarks", "--data-dir", data_dir]
    )
    assert result_norm.exit_code == 0
    assert "routing-benchmarks" in result_norm.output
