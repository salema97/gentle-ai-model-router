"""Tests for external ground-truth dataset collectors (RouterBench, RouteLLM, SWE-Traces).

100% offline, deterministic tests verifying:
1. RouterBenchCollector: parsing, SDD phase mapping, cost/latency/correctness extraction,
   snapshot persistence, and offline fallback.
2. RouteLLMCollector: parsing, choice & noul calibration mapping, phase mapping,
   snapshot persistence, and offline fallback.
3. SWETracesCollector: parsing, SDD phase mapping (apply, verify, tasks), diffs/test extraction,
   snapshot persistence, and offline fallback.
4. CLI Integration: router collect --source routerbench | routellm | swe-traces | all.
5. Telemetry bridge & dataset builder: ingestion into DatasetV1 with ground_truth_traces provenance.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import respx
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.collector.routellm import (
    RouteLLMCollectionError,
    RouteLLMCollector,
    extract_routellm_sample,
    map_routellm_task_to_phase,
)
from gentle_ai_model_router.collector.routerbench import (
    RouterBenchCollectionError,
    RouterBenchCollector,
    extract_routerbench_sample,
    map_routerbench_task_to_phase,
)
from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.collector.swe_traces import (
    SWETracesCollectionError,
    SWETracesCollector,
    extract_swe_trace_sample,
    map_swe_trace_to_phase,
)
from gentle_ai_model_router.dataset.builder import (
    DatasetBuilderConfig,
    build_examples,
)
from gentle_ai_model_router.dataset.schema import (
    GROUND_TRUTH_PROVENANCE_ADDENDUM,
    PROVENANCE_GROUND_TRUTH,
)
from gentle_ai_model_router.dataset.telemetry_bridge import (
    extract_ground_truth_executions,
)
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.registry.models import (
    Deployment,
    Model,
    ModelBenchmark,
    ModelPrice,
    ModelSnapshot,
    ModelVariant,
    Provider,
)
from gentle_ai_model_router.router.config import (
    RouteLLMConfig,
    RouterBenchConfig,
    RouterConfig,
    SWETracesConfig,
)

NOW = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)


# =========================================================================== #
# 1. RouterBench Collector Tests
# =========================================================================== #


def test_routerbench_phase_mapping() -> None:
    """Test mapping of tasks/domains to SDD canonical phases."""
    assert map_routerbench_task_to_phase("humaneval") == "apply"
    assert map_routerbench_task_to_phase("mbpp_coding") == "apply"
    assert map_routerbench_task_to_phase("gsm8k") == "design"
    assert map_routerbench_task_to_phase("math_geometry") == "design"
    assert map_routerbench_task_to_phase("r2bench") == "spec"
    assert map_routerbench_task_to_phase("formal_logic") == "spec"
    assert map_routerbench_task_to_phase("unit_test_eval") == "verify"
    assert map_routerbench_task_to_phase("planning_steps") == "tasks"
    assert map_routerbench_task_to_phase("web_search") == "research"
    assert map_routerbench_task_to_phase("drop") == "explore"
    assert map_routerbench_task_to_phase("summarization") == "propose"
    assert map_routerbench_task_to_phase("unknown_benchmark_xyz") == "explore"


def test_routerbench_sample_extraction() -> None:
    """Test extraction and normalization of RouterBench records."""
    raw = {
        "prompt": "def add(a, b):",
        "task_name": "humaneval",
        "model_name": "openai/gpt-4",
        "cost": 0.004,
        "latency": 0.85,
        "correctness": 1.0,
        "input_tokens": 50,
        "output_tokens": 20,
    }
    sample = extract_routerbench_sample(raw)
    assert sample is not None
    assert sample["prompt"] == "def add(a, b):"
    assert sample["task_name"] == "humaneval"
    assert sample["model_name"] == "openai/gpt-4"
    assert sample["cost"] == pytest.approx(0.004)
    assert sample["latency"] == pytest.approx(0.85)
    assert sample["correctness"] == 1.0
    assert sample["mapped_phase"] == "apply"

    # Boolean correctness handling
    raw_bool = {
        "query": "Solve 2+2",
        "domain": "gsm8k",
        "model": "anthropic/claude-3-5-sonnet",
        "win": True,
    }
    sample_bool = extract_routerbench_sample(raw_bool)
    assert sample_bool is not None
    assert sample_bool["correctness"] == 1.0
    assert sample_bool["mapped_phase"] == "design"

    # Empty row check
    assert extract_routerbench_sample({}) is None


@respx.mock
def test_routerbench_collector_success(store: SnapshotStore) -> None:
    """Test successful RouterBench snapshot collection via HTTP JSONL."""
    url = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/test.jsonl"
    payload_lines = (
        json.dumps({
            "prompt": "Sort list",
            "task": "humaneval",
            "model": "gpt-4",
            "cost": 0.002,
            "latency": 0.5,
            "correctness": 1.0,
        })
        + "\n"
        + json.dumps({
            "prompt": "What is 7 * 8?",
            "task": "gsm8k",
            "model": "llama-3-8b",
            "cost": 0.0001,
            "latency": 0.2,
            "correctness": 0.0,
        })
    )
    respx.get(url).mock(return_value=httpx.Response(200, text=payload_lines))

    config = RouterBenchConfig()
    with RouterBenchCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.source == "routerbench"
    assert record.record_count == 2
    assert record.errors == []
    assert record.path.is_file()

    doc = json.loads(record.path.read_text(encoding="utf-8"))
    assert doc["source"] == "routerbench"
    samples = doc["data"]["samples"]
    assert len(samples) == 2
    assert samples[0]["mapped_phase"] == "apply"
    assert samples[1]["mapped_phase"] == "design"


@respx.mock
def test_routerbench_collector_parquet(store: SnapshotStore) -> None:
    """Test RouterBench parquet file parsing."""
    url = "https://example.com/routerbench.parquet"
    table = pa.Table.from_pydict({
        "prompt": ["Test prompt"],
        "task_name": ["code_eval"],
        "model_name": ["gpt-4o"],
        "cost": [0.003],
        "latency": [0.4],
        "eval_score": [0.95],
    })
    buf = io.BytesIO()
    pq.write_table(table, buf)
    respx.get(url).mock(return_value=httpx.Response(200, content=buf.getvalue()))

    config = RouterBenchConfig(url=url)
    with RouterBenchCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.source == "routerbench"
    assert record.record_count == 1
    doc = json.loads(record.path.read_text(encoding="utf-8"))
    assert doc["data"]["samples"][0]["mapped_phase"] == "apply"


def test_routerbench_collector_load_fn(store: SnapshotStore) -> None:
    """Test RouterBench collector with injected load_fn."""
    mock_data = [
        {"prompt": "Hello", "task": "chat", "model": "gpt-4", "score": 1.0},
    ]

    def mock_load(dataset: str, split: str) -> list[dict]:
        return mock_data

    config = RouterBenchConfig()
    collector = RouterBenchCollector(config, store, load_fn=mock_load)
    record = collector.collect(now=NOW)
    assert record.record_count == 1
    assert record.source == "routerbench"


@respx.mock
def test_routerbench_collector_fallback(store: SnapshotStore) -> None:
    """Test RouterBench offline fallback to existing cached snapshot on failure."""
    # Pre-save an older snapshot
    prior_record = store.save(
        "routerbench",
        data={"source": "routerbench", "samples": [{"prompt": "cached"}]},
        meta={"record_count": 1, "errors": []},
        fetched_at=datetime(2026, 9, 18, tzinfo=UTC),
    )

    url = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/test.jsonl"
    respx.get(url).mock(return_value=httpx.Response(500))

    config = RouterBenchConfig()
    with RouterBenchCollector(config, store) as collector:
        record = collector.collect(now=NOW, force=True)

    assert record.snapshot_id == prior_record.snapshot_id
    assert len(record.errors) == 1
    assert "500" in record.errors[0]


@respx.mock
def test_routerbench_collector_no_cache_error(store: SnapshotStore) -> None:
    """Test error raised when download fails and no cached snapshot exists."""
    url = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/test.jsonl"
    respx.get(url).mock(return_value=httpx.Response(502))

    config = RouterBenchConfig()
    with RouterBenchCollector(config, store) as collector:
        with pytest.raises(RouterBenchCollectionError):
            collector.collect(now=NOW, force=True)


# =========================================================================== #
# 2. RouteLLM Collector Tests
# =========================================================================== #


def test_routellm_phase_mapping() -> None:
    """Test mapping of RouteLLM benchmarks to SDD canonical phases."""
    assert map_routellm_task_to_phase("coding") == "apply"
    assert map_routellm_task_to_phase("humaneval") == "apply"
    assert map_routellm_task_to_phase("math") == "design"
    assert map_routellm_task_to_phase("logic") == "spec"
    assert map_routellm_task_to_phase("planning") == "tasks"
    assert map_routellm_task_to_phase("verify") == "verify"
    assert map_routellm_task_to_phase("arena") == "explore"
    assert map_routellm_task_to_phase("unknown") == "explore"


def test_routellm_choice_and_noul_calibration() -> None:
    """Test extraction of pairwise battles into Choice and Noul calibration labels."""
    # Scenario A: Economy model (model_a: 8b) wins against Frontier model (model_b: gpt-4)
    # Fast success viable (Noul = 1.0), Choice = model_a
    row_a = {
        "prompt": "Summarize the text",
        "small_model": "meta/llama-3-8b",
        "large_model": "openai/gpt-4",
        "winner": "model_a",
        "score_a": 0.9,
        "score_b": 0.7,
        "task": "writing",
    }
    sample_a = extract_routellm_sample(row_a)
    assert sample_a is not None
    assert sample_a["choice_label"] == "model_a"
    assert sample_a["noul_label"] == 1.0
    assert sample_a["mapped_phase"] == "propose"

    # Scenario B: Frontier model wins against economy model
    # Escalation needed (Noul = 0.0), Choice = model_b
    row_b = {
        "prompt": "Write a kernel driver",
        "model_a": "meta/llama-3-8b-instruct",
        "model_b": "anthropic/claude-3-opus",
        "winner": "model_b",
        "score_a": 0.3,
        "score_b": 0.95,
        "task": "coding",
    }
    sample_b = extract_routellm_sample(row_b)
    assert sample_b is not None
    assert sample_b["choice_label"] == "model_b"
    assert sample_b["noul_label"] == 0.0
    assert sample_b["mapped_phase"] == "apply"

    # Scenario C: Tie
    row_c = {
        "prompt": "What is the capital of France?",
        "model_a": "gpt-3.5-turbo",
        "model_b": "gpt-4",
        "winner": "tie",
        "score_a": 1.0,
        "score_b": 1.0,
        "task": "qa",
    }
    sample_c = extract_routellm_sample(row_c)
    assert sample_c is not None
    assert sample_c["choice_label"] == "tie"
    assert sample_c["noul_label"] == 1.0
    assert sample_c["mapped_phase"] == "explore"


@respx.mock
def test_routellm_collector_success(store: SnapshotStore) -> None:
    """Test successful RouteLLM snapshot collection."""
    url = "https://huggingface.co/datasets/lmsys/routellm-eval/resolve/main/test.jsonl"
    payload = (
        json.dumps({
            "prompt": "Compute fibonacci",
            "model_a": "meta/llama-3-8b",
            "model_b": "openai/gpt-4",
            "winner": "model_a",
            "score_a": 0.95,
            "score_b": 0.90,
            "task": "code",
            "threshold": 0.5,
        })
        + "\n"
    )
    respx.get(url).mock(return_value=httpx.Response(200, text=payload))

    config = RouteLLMConfig()
    with RouteLLMCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.source == "routellm"
    assert record.record_count == 1
    assert record.errors == []
    doc = json.loads(record.path.read_text(encoding="utf-8"))
    s = doc["data"]["samples"][0]
    assert s["choice_label"] == "model_a"
    assert s["noul_label"] == 1.0
    assert s["mapped_phase"] == "apply"


@respx.mock
def test_routellm_collector_fallback(store: SnapshotStore) -> None:
    """Test RouteLLM fallback to cached snapshot on download failure."""
    store.save(
        "routellm",
        data={"source": "routellm", "samples": [{"prompt": "cached_battle"}]},
        meta={"record_count": 1, "errors": []},
        fetched_at=datetime(2026, 9, 18, tzinfo=UTC),
    )

    url = "https://huggingface.co/datasets/lmsys/routellm-eval/resolve/main/test.jsonl"
    respx.get(url).mock(return_value=httpx.Response(500))

    config = RouteLLMConfig()
    with RouteLLMCollector(config, store) as collector:
        record = collector.collect(now=NOW, force=True)

    assert record.record_count == 1
    assert record.source == "routellm"
    assert len(record.errors) == 1


@respx.mock
def test_routellm_collector_no_cache_error(store: SnapshotStore) -> None:
    """Test error raised when RouteLLM download fails and no cached snapshot exists."""
    url = "https://huggingface.co/datasets/lmsys/routellm-eval/resolve/main/test.jsonl"
    respx.get(url).mock(return_value=httpx.Response(502))

    config = RouteLLMConfig()
    with RouteLLMCollector(config, store) as collector:
        with pytest.raises(RouteLLMCollectionError):
            collector.collect(now=NOW, force=True)


# =========================================================================== #
# 3. SWE-Traces Collector Tests
# =========================================================================== #


def test_swe_traces_phase_mapping() -> None:
    """Test SDD phase mapping for SWE and Aider trajectories."""
    assert map_swe_trace_to_phase({"step": "apply_patch"}) == "apply"
    assert map_swe_trace_to_phase({"diff": "--- a/test.py\n+++ b/test.py"}) == "apply"
    assert map_swe_trace_to_phase({"step": "run_test_suite"}) == "verify"
    assert map_swe_trace_to_phase({"eval_cmd": "pytest"}) == "verify"
    assert map_swe_trace_to_phase({"step": "task_planning"}) == "tasks"
    assert map_swe_trace_to_phase({"step": "explore_repo"}) == "explore"
    assert map_swe_trace_to_phase({}) == "apply"


def test_swe_traces_sample_extraction() -> None:
    """Test extraction of agent trajectory fields."""
    row = {
        "instance_id": "django__django-11111",
        "problem_statement": "Fix query set leak",
        "repo": "django/django",
        "model": "anthropic/claude-3-5-sonnet",
        "input_tokens": 2000,
        "output_tokens": 400,
        "diff": "--- a/django/db.py\n+++ b/django/db.py\n@@ -1 +1 @@\n-old\n+new",
        "test_passed": True,
        "cost": 0.015,
        "phase": "apply",
    }
    sample = extract_swe_trace_sample(row)
    assert sample is not None
    assert sample["instance_id"] == "django__django-11111"
    assert sample["task_description"] == "Fix query set leak"
    assert sample["repo"] == "django/django"
    assert sample["model_name"] == "anthropic/claude-3-5-sonnet"
    assert sample["input_tokens"] == 2000
    assert sample["output_tokens"] == 400
    assert sample["total_tokens"] == 2400
    assert "django/db.py" in sample["diff"]
    assert sample["test_passed"] is True
    assert sample["cost"] == pytest.approx(0.015)
    assert sample["mapped_phase"] == "apply"


@respx.mock
def test_swe_traces_collector_success(store: SnapshotStore) -> None:
    """Test successful SWE-Traces snapshot collection."""
    url = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/test.jsonl"
    payload = (
        json.dumps({
            "instance_id": "pytest-dev__pytest-1234",
            "problem_statement": "Parametrize fixture bug",
            "repo": "pytest-dev/pytest",
            "model": "openai/gpt-4o",
            "input_tokens": 1200,
            "output_tokens": 300,
            "diff": "--- a/src/pytest.py\n+++ b/src/pytest.py",
            "test_passed": True,
        })
        + "\n"
    )
    respx.get(url).mock(return_value=httpx.Response(200, text=payload))

    config = SWETracesConfig()
    with SWETracesCollector(config, store) as collector:
        record = collector.collect(now=NOW)

    assert record.source == "swe-traces"
    assert record.record_count == 1
    assert record.errors == []
    doc = json.loads(record.path.read_text(encoding="utf-8"))
    assert doc["data"]["trajectories"][0]["instance_id"] == "pytest-dev__pytest-1234"


@respx.mock
def test_swe_traces_collector_fallback(store: SnapshotStore) -> None:
    """Test SWE-Traces fallback to cached snapshot on download failure."""
    store.save(
        "swe-traces",
        data={"source": "swe-traces", "trajectories": [{"instance_id": "cached_trace"}]},
        meta={"record_count": 1, "errors": []},
        fetched_at=datetime(2026, 9, 18, tzinfo=UTC),
    )

    url = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/test.jsonl"
    respx.get(url).mock(return_value=httpx.Response(503))

    config = SWETracesConfig()
    with SWETracesCollector(config, store) as collector:
        record = collector.collect(now=NOW, force=True)

    assert record.record_count == 1
    assert record.source == "swe-traces"
    assert len(record.errors) == 1


@respx.mock
def test_swe_traces_collector_no_cache_error(store: SnapshotStore) -> None:
    """Test error raised when SWE-Traces download fails and no cached snapshot exists."""
    url = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/test.jsonl"
    respx.get(url).mock(return_value=httpx.Response(502))

    config = SWETracesConfig()
    with SWETracesCollector(config, store) as collector:
        with pytest.raises(SWETracesCollectionError):
            collector.collect(now=NOW, force=True)


# =========================================================================== #
# 4. CLI Collect Command Tests
# =========================================================================== #


@respx.mock
def test_cli_collect_routerbench(tmp_path: Path) -> None:
    """Test CLI router collect --source routerbench."""
    url = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/test.jsonl"
    respx.get(url).mock(
        return_value=httpx.Response(200, text=json.dumps({"prompt": "p", "model": "m"}) + "\n")
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["collect", "--source", "routerbench", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "routerbench" in result.output


@respx.mock
def test_cli_collect_routellm(tmp_path: Path) -> None:
    """Test CLI router collect --source routellm."""
    url = "https://huggingface.co/datasets/lmsys/routellm-eval/resolve/main/test.jsonl"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, text=json.dumps({"prompt": "p", "model_a": "a", "model_b": "b"}) + "\n"
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["collect", "--source", "routellm", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "routellm" in result.output


@respx.mock
def test_cli_collect_swe_traces(tmp_path: Path) -> None:
    """Test CLI router collect --source swe-traces."""
    url = "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/test.jsonl"
    respx.get(url).mock(
        return_value=httpx.Response(
            200, text=json.dumps({"instance_id": "i1", "problem_statement": "p"}) + "\n"
        )
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["collect", "--source", "swe-traces", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "swe-traces" in result.output


@respx.mock
def test_cli_collect_all_graceful_degradation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test CLI router collect --source all continues gracefully when external sources fail."""
    respx.get("https://artificialanalysis.ai/api/v2/language/models/free").mock(
        return_value=httpx.Response(401)
    )
    respx.route().mock(return_value=httpx.Response(500))

    def mock_load_dataset(*args, **kwargs):
        raise RuntimeError("Offline test failure")

    monkeypatch.setattr(
        "gentle_ai_model_router.collector.arena.load_dataset", mock_load_dataset
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["collect", "--source", "all", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "warning:" in result.output or "local discovery" in result.output


# =========================================================================== #
# 5. Telemetry Bridge Ground-Truth Conversion Tests
# =========================================================================== #


def test_telemetry_bridge_extract_ground_truth_executions() -> None:
    """Test converting ground-truth snapshot documents into TelemetryExecution records."""
    swe_doc = {
        "fetched_at": "2026-09-19T10:00:00Z",
        "data": {
            "source": "swe-traces",
            "trajectories": [
                {
                    "instance_id": "repo__1",
                    "task_description": "Fix bug",
                    "repo": "owner/repo",
                    "model_name": "gpt-4",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "test_passed": True,
                    "cost": 0.005,
                }
            ],
        },
    }
    execs = extract_ground_truth_executions(swe_doc, default_phase="apply")
    assert len(execs) == 1
    ex = execs[0]
    assert ex.execution_id == "gt-swe-repo__1"
    assert ex.model == "gpt-4"
    assert ex.task_success == 1
    assert ex.quality_score == 1.0
    assert ex.input_tokens == 100
    assert ex.output_tokens == 50
    assert ex.phase == "apply"


# =========================================================================== #
# 6. Dataset Builder Ingestion & Provenance Tests
# =========================================================================== #


def _setup_test_registry(engine) -> None:
    """Seed test database with providers, models, deployments, and variants."""
    registry_db.init_schema(engine)
    with Session(engine) as session:
        # Snapshots
        snap1 = ModelSnapshot(
            snapshot_id="2026-09-18-artificial-analysis",
            source="artificial-analysis",
            fetched_at=datetime(2026, 9, 18, tzinfo=UTC),
            record_count=1,
        )
        session.add(snap1)

        prov = Provider(registry_key="openai", name="OpenAI")
        session.add(prov)
        session.flush()

        model1 = Model(
            canonical_id="openai/gpt-4",
            name="GPT-4",
            org="openai",
        )
        session.add(model1)
        session.flush()

        dep1 = Deployment(
            model_id=model1.id,
            provider_id=prov.id,
            deployment_ref="openai/gpt-4",
        )
        session.add(dep1)
        session.flush()

        var1 = ModelVariant(
            deployment_id=dep1.id, effort="medium", provider_value="openai/gpt-4"
        )
        session.add(var1)

        bench1 = ModelBenchmark(
            model_id=model1.id,
            benchmark="artificial_analysis_intelligence_index",
            score=85.0,
            source_snapshot_id=snap1.snapshot_id,
        )
        session.add(bench1)

        price1 = ModelPrice(
            deployment_id=dep1.id,
            input_price=5.0,
            output_price=15.0,
            source_snapshot_id=snap1.snapshot_id,
        )
        session.add(price1)
        session.commit()


def test_dataset_builder_ground_truth_provenance(tmp_path: Path) -> None:
    """Test DatasetBuilder ingests ground-truth snapshots and assigns provenance."""
    db_url = f"sqlite:///{tmp_path}/test.db"
    engine, _ = registry_db.get_engine_with_fallback(db_url, db_url)
    _setup_test_registry(engine)

    store = SnapshotStore(tmp_path / "snapshots")

    # Store a ground-truth snapshot under routerbench
    rb_snapshot = store.save(
        "routerbench",
        data={
            "source": "routerbench",
            "samples": [
                {
                    "prompt": "def solve(): pass",
                    "task_name": "humaneval",
                    "model_name": "openai/gpt-4",
                    "cost": 0.005,
                    "latency": 0.5,
                    "correctness": 0.95,
                    "mapped_phase": "apply",
                }
            ],
        },
        meta={"record_count": 1, "errors": []},
        fetched_at=datetime(2026, 9, 18, tzinfo=UTC),
    )

    # Store a ground-truth snapshot under swe-traces
    swe_snapshot = store.save(
        "swe-traces",
        data={
            "source": "swe-traces",
            "trajectories": [
                {
                    "instance_id": "repo__issue-1",
                    "task_description": "Implement feature X",
                    "repo": "fastapi/fastapi",
                    "model_name": "openai/gpt-4",
                    "input_tokens": 1500,
                    "output_tokens": 200,
                    "diff": "--- a/main.py\n+++ b/main.py",
                    "test_passed": True,
                    "cost": 0.01,
                    "mapped_phase": "apply",
                }
            ],
        },
        meta={"record_count": 1, "errors": []},
        fetched_at=datetime(2026, 9, 18, tzinfo=UTC),
    )

    config = RouterConfig()
    builder_cfg = DatasetBuilderConfig(
        name="test-ground-truth-dataset",
        train_end=date(2026, 9, 20),
        as_of=date(2026, 9, 20),
        include_ground_truth_traces=True,
        ground_truth_snapshot_sources=["routerbench", "swe-traces"],
    )

    with Session(engine) as session:
        dataset = build_examples(session, config, builder_cfg, store=store)

    assert len(dataset.examples) > 0
    # Check that ground-truth snapshot IDs are recorded
    assert rb_snapshot.snapshot_id in dataset.source_snapshot_ids
    assert swe_snapshot.snapshot_id in dataset.source_snapshot_ids

    # Check for examples with PROVENANCE_GROUND_TRUTH
    gt_examples = [
        e for e in dataset.examples if e.label_provenance == PROVENANCE_GROUND_TRUTH
    ]
    assert len(gt_examples) > 0

    # Verify quality matches empirical measurement
    assert any(e.label_quality_estimate == pytest.approx(0.95) for e in gt_examples)
    assert any(e.label_quality_estimate == pytest.approx(1.0) for e in gt_examples)

    # Check dataset manifest provenance
    manifest = dataset.manifest()
    assert "ground_truth_traces" in manifest["label_provenance"]
    assert GROUND_TRUTH_PROVENANCE_ADDENDUM in dataset.label_provenance_statement
