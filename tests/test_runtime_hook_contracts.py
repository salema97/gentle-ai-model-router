"""Contract and integration tests for runtime hook plugins (OpenCode and Pi).

Validates:
- JS plugin source files exist and pass syntax validation (node --check).
- Emitted JSON payloads strictly conform to ExecutionRecord schema fields.
- Fallback spool mechanism creates valid JSONL ingestible by the telemetry shim.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from io import StringIO
from pathlib import Path

import pytest

from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.integration.telemetry_shim import ExecutionRecord

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "src" / "gentle_ai_model_router" / "plugins"
OPENCODE_JS = PLUGIN_DIR / "opencode" / "router_telemetry.js"
OPENCODE_DTS = PLUGIN_DIR / "opencode" / "router_telemetry.d.ts"
PI_JS = PLUGIN_DIR / "pi" / "router_telemetry.js"
PI_DTS = PLUGIN_DIR / "pi" / "router_telemetry.d.ts"

NODE_AVAILABLE = shutil.which("node") is not None


def test_plugin_source_files_exist() -> None:
    assert OPENCODE_JS.is_file(), f"Missing {OPENCODE_JS}"
    assert OPENCODE_DTS.is_file(), f"Missing {OPENCODE_DTS}"
    assert PI_JS.is_file(), f"Missing {PI_JS}"
    assert PI_DTS.is_file(), f"Missing {PI_DTS}"


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node is not installed")
def test_javascript_syntax_valid() -> None:
    res_opencode = subprocess.run(
        ["node", "--check", str(OPENCODE_JS)], capture_output=True, text=True
    )
    assert res_opencode.returncode == 0, f"OpenCode JS syntax error: {res_opencode.stderr}"

    res_pi = subprocess.run(["node", "--check", str(PI_JS)], capture_output=True, text=True)
    assert res_pi.returncode == 0, f"Pi JS syntax error: {res_pi.stderr}"


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node is not installed")
def test_opencode_hook_payload_contract(tmp_path: Path) -> None:
    node_script = f"""
    const plugin = require({json.dumps(str(OPENCODE_JS))});
    plugin.onMessageUpdated({{
        session_id: 'sess-123',
        model: 'anthropic/claude-3-5-sonnet',
        deployment: 'default',
        effort: 'high',
        tokens: {{ input: 150, output: 250, reasoning: 50, cached: 20, total: 400 }},
        latency_ms: 1200
    }});

    const payload = plugin.onSubagentStop({{
        session_id: 'sess-123',
        phase: 'sdd-apply',
        task_type: 'code_gen',
        tool_calls: 3,
        tool_errors: 0,
        tests_passed: 12,
        tests_failed: 0,
        decision_id: 'dec-abc'
    }});

    console.log(JSON.stringify(payload));
    """
    res = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
    payload = json.loads(res.stdout)

    assert payload["session_id"] == "sess-123"
    assert payload["phase"] == "apply"
    assert payload["model"] == "anthropic/claude-3-5-sonnet"
    assert payload["deployment"] == "default"
    assert payload["effort"] == "high"
    assert payload["input_tokens"] == 150
    assert payload["output_tokens"] == 250
    assert payload["reasoning_tokens"] == 50
    assert payload["cached_tokens"] == 20
    assert payload["total_tokens"] == 400
    assert payload["latency_ms"] == 1200
    assert payload["tool_calls"] == 3
    assert payload["tool_errors"] == 0
    assert payload["tests_passed"] == 12
    assert payload["tests_failed"] == 0
    assert payload["decision_id"] == "dec-abc"
    assert payload["router_version"] == "opencode-hook-v1"

    # Schema validation against SQLAlchemy ExecutionRecord table
    valid_columns = set(ExecutionRecord.__table__.columns.keys())
    for key in payload:
        assert key in valid_columns, f"Field {key} not in ExecutionRecord schema"


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node is not installed")
def test_pi_hook_payload_contract(tmp_path: Path) -> None:
    node_script = f"""
    const plugin = require({json.dumps(str(PI_JS))});
    plugin.onTurnContext({{
        turn_id: 'turn-456',
        phase: 'sdd-verify',
        model: 'openai/gpt-4o',
        thinking: 'medium',
        tokens: {{ input: 100, output: 200, total: 300 }},
        latency_ms: 800
    }});

    const payload = plugin.onReviewComplete({{
        turn_id: 'turn-456',
        decision: 'approved',
        tool_calls: 2,
        tool_errors: 0
    }});

    console.log(JSON.stringify(payload));
    """
    res = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
    payload = json.loads(res.stdout)

    assert payload["session_id"] == "turn-456"
    assert payload["phase"] == "verify"
    assert payload["model"] == "openai/gpt-4o"
    assert payload["effort"] == "medium"
    assert payload["task_success"] == 1
    assert payload["quality_score"] == 100.0
    assert payload["escalation_count"] == 0
    assert payload["total_tokens"] == 300
    assert payload["router_version"] == "pi-hook-v1"

    valid_columns = set(ExecutionRecord.__table__.columns.keys())
    for key in payload:
        assert key in valid_columns, f"Field {key} not in ExecutionRecord schema"


@pytest.mark.skipif(not NODE_AVAILABLE, reason="node is not installed")
def test_spool_fallback_and_ingestion(tmp_path: Path) -> None:
    spool_file = tmp_path / "telemetry-spool.jsonl"
    node_script = f"""
    const plugin = require({json.dumps(str(OPENCODE_JS))});
    async function run() {{
        const payload = plugin.createExecutionPayload({{
            execution_id: 'spool-test-01',
            phase: 'explore',
            total_tokens: 750
        }});
        // Attempt dispatch to offline port, triggers spool write
        await plugin.dispatchExecution(payload, {{
            endpoint: 'http://127.0.0.1:59999/shim/execution',
            timeoutMs: 150,
            spoolFile: {json.dumps(str(spool_file))}
        }});
    }}
    run();
    """
    subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)

    assert spool_file.is_file(), "Spool file was not created"
    content = spool_file.read_text(encoding="utf-8")
    assert "spool-test-01" in content

    # Verify ingestion into ShimStore
    store = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'test_spool.sqlite'}")
    store.init_schema()

    with store.session() as session:
        counts = store.ingest_jsonl(session, StringIO(content))
        assert counts["inserted"] == 1
        assert counts["skipped"] == 0

    # Verify DB row
    with store.session() as session:
        record = session.query(ExecutionRecord).filter_by(execution_id="spool-test-01").first()
        assert record is not None
        assert record.phase == "explore"
        assert record.total_tokens == 750
