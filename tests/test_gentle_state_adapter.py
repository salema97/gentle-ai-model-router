"""Gentle AI state adapter tests: full cycle, both key casings, refusal paths.

All writes target tmp_path state files — the real ~/.gentle-ai/state.json is
never touched (same rule as the opencode adapter tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration import gentle_state_adapter as gs

runner = CliRunner()

# Synthetic state file matching the REAL observed top-level structure of
# ~/.gentle-ai/state.json (read-only inspection, Phase 6a): the real file has
# NO model_assignments key at all; the shape below adds one assignment so
# overwrite/preserve paths are exercised. No private content is replicated.
REAL_SHAPE_STATE = {
    "installed_agents": ["opencode", "claude"],
    "installed_binary_version": "3.0.0-test",
    "managed_asset_digest": "abc123",
    "selection_configured": True,
    "components": ["sdd"],
    "skills": ["sdd-explore"],
    "preset": "balanced",
    "sdd_mode": "multi-profile",
    "strict_tdd": False,
    "community_tools": [],
    "community_tools_configured": False,
    "persona": "default",
    "last_update_check": "2026-09-01T00:00:00Z",
    "opencode_background_subagents": "on",
    "pi_background_subagents": "on",
    "model_assignments": {
        "sdd-spec": {"provider_id": "openai", "model_id": "gpt-5", "effort": "medium"},
    },
}


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(REAL_SHAPE_STATE, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Full cycle: write -> verify -> rollback -> verify restored
# --------------------------------------------------------------------------- #


def test_apply_preserves_everything_else(state_file: Path) -> None:
    before_text = state_file.read_text(encoding="utf-8")
    result = gs.apply_assignment(
        state_file, "explore", "anthropic/claude-sonnet-4", "high"
    )
    assert result.wrote
    assert result.key_used == "model_assignments"
    assert result.backup_path is not None and result.backup_path.is_file()
    after = json.loads(state_file.read_text(encoding="utf-8"))
    # Untouched top-level keys survive verbatim.
    for key, value in REAL_SHAPE_STATE.items():
        if key != "model_assignments":
            assert after[key] == value
    # Pre-existing assignment survives.
    assert after["model_assignments"]["sdd-spec"] == {
        "provider_id": "openai",
        "model_id": "gpt-5",
        "effort": "medium",
    }
    # New assignment has the exact probe-verified shape.
    assert after["model_assignments"]["sdd-explore"] == {
        "provider_id": "anthropic",
        "model_id": "claude-sonnet-4",
        "effort": "high",
    }
    assert before_text != state_file.read_text(encoding="utf-8")


def test_verify_round_trip(state_file: Path) -> None:
    gs.apply_assignment(state_file, "explore", "anthropic/claude-sonnet-4", "high")
    ok, entry = gs.verify_assignment(
        state_file, "explore", "anthropic/claude-sonnet-4", "high"
    )
    assert ok and entry is not None
    ok, entry = gs.verify_assignment(state_file, "explore", "anthropic/other", "high")
    assert not ok
    ok, entry = gs.verify_assignment(state_file, "apply", "anthropic/other", "high")
    assert not ok and entry is None


def test_rollback_restores_original(state_file: Path) -> None:
    original_text = state_file.read_text(encoding="utf-8")
    gs.apply_assignment(state_file, "explore", "anthropic/claude-sonnet-4", "high")
    restored = gs.rollback(state_file)
    assert restored is not None
    assert state_file.read_text(encoding="utf-8") == original_text
    # Post-rollback the new assignment is gone.
    ok, _ = gs.verify_assignment(state_file, "explore", "anthropic/claude-sonnet-4", "high")
    assert not ok


def test_rollback_without_backup(state_file: Path) -> None:
    assert gs.rollback(state_file) is None


# --------------------------------------------------------------------------- #
# Key casing: both model_assignments and ModelAssignments accepted
# --------------------------------------------------------------------------- #


def test_camel_case_assignments_key_accepted(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {"preset": "balanced", "ModelAssignments": {"sdd-spec": {
                "provider_id": "openai", "model_id": "gpt-5", "effort": "medium"}}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = gs.apply_assignment(path, "explore", "anthropic/claude-sonnet-4", "low")
    assert result.key_used == "ModelAssignments"
    after = json.loads(path.read_text(encoding="utf-8"))
    assert "model_assignments" not in after  # did NOT create a second key
    assert after["ModelAssignments"]["sdd-explore"]["model_id"] == "claude-sonnet-4"


def test_read_assignment_accepts_both_casings(tmp_path: Path) -> None:
    for key in ("model_assignments", "ModelAssignments"):
        path = tmp_path / f"state-{key}.json"
        path.write_text(
            json.dumps({key: {"sdd-apply": {
                "provider_id": "openai", "model_id": "gpt-5", "effort": "high"}}}),
            encoding="utf-8",
        )
        entry = gs.read_assignment(path, "apply")
        assert entry is not None and entry["model_id"] == "gpt-5"


def test_create_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    result = gs.apply_assignment(
        path, "explore", "anthropic/claude-sonnet-4", "high", create=True
    )
    assert result.wrote
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "model_assignments": {
            "sdd-explore": {
                "provider_id": "anthropic",
                "model_id": "claude-sonnet-4",
                "effort": "high",
            }
        }
    }


# --------------------------------------------------------------------------- #
# Refusal paths (AdapterError)
# --------------------------------------------------------------------------- #


def test_missing_file_refused(tmp_path: Path) -> None:
    with pytest.raises(gs.AdapterError, match="does not exist"):
        gs.apply_assignment(tmp_path / "state.json", "explore", "a/b", "high")


def test_malformed_json_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(gs.AdapterError, match="malformed JSON"):
        gs.apply_assignment(path, "explore", "a/b", "high")


def test_non_dict_existing_entry_refused(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["model_assignments"]["sdd-explore"] = "anthropic/claude-sonnet-4"
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(gs.AdapterError, match="does not look like a model assignment"):
        gs.apply_assignment(state_file, "explore", "a/b", "high")


def test_unrecognized_dict_shape_refused(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["model_assignments"]["sdd-explore"] = {"unrelated": True}
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(gs.AdapterError, match="does not look like a model assignment"):
        gs.apply_assignment(state_file, "explore", "a/b", "high")


def test_assignments_key_not_an_object_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"model_assignments": []}), encoding="utf-8")
    with pytest.raises(gs.AdapterError, match="not an object"):
        gs.apply_assignment(path, "explore", "a/b", "high")


def test_invalid_effort_refused_with_vocabulary(state_file: Path) -> None:
    with pytest.raises(gs.AdapterError) as excinfo:
        gs.apply_assignment(state_file, "explore", "a/b", "super-high")
    for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        assert level in str(excinfo.value)


def test_model_spec_without_slash(state_file: Path) -> None:
    result = gs.apply_assignment(state_file, "explore", "claude-sonnet-4", "high")
    assert result.provider_id is None
    assert result.model_id == "claude-sonnet-4"
    entry = gs.read_assignment(state_file, "explore")
    assert entry == {"provider_id": None, "model_id": "claude-sonnet-4", "effort": "high"}


def test_model_spec_splits_on_first_slash() -> None:
    provider, model_id = gs.parse_model_spec("openai/azure/gpt-5")
    assert provider == "openai" and model_id == "azure/gpt-5"


def test_existing_entry_extra_keys_preserved(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["model_assignments"]["sdd-explore"] = {
        "model_id": "old-model", "variant": "custom-variant", "note": "keep me",
    }
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    gs.apply_assignment(state_file, "explore", "anthropic/new-model", "low")
    entry = gs.read_assignment(state_file, "explore")
    assert entry["variant"] == "custom-variant"
    assert entry["note"] == "keep me"
    assert entry["model_id"] == "new-model"


def test_indent_preserved(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"preset": "x"}, indent=4) + "\n", encoding="utf-8")
    gs.apply_assignment(path, "explore", "a/b", "high")
    text = path.read_text(encoding="utf-8")
    assert '\n    "model_assignments"' in text  # 4-space indent kept


def test_dry_run_writes_nothing(state_file: Path) -> None:
    before = state_file.read_text(encoding="utf-8")
    result = gs.apply_assignment(
        state_file, "explore", "anthropic/claude-sonnet-4", "high", dry_run=True
    )
    assert not result.wrote
    assert result.diff  # diff still shown
    assert state_file.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_cli_full_cycle(tmp_path: Path, state_file: Path) -> None:
    original = state_file.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "gentle-state",
            "--phase", "explore",
            "--model", "anthropic/claude-sonnet-4",
            "--effort", "high",
            "--state", str(state_file),
            "--verify",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "verify OK" in result.output
    assert "gentle-ai sync" in result.output
    assert "external-single-active" in result.output
    assert "__managed_by" in result.output

    result = runner.invoke(
        app, ["integrate", "gentle-state", "rollback", "--state", str(state_file)]
    )
    assert result.exit_code == 0, result.output
    assert state_file.read_text(encoding="utf-8") == original

    result = runner.invoke(
        app, ["integrate", "gentle-state", "status", "--state", str(state_file)]
    )
    assert result.exit_code == 0
    assert "sdd-spec" in result.output


def test_cli_missing_file_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "gentle-state",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--state", str(tmp_path / "missing.json"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_malformed_json_exit_2(state_file: Path) -> None:
    state_file.write_text("{ broken", encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "gentle-state",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--state", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "malformed JSON" in result.output


def test_cli_conflicting_shape_exit_2(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["model_assignments"]["sdd-explore"] = "not-a-dict"
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "gentle-state",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--state", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output


def test_cli_invalid_effort_exit_2(state_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "gentle-state",
            "--phase", "explore", "--model", "a/b", "--effort", "turbo",
            "--state", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "expected one of" in result.output


def test_cli_unknown_phase_exit_2(state_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "gentle-state",
            "--phase", "bogus", "--model", "a/b", "--effort", "high",
            "--state", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "unknown phase" in result.output


def test_cli_rollback_without_backup_exit_1(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["integrate", "gentle-state", "rollback", "--state", str(path)]
    )
    assert result.exit_code == 1
    assert "no backup found" in result.output


def test_cli_dry_run(state_file: Path) -> None:
    before = state_file.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "gentle-state",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--state", str(state_file), "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "nothing written" in result.output
    assert state_file.read_text(encoding="utf-8") == before
