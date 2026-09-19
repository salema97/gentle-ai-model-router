"""Codex adapter tests: effort-subset mapping, underscore keys, carril writes,
state preservation, backup/rollback, dry-run, fail-closed refusals, CLI smoke.

All writes target tmp_path state.json files — the real
~/.gentle-ai/state.json is never touched (same rule as the other adapter
tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration import codex_adapter as cx

runner = CliRunner()

# Synthetic state.json matching the documented Codex blocks (research doc
# §2.2/§3): CamelCase assignment maps keyed by the Codex transport
# identifiers (underscores for phases).
REAL_SHAPE_STATE = {
    "ModelAssignments": {
        "sdd-explore": {
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4",
            "effort": "high",
        },
    },
    "CodexPhaseModelAssignments": {
        "sdd_spec": {"provider_id": "openai", "model_id": "gpt-5", "effort": "medium"},
    },
    "CodexModelAssignments": {
        "sdd_strong": {"provider_id": "anthropic", "model_id": "claude-opus-4", "effort": "high"},
    },
    "CodexOrchestratorAssignment": {"model": "anthropic/claude-opus-4", "effort": "high"},
    "user_note": "keep me",
}


@pytest.fixture
def state_file(tmp_path: Path) -> Path:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(REAL_SHAPE_STATE, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Effort-subset mapping (Codex only supports low|medium|high|xhigh)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("internal", "codex", "reason"),
    [
        ("off", "low", "codex_effort_mapped:off->low"),
        ("minimal", "low", "codex_effort_mapped:minimal->low"),
        ("low", "low", None),
        ("medium", "medium", None),
        ("high", "high", None),
        ("xhigh", "xhigh", None),
        ("max", "xhigh", "codex_effort_mapped:max->xhigh"),
    ],
)
def test_effort_mapping_table(
    state_file: Path, internal: str, codex: str, reason: str | None
) -> None:
    result = cx.apply_assignment(state_file, "tasks", "a/b", internal)
    assert result.codex_effort == codex
    assert result.effort == internal
    after = json.loads(state_file.read_text(encoding="utf-8"))
    assert after["CodexPhaseModelAssignments"]["sdd_tasks"]["effort"] == codex
    if reason is None:
        assert result.reason_codes == []
    else:
        assert reason in result.reason_codes


def test_to_codex_effort_rejects_unknown() -> None:
    with pytest.raises(cx.AdapterError, match="unknown effort"):
        cx.to_codex_effort("turbo")


# --------------------------------------------------------------------------- #
# Underscore phase keys (Codex transport convention)
# --------------------------------------------------------------------------- #


def test_phase_keys_use_underscores(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "apply", "a/b", "high")
    assert result.phase_key == "sdd_apply"
    after = json.loads(state_file.read_text(encoding="utf-8"))
    assert "sdd_apply" in after["CodexPhaseModelAssignments"]
    assert "sdd-apply" not in after["CodexPhaseModelAssignments"]


def test_phase_key_normalizer() -> None:
    assert cx.phase_key("apply") == "sdd_apply"
    assert cx.phase_key("sdd-apply") == "sdd_apply"
    assert cx.phase_key("sdd_apply") == "sdd_apply"


# --------------------------------------------------------------------------- #
# Carril writes (strong/mid/cheap)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("carril", ["strong", "mid", "cheap"])
def test_carril_write_each(state_file: Path, carril: str) -> None:
    result = cx.apply_assignment(
        state_file, "apply", "deepseek/deepseek-v4.1", "max", carril=carril
    )
    assert result.carril_key == f"sdd_{carril}"
    assert f"codex_carril:sdd_{carril}" in result.reason_codes
    after = json.loads(state_file.read_text(encoding="utf-8"))
    assert after["CodexModelAssignments"][f"sdd_{carril}"] == {
        "provider_id": "deepseek",
        "model_id": "deepseek-v4.1",
        "effort": "xhigh",  # max clamped into the Codex vocabulary
    }
    # The phase block got the same assignment in the same write.
    assert after["CodexPhaseModelAssignments"]["sdd_apply"]["model_id"] == "deepseek-v4.1"


def test_carril_prefix_accepted(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "apply", "a/b", "high", carril="sdd-mid")
    assert result.carril_key == "sdd_mid"


def test_invalid_carril_refused(state_file: Path) -> None:
    with pytest.raises(cx.AdapterError, match="unknown carril"):
        cx.apply_assignment(state_file, "apply", "a/b", "high", carril="platinum")
    assert json.loads(state_file.read_text(encoding="utf-8")) == REAL_SHAPE_STATE


def test_carril_entry_merges_existing_keys(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "apply", "a/b", "high", carril="strong")
    assert result.wrote
    after = json.loads(state_file.read_text(encoding="utf-8"))
    entry = after["CodexModelAssignments"]["sdd_strong"]
    assert entry["model_id"] == "b"
    assert entry["effort"] == "high"


# --------------------------------------------------------------------------- #
# State preservation + provider/model split
# --------------------------------------------------------------------------- #


def test_all_other_keys_preserved(state_file: Path) -> None:
    cx.apply_assignment(state_file, "apply", "google/gemini-3.8-flash", "low")
    after = json.loads(state_file.read_text(encoding="utf-8"))
    for key, value in REAL_SHAPE_STATE.items():
        if key == "CodexPhaseModelAssignments":
            # Pre-existing phase entries survive verbatim; the new sdd_apply
            # entry is the one intentional addition.
            for entry_key, entry in value.items():
                assert after[key][entry_key] == entry
        else:
            assert after[key] == value


def test_provider_model_split_on_first_slash(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "apply", "a/b/c", "high")
    assert result.provider_id == "a"
    assert result.model_id == "b/c"
    after = json.loads(state_file.read_text(encoding="utf-8"))
    entry = after["CodexPhaseModelAssignments"]["sdd_apply"]
    assert entry == {"provider_id": "a", "model_id": "b/c", "effort": "high"}


def test_bare_model_id_provider_none(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "apply", "gpt-5.6-sol", "high")
    assert result.provider_id is None
    assert result.model_id == "gpt-5.6-sol"


def test_overwrite_existing_phase_entry(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "spec", "openai/gpt-5-turbo", "high")
    assert result.wrote
    after = json.loads(state_file.read_text(encoding="utf-8"))
    assert after["CodexPhaseModelAssignments"]["sdd_spec"] == {
        "provider_id": "openai",
        "model_id": "gpt-5-turbo",
        "effort": "high",
    }


def test_noop_write_returns_wrote_false(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "spec", "openai/gpt-5", "medium")
    assert not result.wrote
    assert result.backup_path is None


def test_create_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    result = cx.apply_assignment(
        path, "explore", "anthropic/claude-sonnet-4", "high", create=True
    )
    assert result.wrote
    assert result.backup_path is None  # nothing to back up
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "CodexPhaseModelAssignments": {
            "sdd_explore": {
                "provider_id": "anthropic",
                "model_id": "claude-sonnet-4",
                "effort": "high",
            }
        }
    }


# --------------------------------------------------------------------------- #
# Refusal paths (fail closed with AdapterError)
# --------------------------------------------------------------------------- #


def test_missing_file_refused(tmp_path: Path) -> None:
    with pytest.raises(cx.AdapterError, match="does not exist"):
        cx.apply_assignment(tmp_path / "state.json", "explore", "a/b", "high")


def test_malformed_json_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(cx.AdapterError, match="malformed JSON"):
        cx.apply_assignment(path, "explore", "a/b", "high")


def test_non_dict_root_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(cx.AdapterError, match="root is not an object"):
        cx.apply_assignment(path, "explore", "a/b", "high")


def test_non_dict_phase_block_refused(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["CodexPhaseModelAssignments"] = ["not", "a", "map"]
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(cx.AdapterError, match="not an object"):
        cx.apply_assignment(state_file, "apply", "a/b", "high")
    assert json.loads(state_file.read_text(encoding="utf-8")) == data


def test_non_dict_carril_block_refused(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["CodexModelAssignments"] = 42
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(cx.AdapterError, match="not an object"):
        cx.apply_assignment(state_file, "apply", "a/b", "high", carril="mid")
    assert json.loads(state_file.read_text(encoding="utf-8")) == data


def test_unknown_entry_shape_refused(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["CodexPhaseModelAssignments"]["sdd_apply"] = {"unrelated": True}
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(cx.AdapterError, match="refusing to overwrite"):
        cx.apply_assignment(state_file, "apply", "a/b", "high")
    assert json.loads(state_file.read_text(encoding="utf-8")) == data


def test_unknown_carril_entry_shape_refused(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["CodexModelAssignments"]["sdd_mid"] = ["not", "an", "entry"]
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(cx.AdapterError, match="refusing to overwrite"):
        cx.apply_assignment(state_file, "apply", "a/b", "high", carril="mid")
    assert json.loads(state_file.read_text(encoding="utf-8")) == data


def test_invalid_effort_refused_with_vocabulary(state_file: Path) -> None:
    with pytest.raises(cx.AdapterError) as excinfo:
        cx.apply_assignment(state_file, "apply", "a/b", "super-high")
    for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        assert level in str(excinfo.value)


def test_empty_model_refused(state_file: Path) -> None:
    with pytest.raises(cx.AdapterError, match="empty"):
        cx.apply_assignment(state_file, "apply", "   ", "high")


# --------------------------------------------------------------------------- #
# Atomic write, backup, rollback, dry-run
# --------------------------------------------------------------------------- #


def test_atomic_write_no_tmp_left(state_file: Path) -> None:
    cx.apply_assignment(state_file, "apply", "a/b", "high")
    assert list(state_file.parent.glob("*.router-tmp-*")) == []


def test_rollback_restores_original(state_file: Path) -> None:
    original_text = state_file.read_text(encoding="utf-8")
    cx.apply_assignment(state_file, "tasks", "a/b", "high")
    restored = cx.rollback(state_file)
    assert restored is not None
    assert state_file.read_text(encoding="utf-8") == original_text
    assert cx.read_assignments(state_file)["phases"].get("sdd_tasks") is None


def test_rollback_without_backup(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    assert cx.rollback(path) is None


def test_backup_disabled_writes_without_backup(state_file: Path) -> None:
    result = cx.apply_assignment(state_file, "apply", "a/b", "high", backup=False)
    assert result.wrote
    assert result.backup_path is None
    assert list(state_file.parent.glob("*.router-backup-*")) == []


def test_dry_run_writes_nothing(state_file: Path) -> None:
    before = state_file.read_text(encoding="utf-8")
    result = cx.apply_assignment(state_file, "apply", "a/b", "high", dry_run=True)
    assert not result.wrote
    assert result.backup_path is None
    assert result.diff  # diff still shown
    assert state_file.read_text(encoding="utf-8") == before


def test_dry_run_carril_writes_nothing(state_file: Path) -> None:
    before = state_file.read_text(encoding="utf-8")
    result = cx.apply_assignment(
        state_file, "apply", "a/b", "high", carril="cheap", dry_run=True
    )
    assert not result.wrote
    assert "codex_carril:sdd_cheap" in result.reason_codes
    assert state_file.read_text(encoding="utf-8") == before


def test_indent_preserved(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"user_note": "x"}, indent=4) + "\n", encoding="utf-8")
    cx.apply_assignment(path, "apply", "a/b", "high")
    text = path.read_text(encoding="utf-8")
    assert '\n    "CodexPhaseModelAssignments"' in text  # 4-space indent kept


# --------------------------------------------------------------------------- #
# read_assignments / verify_assignment
# --------------------------------------------------------------------------- #


def test_read_assignments_returns_both_blocks(state_file: Path) -> None:
    assignments = cx.read_assignments(state_file)
    assert assignments["phases"] == {
        "sdd_spec": {"provider_id": "openai", "model_id": "gpt-5", "effort": "medium"},
    }
    assert assignments["carriles"] == {
        "sdd_strong": {
            "provider_id": "anthropic",
            "model_id": "claude-opus-4",
            "effort": "high",
        },
    }


def test_read_assignments_tolerates_junk(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["CodexPhaseModelAssignments"]["sdd_junk"] = {"unrelated": True}
    data["CodexPhaseModelAssignments"]["sdd_list"] = [1, 2]
    data["CodexModelAssignments"] = "broken"
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    assignments = cx.read_assignments(state_file)
    assert "sdd_junk" not in assignments["phases"]
    assert "sdd_list" not in assignments["phases"]
    assert assignments["carriles"] == {}


def test_read_assignments_missing_or_malformed(tmp_path: Path) -> None:
    assert cx.read_assignments(tmp_path / "missing.json") == {
        "phases": {},
        "carriles": {},
    }
    path = tmp_path / "state.json"
    path.write_text("{ broken", encoding="utf-8")
    assert cx.read_assignments(path) == {"phases": {}, "carriles": {}}


def test_verify_assignment_round_trip(state_file: Path) -> None:
    cx.apply_assignment(state_file, "tasks", "a/b", "max")
    ok, entry = cx.verify_assignment(state_file, "tasks", "a/b", "max")
    assert ok and entry is not None
    # Internal effort is compared AFTER the Codex mapping (max -> xhigh).
    ok, _ = cx.verify_assignment(state_file, "tasks", "a/b", "xhigh")
    assert ok
    ok, _ = cx.verify_assignment(state_file, "tasks", "a/other", "xhigh")
    assert not ok
    ok, entry = cx.verify_assignment(state_file, "archive", "a/b", "high")
    assert not ok and entry is None


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_cli_full_cycle(tmp_path: Path, state_file: Path) -> None:
    original = state_file.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "codex",
            "--phase", "tasks",
            "--model", "deepseek/deepseek-v4.1",
            "--effort", "max",
            "--carril", "mid",
            "--state-path", str(state_file),
            "--verify",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "verify OK" in result.output
    assert "sdd_tasks" in result.output
    assert "codex_effort_mapped:max->xhigh" in result.output
    assert "codex_carril:sdd_mid" in result.output
    # The unsupported-TOML note is printed on real writes.
    assert "config.toml" in result.output
    assert "intentionally" in result.output
    after = json.loads(state_file.read_text(encoding="utf-8"))
    assert after["CodexPhaseModelAssignments"]["sdd_tasks"]["effort"] == "xhigh"
    assert after["CodexModelAssignments"]["sdd_mid"]["model_id"] == "deepseek-v4.1"

    result = runner.invoke(
        app, ["integrate", "codex", "rollback", "--state-path", str(state_file)]
    )
    assert result.exit_code == 0, result.output
    assert state_file.read_text(encoding="utf-8") == original

    result = runner.invoke(
        app, ["integrate", "codex", "status", "--state-path", str(state_file)]
    )
    assert result.exit_code == 0, result.output
    assert "sdd_spec" in result.output
    assert "sdd_strong" in result.output


def test_cli_dry_run(state_file: Path) -> None:
    before = state_file.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "codex",
            "--phase", "apply", "--model", "a/b", "--effort", "minimal",
            "--state-path", str(state_file), "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "nothing written" in result.output
    assert "codex_effort_mapped:minimal->low" in result.output
    assert state_file.read_text(encoding="utf-8") == before


def test_cli_missing_file_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "codex",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--state-path", str(tmp_path / "missing.json"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_unknown_shape_exit_2(state_file: Path) -> None:
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["CodexPhaseModelAssignments"]["sdd_apply"] = {"unrelated": True}
    state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "codex",
            "--phase", "apply", "--model", "a/b", "--effort", "high",
            "--state-path", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output


def test_cli_invalid_effort_exit_2(state_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "codex",
            "--phase", "apply", "--model", "a/b", "--effort", "turbo",
            "--state-path", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "expected one of" in result.output


def test_cli_unknown_carril_exit_2(state_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "codex",
            "--phase", "apply", "--model", "a/b", "--effort", "high",
            "--carril", "platinum",
            "--state-path", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "unknown carril" in result.output


def test_cli_unknown_phase_exit_2(state_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "codex",
            "--phase", "bogus", "--model", "a/b", "--effort", "high",
            "--state-path", str(state_file),
        ],
    )
    assert result.exit_code == 2
    assert "unknown phase" in result.output


def test_cli_rollback_without_backup_exit_1(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["integrate", "codex", "rollback", "--state-path", str(path)]
    )
    assert result.exit_code == 1
    assert "no backup found" in result.output
