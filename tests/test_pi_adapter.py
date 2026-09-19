"""Pi (gentle-pi) adapter tests: path precedence, object write, upgrade, refusals.

All writes target tmp_path models.json files — the real
~/.pi/gentle-ai/models.json is never touched (same rule as the other
adapter tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration import pi_adapter as pi

runner = CliRunner()

# Synthetic models.json matching the documented shape (research doc §5):
# flat map <agentName> -> "provider/model" OR {"model", "thinking"}.
REAL_SHAPE_MODELS = {
    "default": "anthropic/claude-sonnet-4",
    "sdd-explore": "anthropic/claude-sonnet-4",
    "sdd-spec": {
        "model": "openai/gpt-5",
        "thinking": "medium",
    },
    "sdd-apply": {
        "model": "google/gemini-3-flash",
        "thinking": "low",
        "note": "keep me",
    },
}


@pytest.fixture
def models_file(tmp_path: Path) -> Path:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(REAL_SHAPE_MODELS, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Path resolution precedence
# --------------------------------------------------------------------------- #


def test_resolve_explicit_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit" / "models.json"
    env = {"GENTLE_PI_CONFIG_HOME": str(tmp_path / "env-home")}
    assert pi.resolve_models_path(explicit, env, tmp_path) == explicit


def test_resolve_env_config_home(tmp_path: Path) -> None:
    env = {"GENTLE_PI_CONFIG_HOME": str(tmp_path / "pi-home")}
    assert pi.resolve_models_path(None, env, tmp_path) == (
        tmp_path / "pi-home" / "gentle-ai" / "models.json"
    )


def test_resolve_default_home(tmp_path: Path) -> None:
    assert pi.resolve_models_path(None, {}, tmp_path) == (
        tmp_path / ".pi" / "gentle-ai" / "models.json"
    )
    # Missing env key entirely behaves like an empty env.
    assert pi.resolve_models_path(None, {}, tmp_path) == pi.resolve_models_path(
        None, {"OTHER": "x"}, tmp_path
    )


# --------------------------------------------------------------------------- #
# Writes: object form, bare-string upgrade, preservation
# --------------------------------------------------------------------------- #


def test_apply_writes_object_form(models_file: Path) -> None:
    result = pi.apply_assignment(models_file, "explore", "anthropic/claude-sonnet-4", "high")
    assert result.wrote
    assert result.agent_key == "sdd-explore"
    assert result.thinking == "high"
    assert result.backup_path is not None and result.backup_path.is_file()
    after = json.loads(models_file.read_text(encoding="utf-8"))
    assert after["sdd-explore"] == {
        "model": "anthropic/claude-sonnet-4",
        "thinking": "high",
    }
    # Existing entries survive verbatim (unknown keys preserved); sdd-explore
    # is the one intentionally-upgraded entry.
    for key, value in REAL_SHAPE_MODELS.items():
        if key != "sdd-explore":
            assert after[key] == value


def test_apply_overwrites_existing_object(models_file: Path) -> None:
    result = pi.apply_assignment(models_file, "spec", "openai/gpt-5-turbo", "high")
    assert result.wrote
    after = json.loads(models_file.read_text(encoding="utf-8"))
    assert after["sdd-spec"] == {"model": "openai/gpt-5-turbo", "thinking": "high"}
    # Untouched bare strings elsewhere stay bare strings.
    assert after["default"] == "anthropic/claude-sonnet-4"
    assert after["sdd-explore"] == "anthropic/claude-sonnet-4"


def test_apply_upgrades_bare_string_entry(models_file: Path) -> None:
    # "sdd-explore" is a bare string in the fixture; writing it must produce
    # the object form with the requested model (upgrade, not refusal), while
    # the unrelated "default" bare string stays untouched.
    result = pi.apply_assignment(models_file, "explore", "google/gemini-3-flash", "low")
    assert result.wrote
    after = json.loads(models_file.read_text(encoding="utf-8"))
    assert after["sdd-explore"] == {"model": "google/gemini-3-flash", "thinking": "low"}
    assert after["default"] == "anthropic/claude-sonnet-4"


def test_existing_object_extra_keys_preserved(models_file: Path) -> None:
    result = pi.apply_assignment(models_file, "apply", "google/gemini-3-pro", "high")
    assert result.wrote
    after = json.loads(models_file.read_text(encoding="utf-8"))
    assert after["sdd-apply"] == {
        "model": "google/gemini-3-pro",
        "thinking": "high",
        "note": "keep me",
    }


def test_create_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "models.json"
    result = pi.apply_assignment(
        path, "explore", "anthropic/claude-sonnet-4", "high", create=True
    )
    assert result.wrote
    assert result.backup_path is None  # nothing to back up
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "sdd-explore": {"model": "anthropic/claude-sonnet-4", "thinking": "high"}
    }


def test_noop_write_returns_wrote_false(models_file: Path) -> None:
    result = pi.apply_assignment(models_file, "spec", "openai/gpt-5", "medium")
    assert not result.wrote
    assert result.backup_path is None


# --------------------------------------------------------------------------- #
# Refusal paths (fail closed with AdapterError)
# --------------------------------------------------------------------------- #


def test_missing_file_refused(tmp_path: Path) -> None:
    with pytest.raises(pi.AdapterError, match="does not exist"):
        pi.apply_assignment(tmp_path / "models.json", "explore", "a/b", "high")


def test_malformed_json_refused(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(pi.AdapterError, match="malformed JSON"):
        pi.apply_assignment(path, "explore", "a/b", "high")


def test_non_dict_root_refused(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(pi.AdapterError, match="root is not an object"):
        pi.apply_assignment(path, "explore", "a/b", "high")


def test_unknown_entry_shape_refused(models_file: Path) -> None:
    data = json.loads(models_file.read_text(encoding="utf-8"))
    data["sdd-explore"] = {"unrelated": True}
    models_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(pi.AdapterError, match="refusing to overwrite"):
        pi.apply_assignment(models_file, "explore", "a/b", "high")
    # File untouched.
    assert json.loads(models_file.read_text(encoding="utf-8")) == data


def test_invalid_effort_refused_with_vocabulary(models_file: Path) -> None:
    with pytest.raises(pi.AdapterError) as excinfo:
        pi.apply_assignment(models_file, "explore", "a/b", "super-high")
    for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        assert level in str(excinfo.value)


def test_empty_model_refused(models_file: Path) -> None:
    with pytest.raises(pi.AdapterError, match="empty"):
        pi.apply_assignment(models_file, "explore", "   ", "high")


# --------------------------------------------------------------------------- #
# Atomic write, backup, rollback, dry-run
# --------------------------------------------------------------------------- #


def test_atomic_write_no_tmp_left(models_file: Path) -> None:
    pi.apply_assignment(models_file, "explore", "a/b", "high")
    assert list(models_file.parent.glob("*.router-tmp-*")) == []


def test_rollback_restores_original(models_file: Path) -> None:
    original_text = models_file.read_text(encoding="utf-8")
    pi.apply_assignment(models_file, "tasks", "a/b", "high")
    restored = pi.rollback(models_file)
    assert restored is not None
    assert models_file.read_text(encoding="utf-8") == original_text
    assert pi.read_assignments(models_file).get("sdd-tasks") is None


def test_rollback_without_backup(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{}", encoding="utf-8")
    assert pi.rollback(path) is None


def test_backup_disabled_writes_without_backup(models_file: Path) -> None:
    result = pi.apply_assignment(
        models_file, "explore", "a/b", "high", backup=False
    )
    assert result.wrote
    assert result.backup_path is None
    assert list(models_file.parent.glob("*.router-backup-*")) == []


def test_dry_run_writes_nothing(models_file: Path) -> None:
    before = models_file.read_text(encoding="utf-8")
    result = pi.apply_assignment(
        models_file, "explore", "a/b", "high", dry_run=True
    )
    assert not result.wrote
    assert result.backup_path is None
    assert result.diff  # diff still shown
    assert models_file.read_text(encoding="utf-8") == before


def test_indent_preserved(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"default": "a/b"}, indent=4) + "\n", encoding="utf-8")
    pi.apply_assignment(path, "explore", "a/b", "high")
    text = path.read_text(encoding="utf-8")
    assert '\n    "sdd-explore"' in text  # 4-space indent kept


# --------------------------------------------------------------------------- #
# read_assignments / verify_assignment
# --------------------------------------------------------------------------- #


def test_read_assignments_normalizes_bare_strings(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "default": "anthropic/claude-sonnet-4",
                "sdd-spec": {"model": "openai/gpt-5", "thinking": "medium"},
                "junk": {"unrelated": True},
                "count": 42,
            }
        ),
        encoding="utf-8",
    )
    assignments = pi.read_assignments(path)
    assert assignments == {
        "default": {"model": "anthropic/claude-sonnet-4", "thinking": None},
        "sdd-spec": {"model": "openai/gpt-5", "thinking": "medium"},
    }


def test_read_assignments_missing_or_malformed(tmp_path: Path) -> None:
    assert pi.read_assignments(tmp_path / "missing.json") == {}
    path = tmp_path / "models.json"
    path.write_text("{ broken", encoding="utf-8")
    assert pi.read_assignments(path) == {}


def test_verify_assignment_round_trip(models_file: Path) -> None:
    pi.apply_assignment(models_file, "tasks", "a/b", "high")
    ok, entry = pi.verify_assignment(models_file, "tasks", "a/b", "high")
    assert ok and entry is not None
    ok, _ = pi.verify_assignment(models_file, "tasks", "a/other", "high")
    assert not ok
    ok, entry = pi.verify_assignment(models_file, "archive", "a/b", "high")
    assert not ok and entry is None


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_cli_full_cycle(tmp_path: Path, models_file: Path) -> None:
    original = models_file.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "pi",
            "--phase", "explore",
            "--model", "anthropic/claude-sonnet-4",
            "--effort", "high",
            "--models", str(models_file),
            "--verify",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "verify OK" in result.output
    assert '"sdd-explore"' in result.output
    assert "gentle-pi" in result.output

    result = runner.invoke(
        app, ["integrate", "pi", "rollback", "--models", str(models_file)]
    )
    assert result.exit_code == 0, result.output
    assert models_file.read_text(encoding="utf-8") == original

    result = runner.invoke(
        app, ["integrate", "pi", "status", "--models", str(models_file)]
    )
    assert result.exit_code == 0, result.output
    assert "sdd-spec" in result.output


def test_cli_dry_run(models_file: Path) -> None:
    before = models_file.read_text(encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "pi",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--models", str(models_file), "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "nothing written" in result.output
    assert models_file.read_text(encoding="utf-8") == before


def test_cli_missing_file_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "pi",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--models", str(tmp_path / "missing.json"),
        ],
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_cli_unknown_shape_exit_2(models_file: Path) -> None:
    data = json.loads(models_file.read_text(encoding="utf-8"))
    data["sdd-explore"] = {"unrelated": True}
    models_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "pi",
            "--phase", "explore", "--model", "a/b", "--effort", "high",
            "--models", str(models_file),
        ],
    )
    assert result.exit_code == 2
    assert "refusing to overwrite" in result.output


def test_cli_invalid_effort_exit_2(models_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "pi",
            "--phase", "explore", "--model", "a/b", "--effort", "turbo",
            "--models", str(models_file),
        ],
    )
    assert result.exit_code == 2
    assert "expected one of" in result.output


def test_cli_unknown_phase_exit_2(models_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "pi",
            "--phase", "bogus", "--model", "a/b", "--effort", "high",
            "--models", str(models_file),
        ],
    )
    assert result.exit_code == 2
    assert "unknown phase" in result.output


def test_cli_rollback_without_backup_exit_1(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        app, ["integrate", "pi", "rollback", "--models", str(path)]
    )
    assert result.exit_code == 1
    assert "no backup found" in result.output
