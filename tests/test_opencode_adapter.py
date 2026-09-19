"""OpenCode write adapter tests: JSON + JSONC round-trips, clamping, rollback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration import opencode_adapter as oa

runner = CliRunner()

PURE_JSON = """{
  "theme": "dark",
  "agent": {
    "sdd-explore": {
      "model": "anthropic/old-model",
      "variant": "high",
      "custom_key": "keep-me"
    },
    "sdd-spec": {
      "model": "openai/gpt-5",
      "variant": "medium"
    }
  },
  "other": {"nested": [1, 2, 3]}
}
"""

JSONC = """{
  // global theme, managed by hand
  "theme": "dark",
  "agent": {
    // explore runs cheap
    "sdd-explore": {
      "model": "anthropic/old-model", // pinned on purpose
      "variant": "high",
    },
  },
}
"""


@pytest.fixture
def pure_json_config(tmp_path: Path) -> Path:
    path = tmp_path / "opencode.json"
    path.write_text(PURE_JSON, encoding="utf-8")
    return path


@pytest.fixture
def jsonc_config(tmp_path: Path) -> Path:
    path = tmp_path / "opencode.jsonc"
    path.write_text(JSONC, encoding="utf-8")
    return path


def test_pure_json_round_trip(pure_json_config: Path) -> None:
    result = oa.apply_decision(
        pure_json_config, "explore", "anthropic/claude-sonnet-4", "low", None
    )
    assert result.wrote
    assert result.backup_path is not None and result.backup_path.exists()
    data = json.loads(pure_json_config.read_text(encoding="utf-8"))
    assert data["agent"]["sdd-explore"]["model"] == "anthropic/claude-sonnet-4"
    # No variants cache -> no variant written, existing variant removed.
    assert "variant" not in data["agent"]["sdd-explore"]
    # Unknown keys and sibling agents survive.
    assert data["agent"]["sdd-explore"]["custom_key"] == "keep-me"
    assert data["agent"]["sdd-spec"]["model"] == "openai/gpt-5"
    assert data["theme"] == "dark"
    assert data["other"]["nested"] == [1, 2, 3]
    assert "variants_unknown:no_variant_written" in result.reason_codes


def test_jsonc_comments_outside_block_survive(jsonc_config: Path) -> None:
    result = oa.apply_decision(
        jsonc_config,
        "explore",
        "anthropic/claude-sonnet-4",
        "low",
        {("anthropic", "claude-sonnet-4"): ["low"]},
    )
    assert result.wrote
    text = jsonc_config.read_text(encoding="utf-8")
    assert "// global theme, managed by hand" in text
    assert "// explore runs cheap" in text
    data = json.loads(oa._strip_jsonc(text))
    assert data["agent"]["sdd-explore"]["model"] == "anthropic/claude-sonnet-4"
    assert data["agent"]["sdd-explore"]["variant"] == "low"
    assert data["theme"] == "dark"


def test_single_line_agent_block_jsonc(tmp_path: Path) -> None:
    """One-line agent objects (legal JSONC) are expanded and patched."""
    path = tmp_path / "opencode.jsonc"
    path.write_text(
        '{\n  // my hand-written config\n  "agent": {\n'
        '    "sdd-explore": { "model": "anthropic/old", "variant": "max" },\n'
        '  },\n}\n',
        encoding="utf-8",
    )
    result = oa.apply_decision(
        path,
        "explore",
        "anthropic/claude-sonnet-4",
        "low",
        {("anthropic", "claude-sonnet-4"): ["low"]},
    )
    assert result.wrote
    text = path.read_text(encoding="utf-8")
    assert "// my hand-written config" in text
    data = json.loads(oa._strip_jsonc(text))
    assert data["agent"]["sdd-explore"]["model"] == "anthropic/claude-sonnet-4"
    assert data["agent"]["sdd-explore"]["variant"] == "low"


def test_effort_clamped_down_to_supported_level(pure_json_config: Path) -> None:
    variants = {("anthropic", "claude-sonnet-4"): ["low", "medium"]}
    result = oa.apply_decision(
        pure_json_config, "spec", "anthropic/claude-sonnet-4", "high", variants
    )
    assert result.effort == "medium"
    assert "effort_clamped:high->medium" in result.reason_codes
    data = json.loads(pure_json_config.read_text(encoding="utf-8"))
    assert data["agent"]["sdd-spec"]["variant"] == "medium"


def test_effort_never_clamped_up(pure_json_config: Path) -> None:
    """Requesting low when only high is supported writes no variant, not high."""
    variants = {("anthropic", "claude-sonnet-4"): ["high"]}
    result = oa.apply_decision(
        pure_json_config, "spec", "anthropic/claude-sonnet-4", "low", variants
    )
    assert result.effort is None
    assert "effort_clamped:low->none" in result.reason_codes


def test_unknown_effort_rejected(pure_json_config: Path) -> None:
    with pytest.raises(oa.AdapterError, match="unknown effort"):
        oa.apply_decision(pure_json_config, "spec", "anthropic/x", "turbo", None)


def test_unknown_model_format_rejected(pure_json_config: Path) -> None:
    with pytest.raises(oa.AdapterError, match="provider/model"):
        oa.apply_decision(pure_json_config, "spec", "no-provider", "low", None)


def test_dry_run_writes_nothing(pure_json_config: Path) -> None:
    before = pure_json_config.read_text(encoding="utf-8")
    result = oa.apply_decision(
        pure_json_config, "explore", "anthropic/new", "low", None, dry_run=True
    )
    assert not result.wrote
    assert result.backup_path is None
    assert pure_json_config.read_text(encoding="utf-8") == before
    assert "anthropic/new" in result.diff


def test_rollback_restores_latest_backup(pure_json_config: Path) -> None:
    original = pure_json_config.read_text(encoding="utf-8")
    oa.apply_decision(pure_json_config, "explore", "anthropic/first", "low", None)
    oa.apply_decision(pure_json_config, "explore", "anthropic/second", "low", None)
    restored = oa.rollback(pure_json_config)
    assert restored is not None
    # The latest backup was taken before the SECOND write.
    data = json.loads(pure_json_config.read_text(encoding="utf-8"))
    assert data["agent"]["sdd-explore"]["model"] == "anthropic/first"
    # The earliest backup holds the pristine original.
    backups = sorted(pure_json_config.parent.glob("opencode.json.router-backup-*"))
    assert backups[0].read_text(encoding="utf-8") == original


def test_rollback_without_backup_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    path.write_text("{}", encoding="utf-8")
    assert oa.rollback(path) is None


def test_managed_by_block_never_touched(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    managed = """{
  "agent": {
    "sdd-apply": {
      "__managed_by": "gentle-ai/sdd",
      "model": "anthropic/managed-model",
      "variant": "max"
    },
    "sdd-explore": {
      "model": "anthropic/free-model"
    }
  }
}
"""
    path.write_text(managed, encoding="utf-8")
    # Writing a DIFFERENT phase leaves the managed block byte-identical.
    oa.apply_decision(path, "explore", "anthropic/new", "low", None)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["agent"]["sdd-apply"] == {
        "__managed_by": "gentle-ai/sdd",
        "model": "anthropic/managed-model",
        "variant": "max",
    }
    # Writing the managed phase itself is refused.
    with pytest.raises(oa.AdapterError, match="__managed_by"):
        oa.apply_decision(path, "apply", "anthropic/new", "low", None)


def test_refuses_missing_path_without_create(tmp_path: Path) -> None:
    missing = tmp_path / "nope" / "opencode.json"
    with pytest.raises(oa.AdapterError, match="does not exist"):
        oa.apply_decision(missing, "explore", "anthropic/x", "low", None)
    result = oa.apply_decision(missing, "explore", "anthropic/x", "low", None, create=True)
    assert result.wrote
    data = json.loads(missing.read_text(encoding="utf-8"))
    assert data["agent"]["sdd-explore"]["model"] == "anthropic/x"


def test_insert_new_agent_into_existing_jsonc(tmp_path: Path) -> None:
    path = tmp_path / "opencode.jsonc"
    path.write_text(
        '{\n  // only spec configured so far\n  "agent": {\n'
        '    "sdd-spec": {\n      "model": "openai/gpt-5",\n    },\n'
        '  },\n}\n',
        encoding="utf-8",
    )
    result = oa.apply_decision(
        path, "verify", "anthropic/claude-sonnet-4", "high",
        {("anthropic", "claude-sonnet-4"): ["high"]},
    )
    assert result.wrote
    data = json.loads(oa._strip_jsonc(path.read_text(encoding="utf-8")))
    assert data["agent"]["sdd-verify"]["model"] == "anthropic/claude-sonnet-4"
    assert data["agent"]["sdd-verify"]["variant"] == "high"
    assert data["agent"]["sdd-spec"]["model"] == "openai/gpt-5"
    assert "// only spec configured so far" in path.read_text(encoding="utf-8")


def test_xdg_resolution(tmp_path: Path) -> None:
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    target = xdg / "opencode" / "opencode.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    resolved = oa.resolve_global_config(env={"XDG_CONFIG_HOME": str(xdg)}, home=home)
    assert resolved == target
    # Without XDG, falls back to ~/.config/opencode.
    resolved = oa.resolve_global_config(env={}, home=home)
    assert resolved == home / ".config" / "opencode" / "opencode.json"


def test_opencode_config_dir_displaces_global(tmp_path: Path) -> None:
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    displaced = xdg / "opencode" / "opencode.json"
    displaced.parent.mkdir(parents=True)
    displaced.write_text("{}", encoding="utf-8")
    custom = tmp_path / "custom"
    custom_target = custom / "opencode.jsonc"
    custom.mkdir()
    custom_target.write_text("{}", encoding="utf-8")
    resolved = oa.resolve_global_config(
        env={"XDG_CONFIG_HOME": str(xdg), "OPENCODE_CONFIG_DIR": str(custom)}, home=home
    )
    assert resolved == custom_target  # JSONC beats JSON in the custom dir too


def test_explicit_config_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.json"
    explicit.write_text("{}", encoding="utf-8")
    resolved = oa.resolve_global_config(
        explicit=explicit, env={"OPENCODE_CONFIG_DIR": "/does/not/matter"}, home=tmp_path
    )
    assert resolved == explicit


def test_load_variants_cache_merges_v1_and_v2(tmp_path: Path) -> None:
    v1 = tmp_path / "model-variants.json"
    v1.write_text(
        json.dumps({"anthropic": {"a": ["low", "high"]}, "openai": {"b": ["medium"]}}),
        encoding="utf-8",
    )
    v2dir = tmp_path / "opencode-v2"
    v2dir.mkdir()
    (v2dir / "aaa.json").write_text(
        json.dumps({"anthropic": {"a": ["medium"]}, "local": {"c": ["off", "low"]}}),
        encoding="utf-8",
    )
    (v2dir / "bbb.json").write_text("{not json", encoding="utf-8")  # skipped gracefully
    variants = oa.load_variants_cache(v1, v2dir)
    assert variants[("anthropic", "a")] == ["low", "high", "medium"]  # merged, deduped
    assert variants[("openai", "b")] == ["medium"]
    assert variants[("local", "c")] == ["off", "low"]


def test_cli_integrate_dry_run(tmp_path: Path) -> None:
    config_path = tmp_path / "opencode.json"
    config_path.write_text('{"agent": {}}', encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "integrate", "opencode",
            "--phase", "explore",
            "--model", "anthropic/claude-sonnet-4",
            "--effort", "low",
            "--config-target", str(config_path),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"agent": {}}


def test_cli_integrate_unknown_phase_exit_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "integrate", "opencode",
            "--phase", "bogus",
            "--model", "anthropic/x",
            "--effort", "low",
        ],
    )
    assert result.exit_code == 2


def test_cli_integrate_status(tmp_path: Path) -> None:
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        '{"agent": {"sdd-explore": {"model": "anthropic/x", "variant": "low"}}}',
        encoding="utf-8",
    )
    result = runner.invoke(app, ["integrate", "status", "--config-target", str(config_path)])
    assert result.exit_code == 0
    assert "sdd-explore" in result.output
    assert "anthropic/x" in result.output
