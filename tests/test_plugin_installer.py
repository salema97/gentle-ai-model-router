"""Tests for plugin installer module and CLI commands (router integrate plugins).

Validates:
- install_opencode_plugin and install_pi_plugin with atomic writes.
- Backup-first mechanics (.<file>.router-backup-<ts>) when overwriting existing files.
- --dry-run mechanics without touching disk.
- Idempotent re-installation when files are identical.
- plugin_status inspection.
- CLI commands: router integrate plugins install / status.
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gentle_ai_model_router.cli.main import app
from gentle_ai_model_router.integration.plugin_installer import (
    install_opencode_plugin,
    install_pi_plugin,
    plugin_status,
)

runner = CliRunner()


def test_install_opencode_plugin_fresh_and_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "opencode_plugins"
    res1 = install_opencode_plugin(target_dir=target, dry_run=False)

    assert res1.installed is True
    assert (target / "router_telemetry.js").is_file()
    assert (target / "router_telemetry.d.ts").is_file()
    assert len(res1.files) == 2
    assert all(f.status == "created" for f in res1.files)

    # Idempotent second run
    res2 = install_opencode_plugin(target_dir=target, dry_run=False)
    assert all(f.status == "unchanged" for f in res2.files)
    assert not any(f.wrote for f in res2.files)


def test_install_pi_plugin_fresh_and_backup_on_update(tmp_path: Path) -> None:
    target = tmp_path / "pi_plugins"
    res1 = install_pi_plugin(target_dir=target, dry_run=False)
    assert res1.installed is True

    js_file = target / "router_telemetry.js"
    assert js_file.is_file()

    # Modify the installed file to trigger backup-first update
    js_file.write_text("// Modified locally", encoding="utf-8")

    res2 = install_pi_plugin(target_dir=target, dry_run=False)
    js_result = next(f for f in res2.files if f.target_path.name == "router_telemetry.js")
    assert js_result.status == "updated"
    assert js_result.backup_path is not None
    assert js_result.backup_path.is_file()
    assert js_result.backup_path.name.startswith(".router_telemetry.js.router-backup-")
    assert js_result.backup_path.read_text(encoding="utf-8") == "// Modified locally"


def test_install_plugin_dry_run(tmp_path: Path) -> None:
    target = tmp_path / "dry_run_plugins"
    res = install_opencode_plugin(target_dir=target, dry_run=True)

    assert not target.exists()
    assert all(f.status == "dry-run" for f in res.files)
    assert not any(f.wrote for f in res.files)


def test_plugin_status_inspection(tmp_path: Path) -> None:
    opencode_dir = tmp_path / "opencode"
    pi_dir = tmp_path / "pi"

    st_before = plugin_status(opencode_dir=opencode_dir, pi_dir=pi_dir)
    assert st_before["opencode"].installed is False
    assert st_before["pi"].installed is False

    # Install OpenCode only
    install_opencode_plugin(target_dir=opencode_dir)

    st_after = plugin_status(opencode_dir=opencode_dir, pi_dir=pi_dir)
    assert st_after["opencode"].installed is True
    assert "router_telemetry.js" in st_after["opencode"].files_present
    assert st_after["opencode"].router_version == "opencode-hook-v1"
    assert st_after["pi"].installed is False


def test_cli_integrate_plugins_install(tmp_path: Path) -> None:
    target = tmp_path / "cli_plugins"
    result = runner.invoke(
        app,
        ["integrate", "plugins", "install", "--opencode", "--target-dir", str(target)],
    )
    assert result.exit_code == 0
    assert "opencode plugin" in result.stdout
    assert (target / "router_telemetry.js").is_file()


def test_cli_integrate_plugins_status(tmp_path: Path) -> None:
    opencode_dir = tmp_path / "cli_opencode"
    pi_dir = tmp_path / "cli_pi"
    install_pi_plugin(target_dir=pi_dir)

    result = runner.invoke(
        app,
        [
            "integrate",
            "plugins",
            "status",
            "--opencode-dir",
            str(opencode_dir),
            "--pi-dir",
            str(pi_dir),
        ],
    )
    assert result.exit_code == 0
    assert "Runtime hook plugins status" in result.stdout
    assert "pi" in result.stdout
    assert "installed" in result.stdout


def test_install_plugin_with_custom_endpoint(tmp_path: Path) -> None:
    target = tmp_path / "custom_endpoint_plugins"
    res = install_opencode_plugin(
        target_dir=target,
        endpoint_url="https://router.example.com/shim/execution",
    )
    assert res.installed is True
    js_content = (target / "router_telemetry.js").read_text(encoding="utf-8")
    assert "const DEFAULT_ENDPOINT = 'https://router.example.com/shim/execution';" in js_content


def test_cli_setup_dry_run() -> None:
    result = runner.invoke(app, ["setup", "--dry-run"])
    assert result.exit_code == 0
    assert "router setup: zero-touch onboarding" in result.stdout
    assert "dry-run" in result.stdout


def test_cli_setup_execution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENCODE_CONFIG_DIR", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "setup",
            "--no-pi",
            "--data-dir",
            str(tmp_path),
            "--endpoint",
            "https://test.salema.dev/shim/execution",
        ],
    )
    assert result.exit_code == 0
    assert "Setup complete" in result.stdout
    assert "synchronized" in result.stdout

