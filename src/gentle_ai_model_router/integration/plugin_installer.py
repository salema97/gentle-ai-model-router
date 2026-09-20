"""Plugin installer for OpenCode and Pi telemetry runtime hooks.

Installs and manages zero-dependency JS/TS telemetry plugins:
- OpenCode: ``router_telemetry.js`` + ``router_telemetry.d.ts``
- Pi: ``router_telemetry.js`` + ``router_telemetry.d.ts``

Guarantees:
- Backup-first: existing files backed up to ``.<filename>.router-backup-<ts>``
- Atomic writes: temporary file + ``os.replace``
- Dry-run support: inspect diff/actions without touching disk
- Idempotent: unchanged files are not re-written or re-backed-up
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PLUGIN_SRC_DIR = Path(__file__).resolve().parent.parent / "plugins"
OPENCODE_SRC_DIR = PLUGIN_SRC_DIR / "opencode"
PI_SRC_DIR = PLUGIN_SRC_DIR / "pi"



@dataclass(frozen=True)
class InstalledFileResult:
    """Outcome for a single installed plugin file."""

    source_path: Path
    target_path: Path
    backup_path: Path | None
    status: str  # "created", "updated", "unchanged", "dry-run"
    wrote: bool


@dataclass(frozen=True)
class PluginInstallResult:
    """Outcome for a plugin suite installation."""

    plugin_name: str
    target_dir: Path
    files: tuple[InstalledFileResult, ...]
    installed: bool


@dataclass(frozen=True)
class PluginStatusInfo:
    """Installation status for a plugin."""

    name: str
    target_dir: Path
    installed: bool
    files_present: tuple[str, ...]
    files_missing: tuple[str, ...]
    backup_count: int
    router_version: str | None


def resolve_opencode_plugin_dir(
    explicit: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve destination directory for the OpenCode telemetry plugin."""
    if explicit is not None:
        return Path(explicit).expanduser()
    environ = os.environ if env is None else env
    home_dir = Path.home() if home is None else home

    if environ.get("OPENCODE_CONFIG_DIR"):
        return Path(environ["OPENCODE_CONFIG_DIR"]).expanduser() / "plugins"

    xdg = environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg).expanduser() / "opencode" / "plugins"

    return home_dir / ".config" / "opencode" / "plugins"


def resolve_pi_plugin_dir(
    explicit: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve destination directory for the Pi telemetry plugin."""
    if explicit is not None:
        return Path(explicit).expanduser()
    environ = os.environ if env is None else env
    home_dir = Path.home() if home is None else home

    pi_home = environ.get("GENTLE_PI_CONFIG_HOME")
    if pi_home:
        return Path(pi_home).expanduser() / "plugins"

    return home_dir / ".pi" / "plugins"


def _atomic_write(target_path: Path, content: str) -> None:
    """Atomically write text content to target_path."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.router-tmp-{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, target_path)


def _install_file(
    source_path: Path,
    target_path: Path,
    dry_run: bool = False,
    endpoint_url: str | None = None,
) -> InstalledFileResult:
    """Install a single file with backup-first and atomic write semantics."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Plugin source file not found: {source_path}")

    new_content = source_path.read_text(encoding="utf-8")
    if endpoint_url and source_path.name in ("router_telemetry.js", "router_telemetry.ts"):
        new_content = re.sub(
            r"const DEFAULT_ENDPOINT = ['\"][^'\"]+['\"];",
            f"const DEFAULT_ENDPOINT = '{endpoint_url}';",
            new_content,
        )

    backup_path: Path | None = None

    if target_path.is_file():
        existing_content = target_path.read_text(encoding="utf-8")
        if existing_content == new_content:
            return InstalledFileResult(
                source_path=source_path,
                target_path=target_path,
                backup_path=None,
                status="unchanged",
                wrote=False,
            )

        if not dry_run:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
            backup_path = target_path.with_name(f".{target_path.name}.router-backup-{ts}")
            shutil.copy2(target_path, backup_path)
            _atomic_write(target_path, new_content)

        return InstalledFileResult(
            source_path=source_path,
            target_path=target_path,
            backup_path=backup_path,
            status="dry-run" if dry_run else "updated",
            wrote=not dry_run,
        )

    # File does not exist yet
    if not dry_run:
        _atomic_write(target_path, new_content)

    return InstalledFileResult(
        source_path=source_path,
        target_path=target_path,
        backup_path=None,
        status="dry-run" if dry_run else "created",
        wrote=not dry_run,
    )


def install_opencode_plugin(
    target_dir: str | Path | None = None,
    dry_run: bool = False,
    endpoint_url: str | None = None,
) -> PluginInstallResult:
    """Install the OpenCode telemetry runtime hook plugin files."""
    dest = resolve_opencode_plugin_dir(target_dir)
    file_names = ("router_telemetry.js", "router_telemetry.d.ts", "router_telemetry.ts")
    results: list[InstalledFileResult] = []

    for name in file_names:
        src = OPENCODE_SRC_DIR / name
        tgt = dest / name
        res = _install_file(src, tgt, dry_run=dry_run, endpoint_url=endpoint_url)
        results.append(res)

    installed = any(r.wrote for r in results) or all(r.status == "unchanged" for r in results)
    return PluginInstallResult(
        plugin_name="opencode",
        target_dir=dest,
        files=tuple(results),
        installed=installed,
    )


def install_pi_plugin(
    target_dir: str | Path | None = None,
    dry_run: bool = False,
    endpoint_url: str | None = None,
) -> PluginInstallResult:
    """Install the Pi telemetry runtime hook plugin files."""
    dest = resolve_pi_plugin_dir(target_dir)
    file_names = ("router_telemetry.js", "router_telemetry.d.ts")
    results: list[InstalledFileResult] = []

    for name in file_names:
        src = PI_SRC_DIR / name
        tgt = dest / name
        res = _install_file(src, tgt, dry_run=dry_run, endpoint_url=endpoint_url)
        results.append(res)

    installed = any(r.wrote for r in results) or all(r.status == "unchanged" for r in results)
    return PluginInstallResult(
        plugin_name="pi",
        target_dir=dest,
        files=tuple(results),
        installed=installed,
    )


def _check_plugin_status(name: str, target_dir: Path, expected_version: str) -> PluginStatusInfo:
    file_names = (
        ("router_telemetry.js", "router_telemetry.d.ts", "router_telemetry.ts")
        if name == "opencode"
        else ("router_telemetry.js", "router_telemetry.d.ts")
    )
    present: list[str] = []
    missing: list[str] = []
    backups: list[Path] = []

    for fname in file_names:
        fpath = target_dir / fname
        if fpath.is_file():
            present.append(fname)
        else:
            missing.append(fname)
        backups.extend(target_dir.glob(f".{fname}.router-backup-*"))

    is_installed = "router_telemetry.js" in present
    return PluginStatusInfo(
        name=name,
        target_dir=target_dir,
        installed=is_installed,
        files_present=tuple(present),
        files_missing=tuple(missing),
        backup_count=len(backups),
        router_version=expected_version if is_installed else None,
    )


def plugin_status(
    opencode_dir: str | Path | None = None,
    pi_dir: str | Path | None = None,
) -> dict[str, PluginStatusInfo]:
    """Inspect installation status for OpenCode and Pi telemetry hook plugins."""
    opencode_dest = resolve_opencode_plugin_dir(opencode_dir)
    pi_dest = resolve_pi_plugin_dir(pi_dir)

    return {
        "opencode": _check_plugin_status("opencode", opencode_dest, "opencode-hook-v1"),
        "pi": _check_plugin_status("pi", pi_dest, "pi-hook-v1"),
    }
