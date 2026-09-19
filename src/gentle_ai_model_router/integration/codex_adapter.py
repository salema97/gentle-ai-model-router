"""Codex write adapter (state-file route): apply router decisions to the
Codex assignment blocks in ``~/.gentle-ai/state.json``.

Why the state file — and NOT ``~/.codex/*.config.toml`` (research doc §2.2/§6):

- Codex takes model+effort as plain argv on ``spawn_agent``
  (``codex/sdd-orchestrator.md:165-181``), driven by a per-phase table that
  ``gentle-ai sync`` REGENERATES from persisted state. An external writer can
  regenerate the same table inputs only via state, so the TOML route is the
  weakest out-of-process surface: hand-edited TOML profiles are overwritten on
  the next sync.
- Whole-session tier TOML profiles (``sdd-strong/sdd-mid/sdd-cheap``) do NOT
  apply to spawned sub-agents (``sdd-orchestrator.md:190``), so writing them
  would not even reach the phase agents. Direct TOML writing is therefore
  intentionally unsupported.
- Codex phase agents are addressed with UNDERSCORE transport identifiers
  (``spawn_agent(task_name="sdd_<phase>")``, research doc §8), so the state
  keys use ``sdd_<phase>`` — unlike the OpenCode/Pi ``sdd-<phase>`` keys.

Write targets (collector/local_discovery.py ``_STATE_ASSIGNMENT_KEYS``):

- ``CodexPhaseModelAssignments``: map ``sdd_<phase>`` → assignment dict
  (per-phase model+effort, what the orchestrator table is generated from).
- ``CodexModelAssignments``: carril (tier) assignments, written when
  ``carril`` is passed. The Codex preset matrix maps 3 carriles
  (``sdd-strong/sdd-mid/sdd-cheap``) → model + effort
  (``internal/model/codex_model.go:166-193``).

Effort vocabulary: Codex supports exactly ``low|medium|high|xhigh``
(``codex_model.go:136-141``). The internal taxonomy is a superset, so it is
mapped DOWN deterministically via ``CODEX_EFFORT_MAP`` — levels with no Codex
equivalent (``off``, ``minimal``) clamp to ``low`` and ``max`` clamps to
``xhigh``. We never invent levels. Every clamp is recorded as a
``codex_effort_mapped:<internal>-><codex>`` reason code (same spirit as the
opencode adapter's ``effort_clamped`` codes).

Safety model (mirrors the gentle-state/pi adapters): every write is atomic
(tmp file + ``os.replace``), a backup copy is made first
(``<path>.router-backup-<ts>``), and ``rollback`` restores the latest backup.
Everything outside the targeted assignment entries is preserved
byte-semantically (pure JSON round trip with the detected indent), and we
REFUSE to overwrite data we do not recognize — non-dict assignment blocks or
entries that do not look like a model assignment fail closed with
``AdapterError``.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gentle_ai_model_router.integration.gentle_state_adapter import (
    parse_model_spec,
    resolve_state_path,
    validate_effort,
)
from gentle_ai_model_router.integration.opencode_adapter import (
    AdapterError,
    latest_backup,
    rollback,
)

STATE_KEY_PHASE = "CodexPhaseModelAssignments"
STATE_KEY_CARRIL = "CodexModelAssignments"
CARRILES = ("strong", "mid", "cheap")
# Codex reasoning_effort vocabulary (codex_model.go:136-141): low|medium|high|xhigh.
# Total deterministic map from the internal taxonomy — missing levels clamp
# DOWN (never up, never invented).
CODEX_EFFORT_MAP = {
    "off": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
}

_INDENT_RE = re.compile(r'^(\s+)"')


@dataclass(frozen=True)
class CodexWriteResult:
    path: Path
    phase_key: str
    carril_key: str | None  # sdd_<carrile> when --carril was passed
    provider_id: str | None
    model_id: str
    effort: str  # internal taxonomy value (as requested)
    codex_effort: str  # mapped into the Codex vocabulary
    reason_codes: list[str]  # codex_effort_mapped:* / codex_carril:* codes
    diff: str
    backup_path: Path | None
    wrote: bool  # False in dry-run mode (or when nothing changed)


# --------------------------------------------------------------------------- #
# Effort mapping
# --------------------------------------------------------------------------- #


def to_codex_effort(effort: str) -> tuple[str, str | None]:
    """Map an internal effort level into the Codex vocabulary.

    Returns ``(codex_effort, reason_code_or_None)``. The reason code is
    ``codex_effort_mapped:<internal>-><codex>`` when the level had to be
    clamped, ``None`` when it survives the mapping unchanged.
    """
    internal = validate_effort(effort)
    codex = CODEX_EFFORT_MAP[internal]
    if codex == internal:
        return codex, None
    return codex, f"codex_effort_mapped:{internal}->{codex}"


# --------------------------------------------------------------------------- #
# Key + file handling
# --------------------------------------------------------------------------- #


def phase_key(phase: str) -> str:
    """Codex transport identifier: ``sdd_<phase>`` with underscores."""
    name = phase.removeprefix("sdd-").removeprefix("sdd_")
    return f"sdd_{name.replace('-', '_')}"


def carril_key(carril: str) -> str:
    """Carril block key: ``sdd_<carrile>`` for strong/mid/cheap."""
    name = carril.removeprefix("sdd-")
    if name not in CARRILES:
        raise AdapterError(
            f"unknown carril '{carril}' (expected one of: {', '.join(CARRILES)})"
        )
    return f"sdd_{name}"


def _detect_indent(text: str) -> int:
    """Indent width of the first indented property line (default 2)."""
    for line in text.splitlines():
        match = _INDENT_RE.match(line)
        if match and " " in match.group(1):
            return len(match.group(1))
    return 2


def _load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"malformed JSON in state file {path}: {exc}") from exc
    except OSError as exc:
        raise AdapterError(f"cannot read state file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterError(f"state file root is not an object: {path}")
    return data


def _assignment_block(data: dict[str, Any], block_key: str) -> dict[str, Any]:
    """Fetch (or create) an assignment block, refusing non-dict shapes."""
    block = data.get(block_key)
    if block is None:
        block = {}
        data[block_key] = block
    if not isinstance(block, dict):
        raise AdapterError(
            f"state key '{block_key}' exists but is not an object "
            f"(got {type(block).__name__}); refusing to overwrite"
        )
    return block


def _looks_like_assignment(value: Any) -> bool:
    """Tolerant shape check so we never clobber unrecognized user data."""
    if not isinstance(value, dict):
        return False
    if "model_id" in value:
        return isinstance(value["model_id"], (str, type(None)))
    # Compact alternative shape some surfaces use: {"model": "provider/model"}.
    if "model" in value:
        return isinstance(value["model"], str)
    return False


def _atomic_write(path: Path, text: str) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.router-tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _set_entry(block: dict[str, Any], entry_key: str, entry: dict[str, Any]) -> None:
    existing = block.get(entry_key)
    if existing is not None and not _looks_like_assignment(existing):
        raise AdapterError(
            f"refusing to overwrite '{entry_key}': existing value does not "
            f"look like a model assignment ({existing!r}); move it aside "
            f"manually to proceed"
        )
    merged: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    merged.update(entry)
    block[entry_key] = merged


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def apply_assignment(
    state_path: str | Path | None,
    phase: str,
    model: str,
    effort: str,
    *,
    carril: str | None = None,
    create: bool = False,
    dry_run: bool = False,
    backup: bool = True,
) -> CodexWriteResult:
    """Set ``CodexPhaseModelAssignments["sdd_<phase>"]`` in the state file.

    When ``carril`` is passed (strong/mid/cheap), the same assignment is also
    written to ``CodexModelAssignments["sdd_<carrile>"]`` in the same atomic
    write. Effort is validated against the internal taxonomy, then mapped into
    the Codex vocabulary (``low|medium|high|xhigh``); clamps are recorded as
    ``codex_effort_mapped:*`` reason codes. Everything outside the targeted
    entries is preserved byte-semantically. Refuses on a missing file (unless
    ``create``), malformed JSON, non-dict assignment blocks, or pre-existing
    entries whose shape we do not recognize.
    """
    path = resolve_state_path(state_path)
    key = phase_key(phase)
    ckey = carril_key(carril) if carril is not None else None
    provider_id, model_id = parse_model_spec(model)
    codex_effort, reason = to_codex_effort(effort)
    reason_codes = [reason] if reason else []
    if ckey is not None:
        reason_codes.append(f"codex_carril:{ckey}")

    if not path.exists():
        if not create:
            raise AdapterError(
                f"state file does not exist: {path} (pass --create to initialize it)"
            )
        original = ""
        data: dict[str, Any] = {}
    else:
        original = path.read_text(encoding="utf-8")
        data = _load_state(path)

    entry: dict[str, Any] = {
        "provider_id": provider_id,
        "model_id": model_id,
        "effort": codex_effort,
    }
    block = _assignment_block(data, STATE_KEY_PHASE)
    _set_entry(block, key, entry)
    if ckey is not None:
        carril_block = _assignment_block(data, STATE_KEY_CARRIL)
        _set_entry(carril_block, ckey, entry)

    indent = _detect_indent(original) if original else 2
    trailing = "\n" if not original or original.endswith("\n") else ""
    new_text = json.dumps(data, indent=indent, ensure_ascii=False) + trailing

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    if dry_run or new_text == original:
        return CodexWriteResult(
            path=path,
            phase_key=key,
            carril_key=ckey,
            provider_id=provider_id,
            model_id=model_id,
            effort=validate_effort(effort),
            codex_effort=codex_effort,
            reason_codes=reason_codes,
            diff=diff,
            backup_path=None,
            wrote=False,
        )

    backup_path: Path | None = None
    if backup and original:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")
        backup_path = path.with_name(f"{path.name}.router-backup-{ts}")
        shutil.copy2(path, backup_path)
    _atomic_write(path, new_text)
    return CodexWriteResult(
        path=path,
        phase_key=key,
        carril_key=ckey,
        provider_id=provider_id,
        model_id=model_id,
        effort=validate_effort(effort),
        codex_effort=codex_effort,
        reason_codes=reason_codes,
        diff=diff,
        backup_path=backup_path,
        wrote=True,
    )


def read_assignments(state_path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Current Codex assignment blocks from the state file (tolerant read).

    Returns ``{"phases": {...}, "carriles": {...}}`` keyed by the transport
    identifiers (``sdd_<phase>`` / ``sdd_<carrile>``). Non-dict blocks and
    unrecognized entry shapes are skipped, never crashed on. Missing or
    malformed files yield empty maps.
    """
    path = resolve_state_path(state_path)
    out: dict[str, dict[str, Any]] = {"phases": {}, "carriles": {}}
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    if not isinstance(data, dict):
        return out
    for block_key, dest in (
        (STATE_KEY_PHASE, "phases"),
        (STATE_KEY_CARRIL, "carriles"),
    ):
        block = data.get(block_key)
        if not isinstance(block, dict):
            continue
        for entry_key, value in block.items():
            if _looks_like_assignment(value):
                out[dest][str(entry_key)] = dict(value)
    return out


def verify_assignment(
    state_path: str | Path | None,
    phase: str,
    model: str,
    effort: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Round-trip check: re-read the state file and compare the assignment.

    Returns (ok, observed_entry_or_None). ``ok`` is True only when the stored
    provider_id/model_id/effort all match the requested spec, with the effort
    compared AFTER the Codex mapping.
    """
    provider_id, model_id = parse_model_spec(model)
    codex_effort, _ = to_codex_effort(effort)
    entry = read_assignments(state_path)["phases"].get(phase_key(phase))
    if entry is None:
        return False, None
    ok = (
        entry.get("provider_id") == provider_id
        and entry.get("model_id") == model_id
        and entry.get("effort") == codex_effort
    )
    return ok, entry


__all__ = [
    "AdapterError",
    "CARRILES",
    "CODEX_EFFORT_MAP",
    "CodexWriteResult",
    "STATE_KEY_CARRIL",
    "STATE_KEY_PHASE",
    "apply_assignment",
    "carril_key",
    "latest_backup",
    "phase_key",
    "read_assignments",
    "resolve_state_path",
    "rollback",
    "to_codex_effort",
    "validate_effort",
    "verify_assignment",
]
