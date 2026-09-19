"""Gentle AI state write adapter: apply router decisions to ``~/.gentle-ai/state.json``.

Verified write-path facts (docs/telemetry-probe.md §2.4, gentle-ai
``internal/state/state.go:59-97``):

- State file: ``~/.gentle-ai/state.json`` — pure JSON written by the Go side
  (atomic tmp+rename), NOT JSONC: a normal ``json.load``/``json.dump`` round
  trip is the correct strategy (unlike the OpenCode adapter's line patching).
- OpenCode model assignments persist in ``model_assignments``: map
  ``agent-name → {provider_id, model_id, effort}`` (snake_case json tags).
  The Go field is ``ModelAssignments``; we accept BOTH casings on read and
  write into whichever casing already exists (snake_case when creating new).
- ``gentle-ai sync`` REGENERATES opencode agent configs from this state file,
  so this is the correct write surface for Gentle-AI-managed setups — unlike
  ``opencode.json`` blocks carrying ``__managed_by: gentle-ai/sdd``, which the
  opencode adapter (correctly) refuses to touch.
- Compatible profile strategy for external tooling:
  ``gentle-ai sync --sdd-profile-strategy external-single-active``
  (docs/opencode-profiles.md, cited in gentle-ai-integration-research.md §6.1).

Safety model (mirrors the opencode adapter): every write is atomic (tmp file +
``os.replace``), a backup copy is made first (``<path>.router-backup-<ts>``),
and ``rollback`` restores the latest backup. We never touch any key other than
the single assignment entry, and we REFUSE to overwrite data we do not
recognize (non-dict entries, or dicts that do not look like a model
assignment) — defensive, user data is never destroyed silently.

Model spec convention: ``provider/model`` splits on the FIRST ``/``
(``a/b/c`` → provider ``a``, model ``b/c``). A spec without ``/`` is accepted
with ``provider_id = null`` and the whole string as ``model_id`` — Gentle AI
treats a bare model id as provider-default. Effort is validated against the
internal taxonomy in ``registry/normalize.py`` (Effort enum).
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

from gentle_ai_model_router.integration.opencode_adapter import (
    AdapterError,
    latest_backup,
    rollback,
)
from gentle_ai_model_router.registry.normalize import Effort

STATE_KEY_SNAKE = "model_assignments"
STATE_KEY_CAMEL = "ModelAssignments"
DEFAULT_STATE_PATH = "~/.gentle-ai/state.json"
ASSIGNMENT_KEYS = ("provider_id", "model_id", "effort")

_INDENT_RE = re.compile(r"^(\s+)\"")


@dataclass(frozen=True)
class StateWriteResult:
    path: Path
    agent_key: str
    provider_id: str | None
    model_id: str
    effort: str
    key_used: str  # STATE_KEY_SNAKE | STATE_KEY_CAMEL (casing found in file)
    diff: str
    backup_path: Path | None
    wrote: bool  # False in dry-run mode


# --------------------------------------------------------------------------- #
# Model spec + effort validation
# --------------------------------------------------------------------------- #


def parse_model_spec(spec: str) -> tuple[str | None, str]:
    """Split ``provider/model`` on the FIRST ``/``.

    A spec without ``/`` yields ``(None, spec)`` — Gentle AI resolves a bare
    model id against the provider default.
    """
    spec = spec.strip()
    if not spec:
        raise AdapterError("model spec is empty")
    if "/" in spec:
        provider, _, model_id = spec.partition("/")
        return provider or None, model_id
    return None, spec


def validate_effort(effort: str) -> str:
    """Validate against the internal taxonomy; refuse with the vocabulary."""
    try:
        return Effort(effort).value
    except ValueError as exc:
        valid = ", ".join(level.value for level in Effort)
        raise AdapterError(
            f"unknown effort '{effort}' (expected one of: {valid})"
        ) from exc


# --------------------------------------------------------------------------- #
# State file handling
# --------------------------------------------------------------------------- #


def resolve_state_path(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    return Path(DEFAULT_STATE_PATH).expanduser()


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


def _select_assignments_key(data: dict[str, Any]) -> str:
    """Pick the casing already used in the file; snake_case when creating.

    Refuses when an existing assignment key is present but is not an object
    (we would have to destroy user data to continue).
    """
    for key in (STATE_KEY_SNAKE, STATE_KEY_CAMEL):
        if key in data:
            if not isinstance(data[key], dict):
                raise AdapterError(
                    f"state key '{key}' exists but is not an object "
                    f"(got {type(data[key]).__name__}); refusing to overwrite"
                )
            return key
    return STATE_KEY_SNAKE


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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def apply_assignment(
    path: Path,
    phase: str,
    model: str,
    effort: str,
    *,
    create: bool = False,
    dry_run: bool = False,
    backup: bool = True,
) -> StateWriteResult:
    """Set ``model_assignments["sdd-<phase>"]`` in the Gentle AI state file.

    Everything outside that single entry is preserved byte-semantically
    (parse → modify one key → dump with the detected indent). Refuses on a
    missing file (unless ``create``), malformed JSON, or a pre-existing entry
    whose shape we do not recognize.
    """
    agent_key = f"sdd-{phase.removeprefix('sdd-')}"
    provider_id, model_id = parse_model_spec(model)
    effort = validate_effort(effort)

    if not path.exists():
        if not create:
            raise AdapterError(
                f"state file does not exist: {path} (pass --create to initialize it)"
            )
        original = ""
        data: dict[str, Any] = {STATE_KEY_SNAKE: {}}
    else:
        original = path.read_text(encoding="utf-8")
        data = _load_state(path)

    key_used = _select_assignments_key(data)
    assignments = data.setdefault(key_used, {})
    existing = assignments.get(agent_key)
    if existing is not None and not _looks_like_assignment(existing):
        raise AdapterError(
            f"refusing to overwrite '{key_used}.{agent_key}': existing value "
            f"does not look like a model assignment ({existing!r}); move it "
            f"aside manually to proceed"
        )

    entry: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    entry["provider_id"] = provider_id
    entry["model_id"] = model_id
    entry["effort"] = effort
    assignments[agent_key] = entry

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
        return StateWriteResult(
            path=path,
            agent_key=agent_key,
            provider_id=provider_id,
            model_id=model_id,
            effort=effort,
            key_used=key_used,
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
    return StateWriteResult(
        path=path,
        agent_key=agent_key,
        provider_id=provider_id,
        model_id=model_id,
        effort=effort,
        key_used=key_used,
        diff=diff,
        backup_path=backup_path,
        wrote=True,
    )


def read_assignment(
    path: Path, phase: str
) -> dict[str, Any] | None:
    """Current assignment for ``sdd-<phase>`` (tolerant read), or None."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    for key in (STATE_KEY_SNAKE, STATE_KEY_CAMEL):
        assignments = data.get(key)
        if isinstance(assignments, dict):
            entry = assignments.get(f"sdd-{phase.removeprefix('sdd-')}")
            return entry if isinstance(entry, dict) else None
    return None


def verify_assignment(
    path: Path,
    phase: str,
    model: str,
    effort: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Round-trip check: re-read the state file and compare the assignment.

    Returns (ok, observed_entry_or_None). ``ok`` is True only when the stored
    provider_id/model_id/effort all match the requested spec.
    """
    provider_id, model_id = parse_model_spec(model)
    effort = validate_effort(effort)
    entry = read_assignment(path, phase)
    if entry is None:
        return False, None
    ok = (
        entry.get("provider_id") == provider_id
        and entry.get("model_id") == model_id
        and entry.get("effort") == effort
    )
    return ok, entry


__all__ = [
    "ASSIGNMENT_KEYS",
    "AdapterError",
    "DEFAULT_STATE_PATH",
    "STATE_KEY_CAMEL",
    "STATE_KEY_SNAKE",
    "StateWriteResult",
    "apply_assignment",
    "latest_backup",
    "parse_model_spec",
    "read_assignment",
    "resolve_state_path",
    "rollback",
    "validate_effort",
    "verify_assignment",
]
