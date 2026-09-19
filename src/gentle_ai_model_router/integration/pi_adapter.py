"""Pi (gentle-pi) write adapter: apply router decisions to ``models.json``.

Verified write-path facts (docs/gentle-ai-integration-research.md §5,
``internal/agents/pi/review_routing.go:17-35``):

- Model assignment ownership is explicitly external: "Pi model assignment
  belongs to ``gentle-pi``, not the Gentle AI installer" (``docs/pi.md:119-121``).
  The file lives at ``~/.pi/gentle-ai/models.json`` (or
  ``$GENTLE_PI_CONFIG_HOME/gentle-ai/models.json``); gentle-pi's
  ``/gentle:models`` modal saves it and applies it as ``--model`` /
  ``--thinking`` argv pairs. Resolution is ONE active file, no merging:
  ``GENTLE_PI_CONFIG_HOME`` → ``~/.pi/gentle-ai/models.json``.
- Shape: a flat JSON map ``<agentName> -> "provider/model"`` OR
  ``{"model": "provider/model", "thinking": "<effort>"}``; SDD phase agents
  are keyed ``sdd-<phase>``.
- Allowed ``--thinking`` levels are exactly ``off, minimal, low, medium,
  high, xhigh, max`` (``review_routing.go:75-79``, validated twice); unknown
  values fail closed on the Go side, so we validate against the same internal
  Effort taxonomy before writing.

Safety model (mirrors the gentle-state/opencode adapters): every write is
atomic (tmp file + ``os.replace``), a backup copy is made first
(``<path>.router-backup-<ts>``), and ``rollback`` restores the latest backup.
We never touch any key other than the single ``sdd-<phase>`` entry, and we
REFUSE to overwrite data we do not recognize (dict entries without a string
``model`` key) — the file may be rewritten by ``/gentle:models`` at any time,
so unknown keys and unknown entry shapes are preserved, never clobbered.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gentle_ai_model_router.integration.gentle_state_adapter import validate_effort
from gentle_ai_model_router.integration.opencode_adapter import (
    AdapterError,
    latest_backup,
    rollback,
)

DEFAULT_MODELS_PATH = "~/.pi/gentle-ai/models.json"
ENV_CONFIG_HOME = "GENTLE_PI_CONFIG_HOME"
AGENT_KEY_PREFIX = "sdd-"

_INDENT_RE = re.compile(r'^(\s+)"')


@dataclass(frozen=True)
class PiWriteResult:
    path: Path
    agent_key: str
    model: str
    thinking: str
    diff: str
    backup_path: Path | None
    wrote: bool  # False in dry-run mode (or when nothing changed)


# --------------------------------------------------------------------------- #
# Path resolution (mirrors gentle-pi review_routing.go: one active file)
# --------------------------------------------------------------------------- #


def resolve_models_path(
    explicit: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve the Pi ``models.json`` path.

    Precedence: explicit argument > ``$GENTLE_PI_CONFIG_HOME`` (whose
    ``gentle-ai/models.json`` is the active file) > ``~/.pi/gentle-ai/models.json``.
    No merging: exactly one file is ever the write target, mirroring the
    gentle-pi Go resolution in ``internal/agents/pi/review_routing.go``.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    environ = os.environ if env is None else env
    pi_home = environ.get(ENV_CONFIG_HOME)
    if pi_home:
        return Path(pi_home).expanduser() / "gentle-ai" / "models.json"
    home_dir = Path.home() if home is None else home
    return home_dir / ".pi" / "gentle-ai" / "models.json"


# --------------------------------------------------------------------------- #
# File handling
# --------------------------------------------------------------------------- #


def _detect_indent(text: str) -> int:
    """Indent width of the first indented property line (default 2)."""
    for line in text.splitlines():
        match = _INDENT_RE.match(line)
        if match and " " in match.group(1):
            return len(match.group(1))
    return 2


def _load_models(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"malformed JSON in pi models file {path}: {exc}") from exc
    except OSError as exc:
        raise AdapterError(f"cannot read pi models file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterError(f"pi models file root is not an object: {path}")
    return data


def _looks_like_pi_entry(value: Any) -> bool:
    """Tolerant shape check so we never clobber unrecognized user data.

    Accepted shapes: a bare model string, or a dict with a string ``model``
    key (the ``{model, thinking}`` object form ``/gentle:models`` writes).
    """
    if isinstance(value, str):
        return True
    return isinstance(value, dict) and isinstance(value.get("model"), str)


def _atomic_write(path: Path, text: str) -> None:
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
) -> PiWriteResult:
    """Set ``models.json["sdd-<phase>"]`` to ``{"model", "thinking"}``.

    The entry is always written in the object form
    ``{"model": "provider/model", "thinking": "<effort>"}``: an existing bare
    string entry is upgraded (its slot kept, model/thinking set from the
    requested assignment), and extra keys inside an existing object entry are
    preserved. Everything outside that single entry survives verbatim (pure
    JSON round trip). Refuses on a missing file (unless ``create``), malformed
    JSON, or a pre-existing entry whose shape we do not recognize — the file
    is owned by gentle-pi and may be rewritten by ``/gentle:models``, so
    unknown shapes are never destroyed.
    """
    agent_key = f"{AGENT_KEY_PREFIX}{phase.removeprefix('sdd-')}"
    model = model.strip()
    if not model:
        raise AdapterError("model spec is empty")
    thinking = validate_effort(effort)

    if not path.exists():
        if not create:
            raise AdapterError(
                f"pi models file does not exist: {path} (pass --create to initialize it)"
            )
        original = ""
        data: dict[str, Any] = {}
    else:
        original = path.read_text(encoding="utf-8")
        data = _load_models(path)

    existing = data.get(agent_key)
    if existing is not None and not _looks_like_pi_entry(existing):
        raise AdapterError(
            f"refusing to overwrite '{agent_key}': existing value does not "
            f"look like a pi model assignment ({existing!r}); move it aside "
            f"manually to proceed"
        )

    entry: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    entry["model"] = model
    entry["thinking"] = thinking
    data[agent_key] = entry

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
        return PiWriteResult(
            path=path,
            agent_key=agent_key,
            model=model,
            thinking=thinking,
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
    return PiWriteResult(
        path=path,
        agent_key=agent_key,
        model=model,
        thinking=thinking,
        diff=diff,
        backup_path=backup_path,
        wrote=True,
    )


def read_assignments(path: Path) -> dict[str, dict[str, Any]]:
    """Current assignments from a Pi ``models.json`` (tolerant read).

    Bare string entries are normalized to ``{"model": <str>, "thinking": None}``;
    unrecognized entry shapes are skipped, never crashed on.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(value, str):
            out[str(key)] = {"model": value, "thinking": None}
        elif isinstance(value, dict) and isinstance(value.get("model"), str):
            out[str(key)] = {"model": value["model"], "thinking": value.get("thinking")}
    return out


def verify_assignment(
    path: Path,
    phase: str,
    model: str,
    thinking: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Round-trip check: re-read the file and compare the assignment.

    Returns (ok, observed_entry_or_None). ``ok`` is True only when the stored
    ``model`` and ``thinking`` both match the requested values.
    """
    thinking = validate_effort(thinking)
    entry = read_assignments(path).get(f"{AGENT_KEY_PREFIX}{phase.removeprefix('sdd-')}")
    if entry is None:
        return False, None
    ok = entry.get("model") == model and entry.get("thinking") == thinking
    return ok, entry


__all__ = [
    "AdapterError",
    "AGENT_KEY_PREFIX",
    "DEFAULT_MODELS_PATH",
    "ENV_CONFIG_HOME",
    "PiWriteResult",
    "apply_assignment",
    "latest_backup",
    "read_assignments",
    "resolve_models_path",
    "rollback",
    "validate_effort",
    "verify_assignment",
]
