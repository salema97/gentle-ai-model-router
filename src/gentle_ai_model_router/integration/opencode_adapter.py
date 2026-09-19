"""OpenCode write adapter: apply router decisions to OpenCode config surfaces.

Verified write-path facts (docs/telemetry-probe.md §2, file:line evidence in
the Gentle AI reference clone):

- Legacy shape ``agent.<name> = {model: "provider/model", variant: "<effort>"}``
  (``internal/opencode/config.go:248-281``); the orchestrator agent is
  ``gentle-orchestrator``, SDD phase agents are ``sdd-<phase>``
  (``:253-254``).
- Effort travels as ``variant`` / ``#variant``. **No** ``reasoningEffort`` key
  exists in OpenCode config (probe §2.1 correction).
- Global config at ``~/.config/opencode/opencode.json``, XDG-aware
  (``internal/opencode/models.go:9-28``); absolute ``OPENCODE_CONFIG_DIR``
  prepends and displaces global (``config.go:41-82``); JSONC beats JSON per
  directory.
- The TUI picker deletes efforts it cannot validate against the model-variants
  cache (``model_picker.go:429-433``), so we never write an effort that is not
  in the discovered variants for that (provider, model) — we clamp DOWN to the
  closest supported level (minimum-sufficient policy), never up.
- ``__managed_by: gentle-ai/sdd`` keys are owned by ``gentle-ai sync``; this
  adapter refuses to touch any agent block carrying that key.

Write strategy (documented choice): pure-JSON files get a normal
``json.load``/``json.dump`` round trip; files containing comments or trailing
commas (JSONC) get a targeted line-level patch of only the agent block being
modified, so comments outside that block survive byte-for-byte. All writes are
atomic (tmp file + ``os.replace``) and a backup copy is made before the first
write to a path (``<path>.router-backup-<ts>``); ``rollback`` restores the
latest backup.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gentle_ai_model_router.registry.normalize import Effort

AGENT_KEY_PREFIX = "sdd-"
ORCHESTRATOR_KEY = "gentle-orchestrator"
MANAGED_BY_KEY = "__managed_by"
MANAGED_BY_VALUE = "gentle-ai/sdd"

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
def _KEY_BLOCK_RE(key: str) -> re.Pattern[str]:
    return re.compile(rf'^(\s*)"{re.escape(key)}"\s*:\s*\{{')
_MODEL_LINE_RE = re.compile(r'^(\s*)"model"\s*:')
_VARIANT_LINE_RE = re.compile(r'^(\s*)"variant"\s*:')
_PROP_LINE_RE = re.compile(r'^(\s*)"[^"]+"\s*:')


class AdapterError(Exception):
    """Fatal adapter failure (bad input, managed block, unreadable config)."""


@dataclass(frozen=True)
class WriteResult:
    path: Path
    agent_key: str
    model: str
    effort: str | None  # effort actually written (clamped) or None
    reason_codes: tuple[str, ...]
    diff: str
    backup_path: Path | None
    wrote: bool  # False in dry-run mode


# --------------------------------------------------------------------------- #
# Target resolution (verified precedence, telemetry-probe.md §2.1)
# --------------------------------------------------------------------------- #


def resolve_global_config(
    explicit: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Resolve the global OpenCode config path.

    Precedence: explicit ``--config`` > absolute ``OPENCODE_CONFIG_DIR``
    (displaces global) > ``$XDG_CONFIG_HOME/opencode`` > ``~/.config/opencode``.
    JSONC beats JSON within the resolved directory. When nothing exists, the
    returned path is ``opencode.json`` (the documented global default); the
    caller still refuses to create it unless ``--create`` was passed.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    environ = os.environ if env is None else env
    home_dir = Path.home() if home is None else home

    def _existing_in(directory: Path) -> Path | None:
        for name in ("opencode.jsonc", "opencode.json"):
            candidate = directory / name
            if candidate.is_file():
                return candidate
        return None

    config_dir_env = environ.get("OPENCODE_CONFIG_DIR")
    if config_dir_env:
        directory = Path(config_dir_env).expanduser()
        existing = _existing_in(directory)
        return existing if existing is not None else directory / "opencode.json"

    xdg = Path(environ.get("XDG_CONFIG_HOME", "")).expanduser() if environ.get(
        "XDG_CONFIG_HOME"
    ) else home_dir / ".config"
    directory = xdg / "opencode"
    existing = _existing_in(directory)
    return existing if existing is not None else directory / "opencode.json"


# --------------------------------------------------------------------------- #
# Variants cache (telemetry-probe.md §2.2)
# --------------------------------------------------------------------------- #


def load_variants_cache(cache_v1: Path, cache_v2_dir: Path) -> dict[tuple[str, str], list[str]]:
    """Discover supported efforts per (provider, model).

    Merges the v1 cache (``~/.gentle-ai/cache/model-variants.json``) and every
    v2 file (``~/.gentle-ai/cache/opencode-v2/*.json``); both have shape
    ``{provider: {model: [variantKey, …]}}``.
    """
    merged: dict[tuple[str, str], list[str]] = {}
    sources: list[Path] = []
    if cache_v1.is_file():
        sources.append(cache_v1)
    if cache_v2_dir.is_dir():
        sources.extend(sorted(cache_v2_dir.glob("*.json")))
    for path in sources:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        for provider, models in data.items():
            if not isinstance(models, dict):
                continue
            for model, variants in models.items():
                if not isinstance(variants, list):
                    continue
                key = (str(provider), str(model))
                known = merged.setdefault(key, [])
                for variant in variants:
                    if isinstance(variant, str) and variant not in known:
                        known.append(variant)
    return merged


_EFFORT_ORDER = [level.value for level in Effort]


def clamp_effort(
    requested: str, supported: Iterable[str] | None
) -> tuple[str | None, str | None]:
    """Clamp an effort DOWN to the closest supported level — never up.

    Returns (effort_or_None, reason_code_or_None). ``supported=None`` means
    "no variants cache data for this (provider, model)": write the model
    without a variant. An empty iterable means the cache knows the model but
    lists no efforts: same behavior, different reason code.
    """
    if requested not in _EFFORT_ORDER:
        raise AdapterError(
            f"unknown effort '{requested}' (expected one of: {', '.join(_EFFORT_ORDER)})"
        )
    if supported is None:
        return None, "variants_unknown:no_variant_written"
    supported_list = sorted(
        (s for s in supported if s in _EFFORT_ORDER),
        key=_EFFORT_ORDER.index,
    )
    if not supported_list:
        return None, "variants_empty:no_variant_written"
    req_rank = _EFFORT_ORDER.index(requested)
    eligible = [s for s in supported_list if _EFFORT_ORDER.index(s) <= req_rank]
    if not eligible:
        return None, f"effort_clamped:{requested}->none"
    clamped = eligible[-1]
    if clamped != requested:
        return clamped, f"effort_clamped:{requested}->{clamped}"
    return clamped, None


# --------------------------------------------------------------------------- #
# JSONC handling
# --------------------------------------------------------------------------- #


def _strip_jsonc(text: str) -> str:
    stripped = _COMMENT_RE.sub("", text)
    return _TRAILING_COMMA_RE.sub(r"\1", stripped)


def _has_jsonc(text: str) -> bool:
    try:
        json.loads(text)
        return False
    except json.JSONDecodeError:
        return True


def _find_block(lines: list[str], key: str, start: int = 0) -> tuple[int, int] | None:
    """Find (start, end) line indices of the object value for ``"<key>": {``."""
    pattern = _KEY_BLOCK_RE(key)
    for idx in range(start, len(lines)):
        match = pattern.match(lines[idx])
        if match:
            depth = 0
            for j in range(idx, len(lines)):
                # Naive brace counting is acceptable for config files; braces
                # inside string literals are vanishingly rare here and the
                # change is verified by a parse check before writing.
                depth += lines[j].count("{") - lines[j].count("}")
                if depth == 0:
                    return idx, j
            return None
    return None


def _normalize_block_commas(lines: list[str], start: int, end: int) -> None:
    """Ensure every property line in [start, end] except the last ends with ','.

    Lines that open a nested object (``"key": {``) are headers, not
    properties, and must not be comma-terminated.
    """
    prop_lines = [
        i
        for i in range(start, end + 1)
        if _PROP_LINE_RE.match(lines[i]) and not lines[i].rstrip().endswith("{")
    ]
    for i in prop_lines:
        stripped = lines[i].rstrip("\n").rstrip()
        if stripped.endswith(","):
            stripped = stripped[:-1]
        lines[i] = stripped + ("\n" if i == prop_lines[-1] else ",\n")


def _patch_agent_block_jsonc(
    text: str, agent_key: str, model: str, effort: str | None
) -> str:
    """Targeted line-level patch of one agent block; other lines untouched."""
    lines = text.splitlines(keepends=True)
    block = _find_block(lines, agent_key)
    if block is None:
        return _insert_agent_block_jsonc(lines, agent_key, model, effort)
    start, end = block
    if end == start:
        # Single-line block: ` "sdd-x": { "model": "...", ... },` — expand it
        # into a multi-line block (comments inside the block do not survive).
        return _patch_single_line_block(lines, start, agent_key, model, effort)
    body = "".join(lines[start : end + 1])
    if f'"{MANAGED_BY_KEY}"' in body:
        raise AdapterError(
            f"refusing to modify agent '{agent_key}': block carries "
            f"'{MANAGED_BY_KEY}: {MANAGED_BY_VALUE}' (owned by gentle-ai sync)"
        )
    indent = _PROP_LINE_RE.match(lines[start]).group(1) + "  "
    model_idx = variant_idx = None
    for i in range(start + 1, end):
        if _MODEL_LINE_RE.match(lines[i]) and model_idx is None:
            model_idx = i
        if _VARIANT_LINE_RE.match(lines[i]) and variant_idx is None:
            variant_idx = i
    new_lines: list[str] = []
    model_pos: int | None = None
    for i in range(start + 1, end):
        if i == model_idx:
            model_pos = len(new_lines)
            new_lines.append(f'{indent}"model": "{model}",\n')
        elif i == variant_idx:
            continue  # re-added below (or dropped when effort is None)
        else:
            new_lines.append(lines[i])
    if effort is not None:
        new_lines.insert(
            (model_pos + 1) if model_pos is not None else 0,
            f'{indent}"variant": "{effort}",\n',
        )
    lines[start + 1 : end] = new_lines
    _normalize_block_commas(lines, start, _find_block(lines, agent_key, start)[1])
    result = "".join(lines)
    # Safety: the patched file must still parse as JSONC.
    try:
        json.loads(_strip_jsonc(result))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"line patch produced unparsable JSONC: {exc}") from exc
    return result


def _patch_single_line_block(
    lines: list[str], idx: int, agent_key: str, model: str, effort: str | None
) -> str:
    """Expand and patch a one-line agent object, keeping everything else."""
    line = lines[idx]
    key_match = _KEY_BLOCK_RE(agent_key).match(line)
    if key_match is None:  # pragma: no cover - guarded by caller
        raise AdapterError(f"agent block not found on line {idx}")
    indent = key_match.group(1)
    open_pos = line.index("{", key_match.start())
    depth = 0
    close_pos: int | None = None
    for pos in range(open_pos, len(line)):
        if line[pos] == "{":
            depth += 1
        elif line[pos] == "}":
            depth -= 1
            if depth == 0:
                close_pos = pos
                break
    if close_pos is None:
        raise AdapterError(f"unbalanced braces on line {idx}")
    fragment = line[open_pos : close_pos + 1]
    if f'"{MANAGED_BY_KEY}"' in fragment:
        raise AdapterError(
            f"refusing to modify agent '{agent_key}': block carries "
            f"'{MANAGED_BY_KEY}: {MANAGED_BY_VALUE}' (owned by gentle-ai sync)"
        )
    try:
        data = json.loads(_strip_jsonc(fragment))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"agent block on line {idx} is unparsable: {exc}") from exc
    if not isinstance(data, dict):
        raise AdapterError(f"agent '{agent_key}' is not an object")
    data["model"] = model
    if effort is not None:
        data["variant"] = effort
    else:
        data.pop("variant", None)
    inner_indent = indent + "  "
    props = [f"{inner_indent}{json.dumps(key)}: {json.dumps(value)}" for key, value in data.items()]
    trailing = line[close_pos + 1 :].rstrip("\n")
    lines[idx] = (
        f'{indent}"{agent_key}": {{\n'
        + ",\n".join(props)
        + f"\n{indent}}}"
        + trailing
        + "\n"
    )
    result = "".join(lines)
    try:
        json.loads(_strip_jsonc(result))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"line patch produced unparsable JSONC: {exc}") from exc
    return result


def _insert_agent_block_jsonc(
    lines: list[str], agent_key: str, model: str, effort: str | None
) -> str:
    """Insert a new agent entry into the ``agent`` object (JSONC file)."""
    agent_block = _find_block(lines, "agent")
    if agent_block is None:
        # No agent object at all: append one at the end of the root object.
        for idx in range(len(lines) - 1, -1, -1):
            if lines[idx].rstrip().endswith("}"):
                prev_match = _PROP_LINE_RE.match(lines[idx - 1]) if idx else None
                indent = prev_match.group(1) if prev_match else "  "
                entry = [f'{indent}"agent": {{\n']
                entry.append(f'{indent}  "{agent_key}": {{\n')
                entry.append(f'{indent}    "model": "{model}"' + ("," if effort else "") + "\n")
                if effort:
                    entry.append(f'{indent}    "variant": "{effort}"\n')
                entry.append(f"{indent}  }}\n{indent}}}\n")
                lines[idx : idx + 1] = entry
                # root closing brace must keep a comma on the previous prop
                _normalize_root_commas(lines)
                return "".join(lines)
        raise AdapterError("could not locate root object to insert agent block")
    start, end = agent_block
    body = "".join(lines[start : end + 1])
    if f'"{MANAGED_BY_KEY}"' in body:
        raise AdapterError("refusing to modify 'agent' block carrying __managed_by")
    inner = [line for line in lines[start + 1 : end] if line.strip()]
    indent = _PROP_LINE_RE.match(lines[start]).group(1) + "  "
    entry = [f'{indent}"{agent_key}": {{\n']
    entry.append(f'{indent}  "model": "{model}"' + ("," if effort else "") + "\n")
    if effort:
        entry.append(f'{indent}  "variant": "{effort}"\n')
    entry.append(f"{indent}}}")
    if inner:
        prev = end - 1
        if prev > start and not lines[prev].rstrip().endswith(","):
            lines[prev] = lines[prev].rstrip("\n") + ",\n"
        entry[-1] = entry[-1] + ",\n"
        lines[end:end] = entry
    else:
        lines[end:end] = entry[:-1] + [entry[-1] + "\n"]
    result = "".join(lines)
    try:
        json.loads(_strip_jsonc(result))
    except json.JSONDecodeError as exc:
        raise AdapterError(f"insert produced unparsable JSONC: {exc}") from exc
    return result


def _normalize_root_commas(lines: list[str]) -> None:
    for i in range(len(lines) - 2, -1, -1):
        if _PROP_LINE_RE.match(lines[i]):
            if not lines[i].rstrip().endswith(","):
                lines[i] = lines[i].rstrip("\n") + ",\n"
            return


def _patch_agent_json(data: dict[str, Any], agent_key: str, model: str, effort: str | None) -> None:
    agent = data.setdefault("agent", {})
    if not isinstance(agent, dict):
        raise AdapterError("'agent' key exists but is not an object")
    block = agent.get(agent_key)
    if isinstance(block, dict) and block.get(MANAGED_BY_KEY) == MANAGED_BY_VALUE:
        raise AdapterError(
            f"refusing to modify agent '{agent_key}': block carries "
            f"'{MANAGED_BY_KEY}: {MANAGED_BY_VALUE}' (owned by gentle-ai sync)"
        )
    if block is None:
        block = {}
        agent[agent_key] = block
    if not isinstance(block, dict):
        raise AdapterError(f"agent '{agent_key}' exists but is not an object")
    block["model"] = model
    if effort is not None:
        block["variant"] = effort
    else:
        block.pop("variant", None)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def apply_decision(
    path: Path,
    phase: str,
    model: str,
    effort: str,
    variants: dict[tuple[str, str], list[str]] | None = None,
    *,
    create: bool = False,
    dry_run: bool = False,
    backup: bool = True,
) -> WriteResult:
    """Write ``agent["sdd-<phase>"].model/variant`` into an OpenCode config.

    ``variants`` maps (provider, model) → supported effort list; ``None`` (or
    a missing key) means "no cache data": write the model without a variant.
    Existing content is preserved; unknown keys are never removed.
    """
    if "/" not in model:
        raise AdapterError(f"model must be 'provider/model', got '{model}'")
    provider, _, model_id = model.partition("/")
    agent_key = f"{AGENT_KEY_PREFIX}{phase.removeprefix('sdd-')}"
    supported = variants.get((provider, model_id)) if variants else None
    clamped, clamp_reason = clamp_effort(effort, supported)
    reason_codes = [c for c in (clamp_reason,) if c]

    if not path.exists():
        if not create:
            raise AdapterError(
                f"config path does not exist: {path} (pass --create to initialize it)"
            )
        original = ""
        new_text = json.dumps({"agent": {agent_key: {}}}, indent=2) + "\n"
        if _has_jsonc(new_text):
            raise AssertionError("unreachable: fresh JSON never has comments")
        data: dict[str, Any] = json.loads(new_text)
        _patch_agent_json(data, agent_key, model, clamped)
        new_text = json.dumps(data, indent=2) + "\n"
    else:
        original = path.read_text(encoding="utf-8")
        if _has_jsonc(original):
            new_text = _patch_agent_block_jsonc(original, agent_key, model, clamped)
        else:
            try:
                data = json.loads(original)
            except json.JSONDecodeError as exc:
                raise AdapterError(f"unparsable config {path}: {exc}") from exc
            if not isinstance(data, dict):
                raise AdapterError(f"config root is not an object: {path}")
            _patch_agent_json(data, agent_key, model, clamped)
            new_text = json.dumps(data, indent=2) + "\n"

    diff = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )
    if dry_run or new_text == original:
        return WriteResult(
            path=path,
            agent_key=agent_key,
            model=model,
            effort=clamped,
            reason_codes=tuple(reason_codes),
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
    return WriteResult(
        path=path,
        agent_key=agent_key,
        model=model,
        effort=clamped,
        reason_codes=tuple(reason_codes),
        diff=diff,
        backup_path=backup_path,
        wrote=True,
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.router-tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def latest_backup(path: Path) -> Path | None:
    """Most recent backup for a config path (sortable timestamp suffix)."""
    backups = sorted(path.parent.glob(f"{path.name}.router-backup-*"))
    return backups[-1] if backups else None


def rollback(path: Path) -> Path | None:
    """Restore the latest backup atomically. Returns the backup path or None."""
    backup = latest_backup(path)
    if backup is None:
        return None
    _atomic_write(path, backup.read_text(encoding="utf-8"))
    return backup


def read_assignments(path: Path) -> dict[str, dict[str, Any]]:
    """Current per-agent assignments from the effective config (tolerant read)."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return {}
    agent = data.get("agent") if isinstance(data, dict) else None
    if not isinstance(agent, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in agent.items():
        if isinstance(value, dict):
            out[str(key)] = {
                "model": value.get("model"),
                "variant": value.get("variant"),
                "managed": value.get(MANAGED_BY_KEY) == MANAGED_BY_VALUE,
            }
    return out
