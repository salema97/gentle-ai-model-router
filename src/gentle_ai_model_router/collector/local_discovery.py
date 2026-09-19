"""Read-only local discovery of available (model, provider, effort) candidates.

Scans, without ever writing to the discovered locations:

- OpenCode config: ``~/.config/opencode/opencode.json`` (+ ``.jsonc``,
  ``$OPENCODE_CONFIG_DIR`` override) — legacy ``agent.<name>.model`` /
  ``variant`` and v2 ``agents.<name>.model`` (``provider/model#variant``).
- Gentle AI model-variants cache: ``~/.gentle-ai/cache/model-variants.json``
  (written by the bundled model-variants plugin; see research doc §4).
- Pi: ``~/.pi/gentle-ai/models.json`` (or ``$GENTLE_PI_CONFIG_HOME``) — entries
  are a model string or ``{model, thinking}`` (research doc §5).
- Gentle AI persisted state: ``~/.gentle-ai/state.json`` — model/effort
  assignment maps parsed defensively, field by field.

Every discovery carries provenance (file path + JSON path within the file).
Missing or malformed files produce a note, never an exception.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gentle_ai_model_router.collector.logging_conf import get_logger
from gentle_ai_model_router.router.config import LocalDiscoveryConfig

logger = get_logger(__name__)

# Assignment-map key prefixes in the Gentle AI state file (research doc §1/§3).
_STATE_ASSIGNMENT_KEYS = (
    "ModelAssignments",
    "ClaudeModelAssignments",
    "ClaudePhaseAssignments",
    "KiroModelAssignments",
    "CodexModelAssignments",
    "CodexOrchestratorAssignment",
    "CodexCarrilModelAssignments",
    "CodexPhaseModelAssignments",
)

_COMMENT_RE = re.compile(r"(?m)//.*?$|/\*.*?\*/")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


@dataclass
class Provenance:
    """Where a candidate was discovered."""

    file: str
    json_path: str  # dotted path within the file, e.g. "agent.sdd-explore.model"


@dataclass
class DiscoveredCandidate:
    """One locally available (model, provider, efforts) candidate."""

    model: str
    provider: str | None = None
    efforts: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)


@dataclass
class DiscoveryResult:
    """Aggregate result of a local discovery run."""

    candidates: list[DiscoveredCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("local_discovery unreadable file=%s error=%s", path, exc)
        return None


def _load_jsonc(path: Path) -> Any | None:
    """Load JSON or JSONC (comments + trailing commas) defensively."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.debug("local_discovery unreadable file=%s error=%s", path, exc)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        stripped = _COMMENT_RE.sub("", text)
        stripped = _TRAILING_COMMA_RE.sub(r"\1", stripped)
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        logger.debug("local_discovery unparsable jsonc file=%s error=%s", path, exc)
        return None


def _split_model_spec(spec: str) -> tuple[str | None, str | None, str | None]:
    """Split ``provider/model#variant`` (also accepts ``:`` separator)."""
    variant: str | None = None
    if "#" in spec:
        spec, _, variant = spec.partition("#")
        variant = variant or None
    for sep in ("/", ":"):
        if sep in spec:
            provider, _, model = spec.partition(sep)
            if provider and model:
                return provider, model, variant
    return None, spec or None, variant


def _merge_candidate(
    result: DiscoveryResult,
    model: str | None,
    provider: str | None,
    effort: str | None,
    provenance: Provenance,
) -> None:
    """Attach a discovery to an existing candidate (matched by provider+model)."""
    if not model:
        return
    for candidate in result.candidates:
        if candidate.model == model and candidate.provider == provider:
            if effort and effort not in candidate.efforts:
                candidate.efforts.append(effort)
            candidate.provenance.append(provenance)
            return
    result.candidates.append(
        DiscoveredCandidate(
            model=model,
            provider=provider,
            efforts=[effort] if effort else [],
            provenance=[provenance],
        )
    )


def _parse_assignment_value(value: Any) -> tuple[str | None, str | None, str | None]:
    """Extract (provider, model, effort) from an assignment entry defensively."""
    if isinstance(value, str):
        provider, model, _variant = _split_model_spec(value)
        return provider, model or value, None
    if isinstance(value, dict):
        model = (
            value.get("model")
            or value.get("model_id")
            or value.get("ModelID")
            or value.get("id")
        )
        effort = (
            value.get("effort")
            or value.get("Effort")
            or value.get("variant")
            or value.get("thinking")
            or value.get("reasoning_effort")
            or value.get("model_reasoning_effort")
        )
        provider = (
            value.get("provider")
            or value.get("provider_id")
            or value.get("ProviderID")
        )
        if isinstance(model, str):
            parsed_provider, parsed_model, variant = _split_model_spec(model)
            return provider or parsed_provider, parsed_model, effort or variant
    return None, None, None


def discover_opencode(config: LocalDiscoveryConfig, result: DiscoveryResult) -> None:
    """Scan OpenCode config files for model/variant assignments."""
    candidates: list[Path] = []
    config_dir_env = os.environ.get("OPENCODE_CONFIG_DIR")
    if config_dir_env:
        for name in ("opencode.json", "opencode.jsonc"):
            candidates.append(Path(config_dir_env).expanduser() / name)
    home = Path.home() / ".config" / "opencode"
    candidates.extend([home / "opencode.json", home / "opencode.jsonc"])
    # Configured path (tests / non-standard locations), plus its JSONC sibling.
    configured = Path(config.opencode_config).expanduser()
    candidates.append(configured)
    if configured.suffix == ".json":
        candidates.append(configured.with_suffix(".jsonc"))

    seen_files: set[Path] = set()
    found_any = False
    for path in candidates:
        if path in seen_files or not path.is_file():
            continue
        seen_files.add(path)
        data = _load_jsonc(path)
        if not isinstance(data, dict):
            continue
        found_any = True
        # Legacy: agent.<name> = {model, variant}
        for name, entry in (data.get("agent") or {}).items():
            if not isinstance(entry, dict):
                continue
            model_spec = entry.get("model")
            if not isinstance(model_spec, str):
                continue
            provider, model, variant = _split_model_spec(model_spec)
            effort = entry.get("variant") or variant
            _merge_candidate(
                result,
                model,
                provider,
                effort,
                Provenance(str(path), f"agent.{name}.model"),
            )
        # Native v2: agents.<name> = {model: "provider/model#variant"}
        for name, entry in (data.get("agents") or {}).items():
            if not isinstance(entry, dict):
                continue
            model_spec = entry.get("model")
            if not isinstance(model_spec, str):
                continue
            provider, model, variant = _split_model_spec(model_spec)
            _merge_candidate(
                result,
                model,
                provider,
                variant,
                Provenance(str(path), f"agents.{name}.model"),
            )
    if not found_any:
        result.notes.append("opencode: no config file found (opencode.json/jsonc)")


def discover_variants_cache(config: LocalDiscoveryConfig, result: DiscoveryResult) -> None:
    """Merge effort levels from the Gentle AI model-variants cache."""
    path = Path(config.opencode_variants_cache).expanduser()
    data = _load_json(path)
    if data is None:
        result.notes.append(f"model-variants cache: missing or unreadable ({path})")
        return
    # Real v1 shape (verified on a live machine, ~/.gentle-ai/cache/
    # model-variants.json): flat map {"<providerId>": {"<modelId>": ["v", ...]}}.
    if isinstance(data, dict) and "providers" not in data:
        handled = 0
        for provider_id, model_map in data.items():
            if not isinstance(provider_id, str) or not isinstance(model_map, dict):
                continue
            for model_id, variants in model_map.items():
                if not isinstance(model_id, str) or not isinstance(variants, list):
                    continue
                for variant in variants:
                    _merge_candidate(
                        result,
                        model_id,
                        provider_id,
                        str(variant),
                        Provenance(str(path), f"{provider_id}.{model_id}"),
                    )
                handled += 1
        if not handled:
            result.notes.append("model-variants cache: unexpected shape, skipped")
        return
    # Alternate catalog-style shape: providers list -> models.*.variants keys.
    providers = data.get("providers") if isinstance(data, dict) else None
    if providers is None and isinstance(data, dict):
        providers = [data]  # tolerate a single-provider dict
    if not isinstance(providers, list):
        result.notes.append("model-variants cache: unexpected shape, skipped")
        return
    for provider_entry in providers:
        if not isinstance(provider_entry, dict):
            continue
        provider = (
            provider_entry.get("id")
            or provider_entry.get("provider")
            or provider_entry.get("name")
        )
        models = provider_entry.get("models") or {}
        if isinstance(models, list):
            models = {m.get("id"): m for m in models if isinstance(m, dict)}
        if not isinstance(models, dict):
            continue
        for model_id, model_entry in models.items():
            if not isinstance(model_entry, dict) or not model_id:
                continue
            variants = model_entry.get("variants") or {}
            efforts = [str(v) for v in variants.keys()] if isinstance(variants, dict) else []
            _merge_candidate(
                result,
                str(model_id),
                str(provider) if provider else None,
                None,
                Provenance(str(path), f"providers.{provider}.models.{model_id}.variants"),
            )
            if efforts:
                for candidate in result.candidates:
                    if candidate.model == str(model_id) and (
                        candidate.provider is None or candidate.provider == provider
                    ):
                        for effort in efforts:
                            if effort not in candidate.efforts:
                                candidate.efforts.append(effort)


def discover_pi_models(config: LocalDiscoveryConfig, result: DiscoveryResult) -> None:
    """Scan Pi ``models.json`` (gentle-pi owns this file; we read only)."""
    pi_home = os.environ.get("GENTLE_PI_CONFIG_HOME")
    paths = []
    if pi_home:
        paths.append(Path(pi_home).expanduser() / "gentle-ai" / "models.json")
    paths.append(Path(config.pi_models).expanduser())
    for path in paths:
        if not path.is_file():
            continue
        data = _load_json(path)
        if not isinstance(data, dict):
            result.notes.append(f"pi models.json: unreadable ({path})")
            continue
        for agent, entry in data.items():
            if isinstance(entry, str):
                provider, model, _ = _split_model_spec(entry)
                _merge_candidate(
                    result, model or entry, provider, None,
                    Provenance(str(path), str(agent)),
                )
            elif isinstance(entry, dict):
                model = entry.get("model")
                thinking = entry.get("thinking")
                if isinstance(model, str):
                    provider, parsed_model, _ = _split_model_spec(model)
                    _merge_candidate(
                        result, parsed_model or model, provider,
                        str(thinking) if thinking else None,
                        Provenance(str(path), f"{agent}.model"),
                    )
        return
    result.notes.append("pi: models.json not found")


def discover_gentle_state(config: LocalDiscoveryConfig, result: DiscoveryResult) -> None:
    """Read Gentle AI persisted state model/effort assignment fields.

    Parsed field by field, defensively: any failure produces a note and the
    scan continues. The router never writes to Gentle AI state (research doc §6).
    """
    path = Path(config.gentle_state).expanduser()
    data = _load_json(path)
    if not isinstance(data, dict):
        result.notes.append(f"gentle-ai state: missing or unreadable ({path})")
        return
    found = False
    for key in _STATE_ASSIGNMENT_KEYS:
        try:
            assignments = data.get(key)
        except Exception:  # pragma: no cover - defensive
            continue
        if not assignments:
            continue
        found = True
        entries: dict[str, Any] = {}
        if isinstance(assignments, dict):
            entries = assignments
        elif isinstance(assignments, list):
            entries = {str(i): v for i, v in enumerate(assignments)}
        for slot, value in entries.items():
            provider, model, effort = _parse_assignment_value(value)
            _merge_candidate(
                result,
                model,
                provider,
                effort,
                Provenance(str(path), f"{key}.{slot}"),
            )
    if not found:
        result.notes.append("gentle-ai state: no model assignment fields present")


def collect_local_candidates(
    config: LocalDiscoveryConfig | None = None,
) -> DiscoveryResult:
    """Run all local discovery scans; aggregate candidates and notes."""
    config = config or LocalDiscoveryConfig()
    result = DiscoveryResult()
    scans = (discover_opencode, discover_variants_cache, discover_pi_models, discover_gentle_state)
    for scan in scans:
        try:
            scan(config, result)
        except Exception as exc:  # never crash the pipeline
            logger.warning("local_discovery scan=%s failed: %s", scan.__name__, exc)
            result.notes.append(f"{scan.__name__}: failed ({exc})")
    logger.info(
        "collect source=local candidates=%s notes=%s",
        len(result.candidates),
        len(result.notes),
    )
    return result
