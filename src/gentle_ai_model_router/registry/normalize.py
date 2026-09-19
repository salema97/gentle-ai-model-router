"""Normalization: internal effort taxonomy + collector payload normalization.

The internal effort vocabulary is the reconciled superset verified across
runtimes (research doc §5): Pi ``--thinking`` uses exactly
``off, minimal, low, medium, high, xhigh, max``. Provider adapters map the
internal level to the provider-native value; new providers plug in by adding
an entry keyed by their registry key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Effort(StrEnum):
    """Internal reasoning-effort taxonomy (superset across runtimes)."""

    OFF = "off"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


# Provider-native effort adapters. ``None`` means the provider has no native
# equivalent for that internal level (callers must not emit it).
_PROVIDER_EFFORTS: dict[str, dict[Effort, str | None]] = {
    # Generic passthrough: provider speaks the internal vocabulary directly.
    "generic": {level: level.value for level in Effort},
    # OpenCode efforts travel as ``variant`` / ``#variant`` provider-native keys.
    "opencode": {level: level.value for level in Effort},
    # Pi --thinking levels (verified: review_routing.go:75-79).
    "pi": {level: level.value for level in Effort},
    # Codex reasoning_effort values (verified: codex_model.go:136-141).
    "codex": {
        Effort.LOW: "low",
        Effort.MEDIUM: "medium",
        Effort.HIGH: "high",
        Effort.XHIGH: "xhigh",
    },
    # Claude frontmatter effort values.
    "claude": {
        Effort.LOW: "low",
        Effort.MEDIUM: "medium",
        Effort.HIGH: "high",
        Effort.XHIGH: "xhigh",
        Effort.MAX: "max",
    },
}

# Fill missing levels with None so lookups never KeyError.
for _adapter in _PROVIDER_EFFORTS.values():
    for _level in Effort:
        _adapter.setdefault(_level, None)


def provider_effort_value(registry_key: str, effort: Effort) -> str | None:
    """Map an internal effort level to the provider-native value (or None)."""
    adapter = _PROVIDER_EFFORTS.get(registry_key, _PROVIDER_EFFORTS["generic"])
    return adapter.get(effort)


def internal_effort(registry_key: str, provider_value: str) -> Effort | None:
    """Reverse-map a provider-native effort value to the internal level."""
    adapter = _PROVIDER_EFFORTS.get(registry_key, _PROVIDER_EFFORTS["generic"])
    for level, value in adapter.items():
        if value == provider_value:
            return level
    return None


def register_effort_adapter(
    registry_key: str, mapping: dict[Effort, str | None]
) -> None:
    """Plug in a new provider adapter keyed by its registry key."""
    complete = {level: mapping.get(level) for level in Effort}
    _PROVIDER_EFFORTS[registry_key] = complete


@dataclass
class NormalizedAAModel:
    """One Artificial Analysis model record, normalized to registry shape."""

    canonical_id: str
    org: str | None
    name: str
    provider_key: str
    deployment_ref: str | None = None
    context_window: int | None = None
    max_output: int | None = None
    modalities: list[str] = field(default_factory=list)
    tool_calling: bool | None = None
    structured_output: bool | None = None
    reasoning_support: bool | None = None
    intelligence_index: float | None = None
    input_price: float | None = None  # USD per 1M tokens
    output_price: float | None = None
    cached_input_price: float | None = None


@dataclass
class NormalizedArenaRecord:
    """One LMArena leaderboard row, normalized to registry shape."""

    model_name: str
    organization: str | None
    rating: float | None
    rank: int | None
    votes: int | None
    ci_lower: float | None
    ci_upper: float | None
    category: str
    leaderboard_publish_date: str | None = None


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else None
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    """Pull the model list out of an AA v2 payload (schema UNVERIFIED)."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for key in ("data", "models", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def _parse_provider_and_name(record: dict[str, Any]) -> tuple[str, str]:
    """Resolve (provider_key, model name) from a raw AA record defensively."""
    name = _first(record, "model", "model_name", "model_id", "id", "name")
    name = str(name) if name else "unknown"
    provider = _first(record, "provider", "provider_name", "provider_id", "company")
    if provider is None and "/" in name:
        provider, _, short = name.partition("/")
        name = short or name
    return str(provider or "unknown"), name


def _extract_pricing(record: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    pricing = record.get("pricing")
    if not isinstance(pricing, dict):
        pricing = record  # some APIs flatten price fields at the top level
    input_price = _as_float(
        _first(pricing, "input", "input_price", "prompt", "price_input", "input_1m")
    )
    output_price = _as_float(
        _first(pricing, "output", "output_price", "completion", "price_output", "output_1m")
    )
    cached = _as_float(
        _first(pricing, "cached_input", "cache_read", "cached", "price_cached_input")
    )
    return input_price, output_price, cached


def normalize_aa_models(payload: Any) -> list[NormalizedAAModel]:
    """Normalize an AA v2 payload into registry-shaped model records.

    The exact v2 JSON schema is **PENDING VERIFICATION** (docs/data-sources.md
    §1), so every field is extracted defensively from the most likely key
    names; the raw payload is preserved in the snapshot regardless.
    """
    out: list[NormalizedAAModel] = []
    for record in _extract_records(payload):
        provider_key, name = _parse_provider_and_name(record)
        input_price, output_price, cached = _extract_pricing(record)
        modalities = _first(record, "modalities", "modality", "input_modalities")
        if isinstance(modalities, str):
            modalities = [modalities]
        out.append(
            NormalizedAAModel(
                canonical_id=f"{provider_key}/{name}",
                org=provider_key if provider_key != "unknown" else None,
                name=name,
                provider_key=provider_key,
                deployment_ref=_first(record, "deployment", "deployment_ref", "endpoint"),
                context_window=_as_int(
                    _first(record, "context_window", "contextWindow", "context_length")
                ),
                max_output=_as_int(_first(record, "max_output", "maxOutput", "max_tokens")),
                modalities=[str(m) for m in modalities] if isinstance(modalities, list) else [],
                tool_calling=_as_bool(_first(record, "tool_calling", "supports_tools")),
                structured_output=_as_bool(
                    _first(record, "structured_output", "supports_structured_output")
                ),
                reasoning_support=_as_bool(
                    _first(record, "reasoning", "reasoning_support", "supports_reasoning")
                ),
                intelligence_index=_as_float(
                    _first(
                        record,
                        "intelligence_index",
                        "intelligenceIndex",
                        "artificial_analysis_intelligence_index",
                        "intelligence",
                    )
                ),
                input_price=input_price,
                output_price=output_price,
                cached_input_price=cached,
            )
        )
    return out


def normalize_arena_records(
    rows: list[dict[str, Any]], category: str
) -> list[NormalizedArenaRecord]:
    """Normalize raw arena rows (already extracted by the collector)."""
    out: list[NormalizedArenaRecord] = []
    for row in rows:
        name = row.get("model_name")
        if not isinstance(name, str) or not name:
            continue
        out.append(
            NormalizedArenaRecord(
                model_name=name,
                organization=(
                    row.get("organization")
                    if isinstance(row.get("organization"), str)
                    else None
                ),
                rating=_as_float(row.get("rating")),
                rank=_as_int(row.get("rank")),
                votes=_as_int(row.get("votes")),
                ci_lower=_as_float(row.get("ci_lower")),
                ci_upper=_as_float(row.get("ci_upper")),
                category=category,
                leaderboard_publish_date=(
                    str(row["leaderboard_publish_date"])
                    if row.get("leaderboard_publish_date") is not None
                    else None
                ),
            )
        )
    return out
