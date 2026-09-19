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
    """One Artificial Analysis model record, normalized to registry shape.

    ``benchmarks`` maps registry benchmark key -> score. The verified free
    endpoint (2026-09-19 snapshot, docs/data-sources.md §1) contributes the
    three AA indices plus informational rows (cost-per-task, tps, ttft);
    the legacy defensive path contributes only the intelligence index.
    """

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
    benchmarks: dict[str, float] = field(default_factory=dict)
    input_price: float | None = None  # USD per 1M tokens
    output_price: float | None = None
    cached_input_price: float | None = None
    cached_write_price: float | None = None  # NOT persisted: registry has no column


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


@dataclass
class NormalizedTelemetryRecord:
    """One Gentle AI telemetry agent-model record, normalized to registry shape."""

    canonical_id: str
    org: str
    name: str
    agent_class: str
    rows: int
    responses: int
    tokens_processed: int
    errored_rows: int
    success_rate: float
    tokens_per_response: float
    tokens_per_success: float
    error_categories: str = ""


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
    name = _first(record, "model", "model_name", "model_id", "name", "id")
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


# Benchmark keys inside ``evaluations`` in the verified free-endpoint payload
# (data/snapshots/2026-09-19/artificial-analysis.json). Stored under their full
# AA key names so router.yaml weights match existing registry rows.
_AA_EVALUATION_KEYS = (
    "artificial_analysis_intelligence_index",
    "artificial_analysis_coding_index",
    "artificial_analysis_agentic_index",
)


def _is_real_free_record(record: dict[str, Any]) -> bool:
    """Detect the VERIFIED free-endpoint shape (slug + model_creator/evaluations)."""
    return isinstance(record.get("model_creator"), dict) or "slug" in record


def _normalize_real_free_record(record: dict[str, Any]) -> NormalizedAAModel:
    """Normalize one record of the real ``/language/models/free`` payload.

    Verified field inventory of the free tier (200 records, 2026-09-19):
    mapped — id, name, slug, release_date, model_creator.name, the three
    ``evaluations`` indices, ``artificial_analysis_intelligence_index_cost``
    (cost_per_task only), pricing ``price_1m_{input,output,cache_hit,
    cache_write}_tokens``, performance ``median_output_tokens_per_second``
    and ``median_time_to_first_token_seconds``.

    NOT MAPPED (absent from the free tier; present only in Pro responses):
    context window, max output, modalities, tool_calling, structured_output,
    reasoning support flags, latency percentiles beyond the median. The free
    tier also carries no provider/deployment info: provider_key is the model
    creator and deployment_ref stays None (-> ``default``).
    """
    creator = record.get("model_creator")
    org = creator.get("name") if isinstance(creator, dict) else None
    org = str(org) if org else None
    slug = record.get("slug")
    slug = str(slug) if slug else None
    name = record.get("name")
    name = str(name) if name else (slug or "unknown")

    benchmarks: dict[str, float] = {}
    evaluations = record.get("evaluations")
    if isinstance(evaluations, dict):
        for key in _AA_EVALUATION_KEYS:
            score = _as_float(evaluations.get(key))
            if score is not None:
                benchmarks[key] = score

    # Informational benchmark rows (NOT part of any default phase weights).
    # Index cost is a nested object; only the per-task figure is a plain number.
    cost = record.get("artificial_analysis_intelligence_index_cost")
    if isinstance(cost, dict):
        per_task = cost.get("cost_per_task")
        value = _as_float(per_task.get("total_cost") if isinstance(per_task, dict) else None)
        if value is not None:
            benchmarks["aa_intelligence_index_cost_per_task"] = value

    pricing = record.get("pricing")
    pricing = pricing if isinstance(pricing, dict) else {}
    # Units: AA prices are USD per 1M tokens — the SAME unit the registry's
    # ModelPrice columns use (estimate_cost divides by 1_000_000), so values
    # are stored verbatim, no conversion.
    input_price = _as_float(pricing.get("price_1m_input_tokens"))
    output_price = _as_float(pricing.get("price_1m_output_tokens"))
    cached = _as_float(pricing.get("price_1m_cache_hit_tokens"))
    cached_write = _as_float(pricing.get("price_1m_cache_write_tokens"))

    # Performance medians are stored as benchmark rows so the policy's
    # speed_benchmark mechanism (inverse-normalized tps proxy) can use them
    # once wired into router.yaml; ttft is informational (lower = better).
    performance = record.get("performance")
    performance = performance if isinstance(performance, dict) else {}
    tps = _as_float(performance.get("median_output_tokens_per_second"))
    if tps is not None:
        benchmarks["aa_median_output_tps"] = tps
    ttft = _as_float(performance.get("median_time_to_first_token_seconds"))
    if ttft is not None:
        benchmarks["aa_median_ttft_seconds"] = ttft

    # Canonical id: ``<creator>/<slug>``. The slug is unique across the
    # verified snapshot (checked: 0 duplicates among 200 records) and is far
    # more stable/matchable than the uuid ``id`` or the display ``name``
    # (which carries variant suffixes like "(Non-reasoning)").
    if org and slug:
        canonical_id = f"{org}/{slug}"
    else:
        canonical_id = slug or f"{org or 'unknown'}/{name}"
    return NormalizedAAModel(
        canonical_id=canonical_id,
        org=org,
        name=name,
        provider_key=org or "unknown",
        deployment_ref=None,
        # context_window / max_output / modalities / capability flags: NOT
        # MAPPED — the free endpoint does not include them.
        intelligence_index=benchmarks.get("artificial_analysis_intelligence_index"),
        benchmarks=benchmarks,
        input_price=input_price,
        output_price=output_price,
        cached_input_price=cached,
        cached_write_price=cached_write,
    )


def normalize_aa_models(payload: Any) -> list[NormalizedAAModel]:
    """Normalize an AA v2 payload into registry-shaped model records.

    Two payload shapes are supported:

    - the VERIFIED ``/language/models/free`` shape (docs/data-sources.md §1,
      observed in data/snapshots/2026-09-19/artificial-analysis.json): a
      wrapper object ``{"tier", "pagination", "data": [records]}`` whose
      records carry ``slug`` / ``model_creator`` / ``evaluations`` /
      ``pricing.price_1m_*`` / ``performance.median_*``;
    - the pre-verification defensive guesses (flat records with
      ``model``/``provider``/``pricing.input`` style keys), kept so older
      fixtures and hypothetical Pro shapes still normalize.
    """
    out: list[NormalizedAAModel] = []
    for record in _extract_records(payload):
        if _is_real_free_record(record):
            out.append(_normalize_real_free_record(record))
            continue
        provider_key, name = _parse_provider_and_name(record)
        input_price, output_price, cached = _extract_pricing(record)
        modalities = _first(record, "modalities", "modality", "input_modalities")
        if isinstance(modalities, str):
            modalities = [modalities]
        intelligence_index = _as_float(
            _first(
                record,
                "intelligence_index",
                "intelligenceIndex",
                "artificial_analysis_intelligence_index",
                "intelligence",
            )
        )
        benchmarks = (
            {"artificial_analysis_intelligence_index": intelligence_index}
            if intelligence_index is not None
            else {}
        )
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
                intelligence_index=intelligence_index,
                benchmarks=benchmarks,
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


_PLACEHOLDERS = {"nan", "none", "null", ""}


def _parse_telemetry_model(raw_model: str) -> tuple[str, str, str]:
    """Resolve (canonical_id, org, name) from a raw telemetry model string."""
    raw = raw_model.strip()
    if "/" in raw:
        org, _, name = raw.partition("/")
        org = org.strip()
        name = name.strip()
    else:
        org = ""
        name = raw

    lowered_org = org.lower()
    lowered_name = name.lower()

    if lowered_org in _PLACEHOLDERS:
        org = "unknown"
    if lowered_name in _PLACEHOLDERS:
        name = "unknown"

    if lowered_name == "custom" and org == "unknown":
        org = "custom"

    canonical_id = f"{org}/{name}"
    return canonical_id, org, name


def normalize_telemetry_records(payload: Any) -> list[NormalizedTelemetryRecord]:
    """Normalize Gentle AI telemetry records into registry-shaped records."""
    if isinstance(payload, list):
        raw_records = [r for r in payload if isinstance(r, dict)]
    elif isinstance(payload, dict):
        if isinstance(payload.get("agent_models"), list):
            raw_records = [r for r in payload["agent_models"] if isinstance(r, dict)]
        elif isinstance(payload.get("data"), dict) and isinstance(
            payload["data"].get("agent_models"), list
        ):
            raw_records = [r for r in payload["data"]["agent_models"] if isinstance(r, dict)]
        elif isinstance(payload.get("data"), list):
            raw_records = [r for r in payload["data"] if isinstance(r, dict)]
        else:
            raw_records = []
    else:
        raw_records = []

    out: list[NormalizedTelemetryRecord] = []
    for record in raw_records:
        raw_model = _first(record, "model", "model_name", "name")
        if raw_model is None:
            continue
        raw_model_str = str(raw_model).strip()
        if not raw_model_str:
            continue

        canonical_id, org, name = _parse_telemetry_model(raw_model_str)

        rows = _as_int(record.get("rows")) or 0
        responses = _as_int(record.get("responses")) or 0
        tokens_processed = _as_int(record.get("tokens_processed")) or 0
        errored_rows = _as_int(record.get("errored_rows")) or 0

        success_rate = _as_float(record.get("success_rate"))
        if success_rate is None:
            success_rate = (rows - errored_rows) / rows if rows > 0 else 0.0

        tokens_per_response = _as_float(record.get("tokens_per_response"))
        if tokens_per_response is None:
            tokens_per_response = (
                tokens_processed / responses if responses > 0 else 0.0
            )

        tokens_per_success = _as_float(record.get("tokens_per_success"))
        if tokens_per_success is None:
            successful = rows - errored_rows
            tokens_per_success = (
                tokens_processed / successful if successful > 0 else 0.0
            )

        agent_class = str(record.get("agent_class") or "")
        error_categories = str(record.get("error_categories") or "")

        out.append(
            NormalizedTelemetryRecord(
                canonical_id=canonical_id,
                org=org,
                name=name,
                agent_class=agent_class,
                rows=rows,
                responses=responses,
                tokens_processed=tokens_processed,
                errored_rows=errored_rows,
                success_rate=success_rate,
                tokens_per_response=tokens_per_response,
                tokens_per_success=tokens_per_success,
                error_categories=error_categories,
            )
        )
    return out
