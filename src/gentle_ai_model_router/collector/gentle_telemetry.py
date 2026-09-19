"""Gentle AI Telemetry collector.

Fetches runtime telemetry datasets (agent models, by-model, by-effort)
from Gentle AI's dataset distribution endpoints and snapshots them into the
versioned JSON snapshot store.
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from typing import Any

import httpx

from gentle_ai_model_router.collector.logging_conf import get_logger
from gentle_ai_model_router.collector.snapshots import SnapshotRecord, SnapshotStore
from gentle_ai_model_router.router.config import GentleTelemetryConfig

logger = get_logger(__name__)

SOURCE_NAME = "gentle-telemetry"

ENDPOINT_KEY_MAP: dict[str, str] = {
    "runtime-agent-models.csv": "agent_models",
    "runtime-by-model.csv": "by_model",
    "runtime-by-effort.csv": "by_effort",
}


def _endpoint_to_key(endpoint: str) -> str:
    """Map a CSV endpoint filename to a structured payload key."""
    name = endpoint.rsplit("/", 1)[-1]
    if name in ENDPOINT_KEY_MAP:
        return ENDPOINT_KEY_MAP[name]
    slug = name.removesuffix(".csv").replace("-", "_")
    if slug.startswith("runtime_"):
        slug = slug.removeprefix("runtime_")
    return slug


def _safe_int(val: Any) -> int:
    """Parse an integer defensively, defaulting to 0 on failure."""
    if val is None:
        return 0
    if isinstance(val, int):
        return val
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        try:
            return int(float(str(val).strip()))
        except (ValueError, TypeError):
            return 0


def _parse_row(raw_row: dict[str, Any], is_agent_models: bool) -> dict[str, Any]:
    """Parse a single CSV row, deriving quality and efficiency metrics if agent models."""
    row: dict[str, Any] = {k: v for k, v in raw_row.items() if k is not None}
    if is_agent_models:
        rows = _safe_int(row.get("rows"))
        responses = _safe_int(row.get("responses"))
        tokens = _safe_int(row.get("tokens_processed"))
        errored = _safe_int(row.get("errored_rows"))

        row["rows"] = rows
        row["responses"] = responses
        row["tokens_processed"] = tokens
        row["errored_rows"] = errored
        row["success_rate"] = (rows - errored) / rows if rows else 0.0
        row["tokens_per_response"] = tokens / responses if responses else 0.0
        row["tokens_per_success"] = (
            tokens / (rows - errored) if (rows - errored) > 0 else 0.0
        )
    return row


class GentleTelemetryCollector:
    """Collect Gentle AI telemetry runtime datasets into the snapshot store."""

    def __init__(
        self,
        config: GentleTelemetryConfig,
        store: SnapshotStore,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds)
        )
        self._owns_client = client is None

    def close(self) -> None:
        """Close the underlying HTTP client if owned."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GentleTelemetryCollector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def collect(self, now: datetime | None = None) -> SnapshotRecord:
        """Fetch all configured CSV endpoints and persist a snapshot.

        Degrades gracefully on network errors by recording errors in snapshot
        meta while preserving whatever tables succeeded.

        Args:
            now: Optional timestamp override (used in tests).

        Returns:
            The stored SnapshotRecord.
        """
        now = (now or datetime.now(tz=UTC)).astimezone(UTC)
        errors: list[str] = []
        payload: dict[str, Any] = {
            "source_url": self.config.base_url,
            "agent_models": [],
            "by_model": [],
            "by_effort": [],
        }

        for endpoint in self.config.endpoints:
            clean_endpoint = endpoint.strip()
            url = f"{self.config.base_url.rstrip('/')}/{clean_endpoint.lstrip('/')}"
            key = _endpoint_to_key(clean_endpoint)

            try:
                response = self._client.get(url)
                response.raise_for_status()
                text = response.text
            except Exception as exc:
                msg = f"failed to fetch {clean_endpoint}: {exc}"
                logger.warning("collect source=%s %s", SOURCE_NAME, msg)
                errors.append(msg)
                continue

            try:
                reader = csv.DictReader(io.StringIO(text))
                is_agent_models = (
                    clean_endpoint.endswith("runtime-agent-models.csv")
                    or key == "agent_models"
                )
                table_rows: list[dict[str, Any]] = []
                for raw_row in reader:
                    if not isinstance(raw_row, dict):
                        continue
                    table_rows.append(_parse_row(raw_row, is_agent_models=is_agent_models))
            except Exception as exc:
                msg = f"failed to parse CSV for {clean_endpoint}: {exc}"
                logger.warning("collect source=%s %s", SOURCE_NAME, msg)
                errors.append(msg)
                continue

            payload[key] = table_rows

        total_rows = sum(len(v) for v in payload.values() if isinstance(v, list))
        record = self.store.save(
            SOURCE_NAME,
            data=payload,
            meta={"record_count": total_rows, "errors": errors},
            fetched_at=now,
        )
        logger.info(
            "collect source=%s snapshot_id=%s record_count=%s errors=%s",
            SOURCE_NAME,
            record.snapshot_id,
            record.record_count,
            len(errors),
        )
        return record
