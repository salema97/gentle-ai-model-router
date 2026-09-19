"""Artificial Analysis API v2 collector.

- Base host is ``artificialanalysis.ai`` (NOT ``api.artificialanalysis.ai`` —
  verified 2026-09-18; the latter 404s).
- Endpoints: ``/api/v2/language/models`` (API key) and
  ``/api/v2/language/models/free`` (no key). Both return 401 without a key
  (verified live via curl).
- Quota-aware by design: the free tier is documented as ~100 requests / 24 h
  (**UNVERIFIED** — treat as true). A local tracker file counts requests in a
  rolling 24 h window; the collector refuses to call the API when the budget
  would be exceeded unless ``force=True``, and prefers serving a fresh-enough
  cached snapshot instead.
- Collectors must never crash the pipeline: a 401 on the free endpoint (or any
  upstream failure) is recorded in the snapshot meta / surfaced as a warning,
  not an exception that kills the run.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from gentle_ai_model_router.collector.logging_conf import get_logger
from gentle_ai_model_router.collector.snapshots import SnapshotRecord, SnapshotStore
from gentle_ai_model_router.router.config import ArtificialAnalysisConfig

logger = get_logger(__name__)

SOURCE_NAME = "artificial-analysis"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_RETRY_WAIT_SECONDS = 30


class QuotaExceededError(Exception):
    """Raised when the local quota budget for the AA free tier is exhausted."""


class AARetryableError(Exception):
    """HTTP error worth retrying (429 / 5xx); carries the Retry-After hint."""

    def __init__(self, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class AASettings(BaseSettings):
    """Secrets for the Artificial Analysis API (env / .env only, never hardcoded)."""

    artificial_analysis_api_key: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


class QuotaTracker:
    """Rolling 24 h request budget persisted as a small JSON file.

    Shape::

        {"window_start": "<ISO-8601>", "count": <int>}
    """

    def __init__(self, path: Path | str, budget: int = 100) -> None:
        self.path = Path(path)
        self.budget = budget

    def _load(self, now: datetime) -> tuple[datetime, int]:
        try:
            data = json_loads(self.path.read_text(encoding="utf-8"))
            start = datetime.fromisoformat(data["window_start"]).astimezone(UTC)
            count = int(data["count"])
        except (OSError, ValueError, KeyError, TypeError):
            return now, 0
        return start, count

    def _save(self, window_start: datetime, count: int) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json_dumps({"window_start": window_start.isoformat(), "count": count}),
            encoding="utf-8",
        )

    def state(self, now: datetime | None = None) -> tuple[datetime, int]:
        """Return ``(window_start, count)`` rolling the window if it expired."""
        now = (now or datetime.now(tz=UTC)).astimezone(UTC)
        start, count = self._load(now)
        if now - start >= timedelta(hours=24):
            return now, 0
        return start, count

    def remaining(self, now: datetime | None = None) -> int:
        _, count = self.state(now)
        return max(0, self.budget - count)

    def consume(self, n: int = 1, now: datetime | None = None) -> None:
        """Consume budget, rolling the window first if the 24 h window expired."""
        now = (now or datetime.now(tz=UTC)).astimezone(UTC)
        start, count = self.state(now)
        self._save(start, count + n)


def json_loads(text: str) -> Any:  # tiny indirection to keep _load readable + typed
    import json

    return json.loads(text)


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2) + "\n"


def _respect_retry_after(retry_state: RetryCallState) -> float:
    """Tenacity wait: exponential backoff, never less than the Retry-After header."""
    attempt = retry_state.attempt_number
    exponential = min(2 ** (attempt - 1), MAX_RETRY_WAIT_SECONDS)
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = getattr(exc, "retry_after", 0.0) or 0.0
    return max(float(exponential), float(retry_after))


def _request_json(client: httpx.Client, path: str, headers: dict[str, str]) -> Any:
    """GET ``path`` with tenacity retries (429/5xx aware, Retry-After respected)."""

    @retry(
        stop=stop_after_attempt(4),
        wait=_respect_retry_after,
        retry=retry_if_exception_type(AARetryableError),
        reraise=True,
    )
    def _do() -> Any:
        response = client.get(path, headers=headers)
        if response.status_code in RETRYABLE_STATUS_CODES:
            raw = response.headers.get("retry-after", "0")
            try:
                retry_after = float(raw)
            except ValueError:
                retry_after = 0.0
            raise AARetryableError(
                f"AA API returned {response.status_code} for {path}", retry_after
            )
        response.raise_for_status()
        return response.json()

    return _do()


class ArtificialAnalysisCollector:
    """Collect Artificial Analysis model data into the snapshot store."""

    def __init__(
        self,
        config: ArtificialAnalysisConfig,
        store: SnapshotStore,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        quota: QuotaTracker | None = None,
        quota_path: Path | str | None = None,
    ) -> None:
        self.config = config
        self.store = store
        settings = AASettings()
        self.api_key = api_key if api_key is not None else settings.artificial_analysis_api_key
        self._client = client or httpx.Client(
            base_url=config.base_url, timeout=httpx.Timeout(30.0)
        )
        self._owns_client = client is None
        if quota is not None:
            self.quota = quota
        else:
            root = store.root.parent  # data/ next to snapshots/
            self.quota = QuotaTracker(
                Path(quota_path) if quota_path else root / "aa-quota.json",
                budget=config.quota_requests_per_24h,
            )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _endpoint(self) -> tuple[str, dict[str, str]]:
        if self.api_key:
            return self.config.models_path, {"x-api-key": self.api_key}
        return self.config.models_free_path, {}

    def collect(
        self,
        force: bool = False,
        max_age_hours: int | None = None,
        now: datetime | None = None,
    ) -> SnapshotRecord:
        """Fetch (or serve from cache) the AA models payload.

        Args:
            force: Skip the fresh-cache shortcut and the quota refusal.
            max_age_hours: Cache TTL; defaults to ``config.cache_ttl_hours``.
            now: Inject the current time (tests).

        Raises:
            QuotaExceededError: Budget exhausted and no fresh cache available.
            httpx.HTTPError: Upstream failure with no cache fallback.
        """
        now = (now or datetime.now(tz=UTC)).astimezone(UTC)
        max_age = max_age_hours if max_age_hours is not None else self.config.cache_ttl_hours

        if not force:
            fresh = self.store.latest_fresh(SOURCE_NAME, max_age)
            if fresh is not None:
                logger.info(
                    "collect source=%s served_from_cache snapshot_id=%s age_ok=%sh",
                    SOURCE_NAME,
                    fresh.snapshot_id,
                    max_age,
                )
                return fresh
            if self.quota.remaining(now) <= 0:
                cached = self.store.latest_record(SOURCE_NAME)
                if cached is not None:
                    logger.warning(
                        "collect source=%s quota_exhausted serving stale cache snapshot_id=%s",
                        SOURCE_NAME,
                        cached.snapshot_id,
                    )
                    return cached
                raise QuotaExceededError(
                    f"AA quota exhausted ({self.quota.budget}/24h) and no cached snapshot exists"
                )

        path, headers = self._endpoint()
        started = datetime.now(tz=UTC)
        fallback: str | None = None
        try:
            payload = _request_json(self._client, path, headers)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403 and self.api_key:
                # Key without Pro access (e.g. Free-tier key): fall back to the
                # free endpoint, still authenticated — the free endpoint
                # rejects unauthenticated calls (verified live: 401 without
                # key). /language/models returns 403 "requires a Pro
                # subscription" for non-Pro keys.
                logger.warning(
                    "collect source=%s authenticated endpoint 403 (no Pro access) "
                    "— falling back to free endpoint",
                    SOURCE_NAME,
                )
                path, headers = self.config.models_free_path, {"x-api-key": self.api_key}
                try:
                    payload = _request_json(self._client, path, headers)
                except httpx.HTTPStatusError:
                    raise exc from None  # surface the original authenticated-endpoint error
                fallback = "free_endpoint_after_403"
            elif exc.response.status_code == 401 and not self.api_key:
                # Free endpoint rejected us: record the error, degrade gracefully.
                logger.warning(
                    "collect source=%s free_endpoint_401 — recording error and continuing",
                    SOURCE_NAME,
                )
                record = self.store.save(
                    SOURCE_NAME,
                    data=None,
                    meta={
                        "request": {"path": path, "authenticated": False},
                        "record_count": 0,
                        "errors": [
                            f"401 Unauthorized on free endpoint {path}; "
                            "set ARTIFICIAL_ANALYSIS_API_KEY to use the authenticated endpoint"
                        ],
                    },
                    fetched_at=now,
                )
                return record
            else:
                raise

        duration = (datetime.now(tz=UTC) - started).total_seconds()
        self.quota.consume(1, now)
        record_count = _record_count(payload)
        record = self.store.save(
            SOURCE_NAME,
            data=payload,
            meta={
                "request": {"path": path, "authenticated": bool(self.api_key)},
                "record_count": record_count,
                "errors": [],
                "duration_seconds": round(duration, 3),
                "fallback": fallback,
            },
            fetched_at=now,
        )
        logger.info(
            "collect source=%s snapshot_id=%s record_count=%s duration=%.3fs errors=0",
            SOURCE_NAME,
            record.snapshot_id,
            record.record_count,
            duration,
        )
        return record


def _record_count(payload: Any) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("data", "models", "results", "records"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return 0
