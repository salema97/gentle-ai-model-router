"""Artificial Analysis collector: all network mocked with respx."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from gentle_ai_model_router.collector.artificial_analysis import (
    ArtificialAnalysisCollector,
    QuotaExceededError,
    QuotaTracker,
)
from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.router.config import ArtificialAnalysisConfig

BASE = "https://artificialanalysis.ai"
NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


def make_collector(
    store: SnapshotStore,
    data_dir: Path,
    api_key: str | None = None,
    client: httpx.Client | None = None,
) -> ArtificialAnalysisCollector:
    config = ArtificialAnalysisConfig()
    quota = QuotaTracker(data_dir / "aa-quota.json", budget=config.quota_requests_per_24h)
    return ArtificialAnalysisCollector(
        config, store, api_key=api_key, client=client, quota=quota
    )


@respx.mock
def test_success_path_authenticated(store: SnapshotStore, data_dir: Path, aa_payload) -> None:
    route = respx.get(f"{BASE}/api/v2/language/models").mock(
        return_value=httpx.Response(200, json=aa_payload)
    )
    collector = make_collector(
        store, data_dir, api_key="secret", client=httpx.Client(base_url=BASE, timeout=5)
    )
    record = collector.collect(now=NOW)
    assert record.record_count == 2
    assert record.errors == []
    assert route.called
    assert collector.quota.remaining(NOW) == 99  # one request consumed


@respx.mock
def test_free_endpoint_used_without_key(store: SnapshotStore, data_dir: Path, aa_payload) -> None:
    route = respx.get(f"{BASE}/api/v2/language/models/free").mock(
        return_value=httpx.Response(200, json=aa_payload)
    )
    collector = make_collector(store, data_dir, api_key=None, client=httpx.Client(base_url=BASE))
    record = collector.collect(now=NOW)
    assert route.called
    assert record.record_count == 2


@respx.mock
def test_429_retry_then_success(store: SnapshotStore, data_dir: Path, aa_payload) -> None:
    route = respx.get(f"{BASE}/api/v2/language/models/free").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json=aa_payload),
        ]
    )
    collector = make_collector(store, data_dir, api_key=None, client=httpx.Client(base_url=BASE))
    record = collector.collect(now=NOW)
    assert route.call_count == 3
    assert record.record_count == 2


@respx.mock
def test_quota_refusal_serves_stale_cache(
    store: SnapshotStore, data_dir: Path, aa_payload
) -> None:
    fresh_time = NOW - timedelta(hours=48)
    store.save("artificial-analysis", data=aa_payload, fetched_at=fresh_time)

    route = respx.get(f"{BASE}/api/v2/language/models/free").mock(
        return_value=httpx.Response(200, json=aa_payload)
    )
    quota = QuotaTracker(data_dir / "aa-quota.json", budget=100)
    quota.consume(100, NOW)
    collector = ArtificialAnalysisCollector(
        ArtificialAnalysisConfig(),
        store,
        api_key=None,
        client=httpx.Client(base_url=BASE),
        quota=quota,
    )
    record = collector.collect(now=NOW)
    assert not route.called  # never hit the API
    assert record.snapshot_id == f"{fresh_time.date().isoformat()}-artificial-analysis"


def test_quota_refusal_without_any_cache_raises(
    store: SnapshotStore, data_dir: Path
) -> None:
    quota = QuotaTracker(data_dir / "aa-quota.json", budget=100)
    quota.consume(100, NOW)
    collector = ArtificialAnalysisCollector(
        ArtificialAnalysisConfig(),
        store,
        api_key=None,
        client=httpx.Client(base_url=BASE, transport=httpx.MockTransport([])),
        quota=quota,
    )
    with pytest.raises(QuotaExceededError):
        collector.collect(now=NOW)


@respx.mock
def test_force_bypasses_quota(store: SnapshotStore, data_dir: Path, aa_payload) -> None:
    respx.get(f"{BASE}/api/v2/language/models/free").mock(
        return_value=httpx.Response(200, json=aa_payload)
    )
    quota = QuotaTracker(data_dir / "aa-quota.json", budget=100)
    quota.consume(100, NOW)
    collector = ArtificialAnalysisCollector(
        ArtificialAnalysisConfig(),
        store,
        api_key=None,
        client=httpx.Client(base_url=BASE),
        quota=quota,
    )
    record = collector.collect(force=True, now=NOW)
    assert record.record_count == 2


@respx.mock
def test_401_free_endpoint_degrades_gracefully(store: SnapshotStore, data_dir: Path) -> None:
    respx.get(f"{BASE}/api/v2/language/models/free").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )
    collector = make_collector(store, data_dir, api_key=None, client=httpx.Client(base_url=BASE))
    record = collector.collect(now=NOW)  # no exception; graceful degradation
    assert record.errors, "expected the 401 to be recorded in meta.errors"
    assert any("401" in e for e in record.errors)
    assert record.record_count == 0


def test_fresh_cache_short_circuits(store: SnapshotStore, data_dir: Path, aa_payload) -> None:
    store.save("artificial-analysis", data=aa_payload, fetched_at=NOW - timedelta(hours=2))
    collector = make_collector(
        store,
        data_dir,
        api_key=None,
        client=httpx.Client(base_url=BASE, transport=httpx.MockTransport([])),
    )
    record = collector.collect(now=NOW)  # would raise if it tried the network
    assert record.record_count == 2


class TestQuotaTracker:
    def test_window_rolls_after_24h(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(tmp_path / "q.json", budget=100)
        tracker.consume(5, NOW)
        assert tracker.remaining(NOW) == 95
        assert tracker.remaining(NOW + timedelta(hours=25)) == 100

    def test_missing_file_is_empty_budget(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(tmp_path / "nope.json", budget=100)
        assert tracker.remaining(NOW) == 100
