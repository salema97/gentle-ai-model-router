"""LMArena collector: datasets.load_dataset fully mocked (no network)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gentle_ai_model_router.collector.arena import (
    ArenaCategoryError,
    ArenaCollectionError,
    LMArenaCollector,
    parse_organization,
)
from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.router.config import LMArenaConfig

NOW = datetime(2026, 9, 18, 12, 0, tzinfo=UTC)


class FakeDataset:
    """Minimal in-memory stand-in for datasets.Dataset."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def to_list(self) -> list[dict]:
        return list(self._rows)


TEXT_ROWS = [
    {
        "model": "anthropic/claude-sonnet-4",
        "rating": 1385.2,
        "rank": 3,
        "votes": 50231,
        "ci_lower": 1379.1,
        "ci_upper": 1391.3,
        "leaderboard_publish_date": "2026-09-16",
    },
    {
        "model": "openai/gpt-5",
        "rating": 1401.7,
        "rank": 1,
        "votes": 61100,
        "ci": 5.0,
        "leaderboard_publish_date": "2026-09-16",
    },
]

WEBDEV_ROWS = [
    {
        "model": "anthropic/claude-sonnet-4",
        "rating": 1450.0,
        "rank": 2,
        "votes": 8000,
        "leaderboard_publish_date": "2026-09-16",
    },
]


def fake_load_dataset(dataset: str, category: str, split: str = "latest"):
    assert split == "latest"
    if category == "text":
        return FakeDataset(TEXT_ROWS)
    if category == "webdev":
        return FakeDataset(WEBDEV_ROWS)
    raise ValueError(
        f"Config '{category}' doesn't exist. Available: ['text', 'webdev', 'agent', 'search']"
    )


def make_collector(store: SnapshotStore, load_fn=fake_load_dataset) -> LMArenaCollector:
    return LMArenaCollector(LMArenaConfig(), store, load_fn=load_fn)


def test_collect_text_and_webdev(store: SnapshotStore) -> None:
    collector = make_collector(store)
    record = collector.collect(categories=["text", "webdev"], now=NOW)
    assert record.record_count == 3
    doc = store.latest("lmarena")
    assert doc is not None
    categories = doc["data"]["categories"]
    assert set(categories) == {"text", "webdev"}

    claude = next(r for r in categories["text"] if r["model_name"] == "anthropic/claude-sonnet-4")
    assert claude["organization"] == "anthropic"
    assert claude["rating"] == 1385.2
    assert claude["rank"] == 3
    assert claude["votes"] == 50231
    assert claude["ci_lower"] == 1379.1
    assert claude["category"] == "text"
    assert claude["leaderboard_publish_date"] == "2026-09-16"

    # ci fallback: rating ± ci
    gpt = next(r for r in categories["text"] if r["model_name"] == "openai/gpt-5")
    assert gpt["ci_lower"] == pytest.approx(1401.7 - 5.0)
    assert gpt["ci_upper"] == pytest.approx(1401.7 + 5.0)


def test_unknown_category_fails_loudly(store: SnapshotStore) -> None:
    collector = make_collector(store)
    with pytest.raises(ArenaCategoryError, match="category 'math' not found"):
        collector.collect(categories=["math"], now=NOW)


def test_unknown_category_skipped_when_configured(store: SnapshotStore) -> None:
    collector = make_collector(store)
    record = collector.collect(categories=["text", "math"], skip_unknown=True, now=NOW)
    assert record.record_count == 2
    assert any("math" in e for e in record.errors)


def test_download_error_falls_back_to_previous_snapshot(store: SnapshotStore) -> None:
    collector = make_collector(store)
    first = collector.collect(categories=["text"], now=NOW)

    def broken_load(dataset: str, category: str, split: str = "latest"):
        raise ConnectionError("HF hub unreachable")

    broken = make_collector(store, load_fn=broken_load)
    record = broken.collect(categories=["text"], now=NOW)
    assert record.snapshot_id == first.snapshot_id
    assert record.errors


def test_download_error_without_fallback_raises(store: SnapshotStore) -> None:
    def broken_load(dataset: str, category: str, split: str = "latest"):
        raise ConnectionError("HF hub unreachable")

    collector = make_collector(store, load_fn=broken_load)
    with pytest.raises(ArenaCollectionError, match="failed to download"):
        collector.collect(categories=["text"], now=NOW)


def test_parse_organization() -> None:
    assert parse_organization("anthropic/claude-sonnet-4") == "anthropic"
    assert parse_organization("openai:gpt-5") == "openai"
    assert parse_organization("gpt-5") is None
