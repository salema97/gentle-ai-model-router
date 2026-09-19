"""Snapshot store: layout, same-day suffixing, latest pointer, index."""

from __future__ import annotations

from datetime import UTC, datetime

from gentle_ai_model_router.collector.snapshots import SnapshotStore


def test_layout_and_document_shape(store: SnapshotStore) -> None:
    fetched = datetime(2026, 9, 18, 10, 0, tzinfo=UTC)
    record = store.save(
        "artificial-analysis",
        data=[{"a": 1}, {"b": 2}],
        meta={"request": {"path": "/free"}, "errors": []},
        fetched_at=fetched,
    )
    assert record.snapshot_id == "2026-09-18-artificial-analysis"
    assert record.path == store.root / "2026-09-18" / "artificial-analysis.json"
    assert record.path.is_file()
    assert record.record_count == 2

    import json

    doc = json.loads(record.path.read_text())
    assert doc["snapshot_id"] == record.snapshot_id
    assert doc["source"] == "artificial-analysis"
    assert doc["meta"]["request"]["path"] == "/free"

    latest = json.loads((store.root / "artificial-analysis" / "latest.json").read_text())
    assert latest["snapshot_id"] == record.snapshot_id
    assert latest["path"] == "2026-09-18/artificial-analysis.json"

    index = store.index("artificial-analysis")
    assert len(index) == 1
    assert index[0]["snapshot_id"] == record.snapshot_id


def test_same_day_never_overwrites(store: SnapshotStore) -> None:
    fetched = datetime(2026, 9, 18, 10, 0, tzinfo=UTC)
    first = store.save("aa", data=[1], fetched_at=fetched)
    second = store.save("aa", data=[1, 2], fetched_at=fetched)
    third = store.save("aa", data=[1, 2, 3], fetched_at=fetched)
    assert first.snapshot_id == "2026-09-18-aa"
    assert second.snapshot_id == "2026-09-18-aa-2"
    assert third.snapshot_id == "2026-09-18-aa-3"
    assert (store.root / "2026-09-18" / "aa.json").is_file()
    assert (store.root / "2026-09-18" / "aa-2.json").is_file()
    assert (store.root / "2026-09-18" / "aa-3.json").is_file()
    assert len(store.index("aa")) == 3


def test_latest_and_freshness(store: SnapshotStore) -> None:
    from datetime import timedelta

    now = datetime.now(tz=UTC)
    old_time = now - timedelta(hours=48)
    store.save("lmarena", data={"categories": {}}, fetched_at=old_time)
    store.save("lmarena", data={"categories": {"text": []}}, fetched_at=now)

    latest = store.latest_record("lmarena")
    assert latest is not None
    assert latest.snapshot_id.startswith(now.date().isoformat())

    assert store.latest_fresh("lmarena", 24) is not None
    assert store.latest_fresh("missing-source", 24) is None

    doc = store.latest("lmarena")
    assert doc is not None
    assert "text" in doc["data"]["categories"]


def test_latest_missing_returns_none(store: SnapshotStore) -> None:
    assert store.latest("nope") is None
    assert store.latest_record("nope") is None
    assert store.index("nope") == []
