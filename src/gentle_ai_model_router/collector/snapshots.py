"""Versioned JSON snapshot store.

Layout::

    data/snapshots/<YYYY-MM-DD>/<source>.json     # one file per snapshot
    data/snapshots/<source>/latest.json           # pointer to newest snapshot
    data/snapshots/<source>/index.json            # append-only list of snapshots

Every snapshot document carries::

    {
      "snapshot_id": "<YYYY-MM-DD>-<source>[-N]",
      "source": "<source>",
      "fetched_at": "<ISO-8601>",
      "data": <raw collector payload>,
      "meta": {"request": {...}, "record_count": int, "errors": [...]}
    }

Snapshots are never silently overwritten: a second snapshot of the same
source on the same day gets a ``-2``, ``-3``, ... suffix.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SLUG_RE = re.compile(r"[^a-z0-9._-]+")


@dataclass
class SnapshotRecord:
    """Handle for a stored snapshot."""

    snapshot_id: str
    source: str
    fetched_at: datetime
    path: Path
    record_count: int = 0
    errors: list[str] = field(default_factory=list)


class SnapshotStore:
    """Filesystem-backed, date-versioned snapshot store."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _safe_slug(self, source: str) -> str:
        slug = SLUG_RE.sub("-", source.strip().lower()).strip("-")
        if not slug:
            raise ValueError(f"invalid snapshot source name: {source!r}")
        return slug

    def _source_dir(self, source: str) -> Path:
        return self.root / self._safe_slug(source)

    def _date_dir(self, fetched_at: datetime) -> Path:
        return self.root / fetched_at.date().isoformat()

    def save(
        self,
        source: str,
        data: Any,
        meta: dict[str, Any] | None = None,
        fetched_at: datetime | None = None,
    ) -> SnapshotRecord:
        """Persist a snapshot; never overwrite an existing same-day file.

        Args:
            source: Collector source name (e.g. ``artificial-analysis``).
            data: Raw payload to keep verbatim.
            meta: Request params, record count, errors, provenance.
            fetched_at: Timestamp of the fetch (defaults to now, UTC).

        Returns:
            The stored :class:`SnapshotRecord`.
        """
        fetched_at = (fetched_at or datetime.now(tz=UTC)).astimezone(UTC)
        meta = dict(meta or {})
        errors = [str(e) for e in meta.get("errors", [])]
        raw_count = meta.get("record_count")
        record_count = int(raw_count if raw_count is not None else _count_records(data))

        slug = self._safe_slug(source)
        date_dir = self._date_dir(fetched_at)
        date_dir.mkdir(parents=True, exist_ok=True)
        n = 1
        target = date_dir / f"{slug}.json"
        while target.exists():  # never overwrite: -2, -3, ... on the same day
            n += 1
            target = date_dir / f"{slug}-{n}.json"
        suffix = f"-{n}" if n > 1 else ""
        snapshot_id = f"{fetched_at.date().isoformat()}-{slug}{suffix}"

        doc = {
            "snapshot_id": snapshot_id,
            "source": source,
            "fetched_at": fetched_at.isoformat(),
            "data": data,
            "meta": meta,
        }
        target.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")

        record = SnapshotRecord(
            snapshot_id=snapshot_id,
            source=source,
            fetched_at=fetched_at,
            path=target,
            record_count=record_count,
            errors=errors,
        )
        self._update_pointers(record)
        return record

    def _update_pointers(self, record: SnapshotRecord) -> None:
        source_dir = self._source_dir(record.source)
        source_dir.mkdir(parents=True, exist_ok=True)
        rel = record.path.relative_to(self.root)
        (source_dir / "latest.json").write_text(
            json.dumps(
                {
                    "snapshot_id": record.snapshot_id,
                    "source": record.source,
                    "fetched_at": record.fetched_at.isoformat(),
                    "path": str(rel),
                    "record_count": record.record_count,
                    "errors": record.errors,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        index_path = source_dir / "index.json"
        entries = self.index(record.source)
        entries.append(
            {
                "snapshot_id": record.snapshot_id,
                "fetched_at": record.fetched_at.isoformat(),
                "path": str(rel),
                "record_count": record.record_count,
                "errors": record.errors,
            }
        )
        index_path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")

    def index(self, source: str) -> list[dict[str, Any]]:
        """Return the per-source index (empty list if none exists)."""
        index_path = self._source_dir(source) / "index.json"
        if not index_path.is_file():
            return []
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []

    def latest(self, source: str) -> dict[str, Any] | None:
        """Return the latest snapshot document, or ``None`` if never collected."""
        pointer = self._source_dir(source) / "latest.json"
        if not pointer.is_file():
            return None
        try:
            info = json.loads(pointer.read_text(encoding="utf-8"))
            doc_path = self.root / info["path"]
            return json.loads(doc_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def latest_record(self, source: str) -> SnapshotRecord | None:
        """Return metadata for the latest snapshot without loading its payload."""
        pointer = self._source_dir(source) / "latest.json"
        if not pointer.is_file():
            return None
        try:
            info = json.loads(pointer.read_text(encoding="utf-8"))
            return SnapshotRecord(
                snapshot_id=info["snapshot_id"],
                source=source,
                fetched_at=datetime.fromisoformat(info["fetched_at"]),
                path=self.root / info["path"],
                record_count=int(info.get("record_count") or 0),
                errors=list(info.get("errors") or []),
            )
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return None

    def latest_fresh(
        self, source: str, max_age_hours: float, now: datetime | None = None
    ) -> SnapshotRecord | None:
        """Return the latest snapshot if younger than ``max_age_hours``, else None."""
        record = self.latest_record(source)
        if record is None:
            return None
        current = (now or datetime.now(tz=UTC)).astimezone(UTC)
        age = current - record.fetched_at.astimezone(UTC)
        return record if age.total_seconds() < max_age_hours * 3600 else None


def _count_records(data: Any) -> int:
    """Best-effort record count for snapshot metadata."""
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("data", "models", "results", "records", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
        return len(data)
    return 0
