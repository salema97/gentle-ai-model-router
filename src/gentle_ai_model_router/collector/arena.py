"""LMArena leaderboard collector (Hugging Face ``lmarena-ai/leaderboard-dataset``).

Verified facts (docs/data-sources.md, 2026-09-18): the dataset is public,
non-gated, parquet, CC-BY-4.0, with configs ``text``, ``webdev``, ``agent``,
``search`` (plus others) and splits ``latest`` / ``full``. **There is no
``math`` config** — requesting an absent category fails loudly with a clear
message unless ``skip_unknown=True``.

Only the configs actually needed are downloaded (one ``load_dataset`` call per
category, split ``latest`` by default). Download failures degrade gracefully:
the collector falls back to the previous local snapshot for that source.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from datasets import load_dataset

from gentle_ai_model_router.collector.logging_conf import get_logger
from gentle_ai_model_router.collector.snapshots import SnapshotRecord, SnapshotStore
from gentle_ai_model_router.router.config import LMArenaConfig

logger = get_logger(__name__)

SOURCE_NAME = "lmarena"

# Verified config list from the dataset card (docs/data-sources.md §2).
KNOWN_CONFIGS: tuple[str, ...] = (
    "agent",
    "agent_bash_recovery_steps",
    "agent_praise_complaint",
    "agent_steerability",
    "agent_tool_hallucination",
    "document",
    "document_style_control",
    "image_edit",
    "image_to_video",
    "search",
    "search_factuality",
    "search_style_control",
    "text",
    "text_factuality",
    "text_style_control",
    "text_to_image",
    "text_to_video",
    "vision",
    "vision_style_control",
    "webdev",
    "video_edit",
)


class ArenaCategoryError(Exception):
    """A configured leaderboard category/config does not exist in the dataset."""


class ArenaCollectionError(Exception):
    """The leaderboard could not be fetched and no local fallback exists."""


def _rows_to_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    """Accept HF Dataset rows, mappings, or plain dicts defensively."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
        elif hasattr(row, "_asdict"):
            out.append(dict(row._asdict()))
        else:
            try:
                out.append(dict(row))
            except (TypeError, ValueError):
                continue
    return out


class LMArenaCollector:
    """Collect per-category leaderboard rows into the snapshot store."""

    def __init__(
        self,
        config: LMArenaConfig,
        store: SnapshotStore,
        load_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._load_fn = load_fn or load_dataset

    def _load_category(self, category: str, split: str) -> list[dict[str, Any]]:
        try:
            dataset = self._load_fn(self.config.dataset, category, split=split)
        except (ValueError, KeyError) as exc:
            available = ", ".join(KNOWN_CONFIGS)
            raise ArenaCategoryError(
                f"category '{category}' not found in dataset '{self.config.dataset}'. "
                f"Available configs: {available}. "
                f"(Original error: {exc})"
            ) from exc
        except Exception as exc:  # HF download/network errors
            raise ArenaCollectionError(
                f"failed to download category '{category}' from '{self.config.dataset}': {exc}"
            ) from exc
        if hasattr(dataset, "to_list"):
            return [r for r in dataset.to_list() if isinstance(r, dict)]
        return _rows_to_dicts(dataset)

    def collect(
        self,
        categories: list[str] | None = None,
        split: str = "latest",
        skip_unknown: bool = False,
        now: datetime | None = None,
    ) -> SnapshotRecord:
        """Fetch the configured categories and store one combined snapshot.

        Args:
            categories: Override the configured category list.
            split: Dataset split (``latest`` default; ``full`` for history).
            skip_unknown: Skip absent categories instead of failing loudly.
            now: Inject the current time (tests).

        Raises:
            ArenaCategoryError: A category is absent and ``skip_unknown`` is off.
            ArenaCollectionError: Download failed and no local fallback exists.
        """
        now = (now or datetime.now(tz=UTC)).astimezone(UTC)
        cats = categories if categories is not None else self.config.categories
        errors: list[str] = []
        data: dict[str, Any] = {"dataset": self.config.dataset, "split": split, "categories": {}}
        record_count = 0

        for category in cats:
            try:
                rows = self._load_category(category, split)
            except ArenaCategoryError:
                if skip_unknown:
                    logger.warning(
                        "collect source=%s skipping unknown category=%s", SOURCE_NAME, category
                    )
                    errors.append(f"skipped unknown category '{category}'")
                    continue
                raise
            except ArenaCollectionError as exc:
                fallback = self.store.latest(SOURCE_NAME)
                if fallback is not None:
                    logger.warning(
                        "collect source=%s download_failed fallback snapshot_id=%s error=%s",
                        SOURCE_NAME,
                        fallback.get("snapshot_id"),
                        exc,
                    )
                    return SnapshotRecord(
                        snapshot_id=str(fallback.get("snapshot_id", "unknown")),
                        source=SOURCE_NAME,
                        fetched_at=now,
                        path=self.store.root / "fallback",
                        record_count=int((fallback.get("meta") or {}).get("record_count") or 0),
                        errors=[str(exc)],
                    )
                raise

            records = [extract_arena_record(row, category) for row in rows]
            records = [r for r in records if r is not None]
            data["categories"][category] = records
            record_count += len(records)
            logger.info(
                "collect source=%s category=%s records=%s", SOURCE_NAME, category, len(records)
            )

        record = self.store.save(
            SOURCE_NAME,
            data=data,
            meta={
                "request": {"dataset": self.config.dataset, "split": split, "categories": cats},
                "record_count": record_count,
                "errors": errors,
            },
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


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def parse_organization(model_name: str) -> str | None:
    """Parse the organization from a model name (``org/model`` or ``org:model``)."""
    for sep in ("/", ":"):
        if sep in model_name:
            org, _, rest = model_name.partition(sep)
            if org and rest:
                return org
    return None


def extract_arena_record(row: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Extract one normalized arena record from a raw leaderboard row.

    Column names are detected defensively because the exact schema is only
    partially verified; rows without a usable model name are skipped (None).
    """
    name = _first(row, "model", "model_name", "Model", "name")
    if not name or not isinstance(name, str):
        return None
    rating = _first(row, "rating", "score", "elo", "Arena Elo", "arena_elo")
    ci_lower = _first(row, "ci_lower", "ci_lb", "lower_ci")
    ci_upper = _first(row, "ci_upper", "ci_ub", "upper_ci")
    if ci_lower is None and isinstance(rating, (int, float)):
        ci = _first(row, "ci", "confidence_interval")
        if isinstance(ci, (int, float)):
            ci_lower, ci_upper = rating - ci, rating + ci
    return {
        "model_name": name,
        "organization": parse_organization(name),
        "rating": float(rating) if isinstance(rating, (int, float)) else None,
        "rank": _first(row, "rank", "Rank"),
        "votes": _first(row, "votes", "num_votes", "num_battles", "Votes"),
        "ci_lower": float(ci_lower) if isinstance(ci_lower, (int, float)) else None,
        "ci_upper": float(ci_upper) if isinstance(ci_upper, (int, float)) else None,
        "category": category,
        "leaderboard_publish_date": _first(
            row, "leaderboard_publish_date", "publish_date", "date"
        ),
    }
