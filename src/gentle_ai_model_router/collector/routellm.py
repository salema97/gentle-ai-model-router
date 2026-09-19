"""RouteLLM dataset collector (lm-sys/RouteLLM).

Collects model pairwise battles and preference thresholds for routing between
lightweight and frontier models, mapping win/loss outcomes to calibration labels
for Choice and Noul and SDD pipeline phases. Persists snapshots into the versioned
JSON snapshot store under source "routellm".
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx
import pyarrow.parquet as pq

from gentle_ai_model_router.collector.logging_conf import get_logger
from gentle_ai_model_router.collector.snapshots import SnapshotRecord, SnapshotStore
from gentle_ai_model_router.router.config import RouteLLMConfig

logger = get_logger(__name__)

SOURCE_NAME = "routellm"

TASK_TO_PHASE: dict[str, str] = {
    # Code & programming -> apply
    "code": "apply",
    "coding": "apply",
    "humaneval": "apply",
    "mbpp": "apply",
    "programming": "apply",
    "python": "apply",
    # Math & reasoning -> design
    "math": "design",
    "gsm8k": "design",
    "reasoning": "design",
    "arc": "design",
    # Spec & Logic -> spec
    "logic": "spec",
    "spec": "spec",
    "specification": "spec",
    # Tasks & Planning -> tasks
    "planning": "tasks",
    "tasks": "tasks",
    "plan": "tasks",
    # Verification & QA -> verify
    "verify": "verify",
    "verification": "verify",
    "eval": "verify",
    "judge": "verify",
    # Exploration & QA -> explore
    "qa": "explore",
    "chat": "explore",
    "general": "explore",
    "conversation": "explore",
    "arena": "explore",
    "mmlu": "explore",
    # Propose -> propose
    "writing": "propose",
    "summarization": "propose",
    "propose": "propose",
}


def map_routellm_task_to_phase(task_name: str) -> str:
    """Map a RouteLLM task or category to an SDD phase."""
    cleaned = task_name.strip().lower()
    if cleaned in TASK_TO_PHASE:
        return TASK_TO_PHASE[cleaned]
    for key in sorted(TASK_TO_PHASE.keys(), key=len, reverse=True):
        if key in cleaned:
            return TASK_TO_PHASE[key]
    return "explore"


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or isinstance(val, bool):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    if val is None or isinstance(val, bool):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default


def extract_routellm_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract and normalize a single RouteLLM pairwise battle sample.

    Maps pairwise battles and thresholds to:
    - Choice: preferred candidate model ('model_a', 'model_b', or 'tie')
    - Noul: fast success viability (1.0 if economy model succeeds,
      0.0 if frontier escalation needed)
    """
    prompt = str(
        row.get("prompt")
        or row.get("instruction")
        or row.get("question")
        or row.get("query")
        or ""
    )
    model_a = str(
        row.get("model_a")
        or row.get("small_model")
        or row.get("weak_model")
        or ""
    )
    model_b = str(
        row.get("model_b")
        or row.get("large_model")
        or row.get("strong_model")
        or ""
    )
    if not model_a and not model_b:
        if "gpt4_response" in row or "mixtral_response" in row:
            model_a = "mistralai/mixtral-8x7b-instruct"
            model_b = "openai/gpt-4"
            m_score = _safe_float(row.get("mixtral_score"), default=3.0)
            score_a = m_score / 5.0
            score_b = 1.0
            winner = "model_a" if m_score >= 4.0 else "model_b"
        else:
            return None
    else:
        winner_raw = row.get("winner") or row.get("winner_model") or row.get("label") or ""
        winner = str(winner_raw).strip().lower()
        score_a = _safe_float(
            row.get("score_a") if row.get("score_a") is not None else row.get("quality_a"),
            default=0.5,
        )
        score_b = _safe_float(
            row.get("score_b") if row.get("score_b") is not None else row.get("quality_b"),
            default=0.5,
        )

    sources = row.get("source")
    task_name = str(
        (sources[0] if isinstance(sources, list) and sources else sources)
        or row.get("task_name")
        or row.get("task")
        or row.get("benchmark")
        or row.get("category")
        or "routellm"
    )
    threshold = _safe_float(row.get("threshold") or row.get("routing_threshold"), default=0.5)

    # Choice mapping (candidate preference)
    if winner in ("model_a", "a", model_a.lower()) or (
        winner not in ("model_b", "b", model_b.lower()) and score_a > score_b
    ):
        choice_label = "model_a"
    elif winner in ("model_b", "b", model_b.lower()) or (
        winner not in ("model_a", "a", model_a.lower()) and score_b > score_a
    ):
        choice_label = "model_b"
    else:
        choice_label = "tie"

    # Noul mapping (P(fast_success without escalation))
    # Check if model_a is designated or known as the lightweight/economy model
    small_hints = ("mini", "haiku", "flash", "8b", "7b", "3.5", "small", "lite")
    is_a_small = (
        bool(row.get("small_model") == model_a or row.get("weak_model") == model_a)
        or any(h in model_a.lower() for h in small_hints)
    )
    large_hints = ("opus", "plus", "70b", "405b", "gpt-4", "sonnet", "large")
    is_b_large = (
        bool(row.get("large_model") == model_b or row.get("strong_model") == model_b)
        or any(h in model_b.lower() for h in large_hints)
    )

    if is_a_small or is_b_large:
        # model_a is economy; model_b is frontier
        if choice_label in ("model_a", "tie") or score_a >= threshold:
            noul_label = 1.0
        else:
            noul_label = 0.0
    else:
        # Fallback when no explicit size distinction is known
        noul_label = 1.0 if choice_label in ("model_a", "tie") else 0.0

    mapped_phase = str(row.get("mapped_phase") or map_routellm_task_to_phase(task_name))

    cost_a = _safe_float(row.get("cost_a") or 0.0)
    cost_b = _safe_float(row.get("cost_b") or 0.0)

    return {
        "prompt": prompt,
        "task_name": task_name,
        "model_a": model_a,
        "model_b": model_b,
        "winner": winner,
        "score_a": score_a,
        "score_b": score_b,
        "choice_label": choice_label,
        "noul_label": noul_label,
        "threshold": threshold,
        "cost_a": cost_a,
        "cost_b": cost_b,
        "mapped_phase": mapped_phase,
    }


class RouteLLMCollectionError(Exception):
    """Raised when RouteLLM collection fails and no offline fallback exists."""


class RouteLLMCollector:
    """Collect RouteLLM datasets into the snapshot store."""

    def __init__(
        self,
        config: RouteLLMConfig,
        store: SnapshotStore,
        client: httpx.Client | None = None,
        load_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds)
        )
        self._owns_client = client is None
        self._load_fn = load_fn

    def close(self) -> None:
        """Close HTTP client if owned."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RouteLLMCollector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve_url(self) -> str:
        if self.config.url:
            return self.config.url
        dataset = self.config.dataset.strip()
        if dataset.startswith("http://") or dataset.startswith("https://"):
            return dataset
        return (
            f"https://huggingface.co/datasets/{dataset}/resolve/main/"
            f"{self.config.split}.jsonl"
        )

    def _fetch_samples(self) -> list[dict[str, Any]]:
        if self._load_fn is not None:
            raw_data = self._load_fn(self.config.dataset, split=self.config.split)
            if hasattr(raw_data, "to_list"):
                rows = [r for r in raw_data.to_list() if isinstance(r, dict)]
            elif isinstance(raw_data, list):
                rows = raw_data
            else:
                rows = [dict(r) for r in raw_data if isinstance(r, (dict, tuple))]
        else:
            url = self._resolve_url()
            resp = self._client.get(url, follow_redirects=True)
            resp.raise_for_status()
            content = resp.content
            if url.endswith(".parquet") or content.startswith(b"PAR1"):
                table = pq.read_table(io.BytesIO(content))
                if table.num_rows > self.config.max_samples:
                    table = table.slice(0, self.config.max_samples)
                rows = table.to_pylist()
            else:
                text = resp.text.strip()
                if text.startswith("["):
                    rows = json.loads(text)
                else:
                    rows = []
                    for line in text.splitlines():
                        if line.strip():
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue

        samples: list[dict[str, Any]] = []
        for r in rows:
            if isinstance(r, dict):
                sample = extract_routellm_sample(r)
                if sample is not None:
                    samples.append(sample)
                if len(samples) >= self.config.max_samples:
                    break
        return samples

    def collect(
        self,
        now: datetime | None = None,
        force: bool = False,
        max_age_hours: int | None = None,
    ) -> SnapshotRecord:
        """Fetch RouteLLM samples and persist a snapshot with offline fallback."""
        now = (now or datetime.now(tz=UTC)).astimezone(UTC)
        errors: list[str] = []

        if not force:
            fresh = self.store.latest_fresh(SOURCE_NAME, max_age_hours or 24, now=now)
            if fresh is not None:
                logger.info(
                    "collect source=%s served_from_cache snapshot_id=%s",
                    SOURCE_NAME,
                    fresh.snapshot_id,
                )
                return fresh

        try:
            samples = self._fetch_samples()
        except Exception as exc:
            logger.warning("collect source=%s failed: %s", SOURCE_NAME, exc)
            fallback = self.store.latest(SOURCE_NAME)
            if fallback is not None:
                logger.warning(
                    "collect source=%s using fallback cached snapshot_id=%s",
                    SOURCE_NAME,
                    fallback.get("snapshot_id"),
                )
                return SnapshotRecord(
                    snapshot_id=str(fallback.get("snapshot_id", "unknown")),
                    source=SOURCE_NAME,
                    fetched_at=now,
                    path=self.store.root / "fallback",
                    record_count=int((fallback.get("meta") or {}).get("record_count") or 0),
                    errors=[str(exc)],
                )
            raise RouteLLMCollectionError(
                f"RouteLLM collection failed and no cached snapshot exists: {exc}"
            ) from exc

        payload = {
            "source": SOURCE_NAME,
            "dataset": self.config.dataset,
            "split": self.config.split,
            "samples": samples,
        }
        meta = {
            "source": SOURCE_NAME,
            "record_count": len(samples),
            "errors": errors,
        }
        return self.store.save(SOURCE_NAME, data=payload, meta=meta, fetched_at=now)
