"""RouterBench dataset collector (withmartian/routerbench).

Collects prompts, model evaluations, costs, latencies, and correctness scores
from the RouterBench benchmark and maps tasks/domains to SDD pipeline phases.
Persists snapshots into the versioned JSON snapshot store under source "routerbench".
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
from gentle_ai_model_router.router.config import RouterBenchConfig

logger = get_logger(__name__)

SOURCE_NAME = "routerbench"

DOMAIN_TO_PHASE: dict[str, str] = {
    # Code generation & programming -> apply
    "code": "apply",
    "coding": "apply",
    "humaneval": "apply",
    "mbpp": "apply",
    "swe": "apply",
    "programming": "apply",
    "python": "apply",
    # Verification & Testing -> verify
    "verification": "verify",
    "verify": "verify",
    "test": "verify",
    "testing": "verify",
    "eval": "verify",
    # Reasoning & Math -> design
    "math": "design",
    "gsm8k": "design",
    "algebra": "design",
    "geometry": "design",
    "reasoning": "design",
    "arc": "design",
    # Logic & Formal specification -> spec
    "logic": "spec",
    "formal": "spec",
    "r2bench": "spec",
    "specification": "spec",
    # Tasks & Planning -> tasks
    "planning": "tasks",
    "plan": "tasks",
    "tasks": "tasks",
    # Research & Information Retrieval -> research
    "research": "research",
    "search": "research",
    # Reading Comprehension & General Exploration -> explore
    "reading_comprehension": "explore",
    "qa": "explore",
    "drop": "explore",
    "drop-800": "explore",
    "squad": "explore",
    "mmlu": "explore",
    "hellaswag": "explore",
    "winogrande": "explore",
    # Propose & Summarization -> propose
    "propose": "propose",
    "summarization": "propose",
    "generation": "propose",
}


def map_routerbench_task_to_phase(task_name: str) -> str:
    """Map a RouterBench task or domain name to an SDD phase."""
    cleaned = task_name.strip().lower()
    if cleaned in DOMAIN_TO_PHASE:
        return DOMAIN_TO_PHASE[cleaned]
    for key in sorted(DOMAIN_TO_PHASE.keys(), key=len, reverse=True):
        if key in cleaned:
            return DOMAIN_TO_PHASE[key]
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


def extract_routerbench_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract and normalize a single RouterBench row."""
    prompt = str(
        row.get("prompt")
        or row.get("query")
        or row.get("input")
        or row.get("question")
        or ""
    )
    task_name = str(
        row.get("task_name")
        or row.get("task")
        or row.get("domain")
        or row.get("dataset")
        or "routerbench"
    )
    model = str(row.get("model_name") or row.get("model") or "")
    models_name = row.get("models_name")
    if not prompt and not model and not models_name:
        return None

    cost = _safe_float(row.get("cost") or row.get("model_cost") or row.get("eval_cost"))
    latency = _safe_float(
        row.get("latency")
        or row.get("response_time")
        or row.get("time")
        or row.get("duration")
    )
    correctness = (
        1.0
        if row.get("win") is True or row.get("correct") is True or row.get("success") is True
        else 0.0
        if row.get("win") is False or row.get("correct") is False or row.get("success") is False
        else _safe_float(
            row.get("correctness")
            if row.get("correctness") is not None
            else row.get("score", 0.5),
            default=0.5,
        )
    )
    input_tokens = _safe_int(
        row.get("input_tokens") or row.get("prompt_tokens") or row.get("tokens_input")
    )
    output_tokens = _safe_int(
        row.get("output_tokens") or row.get("completion_tokens") or row.get("tokens_output")
    )
    mapped_phase = map_routerbench_task_to_phase(task_name)

    sample: dict[str, Any] = {
        "prompt": prompt,
        "task_name": task_name,
        "model_name": model,
        "cost": cost,
        "latency": latency,
        "correctness": correctness,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "mapped_phase": mapped_phase,
    }
    if isinstance(models_name, list):
        sample["models_name"] = [str(m) for m in models_name]
        sample["models_performance"] = [
            _safe_float(p, 0.5) for p in (row.get("models_performance") or [])
        ]
    return sample


class RouterBenchCollectionError(Exception):
    """Raised when RouterBench collection fails and no offline fallback exists."""


class RouterBenchCollector:
    """Collect RouterBench datasets into the snapshot store."""

    def __init__(
        self,
        config: RouterBenchConfig,
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

    def __enter__(self) -> RouterBenchCollector:
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
                sample = extract_routerbench_sample(r)
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
        """Fetch RouterBench samples and persist a snapshot with offline fallback."""
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
            raise RouterBenchCollectionError(
                f"RouterBench collection failed and no cached snapshot exists: {exc}"
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
