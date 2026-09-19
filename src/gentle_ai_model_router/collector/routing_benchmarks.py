"""Empirical routing benchmarks collector (DARS, xRouteBench, RoutingCompendium).

Collects benchmark samples across diverse reasoning, code, and math tasks
and maps them to SDD pipeline phases. Persists snapshots into the
versioned JSON snapshot store under source "routing-benchmarks".
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pyarrow.parquet as pq

from gentle_ai_model_router.collector.logging_conf import get_logger
from gentle_ai_model_router.collector.snapshots import SnapshotRecord, SnapshotStore
from gentle_ai_model_router.router.config import RoutingBenchmarksConfig

logger = get_logger(__name__)

SOURCE_NAME = "routing-benchmarks"

TASK_TO_PHASE: dict[str, str] = {
    "mbpp": "apply",
    "humaneval": "apply",
    "code_eval": "apply",
    "gsm8k": "design",
    "math": "design",
    "gpqa": "design",
    "drop": "explore",
    "drop-800": "explore",
    "reading_comprehension": "explore",
    "abstract2title": "propose",
    "r2bench": "spec",
    "sprout": "tasks",
    "fusionbench": "verify",
}


def map_task_to_phase(task_name: str) -> str:
    """Map a benchmark task or dataset name to an SDD phase."""
    cleaned = task_name.strip().lower()
    if cleaned in TASK_TO_PHASE:
        return TASK_TO_PHASE[cleaned]
    for key, phase in TASK_TO_PHASE.items():
        if key in cleaned:
            return phase
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


class RoutingBenchmarksCollector:
    """Collect empirical routing benchmark datasets into the snapshot store."""

    def __init__(
        self,
        config: RoutingBenchmarksConfig,
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

    def __enter__(self) -> RoutingBenchmarksCollector:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _dars_url(self) -> str:
        dataset = self.config.dars_dataset.strip()
        if dataset.startswith("http://") or dataset.startswith("https://"):
            return dataset
        return (
            f"https://huggingface.co/datasets/{dataset}/resolve/main/drop-800/"
            "train_scored_generations.jsonl"
        )

    def _xroutebench_url(self) -> str:
        dataset = self.config.xroutebench_dataset.strip()
        if dataset.startswith("http://") or dataset.startswith("https://"):
            return dataset
        return (
            f"https://huggingface.co/datasets/{dataset}/resolve/main/"
            "llmrouter_generic/train.parquet"
        )

    def _compendium_url(self) -> str:
        dataset = self.config.compendium_dataset.strip()
        if dataset.startswith("http://") or dataset.startswith("https://"):
            return dataset
        return (
            f"https://huggingface.co/datasets/{dataset}/resolve/main/"
            "data/RouterBench-00000-of-00001.parquet"
        )

    def _fetch_dars(self) -> list[dict[str, Any]]:
        url = self._dars_url()
        samples: list[dict[str, Any]] = []
        with self._client.stream("GET", url, follow_redirects=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue

                query_id = str(row.get("query_id") or row.get("id") or "")
                prompt = str(
                    row.get("input_question")
                    or row.get("prompt")
                    or row.get("question")
                    or ""
                )
                model = str(row.get("model") or row.get("model_name") or "")
                score = row.get("score") if row.get("score") is not None else row.get("quality")
                quality = _safe_float(score)
                cost = _safe_float(row.get("cost"))
                prompt_tokens = _safe_int(row.get("prompt_tokens") or row.get("input_tokens"))
                completion_tokens = _safe_int(
                    row.get("completion_tokens") or row.get("output_tokens")
                )
                task_name = str(row.get("task_name") or row.get("task") or "drop-800")
                mapped_phase = str(row.get("mapped_phase") or map_task_to_phase(task_name))

                samples.append(
                    {
                        "query_id": query_id,
                        "prompt": prompt,
                        "model": model,
                        "quality": quality,
                        "cost": cost,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "task_name": task_name,
                        "mapped_phase": mapped_phase,
                    }
                )
                if len(samples) >= self.config.max_samples_per_source:
                    break
        return samples

    def _fetch_xroutebench(self) -> list[dict[str, Any]]:
        url = self._xroutebench_url()
        samples: list[dict[str, Any]] = []
        resp = self._client.get(url, follow_redirects=True)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))
        if table.num_rows > self.config.max_samples_per_source:
            table = table.slice(0, self.config.max_samples_per_source)
        rows = table.to_pylist()
        for row in rows:
            task_name = str(row.get("task_name") or row.get("task") or "")
            query = str(row.get("query") or row.get("prompt") or "")
            model_name = str(row.get("model_name") or row.get("model") or "")
            perf = (
                row.get("performance")
                if row.get("performance") is not None
                else row.get("score")
            )
            performance = _safe_float(perf)
            input_tokens = _safe_int(row.get("input_tokens") or row.get("prompt_tokens"))
            output_tokens = _safe_int(row.get("output_tokens") or row.get("completion_tokens"))
            resp_time = (
                row.get("response_time")
                if row.get("response_time") is not None
                else row.get("latency")
            )
            response_time = _safe_float(resp_time)
            mapped_phase = str(row.get("mapped_phase") or map_task_to_phase(task_name))

            samples.append(
                {
                    "task_name": task_name,
                    "query": query,
                    "model_name": model_name,
                    "performance": performance,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "response_time": response_time,
                    "mapped_phase": mapped_phase,
                }
            )
        return samples

    def _fetch_compendium(self) -> list[dict[str, Any]]:
        url = self._compendium_url()
        samples: list[dict[str, Any]] = []
        resp = self._client.get(url, follow_redirects=True)
        resp.raise_for_status()
        table = pq.read_table(io.BytesIO(resp.content))
        if table.num_rows > self.config.max_samples_per_source:
            table = table.slice(0, self.config.max_samples_per_source)
        rows = table.to_pylist()
        for row in rows:
            prompt = str(row.get("prompt") or row.get("query") or "")
            dataset = str(row.get("dataset") or row.get("task_name") or row.get("task") or "")
            raw_models = row.get("models_name") or row.get("models") or []
            if isinstance(raw_models, (list, tuple)):
                models_name = [str(m) for m in raw_models]
            elif hasattr(raw_models, "__iter__") and not isinstance(raw_models, str):
                models_name = [str(m) for m in raw_models]
            else:
                models_name = [str(raw_models)] if raw_models else []

            raw_perf = row.get("models_performance") or row.get("performance") or []
            if isinstance(raw_perf, (list, tuple)):
                models_performance = [_safe_float(p) for p in raw_perf]
            elif hasattr(raw_perf, "__iter__") and not isinstance(raw_perf, str):
                models_performance = [_safe_float(p) for p in raw_perf]
            else:
                models_performance = [_safe_float(raw_perf)] if raw_perf is not None else []

            mapped_phase = str(row.get("mapped_phase") or map_task_to_phase(dataset))

            samples.append(
                {
                    "prompt": prompt,
                    "dataset": dataset,
                    "models_name": models_name,
                    "models_performance": models_performance,
                    "mapped_phase": mapped_phase,
                }
            )
        return samples

    def collect(self, now: datetime | None = None) -> SnapshotRecord:
        """Fetch all configured benchmark endpoints and persist a snapshot.

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
            "source": "huggingface",
            "dars": [],
            "xroutebench": [],
            "compendium": [],
        }

        # 1. DARS
        try:
            payload["dars"] = self._fetch_dars()
        except Exception as exc:
            msg = f"failed to fetch DARS: {exc}"
            logger.warning("collect source=%s %s", SOURCE_NAME, msg)
            errors.append(msg)

        # 2. xRouteBench
        try:
            payload["xroutebench"] = self._fetch_xroutebench()
        except Exception as exc:
            msg = f"failed to fetch xRouteBench: {exc}"
            logger.warning("collect source=%s %s", SOURCE_NAME, msg)
            errors.append(msg)

        # 3. Compendium
        try:
            payload["compendium"] = self._fetch_compendium()
        except Exception as exc:
            msg = f"failed to fetch Compendium: {exc}"
            logger.warning("collect source=%s %s", SOURCE_NAME, msg)
            errors.append(msg)

        record_count = (
            len(payload["dars"])
            + len(payload["xroutebench"])
            + len(payload["compendium"])
        )
        meta = {
            "source": SOURCE_NAME,
            "record_count": record_count,
            "errors": errors,
        }

        return self.store.save(
            SOURCE_NAME,
            data=payload,
            meta=meta,
            fetched_at=now,
        )
