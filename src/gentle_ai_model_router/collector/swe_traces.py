"""SWE-Traces & Aider trajectory collector.

Collects real software engineering trajectories and benchmark outcomes
(task description, repo, model, tokens, diffs, test pass/fail) and maps them
to SDD pipeline phases (apply, verify, tasks). Persists snapshots into the
versioned JSON snapshot store under source "swe-traces".
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
from gentle_ai_model_router.router.config import SWETracesConfig

logger = get_logger(__name__)

SOURCE_NAME = "swe-traces"

SWE_PHASE_KEYWORDS: dict[str, str] = {
    # Verification & Testing -> verify
    "verify": "verify",
    "verification": "verify",
    "test": "verify",
    "eval": "verify",
    "evaluating": "verify",
    "check": "verify",
    # Implementation & Patching -> apply
    "apply": "apply",
    "patch": "apply",
    "diff": "apply",
    "edit": "apply",
    "code": "apply",
    "coding": "apply",
    "implement": "apply",
    # Planning & Task Decomposition -> tasks
    "tasks": "tasks",
    "plan": "tasks",
    "planning": "tasks",
    "issue": "tasks",
    "triage": "tasks",
    "breakdown": "tasks",
    # Exploration & Discovery -> explore
    "explore": "explore",
    "search": "explore",
    "repo": "explore",
    "discovery": "explore",
}


def map_swe_trace_to_phase(row: dict[str, Any]) -> str:
    """Map a trajectory's step, diff, or action to an SDD phase."""
    step = str(
        row.get("phase")
        or row.get("step")
        or row.get("stage")
        or row.get("action")
        or ""
    ).strip().lower()

    if step:
        for kw, phase in SWE_PHASE_KEYWORDS.items():
            if kw in step:
                return phase

    # If test command or test results are evaluated
    if row.get("test_command") or row.get("eval_cmd"):
        return "verify"

    # If diff/patch is present and being edited/applied
    if row.get("diff") or row.get("patch") or row.get("changes"):
        return "apply"

    # Default for SWE/Aider coding trajectories
    return "apply"


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


def extract_swe_trace_sample(row: dict[str, Any]) -> dict[str, Any] | None:
    """Extract and normalize a single SWE/Aider agent trajectory."""
    instance_id = str(row.get("instance_id") or row.get("task_id") or row.get("id") or "")
    task_desc = str(
        row.get("task_description")
        or row.get("problem_statement")
        or row.get("prompt")
        or row.get("issue")
        or ""
    )
    repo = str(row.get("repo") or row.get("repository") or "unknown/repo")
    model = str(row.get("model_name") or row.get("model") or "")

    if not task_desc and not instance_id:
        return None

    in_tokens = _safe_int(row.get("input_tokens") or row.get("prompt_tokens") or 0)
    out_tokens = _safe_int(row.get("output_tokens") or row.get("completion_tokens") or 0)
    tot_tokens = _safe_int(row.get("total_tokens") or (in_tokens + out_tokens))

    diff = str(
        row.get("diff")
        or row.get("patch")
        or row.get("changes")
        or row.get("model_patch")
        or ""
    )

    raw_test = (
        row.get("test_passed")
        if row.get("test_passed") is not None
        else row.get("pass")
        if row.get("pass") is not None
        else row.get("resolved")
        if row.get("resolved") is not None
        else row.get("success")
        if row.get("success") is not None
        else row.get("task_success")
    )
    if isinstance(raw_test, bool):
        test_passed = raw_test
    elif isinstance(raw_test, (int, float)):
        test_passed = raw_test > 0
    elif isinstance(raw_test, str):
        test_passed = raw_test.strip().lower() in ("true", "1", "passed", "resolved", "success")
    else:
        test_passed = False

    cost = _safe_float(row.get("cost") or row.get("estimated_cost") or 0.0)
    latency = _safe_float(
        row.get("latency")
        or row.get("duration")
        or row.get("response_time")
        or 0.0
    )
    mapped_phase = str(row.get("mapped_phase") or map_swe_trace_to_phase(row))

    return {
        "instance_id": instance_id,
        "task_description": task_desc,
        "repo": repo,
        "model_name": model,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": tot_tokens,
        "diff": diff,
        "test_passed": test_passed,
        "cost": cost,
        "latency": latency,
        "mapped_phase": mapped_phase,
    }


class SWETracesCollectionError(Exception):
    """Raised when SWE-Traces collection fails and no offline fallback exists."""


class SWETracesCollector:
    """Collect SWE agent trajectories into the snapshot store."""

    def __init__(
        self,
        config: SWETracesConfig,
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

    def __enter__(self) -> SWETracesCollector:
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
                sample = extract_swe_trace_sample(r)
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
        """Fetch SWE trajectories and persist a snapshot with offline fallback."""
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
            raise SWETracesCollectionError(
                f"SWE-Traces collection failed and no cached snapshot exists: {exc}"
            ) from exc

        payload = {
            "source": SOURCE_NAME,
            "dataset": self.config.dataset,
            "split": self.config.split,
            "trajectories": samples,
        }
        meta = {
            "source": SOURCE_NAME,
            "record_count": len(samples),
            "errors": errors,
        }
        return self.store.save(SOURCE_NAME, data=payload, meta=meta, fetched_at=now)
