"""Deterministic registry content fingerprint.

:func:`registry_fingerprint` answers "did the registry change?" for API
determinism guarantees (``registry_hash`` in responses) and policy-cache
invalidation — WITHOUT hashing every row on each request.

Approximation (documented, deliberate): the fingerprint covers, per table,
``(row_count, max(id))``. That detects every insert (count and max id move)
and every delete (count moves). It does NOT detect in-place updates of
existing rows (e.g. a benchmark score refreshed in the same row) — there is
no ``updated_at`` column in the registry schema to capture those cheaply.
A full-content hash over 14k+ variant rows would dominate request latency;
callers needing update-detection can force a fingerprint bump by re-running
``router normalize`` (idempotent upserts re-write changed rows) or fall back
to a content hash. This trade-off is intentional for a per-request call.
"""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from gentle_ai_model_router.registry.models import (
    Deployment,
    Model,
    ModelBenchmark,
    ModelPrice,
    ModelSnapshot,
    ModelVariant,
    Provider,
)

# Order is fixed so the serialized payload is deterministic.
_FINGERPRINT_TABLES: tuple[tuple[str, type], ...] = (
    ("providers", Provider),
    ("models", Model),
    ("deployments", Deployment),
    ("model_variants", ModelVariant),
    ("model_benchmarks", ModelBenchmark),
    ("model_prices", ModelPrice),
    ("model_snapshots", ModelSnapshot),
)


def registry_fingerprint(engine: Engine) -> str:
    """Cheap deterministic fingerprint of the registry content.

    Returns ``reg-<sha256[:16]>`` over ordered ``table:count:max_id`` segments.
    See the module docstring for the documented approximation.
    """
    parts: list[str] = []
    with Session(engine) as session:
        for name, table in _FINGERPRINT_TABLES:
            row = session.execute(
                select(func.count(), func.coalesce(func.max(table.id), 0)).select_from(table)
            ).one()
            parts.append(f"{name}:{int(row[0])}:{int(row[1])}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"reg-{digest}"
