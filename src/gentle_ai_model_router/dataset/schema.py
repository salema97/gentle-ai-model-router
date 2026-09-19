"""Dataset V1 schema: examples, preference pairs, manifest.

HONESTY RULE (docs/training.md): by default every label produced by the
builder is a **bootstrap prior** derived from external benchmark data — NOT
ground truth. The optional telemetry bridge (dataset/telemetry_bridge.py)
can add rows with label_provenance='telemetry' carrying measured shim
outcomes. ``label_provenance`` makes the distinction explicit per row, and
the manifest carries a loud provenance statement so no downstream consumer
can mistake priors for measured outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FEATURE_SCHEMA_VERSION = "2"
NORMALIZATION_VERSION = "1"  # internal effort taxonomy (registry/normalize.py)
PROVENANCE_BOOTSTRAP = "bootstrap_prior"
PROVENANCE_TELEMETRY = "telemetry"
PROVENANCE_EMPIRICAL = "empirical_benchmark"

# Schema history (cost_features key set is part of the version):
# - v1: est_input_tokens, est_output_tokens, est_total_tokens, est_cost.
# - v2: adds OPTIONAL actual_input_tokens, actual_output_tokens,
#   actual_total_tokens, actual_cost, latency_ms. These are populated on
#   telemetry-provenance rows (real measured shim values) and None on
#   bootstrap/empirical rows. v1 datasets still load: the loader keeps only
#   present keys, and consumers read est_* keys that exist in every version.
# Parquet note: within one written dataset every row carries the SAME key set
# (None where unmeasured) because pyarrow infers the struct type from the
# first rows — heterogeneous dicts would silently drop keys.

# Fixed-width numeric vectors. Order is part of FEATURE_SCHEMA_VERSION.
MODEL_FEATURE_NAMES: tuple[str, ...] = (
    "log_context_window",
    "log_max_output",
    "input_price_per_1m",
    "output_price_per_1m",
    "tool_calling",
    "structured_output",
    "tps_proxy",  # 0.0 = no speed data / no speed_benchmark configured
)
BENCHMARK_FEATURE_SUFFIX_MISSING = "__missing"


@dataclass(frozen=True)
class CandidateRef:
    """One (model, deployment, effort) candidate."""

    model: str  # canonical id, e.g. "anthropic/claude-sonnet-4"
    deployment: str
    effort: str

    @property
    def key(self) -> str:
        return f"{self.model}#{self.deployment}#{self.effort}"


@dataclass
class DatasetExample:
    """One training example: (phase, task_context, candidate) -> utility label."""

    example_id: str
    task_id: str  # synthetic per (phase, task context) group
    phase: str
    task_type: str | None
    context_tokens: int  # task context size used for cost estimation
    task_text: str
    candidate: CandidateRef
    model_features: list[float]
    benchmark_features: list[float]  # width = len(benchmark feature names)
    benchmark_feature_names: tuple[str, ...]
    # cost keys (schema v1): est_input_tokens, est_output_tokens,
    # est_total_tokens, est_cost. Schema v2 adds optional actual_* keys and
    # latency_ms (float, milliseconds) on telemetry-provenance rows; None on
    # rows without measurements.
    cost_features: dict[str, float | None]
    label_utility: float  # 0..1
    # bootstrap/empirical rows: quality model estimate; telemetry rows:
    # MEASURED outcome (quality_score, or task_success coerced to 0.0/1.0).
    label_quality_estimate: float  # 0..1
    label_provenance: str  # PROVENANCE_BOOTSTRAP | PROVENANCE_TELEMETRY | PROVENANCE_EMPIRICAL
    snapshot_date: str  # ISO date: max date of this example's source snapshots
    split: str = ""  # train | validation | test | temporal_test


@dataclass(frozen=True)
class PreferencePair:
    """Pairwise label: candidate A beats candidate B for (phase, task context)."""

    pair_id: str
    task_id: str
    phase: str
    candidate_a: CandidateRef
    candidate_b: CandidateRef
    score_a: float
    score_b: float
    margin: float
    label_provenance: str
    snapshot_date: str
    split: str = ""


@dataclass
class DatasetV1:
    """In-memory dataset plus build metadata."""

    name: str
    version: int
    examples: list[DatasetExample] = field(default_factory=list)
    pairs: list[PreferencePair] = field(default_factory=list)
    model_feature_names: tuple[str, ...] = MODEL_FEATURE_NAMES
    benchmark_feature_names: tuple[str, ...] = ()
    source_snapshot_ids: list[str] = field(default_factory=list)
    label_provenance_statement: str = ""
    git_commit: str = "unknown"
    created_at: str = ""
    # Anti-leakage bookkeeping (surfaced in the manifest).
    dropped_train_task_overlap: int = 0
    # Telemetry bridge bookkeeping (empty unless the bridge was enabled):
    # read = scored shim rows seen, emitted = telemetry examples produced,
    # skipped = invalid rows dropped with a counted warning.
    telemetry_stats: dict[str, int] = field(default_factory=dict)
    # Quality threshold conditioning:
    threshold_penalty: float = 0.0
    hard_threshold: bool = False

    def split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for example in self.examples:
            counts[example.split] = counts.get(example.split, 0) + 1
        return counts

    def pair_split_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pair in self.pairs:
            counts[pair.split] = counts.get(pair.split, 0) + 1
        return counts

    def manifest(self) -> dict[str, Any]:
        provenances = sorted({e.label_provenance for e in self.examples if e.label_provenance})
        if not provenances:
            label_prov = PROVENANCE_BOOTSTRAP
        elif len(provenances) == 1:
            label_prov = provenances[0]
        else:
            label_prov = "+".join(provenances)

        manifest = {
            "dataset_version": self.version,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "name": self.name,
            "source_snapshot_ids": sorted(self.source_snapshot_ids),
            "split_counts": self.split_counts(),
            "pair_split_counts": self.pair_split_counts(),
            "example_count": len(self.examples),
            "pair_count": len(self.pairs),
            "model_feature_names": list(self.model_feature_names),
            "benchmark_feature_names": list(self.benchmark_feature_names),
            "label_provenance": label_prov,
            "label_provenance_statement": self.label_provenance_statement,
            "dropped_train_task_overlap": self.dropped_train_task_overlap,
            "git_commit": self.git_commit,
            "created_at": self.created_at,
            "threshold_penalty": self.threshold_penalty,
            "hard_threshold": self.hard_threshold,
        }
        if self.telemetry_stats:
            manifest["telemetry"] = dict(self.telemetry_stats)
        return manifest

PROVENANCE_STATEMENT = (
    "ALL labels in this dataset are BOOTSTRAP PRIORS derived from external "
    "benchmark data (Artificial Analysis, LMArena) via the shared quality "
    "model in router/policy.py. They are NOT ground truth: there is (almost) "
    "no real execution telemetry behind them. A learned ranker trained on "
    "these labels learns the deterministic policy's opinion of the priors, "
    "not measured task success. Treat every metric computed against these "
    "labels as an internal-consistency check, never as expected production "
    "quality. Replace with telemetry-derived labels (label_provenance="
    "'telemetry') as soon as the shim data exists."
)

# Appended to PROVENANCE_STATEMENT when the dataset mixes in telemetry rows.
TELEMETRY_PROVENANCE_ADDENDUM = (
    "ADDENDUM (telemetry rows): rows with label_provenance='telemetry' carry "
    "MEASURED outcomes from the router-owned telemetry shim (actual "
    "tokens/latency in cost_features actual_* keys; utility computed with "
    "actual total tokens, threshold-conditioned exactly like the bootstrap "
    "labels). They are real executions, not priors — but volume may be small "
    "and phase-biased toward whatever the router actually routed. Bootstrap "
    "and telemetry rows are comparable by construction (same utility "
    "formula); keep provenance per row when slicing metrics."
)
