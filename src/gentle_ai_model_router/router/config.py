"""Central configuration loading for the model router.

``router.yaml`` is the primary configuration surface; when it is absent the
bundled ``router.yaml.example`` values are used as fallback. Environment
variables win for secrets and deployment-specific overrides (see
:mod:`gentle_ai_model_router.collector.artificial_analysis` for the API key).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_FILENAMES = ("router.yaml", "router.yaml.example")
ENV_CONFIG_VAR = "ROUTER_CONFIG"

# Shared by training and evaluation: "auto" = cuda when available, else cpu.
DeviceSetting = Literal["auto", "cpu", "cuda"]


class ArtificialAnalysisConfig(BaseModel):
    """Artificial Analysis API v2 settings."""

    base_url: str = "https://artificialanalysis.ai"
    models_path: str = "/api/v2/language/models"
    models_free_path: str = "/api/v2/language/models/free"
    cache_ttl_hours: int = 24
    # Free tier is documented as ~100 requests / 24 h (UNVERIFIED — treat as
    # the safe default and never exceed it without --force).
    quota_requests_per_24h: int = 100


class LMArenaConfig(BaseModel):
    """LMArena leaderboard-dataset settings."""

    dataset: str = "lmarena-ai/leaderboard-dataset"
    splits: list[str] = Field(default_factory=lambda: ["latest"])
    # Verified configs (2026-09-18). NOTE: no "math" config exists.
    categories: list[str] = Field(
        default_factory=lambda: ["text", "webdev", "agent", "search"]
    )


class LocalDiscoveryConfig(BaseModel):
    """Read-only local discovery paths (``~`` expanded at scan time)."""

    opencode_config: str = "~/.config/opencode/opencode.json"
    opencode_variants_cache: str = "~/.gentle-ai/cache/model-variants.json"
    pi_models: str = "~/.pi/gentle-ai/models.json"
    gentle_state: str = "~/.gentle-ai/state.json"
    codex_profiles_dir: str = "~/.codex"
    claude_agents_dir: str = "~/.claude/agents"


class GentleTelemetryConfig(BaseModel):
    """Gentle AI Telemetry dataset settings."""

    enabled: bool = True
    base_url: str = "https://gentlemanprogramming.com/data/datasets"
    endpoints: list[str] = Field(
        default_factory=lambda: [
            "runtime-agent-models.csv",
            "runtime-by-model.csv",
            "runtime-by-effort.csv",
        ]
    )
    timeout_seconds: float = 30.0


class RoutingBenchmarksConfig(BaseModel):
    """Routing benchmarks dataset settings (DARS, xRouteBench, RoutingCompendium)."""

    enabled: bool = True
    dars_dataset: str = "AIGNLAI/DARS"
    xroutebench_dataset: str = "ulab-ai/xRouteBench"
    compendium_dataset: str = "Wikit/RoutingCompendium-perf"
    max_samples_per_source: int = 500
    timeout_seconds: float = 30.0


class RouterBenchConfig(BaseModel):
    """RouterBench dataset settings (withmartian/routerbench via Parquet mirror)."""

    enabled: bool = True
    dataset: str = "Wikit/RoutingCompendium-perf"
    split: str = "train"
    url: str | None = (
        "https://huggingface.co/datasets/Wikit/RoutingCompendium-perf/resolve/main/"
        "data/RouterBench-00000-of-00001.parquet"
    )
    max_samples: int = 500
    timeout_seconds: float = 30.0


class RouteLLMConfig(BaseModel):
    """RouteLLM dataset settings (lm-sys/RouteLLM)."""

    enabled: bool = True
    dataset: str = "routellm/gpt4_dataset"
    split: str = "train"
    url: str | None = (
        "https://huggingface.co/datasets/routellm/gpt4_dataset/resolve/main/train.jsonl"
    )
    max_samples: int = 500
    timeout_seconds: float = 30.0


class SWETracesConfig(BaseModel):
    """SWE-Traces & Aider trajectory dataset settings."""

    enabled: bool = True
    dataset: str = "princeton-nlp/SWE-bench_Lite"
    split: str = "test"
    url: str | None = (
        "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/resolve/main/"
        "data/test-00000-of-00001.parquet"
    )
    max_samples: int = 500
    timeout_seconds: float = 30.0


class DataSourcesConfig(BaseModel):
    artificial_analysis: ArtificialAnalysisConfig = Field(
        default_factory=ArtificialAnalysisConfig
    )
    lmarena: LMArenaConfig = Field(default_factory=LMArenaConfig)
    local_discovery: LocalDiscoveryConfig = Field(default_factory=LocalDiscoveryConfig)
    gentle_telemetry: GentleTelemetryConfig = Field(
        default_factory=GentleTelemetryConfig
    )
    routing_benchmarks: RoutingBenchmarksConfig = Field(
        default_factory=RoutingBenchmarksConfig
    )
    routerbench: RouterBenchConfig = Field(default_factory=RouterBenchConfig)
    routellm: RouteLLMConfig = Field(default_factory=RouteLLMConfig)
    swe_traces: SWETracesConfig = Field(default_factory=SWETracesConfig)


class RegistryConfig(BaseModel):
    """Registry database selection.

    Precedence: ``registry.database_url`` in ``router.yaml`` > ``DATABASE_URL``
    env > SQLite under ``data_dir``.
    """

    database_url: str | None = None
    sqlite_filename: str = "router.db"


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8377
    route_path: str = "/route"
    # Telemetry shim SQLite used to persist /route decisions. Accepts a plain
    # filesystem path or a SQLAlchemy URL. None falls back to
    # ``RouterConfig.telemetry_url`` (data_dir/telemetry.sqlite); the app
    # factory also accepts no shim at all (structured log only).
    shim_db_path: str | None = None


class LoggingConfig(BaseModel):
    level: str = "INFO"
    provenance: bool = True


# Default per-phase quality floors (all configurable via router.yaml
# ``phases.<name>.threshold_quality``). apply/verify test success is handled
# by escalation, not by the threshold, hence the lower floor there.
DEFAULT_PHASE_THRESHOLDS: dict[str, float] = {
    "init": 0.6,
    "explore": 0.6,
    "research": 0.65,
    "propose": 0.75,
    "spec": 0.8,
    "design": 0.85,
    "tasks": 0.7,
    "apply": 0.75,
    "verify": 0.8,
    "archive": 0.6,
    "onboard": 0.6,
}

DEFAULT_PHASE_WEIGHTS: dict[str, float] = {
    "artificial_analysis_intelligence_index": 1.0,
}

DEFAULT_EFFORT_QUALITY_GAIN: dict[str, float] = {
    "off": 0.0,
    "minimal": 0.2,
    "low": 0.4,
    "medium": 0.6,
    "high": 0.75,
    "xhigh": 0.85,
    "max": 0.92,
}

# Superlinear token cost multiplier per effort (diminishing returns: cost
# grows faster than the quality gain from DEFAULT_EFFORT_QUALITY_GAIN).
DEFAULT_EFFORT_TOKEN_MULTIPLIER: dict[str, float] = {
    "off": 1.0,
    "minimal": 1.15,
    "low": 1.4,
    "medium": 1.9,
    "high": 2.6,
    "xhigh": 3.5,
    "max": 5.0,
}


class BanditConfig(BaseModel):
    """Constrained bandit parameters (loadable from router.yaml ``bandit:``).

    Lives here (not in bandit.py) so :class:`RouterConfig` can own it without
    an import cycle — bandit.py already imports config.py transitively via
    policy.py. ``router.bandit`` re-exports this class; the public import
    path is unchanged.

    ``seed`` is currently unused by the deterministic UCB score; it is kept
    in the config (and hashed into ``bandit_version``) so any future
    randomized tie-breaking is seeded and reproducible by construction.

    Fail closed: invalid values (negative exploration weight, zero cold-start
    threshold, floors outside (0, 1]) are rejected at load time with a
    pydantic ValidationError — never silently clamped.
    """

    exploration_weight: float = 1.0
    min_executions_before_exploit: int = 3
    seed: int = 42
    quality_floor: float = 0.5  # minimum success_rate to stay UCB-eligible
    phases: dict[str, float] = Field(
        default_factory=dict
    )  # optional per-phase quality_floor overrides

    @field_validator("exploration_weight")
    @classmethod
    def _exploration_weight_non_negative(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError(f"exploration_weight must be >= 0, got {value}")
        return value

    @field_validator("min_executions_before_exploit")
    @classmethod
    def _min_executions_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"min_executions_before_exploit must be >= 1, got {value}")
        return value

    @field_validator("quality_floor")
    @classmethod
    def _quality_floor_in_unit_interval(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError(f"quality_floor must be in (0, 1], got {value}")
        return value

    @field_validator("phases")
    @classmethod
    def _per_phase_floors_in_unit_interval(cls, value: dict[str, float]) -> dict[str, float]:
        for phase, floor in value.items():
            if not 0.0 < floor <= 1.0:
                raise ValueError(f"bandit.phases[{phase!r}] must be in (0, 1], got {floor}")
        return value


class PhaseConfig(BaseModel):
    """Per-phase policy: quality floor + benchmark prior weights.

    Weight keys are registry benchmark rows: ``<benchmark>`` for rows without
    category, ``<benchmark>:<category>`` for categorized ones (e.g.
    ``lmarena_elo:text``).
    """

    threshold_quality: float
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_PHASE_WEIGHTS))


class PolicyConfig(BaseModel):
    """Deterministic baseline ranker parameters."""

    flat_prior: float = 0.5  # prior when a candidate has no benchmark data
    effort_ceiling: float = 0.98  # quality asymptote at max effort
    effort_quality_gain: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_EFFORT_QUALITY_GAIN)
    )
    effort_token_multiplier: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_EFFORT_TOKEN_MULTIPLIER)
    )
    lambda_price: float = 1.0
    lambda_latency: float = 0.0  # off until the registry carries tps data
    speed_benchmark: str | None = None  # benchmark key used as tps proxy
    base_tokens: int = 10_000
    output_fraction: float = 0.3  # share of estimated tokens priced as output
    default_input_price: float = 5.0  # USD/1M, used when price unknown
    default_output_price: float = 15.0
    zero_cost_providers: list[str] = Field(
        default_factory=lambda: [
            "ollama",
            "local",
            "vllm",
            "llama.cpp",
            "opencode",
            "lmstudio",
            "exo",
        ]
    )
    zero_cost_patterns: list[str] = Field(
        default_factory=lambda: [
            "*free*",
            "*:free",
            "*-free",
        ]
    )
    pricing_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    neural_cost_weight: float = 10.0
    top_k: int = 5


class IntegrateConfig(BaseModel):
    """OpenCode write-adapter settings (paths support ~ expansion)."""

    variants_cache_v1: str = "~/.gentle-ai/cache/model-variants.json"
    variants_cache_v2_dir: str = "~/.gentle-ai/cache/opencode-v2"
    backup: bool = True


class ShimConfig(BaseModel):
    """Telemetry shim settings."""

    database_filename: str = "telemetry.sqlite"


class TrainingConfig(BaseModel):
    """ModernBERT ranker training settings (docs/training.md).

    Heavy dependencies (torch/transformers) are imported lazily inside
    training/ — the base install never needs them.
    """

    model_name: str = "answerdotai/ModernBERT-base"
    objective: str = "pointwise"  # pointwise | pairwise (listwise: future work)
    output_dir: str = "models/modernbert-router"
    learning_rate: float = 2e-5
    epochs: float = 2.0
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_length: int = 512
    seed: int = 42
    device: DeviceSetting = "auto"  # auto = cuda when available, else cpu
    telemetry_weight: float = 1.0
    bf16: bool = True
    max_vram_fraction: float | None = 0.5
    disable_tqdm: bool = False


class EvaluationConfig(BaseModel):
    """Ranker scoring settings for `router evaluate` (docs/evaluation.md).

    ``batch_size`` controls how many candidates of a task group are scored
    per forward pass; ``device`` mirrors the training setting. Both only
    apply when a checkpoint is being evaluated — baselines-only runs never
    import torch.
    """

    batch_size: int = 64
    device: DeviceSetting = "auto"


class RouterConfig(BaseModel):
    """Top-level router configuration."""

    data_dir: Path = Path("data")
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    phase_thresholds: dict[str, dict[str, Any]] = Field(default_factory=dict)
    phases: dict[str, PhaseConfig] = Field(default_factory=dict)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    integrate: IntegrateConfig = Field(default_factory=IntegrateConfig)
    shim: ShimConfig = Field(default_factory=ShimConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    token_weights: dict[str, float] = Field(default_factory=dict)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @property
    def snapshot_root(self) -> Path:
        """Root directory for the versioned snapshot store."""
        return self.data_dir / "snapshots"

    @property
    def quota_file(self) -> Path:
        """Local Artificial Analysis quota tracker file."""
        return self.data_dir / "aa-quota.json"

    @property
    def database_url(self) -> str:
        """Resolve the registry database URL with documented precedence."""
        if self.registry.database_url:
            return self.registry.database_url
        env_url = os.environ.get("DATABASE_URL")
        if env_url:
            return env_url
        return f"sqlite:///{self.data_dir / self.registry.sqlite_filename}"

    @property
    def sqlite_fallback_url(self) -> str:
        """SQLite URL used when the configured Postgres is unreachable."""
        return f"sqlite:///{self.data_dir / self.registry.sqlite_filename}"

    def phase_config(self, phase: str) -> PhaseConfig:
        """Resolve per-phase config; accepts ``explore`` or ``sdd-explore``.

        ``phases.<name>`` in router.yaml wins; missing phases fall back to the
        default threshold table with the default weights.
        """
        name = phase.removeprefix("sdd-")
        for key in (phase, name):
            if key in self.phases:
                return self.phases[key]
        return PhaseConfig(threshold_quality=DEFAULT_PHASE_THRESHOLDS[name])

    @property
    def telemetry_url(self) -> str:
        """SQLite URL for the telemetry shim store."""
        return f"sqlite:///{self.data_dir / self.shim.database_filename}"


def find_config_file(config_path: str | Path | None = None) -> Path | None:
    """Resolve the config file: explicit path, ``ROUTER_CONFIG``, or defaults."""
    if config_path is not None:
        path = Path(config_path)
        return path if path.is_file() else None
    env_path = os.environ.get(ENV_CONFIG_VAR)
    if env_path and Path(env_path).is_file():
        return Path(env_path)
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = Path.cwd() / name
        if candidate.is_file():
            return candidate
    # Fall back to the example shipped next to this package's project root.
    example = Path(__file__).resolve().parents[3] / "router.yaml.example"
    return example if example.is_file() else None


def load_config(
    config_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> RouterConfig:
    """Load ``router.yaml`` (or example fallback) into a :class:`RouterConfig`.

    Missing files are not an error: built-in defaults apply. ``data_dir``
    overrides the default ``./data`` location (used by tests and the CLI).
    """
    raw: dict[str, Any] = {}
    path = find_config_file(config_path)
    if path is not None:
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            raw = loaded
    merged = dict(raw)
    if data_dir is not None:
        merged["data_dir"] = Path(data_dir)
    return RouterConfig(**merged)
