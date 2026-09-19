"""Central configuration loading for the model router.

``router.yaml`` is the primary configuration surface; when it is absent the
bundled ``router.yaml.example`` values are used as fallback. Environment
variables win for secrets and deployment-specific overrides (see
:mod:`gentle_ai_model_router.collector.artificial_analysis` for the API key).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_FILENAMES = ("router.yaml", "router.yaml.example")
ENV_CONFIG_VAR = "ROUTER_CONFIG"


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


class DataSourcesConfig(BaseModel):
    artificial_analysis: ArtificialAnalysisConfig = Field(
        default_factory=ArtificialAnalysisConfig
    )
    lmarena: LMArenaConfig = Field(default_factory=LMArenaConfig)
    local_discovery: LocalDiscoveryConfig = Field(default_factory=LocalDiscoveryConfig)


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


class LoggingConfig(BaseModel):
    level: str = "INFO"
    provenance: bool = True


class RouterConfig(BaseModel):
    """Top-level router configuration."""

    data_dir: Path = Path("data")
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    phase_thresholds: dict[str, dict[str, Any]] = Field(default_factory=dict)
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
    overrides: dict[str, Any] = {}
    if data_dir is not None:
        overrides["data_dir"] = Path(data_dir)
    return RouterConfig(**raw, **overrides)
