"""Shared fixtures: temp data dirs, snapshot stores, fake AA payloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.router.config import RouterConfig, load_config


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def store(data_dir: Path) -> SnapshotStore:
    return SnapshotStore(data_dir / "snapshots")


@pytest.fixture
def config(tmp_path: Path) -> RouterConfig:
    return load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")


@pytest.fixture
def aa_payload() -> list[dict]:
    """Minimal AA v2-shaped payload (schema UNVERIFIED — defensive keys)."""
    return [
        {
            "model": "claude-sonnet-4",
            "provider": "anthropic",
            "context_window": 200000,
            "max_output": 64000,
            "modalities": ["text"],
            "tool_calling": True,
            "structured_output": True,
            "reasoning": True,
            "intelligence_index": 42.5,
            "pricing": {"input": 3.0, "output": 15.0, "cached_input": 0.3},
        },
        {
            "model": "gpt-5",
            "provider": "openai",
            "context_window": 256000,
            "intelligence_index": 50.1,
            "pricing": {"input": 1.25, "output": 10.0},
        },
    ]
