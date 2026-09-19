"""Shared fixtures: temp data dirs, snapshot stores, fake AA payloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from gentle_ai_model_router.collector.snapshots import SnapshotStore
from gentle_ai_model_router.router.config import RouterConfig, load_config


@pytest.fixture(autouse=True)
def _isolate_aa_api_key(monkeypatch):
    """Never let a real .env ARTIFICIAL_ANALYSIS_API_KEY leak into tests.

    pydantic-settings precedence: environment variables beat the .env file,
    so forcing the env var to empty neutralizes any ambient key (an empty
    value is falsy -> the collector uses the free endpoint, as the no-key
    tests assume). Tests that need a key pass it explicitly.
    """
    monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "")


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


@pytest.fixture
def aa_free_payload() -> dict:
    """VERIFIED real free-endpoint payload shape (trimmed to 2 records).

    Hand-written to match data/snapshots/2026-09-19/artificial-analysis.json:
    wrapper {"tier", "pagination", "data": [records]} with slug/model_creator/
    evaluations/pricing.price_1m_*/performance.median_* per record.
    """
    return {
        "tier": "free",
        "intelligence_index_version": 4.3,
        "pagination": {
            "page": 1,
            "page_size": 200,
            "total_pages": 4,
            "has_more": True,
        },
        "data": [
            {
                "id": "0081ab31-d10a-44a0-a10d-eee5533fec65",
                "name": "GLM-4.5V (Non-reasoning)",
                "slug": "glm-4-5v",
                "release_date": "2025-08-11",
                "model_creator": {
                    "id": "67437eb6-7dc1-4e93-befd-22c8b8ec2065",
                    "name": "Z AI",
                },
                "evaluations": {
                    "artificial_analysis_intelligence_index": 6.7,
                    "artificial_analysis_coding_index": None,
                    "artificial_analysis_agentic_index": None,
                },
                "artificial_analysis_intelligence_index_cost": None,
                "pricing": {
                    "price_1m_input_tokens": 0.6,
                    "price_1m_output_tokens": 1.8,
                    "price_1m_cache_hit_tokens": None,
                    "price_1m_cache_write_tokens": None,
                },
                "performance": {
                    "median_output_tokens_per_second": 37.71,
                    "median_time_to_first_token_seconds": 2.71,
                    "median_time_to_first_answer_token_seconds": 2.71,
                    "median_end_to_end_response_time_seconds": 15.97,
                },
            },
            {
                "id": "0097ebf5-124f-42f6-9463-33b00e711f03",
                "name": "Gemini 3.5 Flash (high)",
                "slug": "gemini-3-5-flash",
                "release_date": "2026-05-19",
                "model_creator": {
                    "id": "faddc6d9-2c14-445f-9b28-56726f59c793",
                    "name": "Google",
                },
                "evaluations": {
                    "artificial_analysis_intelligence_index": 33.0,
                    "artificial_analysis_coding_index": 70.1,
                    "artificial_analysis_agentic_index": 27.3,
                },
                "artificial_analysis_intelligence_index_cost": {
                    "total_cost": 2172.43,
                    "cost_per_task": {"total_cost": 1.5625},
                },
                "pricing": {
                    "price_1m_input_tokens": 1.5,
                    "price_1m_output_tokens": 9.0,
                    "price_1m_cache_hit_tokens": 0.15,
                    "price_1m_cache_write_tokens": None,
                },
                "performance": {
                    "median_output_tokens_per_second": 207.83,
                    "median_time_to_first_token_seconds": 14.38,
                    "median_time_to_first_answer_token_seconds": 14.38,
                    "median_end_to_end_response_time_seconds": 16.79,
                },
            },
        ],
    }
