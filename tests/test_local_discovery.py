"""Local discovery: fake config trees under tmp_path (no real home files)."""

from __future__ import annotations

import json
from pathlib import Path

from gentle_ai_model_router.collector.local_discovery import (
    LocalDiscoveryConfig,
    collect_local_candidates,
)

OPENCODE_JSONC = """{
  // legacy agent entries
  "agent": {
    "sdd-explore": {"model": "anthropic/claude-sonnet-4", "variant": "high"},
    "general": {"model": "openai/gpt-5"}
  },
  "agents": {
    "sdd-apply": {"model": "anthropic/claude-opus-4#max"},
  },
}
"""


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> LocalDiscoveryConfig:
    return LocalDiscoveryConfig(
        opencode_config=str(tmp_path / "opencode" / "opencode.jsonc"),
        opencode_variants_cache=str(tmp_path / "gentle-ai" / "cache" / "model-variants.json"),
        pi_models=str(tmp_path / "pi" / "gentle-ai" / "models.json"),
        gentle_state=str(tmp_path / "gentle-ai" / "state.json"),
        codex_profiles_dir=str(tmp_path / "codex"),
        claude_agents_dir=str(tmp_path / "claude" / "agents"),
    )


def test_full_discovery(tmp_path: Path) -> None:
    _write(tmp_path / "opencode" / "opencode.jsonc", OPENCODE_JSONC)
    _write(
        tmp_path / "gentle-ai" / "cache" / "model-variants.json",
        {
            "providers": [
                {
                    "id": "anthropic",
                    "models": {
                        "claude-sonnet-4": {"variants": {"low": {}, "high": {}, "max": {}}},
                    },
                }
            ]
        },
    )
    _write(
        tmp_path / "pi" / "gentle-ai" / "models.json",
        {
            "sdd-explore": {"model": "google/gemini-2.5-pro", "thinking": "high"},
            "sdd-spec": "anthropic/claude-sonnet-4",
        },
    )
    _write(
        tmp_path / "gentle-ai" / "state.json",
        {
            "ModelAssignments": {
                "sdd-verify": {
                    "ProviderID": "anthropic",
                    "ModelID": "claude-opus-4",
                    "Effort": "high",
                }
            },
            "CodexPhaseModelAssignments": {
                "sdd_design": {"model": "openai/gpt-5", "reasoning_effort": "high"}
            },
            "unrelated": "ignored",
        },
    )

    result = collect_local_candidates(_config(tmp_path))
    by_model = {c.model: c for c in result.candidates}

    sonnet = by_model["claude-sonnet-4"]
    assert sonnet.provider == "anthropic"
    assert set(sonnet.efforts) == {"low", "high", "max"}  # merged from variants cache
    prov_paths = {p.json_path for p in sonnet.provenance}
    assert "agent.sdd-explore.model" in prov_paths
    assert any(p.startswith("providers.anthropic.models") for p in prov_paths)

    apply = by_model["claude-opus-4"]
    assert apply.provider == "anthropic"
    assert "max" in apply.efforts  # from agents.*.model #max variant
    assert any(p.json_path == "agents.sdd-apply.model" for p in apply.provenance)
    # effort from Gentle AI state merges too
    assert "high" in apply.efforts

    gpt5 = by_model["gpt-5"]
    assert gpt5.provider == "openai"
    assert "high" in gpt5.efforts  # codex reasoning_effort from state

    gemini = by_model["gemini-2.5-pro"]
    assert gemini.provider == "google"
    assert gemini.efforts == ["high"]

    assert result.notes == []


def test_missing_files_are_notes_not_errors(tmp_path: Path) -> None:
    result = collect_local_candidates(_config(tmp_path))
    assert result.candidates == []
    assert len(result.notes) >= 3  # opencode, variants cache, pi, state


def test_malformed_state_does_not_crash(tmp_path: Path) -> None:
    _write(tmp_path / "gentle-ai" / "state.json", "not json")  # invalid JSON
    result = collect_local_candidates(_config(tmp_path))
    assert result.candidates == []
    assert any("state" in n for n in result.notes)


def test_state_with_weird_assignment_values(tmp_path: Path) -> None:
    _write(
        tmp_path / "gentle-ai" / "state.json",
        {
            "ModelAssignments": {
                "ok": {"model": "anthropic/claude-sonnet-4", "effort": "low"},
                "broken": 42,
                "null": None,
            },
            "ClaudeModelAssignments": "not-a-mapping",
        },
    )
    result = collect_local_candidates(_config(tmp_path))
    assert [c.model for c in result.candidates] == ["claude-sonnet-4"]
    assert result.candidates[0].efforts == ["low"]


def test_variants_cache_flat_shape(tmp_path: Path) -> None:
    """Real v1 cache shape on a live machine: provider -> model -> [variants]."""
    _write(
        tmp_path / "gentle-ai" / "cache" / "model-variants.json",
        {
            "tokengo": {
                "deepseek/deepseek-v4-flash": ["high", "low", "max"],
            },
            "modelis": {
                "claude-sonnet-4-6": ["high", "low"],
            },
        },
    )
    result = collect_local_candidates(_config(tmp_path))
    by_key = {(c.provider, c.model): sorted(c.efforts) for c in result.candidates}
    assert by_key[("tokengo", "deepseek/deepseek-v4-flash")] == ["high", "low", "max"]
    assert by_key[("modelis", "claude-sonnet-4-6")] == ["high", "low"]
