"""Tests for zero-cost provider resolution and cost-aware neural routing."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import PolicyConfig, load_config
from gentle_ai_model_router.router.neural import neural_rerank
from gentle_ai_model_router.router.policy import (
    rank_candidates,
    resolve_candidate_price,
)


def _engine(tmp_path: Path):
    engine = registry_db.get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    registry_db.init_schema(engine)
    return engine


def test_resolve_candidate_price_precedence(tmp_path: Path):
    engine = _engine(tmp_path)
    policy = PolicyConfig(
        default_input_price=5.0,
        default_output_price=15.0,
        zero_cost_providers=["ollama", "local", "vllm", "opencode"],
        zero_cost_patterns=["*free*", "*:free"],
        pricing_overrides={
            "custom/paid": {"input": 1.5, "output": 3.0},
            "special-prov": {"input": 0.5, "output": 1.0},
        },
    )

    with registry_db.Session(engine) as session:
        prov_local = registry_db.get_or_create_provider(session, "ollama")
        prov_paid = registry_db.get_or_create_provider(session, "anthropic")
        m_local = registry_db.upsert_model(session, canonical_id="ollama/llama3")
        m_paid = registry_db.upsert_model(session, canonical_id="anthropic/claude-3-opus")
        m_free_pattern = registry_db.upsert_model(
            session, canonical_id="openrouter/gemini-flash:free"
        )
        m_override = registry_db.upsert_model(session, canonical_id="custom/paid")

        dep_local = registry_db.get_or_create_deployment(session, m_local, prov_local, "default")
        dep_paid = registry_db.get_or_create_deployment(session, m_paid, prov_paid, "default")
        dep_free_pat = registry_db.get_or_create_deployment(
            session, m_free_pattern, prov_paid, "default"
        )
        dep_override = registry_db.get_or_create_deployment(
            session, m_override, prov_paid, "default"
        )

        # Set a DB price on dep_paid
        registry_db.upsert_price(session, dep_paid, "snap", 10.0, 30.0, None)
        session.commit()

        # 1. Declarative Override
        in_p, out_p, reasons = resolve_candidate_price(
            session, dep_override.id, "custom/paid", "anthropic", policy
        )
        assert (in_p, out_p) == (1.5, 3.0)
        assert reasons == ["pricing_override"]

        # 2. Zero-cost Provider (by provider name or canonical prefix)
        in_p, out_p, reasons = resolve_candidate_price(
            session, dep_local.id, "ollama/llama3", "ollama", policy
        )
        assert (in_p, out_p) == (0.0, 0.0)
        assert reasons == ["zero_cost_provider"]

        # 3. Zero-cost Pattern
        in_p, out_p, reasons = resolve_candidate_price(
            session, dep_free_pat.id, "openrouter/gemini-flash:free", "openrouter", policy
        )
        assert (in_p, out_p) == (0.0, 0.0)
        assert reasons == ["zero_cost_pattern"]

        # 4. Database Price
        in_p, out_p, reasons = resolve_candidate_price(
            session, dep_paid.id, "anthropic/claude-3-opus", "anthropic", policy
        )
        assert (in_p, out_p) == (10.0, 30.0)
        assert reasons == []

        # 5. Default fallback
        m_unknown = registry_db.upsert_model(session, canonical_id="unknown/model")
        dep_unknown = registry_db.get_or_create_deployment(
            session, m_unknown, prov_paid, "default"
        )
        session.commit()
        in_p, out_p, reasons = resolve_candidate_price(
            session, dep_unknown.id, "unknown/model", "anthropic", policy
        )
        assert (in_p, out_p) == (5.0, 15.0)
        assert reasons == ["missing_price_data:using_default"]


def test_deterministic_cost_tie_breaking(tmp_path: Path):
    """When models have identical tokens and quality, cheaper model wins."""
    engine = _engine(tmp_path)
    config = load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")

    with registry_db.Session(engine) as session:
        prov = registry_db.get_or_create_provider(session, "test")
        # Model 1: Paid ($5.0/$15.0)
        m1 = registry_db.upsert_model(session, canonical_id="a-paid-model", tool_calling=True)
        dep1 = registry_db.get_or_create_deployment(session, m1, prov, "default")
        registry_db.upsert_variant(session, dep1, "low", "low")
        registry_db.upsert_price(session, dep1, "snap", 5.0, 15.0, None)

        # Model 2: Free ($0.0/$0.0)
        m2 = registry_db.upsert_model(session, canonical_id="z-free-model", tool_calling=True)
        dep2 = registry_db.get_or_create_deployment(session, m2, prov, "default")
        registry_db.upsert_variant(session, dep2, "low", "low")
        registry_db.upsert_price(session, dep2, "snap", 0.0, 0.0, None)
        session.commit()

        ranking = rank_candidates(session, "explore", config)
        # Even though "a-paid-model" comes first alphabetically, "z-free-model" wins due to cost=0.0
        assert ranking.candidates[0].model.canonical_id == "z-free-model"
        assert ranking.candidates[0].estimated_cost == 0.0
        assert ranking.candidates[1].model.canonical_id == "a-paid-model"
        assert ranking.candidates[1].estimated_cost > 0.0


def test_neural_rerank_cost_awareness(tmp_path: Path):
    """Neural reranking balances affinity score with monetary cost."""
    engine = _engine(tmp_path)
    config = load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")
    config.policy.neural_cost_weight = 20.0
    config.policy.lambda_price = 1.0

    with registry_db.Session(engine) as session:
        prov = registry_db.get_or_create_provider(session, "test")
        # Free model: score=1.0, cost=0.0
        m_free = registry_db.upsert_model(session, canonical_id="model-free", tool_calling=True)
        dep_free = registry_db.get_or_create_deployment(session, m_free, prov, "default")
        registry_db.upsert_variant(session, dep_free, "low", "low")
        registry_db.upsert_price(session, dep_free, "snap", 0.0, 0.0, None)

        # Paid model: score=1.5, cost=0.10 (penalty = 20 * 1.0 * 0.10 = 2.0 -> net utility = -0.5)
        m_paid = registry_db.upsert_model(session, canonical_id="model-paid", tool_calling=True)
        dep_paid = registry_db.get_or_create_deployment(session, m_paid, prov, "default")
        registry_db.upsert_variant(session, dep_paid, "low", "low")
        registry_db.upsert_price(session, dep_paid, "snap", 10.0, 10.0, None)
        session.commit()

        ranking = rank_candidates(session, "explore", config)

        mock_ranker = MagicMock()
        # Paid gets raw 1.5, Free gets raw 1.0
        mock_ranker.score.return_value = [1.0, 1.5]
        mock_ranker.benchmark_feature_names = None

        reranked = neural_rerank(
            session=session,
            ranking=ranking,
            ranker=mock_ranker,
            task="Read local config",
            config=config,
        )

        # Free model wins because 1.0 - 0.0 > 1.5 - (20 * 0.10)
        assert reranked.candidates[0].model.canonical_id == "model-free"
