"""Tests for OnnxRanker integration and neural reranking pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gentle_ai_model_router.api.server import create_app
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.config import RouterConfig, load_config
from gentle_ai_model_router.router.neural import build_candidate_features, neural_rerank
from gentle_ai_model_router.router.policy import rank_candidates

AA = "artificial_analysis_intelligence_index"


def _seed_registry(session: Session) -> None:
    """Seed test registry with two distinct models."""
    rows = [
        # canonical, aa score, (in, out) price, context, tool_calling
        ("test/model-a", 30.0, (1.0, 2.0), 128_000, True),
        ("test/model-b", 80.0, (5.0, 15.0), 256_000, True),
    ]
    for canonical, aa, (in_p, out_p), ctx, tools in rows:
        provider = registry_db.get_or_create_provider(session, "test")
        model = registry_db.upsert_model(
            session,
            canonical_id=canonical,
            context_window=ctx,
            max_output=16_000,
            tool_calling=tools,
            structured_output=True,
        )
        deployment = registry_db.get_or_create_deployment(session, model, provider, "default")
        for effort in ("low", "medium", "high"):
            registry_db.upsert_variant(session, deployment, effort, effort)
        registry_db.upsert_benchmark(session, model, AA, aa, None, "snap-test")
        registry_db.upsert_price(session, deployment, "snap-test", in_p, out_p, None)


@pytest.fixture
def config(tmp_path: Path) -> RouterConfig:
    return load_config(config_path="/nonexistent/router.yaml", data_dir=tmp_path / "data")


@pytest.fixture
def engine(tmp_path: Path):
    eng = registry_db.get_engine(f"sqlite:///{tmp_path / 'router.db'}")
    registry_db.init_schema(eng)
    with registry_db.Session(eng) as session:
        _seed_registry(session)
        session.commit()
    return eng


class MockRanker:
    """Mock ranker that scores candidates by preferential substring.

    ``benchmark_feature_names`` emulates the recording training writes into
    the checkpoint metrics.json and OnnxRanker loads onto the instance:
    the artifact-driven serving path (router/neural.py) must build the
    numeric feature vector with EXACTLY these names.
    """

    def __init__(
        self,
        preferred_model: str = "test/model-b",
        high_score: float = 0.95,
        benchmark_feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self.preferred_model = preferred_model
        self.high_score = high_score
        self.benchmark_feature_names = benchmark_feature_names
        self.call_count = 0
        self.last_numeric_features: list[list[float]] = []

    def score(
        self,
        texts: list[str],
        numeric_features: list[list[float]],
        max_length: int | None = None,
    ) -> list[float]:
        self.call_count += 1
        self.last_numeric_features = [list(row) for row in numeric_features]
        scores: list[float] = []
        for text in texts:
            if self.preferred_model in text:
                scores.append(self.high_score)
            else:
                scores.append(0.1)
        return scores


def test_build_candidate_features_shape(engine, config: RouterConfig) -> None:
    """build_candidate_features produces 22 features per candidate."""
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        candidates = list(ranking.candidates)
        assert len(candidates) >= 2

        features = build_candidate_features(session, candidates, config)
        assert len(features) == len(candidates)
        for vec in features:
            assert len(vec) == 22
            # 7 model features + 15 benchmark features
            assert all(isinstance(val, float) for val in vec)

        # Empty candidates produces empty features list
        empty_features = build_candidate_features(session, [], config)
        assert empty_features == []


def test_neural_rerank_with_mock(engine, config: RouterConfig) -> None:
    """neural_rerank re-orders candidates and updates winner reason codes."""
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        # In baseline policy, test/model-b meets floor at low effort, winning on tokens
        assert ranking.candidates[0].model.canonical_id == "test/model-b"

        mock_ranker = MockRanker(preferred_model="test/model-a", high_score=0.987654)
        reranked = neural_rerank(
            session=session,
            ranking=ranking,
            ranker=mock_ranker,
            task="Refactor neural pipeline",
            config=config,
        )

        assert mock_ranker.call_count == 1
        # Now test/model-a should be winner
        winner = reranked.candidates[0]
        assert winner.model.canonical_id == "test/model-a"
        assert winner.score == 0.987654
        assert "neural_ranker:onnx" in winner.reason_codes
        assert reranked.policy_version == f"{ranking.policy_version}+neural"
        assert len(reranked.candidates) == len(ranking.candidates)


# --------------------------------------------------------------------------- #
# R2: serving feature layout must match the trained dim (artifact-driven names)
# --------------------------------------------------------------------------- #


def _rerank_with_names(engine, config: RouterConfig, names: tuple[str, ...] | None) -> MockRanker:
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        mock_ranker = MockRanker(benchmark_feature_names=names)
        reranked = neural_rerank(
            session=session,
            ranking=ranking,
            ranker=mock_ranker,
            task="Refactor neural pipeline",
            config=config,
        )
    assert reranked.candidates[0].model.canonical_id == "test/model-b"
    return mock_ranker


def test_neural_rerank_with_3_name_artifact(engine, config: RouterConfig) -> None:
    names3 = (
        "artificial_analysis_intelligence_index",
        "lmarena_elo:text",
        "lmarena_elo:webdev",
    )
    mock_ranker = _rerank_with_names(engine, config, names3)
    assert mock_ranker.call_count == 1
    # 7 model features + 3 artifact benchmark features.
    assert all(len(row) == 10 for row in mock_ranker.last_numeric_features)


def test_neural_rerank_with_15_name_artifact(engine, config: RouterConfig) -> None:
    from gentle_ai_model_router.router.neural import DEFAULT_BENCHMARK_NAMES

    mock_ranker = _rerank_with_names(engine, config, DEFAULT_BENCHMARK_NAMES)
    assert all(len(row) == 22 for row in mock_ranker.last_numeric_features)


def test_neural_rerank_fallback_reason_code_when_artifact_lacks_names(
    engine, config: RouterConfig
) -> None:
    """Legacy artifact (no recorded names): fall back to the 15-name default
    but stamp the winner reason codes with the fallback explicitly."""
    from gentle_ai_model_router.router.neural import (
        DEFAULT_BENCHMARK_NAMES,
        REASON_BENCH_NAMES_FALLBACK,
        resolve_ranker_benchmark_names,
    )

    names, source = resolve_ranker_benchmark_names(MockRanker(), config)
    assert source == "legacy_default"
    assert names == tuple(DEFAULT_BENCHMARK_NAMES)

    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        mock_ranker = MockRanker()  # no benchmark_feature_names recorded
        reranked = neural_rerank(
            session=session,
            ranking=ranking,
            ranker=mock_ranker,
            task="Refactor neural pipeline",
            config=config,
        )
    winner = reranked.candidates[0]
    assert REASON_BENCH_NAMES_FALLBACK in winner.reason_codes
    assert "neural_ranker:onnx" in winner.reason_codes
    assert all(len(row) == 22 for row in mock_ranker.last_numeric_features)


def test_neural_rerank_artifact_names_no_fallback_reason(engine, config: RouterConfig) -> None:
    from gentle_ai_model_router.router.neural import REASON_BENCH_NAMES_FALLBACK

    names3 = ("artificial_analysis_intelligence_index", "lmarena_elo:text", "lmarena_elo:webdev")
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        mock_ranker = MockRanker(benchmark_feature_names=names3)
        reranked = neural_rerank(
            session=session,
            ranking=ranking,
            ranker=mock_ranker,
            task="Refactor neural pipeline",
            config=config,
        )
    winner = reranked.candidates[0]
    assert REASON_BENCH_NAMES_FALLBACK not in winner.reason_codes


def test_resolve_ranker_benchmark_names_empty_fails_closed(config: RouterConfig) -> None:
    """An artifact that records names but cannot provide a usable list must
    fail closed, not serve with a zero-width feature vector."""
    from gentle_ai_model_router.router.neural import resolve_ranker_benchmark_names

    with pytest.raises(ValueError, match="empty benchmark_feature_names"):
        resolve_ranker_benchmark_names(MockRanker(benchmark_feature_names=()), config)


def test_onnx_ranker_loads_benchmark_feature_names(tmp_path: Path) -> None:
    """OnnxRanker exposes the names training recorded in metrics.json; legacy
    artifacts without the recording get None (fallback decided by neural.py)."""
    import json

    from gentle_ai_model_router.training.onnx_export import load_benchmark_feature_names

    # No metrics.json at all -> legacy artifact.
    assert load_benchmark_feature_names(tmp_path) is None

    names = ["aa_index", "lmarena_elo:text"]
    (tmp_path / "metrics.json").write_text(
        json.dumps({"benchmark_feature_names": names}), encoding="utf-8"
    )
    assert load_benchmark_feature_names(tmp_path) == tuple(names)

    # Key present but unusable -> fail closed.
    (tmp_path / "metrics.json").write_text(
        json.dumps({"benchmark_feature_names": []}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid benchmark_feature_names"):
        load_benchmark_feature_names(tmp_path)


def test_neural_rerank_noop_without_task(engine, config: RouterConfig) -> None:
    """neural_rerank is a no-op if task is empty or ranking has no candidates."""
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        mock_ranker = MockRanker()
        unchanged = neural_rerank(
            session=session,
            ranking=ranking,
            ranker=mock_ranker,
            task="",
            config=config,
        )
        assert mock_ranker.call_count == 0
        assert unchanged == ranking


def test_api_health_endpoint_reports_ranker_status(engine, config: RouterConfig) -> None:
    """GET /health reports ranker: onnx when ranker is wired, else ranker: none."""
    # App without ranker
    app_none = create_app(config, engine, ranker=None)
    client_none = TestClient(app_none)
    res_none = client_none.get("/health")
    assert res_none.status_code == 200
    assert res_none.json()["ranker"] == "none"

    # App with ranker
    mock_ranker = MockRanker()
    app_onnx = create_app(config, engine, ranker=mock_ranker)
    client_onnx = TestClient(app_onnx)
    res_onnx = client_onnx.get("/health")
    assert res_onnx.status_code == 200
    assert res_onnx.json()["ranker"] == "onnx"


def test_api_route_with_task_neural_reranking(engine, config: RouterConfig, tmp_path: Path) -> None:
    """POST /route with task triggers neural reranking and updates reason codes."""
    shim = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    shim.init_schema()
    mock_ranker = MockRanker(preferred_model="test/model-a", high_score=0.912345)
    app = create_app(config, engine, shim_store=shim, ranker=mock_ranker)
    client = TestClient(app)

    # 1. Route with ranker wired -> neural reranking happens
    response = client.post(
        "/route",
        json={"task": "implement routing", "phase": "explore"},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["model"] == "test/model-a"
    assert data["score"] == 0.912345
    assert "neural_ranker:onnx" in data["reason_codes"]
    assert data["policy_version"].endswith("+neural")
    assert mock_ranker.call_count == 1

    # 2. Route with app having no ranker -> deterministic baseline (no neural rerank)
    app_unranked = create_app(config, engine, shim_store=shim, ranker=None)
    client_unranked = TestClient(app_unranked)
    response_unranked = client_unranked.post(
        "/route",
        json={"task": "implement routing", "phase": "explore"},
    )
    assert response_unranked.status_code == 200
    data_unranked = response_unranked.json()
    assert data_unranked["model"] == "test/model-b"
    assert "neural_ranker:onnx" not in data_unranked["reason_codes"]
    assert "cheapest_of_meeting" in data_unranked["reason_codes"]
    assert not data_unranked["policy_version"].endswith("+neural")


def test_neural_rerank_with_real_onnx_ranker(engine, config: RouterConfig) -> None:
    """Test neural reranking end-to-end with the trained ONNX checkpoint."""
    model_path = Path("models/modernbert-router/v9/model.quant.onnx")
    if not model_path.exists():
        model_path = Path("models/deberta-router/v9/model.quant.onnx")
    if not model_path.exists():
        pytest.skip(f"ONNX checkpoint {model_path} not present")

    from gentle_ai_model_router.training.onnx_export import OnnxRanker

    ranker = OnnxRanker(model_path.parent, model_path=model_path)
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        reranked = neural_rerank(
            session=session,
            ranking=ranking,
            ranker=ranker,
            task="Analyze architectural tradeoffs in database indexing",
            config=config,
        )
        assert len(reranked.candidates) == len(ranking.candidates)
        winner = reranked.candidates[0]
        assert "neural_ranker:onnx" in winner.reason_codes
        assert reranked.policy_version.endswith("+neural")
        assert isinstance(winner.score, float)


def test_cli_serve_help() -> None:
    """Verify router serve --help documents --ranker option."""
    from typer.testing import CliRunner

    from gentle_ai_model_router.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--ranker" in result.output
