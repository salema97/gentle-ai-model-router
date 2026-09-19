"""Unit and integration tests for Jev System One Calibrated Decision Routing.

Tests:
1. SystemOneModernBERT multi-task architecture and heads with tiny offline config.
2. System One evaluation engine (probability sum, confidence, rubric expectation, noul bounds).
3. FastAPI /route endpoint integration and backward compatibility.
4. Calibration metrics (Brier score, ECE).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from gentle_ai_model_router.api.schemas import RouteResponse, SystemOneMeta
from gentle_ai_model_router.api.server import create_app
from gentle_ai_model_router.integration import telemetry_shim
from gentle_ai_model_router.registry import db as registry_db
from gentle_ai_model_router.router.calibrate import (
    compute_brier_score,
    compute_expected_calibration_error,
)
from gentle_ai_model_router.router.config import RouterConfig, load_config
from gentle_ai_model_router.router.decision import TaskContext
from gentle_ai_model_router.router.policy import rank_candidates
from gentle_ai_model_router.router.system_one import (
    RUBRIC_EFFORT_BINS,
    SystemOneDecision,
    evaluate_system_one,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gentle_ai_model_router.training.model import (  # noqa: E402
    build_system_one_model,
    tiny_modernbert_config,
)

AA = "artificial_analysis_intelligence_index"


def _seed(session: Session) -> None:
    """Seed test registry with multiple models and effort ladders."""
    rows = [
        ("test/cheap-1", 30.0, (1.0, 2.0), 128_000, True),
        ("test/strong", 90.0, (8.0, 30.0), 256_000, True),
        ("test/no-tools", 50.0, (2.0, 6.0), 128_000, False),
    ]
    for canonical, aa, (in_p, out_p), ctx, tools in rows:
        provider = registry_db.get_or_create_provider(session, "test")
        model = registry_db.upsert_model(
            session,
            canonical_id=canonical,
            context_window=ctx,
            tool_calling=tools,
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
        _seed(session)
        session.commit()
    return eng


@pytest.fixture
def client(config: RouterConfig, engine, tmp_path: Path) -> TestClient:
    shim = telemetry_shim.ShimStore(f"sqlite:///{tmp_path / 'telemetry.sqlite'}")
    shim.init_schema()
    return TestClient(create_app(config, engine, shim))


# ---------------------------------------------------------------------------
# 1. SystemOneModernBERT Multi-Task Architecture Tests
# ---------------------------------------------------------------------------


def test_system_one_modernbert_forward() -> None:
    """Test SystemOneModernBERT multi-task forward pass with tiny config."""
    torch.manual_seed(42)
    tiny_cfg = tiny_modernbert_config()
    model = build_system_one_model(tiny_config=tiny_cfg, numeric_dim=4)

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(2, 1024, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    numeric_features = torch.randn(batch_size, 4)

    out = model(input_ids, attention_mask, numeric_features)

    # Choice head: scalar affinity per batch item
    assert out.choice_logits.shape == (batch_size,)

    # Score head: 4 rubric bins, probabilities sum to 1.0
    assert out.score_logits.shape == (batch_size, 4)
    assert out.score_probs.shape == (batch_size, 4)
    assert torch.allclose(
        out.score_probs.sum(dim=-1), torch.ones(batch_size), atol=1e-5
    )

    # Expected effort score: sum(k * p_k)
    k_weights = torch.tensor([0.0, 1.0, 2.0, 3.0])
    manual_expected = (out.score_probs * k_weights).sum(dim=-1)
    assert torch.allclose(out.expected_score, manual_expected, atol=1e-5)
    assert torch.all((out.expected_score >= 0.0) & (out.expected_score <= 3.0))

    # Noul head: binary fast-success probability in [0, 1]
    assert out.noul_logits.shape == (batch_size,)
    assert out.noul_prob.shape == (batch_size,)
    assert torch.all((out.noul_prob >= 0.0) & (out.noul_prob <= 1.0))

    # Dict-like access
    assert torch.equal(out["choice_logits"], out.choice_logits)
    assert torch.equal(out["expected_score"], out.expected_score)


def test_system_one_modernbert_heads_isolated() -> None:
    """Test isolated ChoiceHead, ScoreHead, and NoulHead outputs."""
    torch.manual_seed(42)
    tiny_cfg = tiny_modernbert_config()
    model = build_system_one_model(tiny_config=tiny_cfg, numeric_dim=0)

    pooled = torch.randn(3, tiny_cfg.hidden_size)

    # Test choice_head
    choice = model.choice_head(pooled)
    assert choice.shape == (3,)

    # Test score_head
    score_logits, score_probs, expected = model.score_head(pooled)
    assert score_logits.shape == (3, 4)
    assert score_probs.shape == (3, 4)
    assert expected.shape == (3,)
    assert torch.allclose(score_probs.sum(dim=-1), torch.ones(3), atol=1e-5)

    # Test noul_head
    noul_logit, noul_prob = model.noul_head(pooled)
    assert noul_logit.shape == (3,)
    assert noul_prob.shape == (3,)
    assert torch.all((noul_prob >= 0.0) & (noul_prob <= 1.0))


def test_system_one_modernbert_save_pretrained(tmp_path: Path) -> None:
    """Test checkpoint saving for SystemOneModernBERT."""
    tiny_cfg = tiny_modernbert_config()
    model = build_system_one_model(tiny_config=tiny_cfg, numeric_dim=2)
    save_dir = tmp_path / "sys1_ckpt"
    model.save_pretrained(str(save_dir))

    assert (save_dir / "config.json").exists()
    assert (save_dir / "system_one_heads.pt").exists()
    heads_sd = torch.load(save_dir / "system_one_heads.pt", weights_only=True)
    assert "linear.weight" in heads_sd["choice_head"]
    assert "linear.weight" in heads_sd["score_head"]
    assert "linear.weight" in heads_sd["noul_head"]
    assert heads_sd["numeric_dim"] == 2
    assert heads_sd["hidden_size"] == tiny_cfg.hidden_size


# ---------------------------------------------------------------------------
# 2. System One Evaluation Engine Tests
# ---------------------------------------------------------------------------


def test_evaluate_system_one_baseline(engine, config: RouterConfig) -> None:
    """Test evaluate_system_one with ranking from deterministic baseline."""
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        context = TaskContext(context_tokens=10_000)

        sys1 = evaluate_system_one(ranking, context)

        # Choice arm probabilities sum to 1.0
        assert isinstance(sys1, SystemOneDecision)
        assert sum(sys1.probabilities.values()) == pytest.approx(1.0, abs=1e-4)

        # Confidence is max arm probability and in [0, 1]
        assert 0.0 <= sys1.confidence <= 1.0
        assert sys1.confidence == pytest.approx(max(sys1.probabilities.values()), abs=1e-6)

        # Effort rubric bins and probabilities sum to 1.0
        assert set(sys1.effort_probabilities.keys()) == set(RUBRIC_EFFORT_BINS)
        assert sum(sys1.effort_probabilities.values()) == pytest.approx(1.0, abs=1e-4)

        # Expected effort score matches sum(k * p_k)
        manual_score = sum(
            k * sys1.effort_probabilities[bin_name]
            for k, bin_name in enumerate(RUBRIC_EFFORT_BINS)
        )
        assert sys1.effort_score == pytest.approx(manual_score, abs=1e-4)
        assert 0.0 <= sys1.effort_score <= 3.0

        # Noul fast-success probability in [0, 1]
        assert 0.0 <= sys1.noul_fast_success <= 1.0
        assert sys1.noul_fast_success_probability == sys1.noul_fast_success
        assert sys1.calibrated is True

        # to_dict compatibility
        meta_dict = sys1.to_dict()
        assert meta_dict["effort_score"] == sys1.effort_score
        assert meta_dict["calibrated"] is True
        meta = SystemOneMeta(**meta_dict)
        assert meta.effort_score == sys1.effort_score


def test_evaluate_system_one_temperature_scaling(engine, config: RouterConfig) -> None:
    """Test temperature scaling effect on confidence and probability spread."""
    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)

        sys1_t1 = evaluate_system_one(ranking, temperature=1.0)
        sys1_high_t = evaluate_system_one(ranking, temperature=10.0)
        sys1_low_t = evaluate_system_one(ranking, temperature=0.1)

        # Higher T spreads probabilities -> lower confidence
        assert sys1_high_t.confidence <= sys1_t1.confidence

        # Lower T sharpens probabilities -> higher confidence
        assert sys1_low_t.confidence >= sys1_t1.confidence

        # Invalid non-positive temperature raises ValueError
        with pytest.raises(ValueError, match="temperature must be positive"):
            evaluate_system_one(ranking, temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be positive"):
            evaluate_system_one(ranking, temperature=-1.0)


def test_evaluate_system_one_with_neural_model(engine, config: RouterConfig) -> None:
    """Test evaluate_system_one integrating a SystemOneModernBERT model."""
    tiny_cfg = tiny_modernbert_config()
    model = build_system_one_model(tiny_config=tiny_cfg, numeric_dim=0)

    with registry_db.Session(engine) as session:
        ranking = rank_candidates(session, "explore", config)
        input_ids = torch.randint(2, 1024, (1, 16))
        attention_mask = torch.ones((1, 16), dtype=torch.long)

        sys1 = evaluate_system_one(
            ranking,
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert sum(sys1.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
        assert sum(sys1.effort_probabilities.values()) == pytest.approx(1.0, abs=1e-4)
        assert 0.0 <= sys1.effort_score <= 3.0
        assert 0.0 <= sys1.noul_fast_success <= 1.0


# ---------------------------------------------------------------------------
# 3. /route Endpoint Integration & Backward Compatibility Tests
# ---------------------------------------------------------------------------


def test_route_endpoint_system_one_response(client: TestClient) -> None:
    """Verify /route returns calibrated primitives and preserves all schema fields."""
    response = client.post(
        "/route",
        json={"task": "implement calibrated routing", "phase": "explore"},
    )
    assert response.status_code == 200, response.text
    data = response.json()

    # Required baseline fields preserved
    assert data["model"] == "test/strong"
    assert data["deployment"] == "default"
    assert data["effort"] == "low"
    assert "alternatives" in data
    assert "reason_codes" in data
    assert "estimated_tokens" in data
    assert "estimated_cost" in data
    assert "policy_version" in data
    assert "registry_hash" in data

    # New System One fields
    assert "confidence" in data and data["confidence"] is not None
    assert 0.0 <= data["confidence"] <= 1.0

    assert "probabilities" in data and data["probabilities"] is not None
    probs = data["probabilities"]
    assert "test/strong" in probs
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert data["confidence"] == pytest.approx(max(probs.values()), abs=1e-6)

    assert "system_one" in data and data["system_one"] is not None
    sys1 = data["system_one"]
    assert "effort_score" in sys1 and sys1["effort_score"] is not None
    assert 0.0 <= sys1["effort_score"] <= 3.0

    assert "effort_probabilities" in sys1 and sys1["effort_probabilities"] is not None
    effort_probs = sys1["effort_probabilities"]
    assert set(effort_probs.keys()) == {"low", "medium", "high", "max"}
    assert sum(effort_probs.values()) == pytest.approx(1.0, abs=1e-4)

    assert "noul_fast_success" in sys1 and sys1["noul_fast_success"] is not None
    assert 0.0 <= sys1["noul_fast_success"] <= 1.0
    assert sys1["calibrated"] is True

    # Validate against Pydantic wire contract
    validated = RouteResponse.model_validate(data)
    assert validated.confidence == data["confidence"]
    assert validated.system_one is not None
    assert (
        validated.system_one.noul_fast_success_probability
        == data["system_one"]["noul_fast_success"]
    )


def test_route_deterministic_bytes_with_system_one(client: TestClient) -> None:
    """Verify /route byte-level determinism with System One fields included."""
    resp1 = client.post("/route", json={"task": "test determinism", "phase": "explore"})
    resp2 = client.post("/route", json={"task": "test determinism", "phase": "explore"})
    assert resp1.status_code == 200 and resp2.status_code == 200
    assert resp1.content == resp2.content


# ---------------------------------------------------------------------------
# 4. Calibration Metrics Tests (Brier Score and ECE)
# ---------------------------------------------------------------------------


def test_compute_brier_score() -> None:
    """Test Brier score computation across scenarios."""
    # Perfect predictions
    assert compute_brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)

    # Worst predictions
    assert compute_brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)

    # Balanced uncertainty
    assert compute_brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)

    # Typical probabilities
    preds = [0.9, 0.8, 0.2, 0.1]
    targets = [1, 1, 0, 0]
    expected_bs = (0.01 + 0.04 + 0.04 + 0.01) / 4
    assert compute_brier_score(preds, targets) == pytest.approx(expected_bs)

    # Error conditions
    with pytest.raises(ValueError, match="same length"):
        compute_brier_score([0.5], [1, 0])
    with pytest.raises(ValueError, match="not be empty"):
        compute_brier_score([], [])


def test_compute_expected_calibration_error() -> None:
    """Test Expected Calibration Error (ECE) computation."""
    # Perfect calibration
    perfect_preds = [0.05, 0.95]
    perfect_targets = [0, 1]
    ece_perfect = compute_expected_calibration_error(perfect_preds, perfect_targets, num_bins=10)
    assert 0.0 <= ece_perfect < 0.1

    # Completely uncalibrated: confident in wrong predictions
    wrong_preds = [1.0, 1.0, 1.0]
    wrong_targets = [0, 0, 0]
    ece_wrong = compute_expected_calibration_error(wrong_preds, wrong_targets, num_bins=10)
    assert ece_wrong == pytest.approx(1.0)

    # Bounded in [0, 1]
    mixed_preds = [0.1, 0.3, 0.6, 0.8, 0.9]
    mixed_targets = [0, 0, 1, 1, 1]
    ece = compute_expected_calibration_error(mixed_preds, mixed_targets, num_bins=5)
    assert 0.0 <= ece <= 1.0

    # Error conditions
    with pytest.raises(ValueError, match="same length"):
        compute_expected_calibration_error([0.5], [1, 0])
    with pytest.raises(ValueError, match="positive"):
        compute_expected_calibration_error([0.5], [1], num_bins=0)
    with pytest.raises(ValueError, match="not be empty"):
        compute_expected_calibration_error([], [])
