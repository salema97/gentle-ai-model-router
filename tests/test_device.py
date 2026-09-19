"""Device resolution + evaluation config tests — pure, no torch, no GPU.

Covers:
- ``TrainingConfig.device`` / ``EvaluationConfig`` parsing (defaults, yaml,
  invalid values rejected by pydantic).
- ``resolve_device`` as a pure function (CUDA availability passed in, so the
  tests monkeypatch nothing and need no GPU).
- Baselines-only evaluation must not import torch (lazy-guard regression test).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gentle_ai_model_router.router.config import (
    EvaluationConfig,
    RouterConfig,
    TrainingConfig,
    load_config,
)
from gentle_ai_model_router.training.device import resolve_device


class TestTrainingConfigDevice:
    def test_default_is_auto(self) -> None:
        assert TrainingConfig().device == "auto"

    def test_explicit_values_parse(self) -> None:
        for value in ("auto", "cpu", "cuda"):
            assert TrainingConfig(device=value).device == value

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(device="gpu")

    def test_yaml_section_parses(self, tmp_path) -> None:
        config_path = tmp_path / "router.yaml"
        config_path.write_text('training:\n  device: "cuda"\n', encoding="utf-8")
        config = load_config(config_path=config_path)
        assert config.training.device == "cuda"


class TestEvaluationConfig:
    def test_defaults(self) -> None:
        evaluation = EvaluationConfig()
        assert evaluation.batch_size == 64
        assert evaluation.device == "auto"

    def test_yaml_section_parses(self, tmp_path) -> None:
        config_path = tmp_path / "router.yaml"
        config_path.write_text(
            "evaluation:\n  batch_size: 128\n  device: cpu\n", encoding="utf-8"
        )
        config = load_config(config_path=config_path)
        assert config.evaluation.batch_size == 128
        assert config.evaluation.device == "cpu"

    def test_router_config_carries_evaluation(self) -> None:
        config = RouterConfig(evaluation={"batch_size": 32, "device": "cuda"})
        assert config.evaluation.batch_size == 32
        assert config.evaluation.device == "cuda"


class TestResolveDevice:
    """Pure function: the CUDA flag is an argument, no monkeypatching needed."""

    def test_auto_prefers_cuda(self) -> None:
        assert resolve_device("auto", cuda_available=True) == "cuda"
        assert resolve_device("auto", cuda_available=False) == "cpu"

    def test_cpu_always_cpu(self) -> None:
        assert resolve_device("cpu", cuda_available=True) == "cpu"
        assert resolve_device("cpu", cuda_available=False) == "cpu"

    def test_cuda_requires_availability(self) -> None:
        assert resolve_device("cuda", cuda_available=True) == "cuda"
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda", cuda_available=False)

    def test_unknown_device_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown device"):
            resolve_device("gpu", cuda_available=True)


class TestBaselinesOnlyNeverImportsTorch:
    """No-checkpoint evaluation must stay torch-free (base-env guard)."""

    def test_baselines_only_runs_with_torch_blocked(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests.test_evaluate import _build  # reuse the tiny fixture

        monkeypatch.setitem(__import__("sys").modules, "torch", None)
        config, out_dir, _ = _build(tmp_path, with_policy_session=False)

        from gentle_ai_model_router.training.evaluate import evaluate_dataset

        results = evaluate_dataset(out_dir, config, session=None, checkpoint=None)
        assert set(results["splits"]) == {"test", "temporal_test"}
