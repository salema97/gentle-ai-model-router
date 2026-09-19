"""ONNX and INT8 quantization export tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from gentle_ai_model_router.cli.main import app  # noqa: E402
from gentle_ai_model_router.training.model import (  # noqa: E402
    build_model,
    load_ranker,
    tiny_modernbert_config,
)
from gentle_ai_model_router.training.onnx_export import OnnxRanker, export_onnx  # noqa: E402

runner = CliRunner()


def _create_tiny_checkpoint(root: Path, numeric_dim: int = 4) -> Path:
    """Create a fully offline tiny ModernBERT ranker checkpoint with tokenizer."""
    ckpt = root / "tiny-ranker"
    ckpt.mkdir(parents=True, exist_ok=True)

    # 1. Save minimal offline tokenizer
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "hello": 5,
        "world": 6,
        "test": 7,
        "query": 8,
    }
    raw_tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    raw_tok.pre_tokenizer = Whitespace()
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    fast_tok.save_pretrained(ckpt)

    # 2. Build and save tiny ranker model
    config = tiny_modernbert_config(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = build_model(tiny_config=config, numeric_dim=numeric_dim)
    model.save_pretrained(ckpt)

    return ckpt


def test_export_onnx_with_quantization(tmp_path: Path) -> None:
    ckpt_dir = _create_tiny_checkpoint(tmp_path, numeric_dim=4)
    fp32_path, quant_path = export_onnx(ckpt_dir, quantize=True)

    assert fp32_path.exists()
    assert fp32_path == ckpt_dir / "model.onnx"
    assert quant_path is not None
    assert quant_path.exists()
    assert quant_path == ckpt_dir / "model.quant.onnx"

    # Verify value_info is stripped on the model graph
    model_proto = onnx.load(str(fp32_path))
    assert len(model_proto.graph.value_info) == 0

    input_names = [inp.name for inp in model_proto.graph.input]
    assert input_names == ["input_ids", "attention_mask", "numeric_features"]
    output_names = [out.name for out in model_proto.graph.output]
    assert output_names == ["score"]


def test_export_onnx_without_quantization(tmp_path: Path) -> None:
    ckpt_dir = _create_tiny_checkpoint(tmp_path, numeric_dim=3)
    out_dir = tmp_path / "custom_out"

    fp32_path, quant_path = export_onnx(ckpt_dir, output_dir=out_dir, quantize=False)

    assert fp32_path.exists()
    assert fp32_path == out_dir / "model.onnx"
    assert quant_path is None
    assert not (out_dir / "model.quant.onnx").exists()
    # Tokenizer files should have been copied to out_dir
    assert (out_dir / "tokenizer.json").exists()


def test_onnx_ranker_scoring_equivalence(tmp_path: Path) -> None:
    numeric_dim = 4
    ckpt_dir = _create_tiny_checkpoint(tmp_path, numeric_dim=numeric_dim)
    fp32_path, quant_path = export_onnx(ckpt_dir, quantize=True)
    assert quant_path is not None

    pt_model, pt_tokenizer = load_ranker(str(ckpt_dir))
    pt_model.eval()

    texts = [
        "hello world",
        "world hello",
        "test query",
        "query hello test",
    ]
    numeric_features = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [-0.1, -0.2, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ]

    # Reference PyTorch scores
    enc = pt_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    num_tensor = torch.tensor(numeric_features, dtype=torch.float32)
    with torch.no_grad():
        pt_scores = pt_model(enc["input_ids"], enc["attention_mask"], num_tensor).numpy()

    # FP32 ONNXRanker scores
    fp32_ranker = OnnxRanker(ckpt_dir, model_path=fp32_path, use_quantized=False)
    fp32_scores = fp32_ranker.score(texts, numeric_features)

    assert len(fp32_scores) == len(texts)
    # FP32 ONNX must match PyTorch scores within 1e-4
    np.testing.assert_allclose(pt_scores, fp32_scores, rtol=1e-4, atol=1e-4)

    # INT8 Quantized ONNXRanker scores
    int8_ranker = OnnxRanker(ckpt_dir, use_quantized=True)
    int8_scores = int8_ranker.score(texts, numeric_features)

    assert len(int8_scores) == len(texts)
    # Quantized scores should be close to FP32 scores
    np.testing.assert_allclose(pt_scores, int8_scores, rtol=0.25, atol=0.15)


def test_onnx_ranker_edge_cases(tmp_path: Path) -> None:
    ckpt_dir = _create_tiny_checkpoint(tmp_path, numeric_dim=2)
    export_onnx(ckpt_dir, quantize=False)

    ranker = OnnxRanker(ckpt_dir)
    assert ranker.score([], []) == []

    with pytest.raises(ValueError, match="texts length .* does not match"):
        ranker.score(["test"], [[0.1, 0.2], [0.3, 0.4]])


def test_cli_export_onnx(tmp_path: Path) -> None:
    ckpt_dir = _create_tiny_checkpoint(tmp_path, numeric_dim=4)
    out_dir = tmp_path / "cli_out"

    result = runner.invoke(
        app,
        [
            "export-onnx",
            "--checkpoint",
            str(ckpt_dir),
            "--output",
            str(out_dir),
            "--quantize",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ONNX Export Summary" in result.output
    assert "FP32 Model" in result.output
    assert "INT8 Quantized Model" in result.output
    assert (out_dir / "model.onnx").exists()
    assert (out_dir / "model.quant.onnx").exists()


def test_v9_checkpoint_onnx_ranker_if_available() -> None:
    v9_path = Path("models/deberta-router/v9")
    if not (v9_path / "model.onnx").exists():
        pytest.skip("models/deberta-router/v9 not present")

    ranker = OnnxRanker(v9_path, use_quantized=True)
    assert ranker.session is not None
    # v9 has 22 numeric dimensions
    dummy_numeric = [[0.0] * 22]
    scores = ranker.score(["[task] design\n[candidate] model=openai/gpt-4o"], dummy_numeric)
    assert len(scores) == 1
    assert isinstance(scores[0], float)
