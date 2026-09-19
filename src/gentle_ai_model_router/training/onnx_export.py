"""ONNX export and INT8 dynamic quantization for ModernBERT rankers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = logging.getLogger(__name__)


def _require_onnx_extra() -> None:
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "onnx export requires the [onnx] extra: pip install 'gentle-ai-model-router[onnx]'"
        ) from exc


def load_benchmark_feature_names(checkpoint: str | Path) -> tuple[str, ...] | None:
    """Read ``benchmark_feature_names`` recorded by training in metrics.json.

    Training writes the exact benchmark feature vector layout into the
    checkpoint's metrics.json so serving (router/neural.py) builds numeric
    features with the SAME names — and therefore the same width — the ranker
    was trained on.

    Returns None when the artifact predates the recording (the caller decides
    on a legacy fallback). Raises ValueError on a present-but-invalid
    recording: a malformed artifact must fail closed, never serve silently
    with a mismatched feature vector.
    """
    metrics_path = Path(checkpoint) / "metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid metrics.json in {checkpoint}: {exc}") from exc
    names = data.get("benchmark_feature_names")
    if names is None:
        return None
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(n, str) and n for n in names)
    ):
        raise ValueError(
            f"invalid benchmark_feature_names in {metrics_path}: {names!r} "
            "(expected a non-empty list of non-empty strings)"
        )
    return tuple(names)


def export_onnx(
    checkpoint: str | Path,
    output_dir: str | Path | None = None,
    quantize: bool = True,
    opset_version: int = 18,
) -> tuple[Path, Path | None]:
    """Export a trained ranker checkpoint to ONNX format with optional INT8 dynamic quantization.

    Args:
        checkpoint: Path to directory containing ranker checkpoint (encoder + head.pt).
        output_dir: Destination directory for exported model(s). Defaults to checkpoint directory.
        quantize: Whether to produce an INT8 dynamically quantized model in addition to FP32.
        opset_version: ONNX opset version (default 18).

    Returns:
        tuple of (fp32_onnx_path, quant_onnx_path | None).
    """
    _require_onnx_extra()
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from gentle_ai_model_router.training.model import load_ranker

    ckpt_path = Path(checkpoint)
    out_dir = Path(output_dir) if output_dir is not None else ckpt_path
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_ranker(str(ckpt_path))
    model.eval()

    hidden_size = model.encoder.config.hidden_size
    head_in_features = int(model.head[0].weight.shape[1])
    numeric_dim = head_in_features - hidden_size
    if numeric_dim < 0:
        raise ValueError(
            f"Invalid numeric_dim computed from head: "
            f"{head_in_features} - {hidden_size} = {numeric_dim}"
        )

    dummy_ids = torch.ones((1, 16), dtype=torch.long)
    dummy_mask = torch.ones((1, 16), dtype=torch.long)
    dummy_num = torch.zeros((1, numeric_dim), dtype=torch.float32)

    model_onnx_path = out_dir / "model.onnx"
    logger.info("Exporting FP32 ONNX model to %s (opset=%d)", model_onnx_path, opset_version)

    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask, dummy_num),
        str(model_onnx_path),
        opset_version=opset_version,
        input_names=["input_ids", "attention_mask", "numeric_features"],
        output_names=["score"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "numeric_features": {0: "batch_size"},
            "score": {0: "batch_size"},
        },
    )

    if out_dir.resolve() != ckpt_path.resolve():
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as exc:
            logger.warning("Could not copy tokenizer to %s: %s", out_dir, exc)
        # Carry the recorded feature layout so the exported artifact is
        # self-describing for serving (router/neural.py).
        metrics_src = ckpt_path / "metrics.json"
        if metrics_src.is_file():
            (out_dir / "metrics.json").write_text(
                metrics_src.read_text(encoding="utf-8"), encoding="utf-8"
            )

    if quantize:
        # Strip value_info so quantize_dynamic won't fail shape inference
        # on dynamic concatenated numeric dimensions
        onnx_model = onnx.load(str(model_onnx_path))
        del onnx_model.graph.value_info[:]
        onnx.save(onnx_model, str(model_onnx_path))

        quant_onnx_path = out_dir / "model.quant.onnx"
        logger.info("Quantizing to INT8 dynamic: %s", quant_onnx_path)
        quantize_dynamic(
            str(model_onnx_path),
            str(quant_onnx_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
        )
        return model_onnx_path, quant_onnx_path

    return model_onnx_path, None


class OnnxRanker:
    """ONNX-based ranker for low-overhead CPU inference without PyTorch."""

    def __init__(
        self,
        checkpoint: str | Path,
        model_path: str | Path | None = None,
        use_quantized: bool = True,
        sess_options: Any | None = None,
        providers: list[str] | None = None,
    ) -> None:
        _require_onnx_extra()
        import onnxruntime as ort
        from transformers import AutoTokenizer

        ckpt_p = Path(checkpoint)
        if ckpt_p.is_file():
            actual_model_path = ckpt_p
            actual_ckpt_dir = ckpt_p.parent
        else:
            actual_ckpt_dir = ckpt_p
            if model_path is not None:
                actual_model_path = Path(model_path)
            else:
                quant_p = actual_ckpt_dir / "model.quant.onnx"
                fp32_p = actual_ckpt_dir / "model.onnx"
                if use_quantized and quant_p.exists():
                    actual_model_path = quant_p
                elif fp32_p.exists():
                    actual_model_path = fp32_p
                elif quant_p.exists():
                    actual_model_path = quant_p
                else:
                    raise FileNotFoundError(
                        f"No ONNX model (model.onnx or model.quant.onnx) found in {actual_ckpt_dir}"
                    )

        if not actual_model_path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {actual_model_path}")

        self.checkpoint = actual_ckpt_dir
        self.model_path = actual_model_path
        # Feature layout recorded by training (None for legacy artifacts that
        # predate the recording — neural.py falls back with a reason code).
        self.benchmark_feature_names = load_benchmark_feature_names(actual_ckpt_dir)

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint))

    def score(
        self,
        texts: list[str],
        numeric_features: list[list[float]],
        max_length: int | None = None,
    ) -> list[float]:
        """Score (query, candidate) text and numeric feature pairs using numpy and ONNX Runtime.

        Args:
            texts: Formatted pair texts (e.g. from encode_pair_text).
            numeric_features: 2D list of float feature vectors, one per text.
            max_length: Optional sequence truncation limit.

        Returns:
            List of scalar float scores.
        """
        if not texts:
            return []
        if len(texts) != len(numeric_features):
            raise ValueError(
                f"texts length ({len(texts)}) does not match numeric_features length "
                f"({len(numeric_features)})"
            )

        import numpy as np

        tok_kwargs: dict[str, Any] = {
            "padding": True,
            "truncation": True,
            "return_tensors": "np",
        }
        if max_length is not None:
            tok_kwargs["max_length"] = max_length

        encoded = self.tokenizer(texts, **tok_kwargs)
        numeric_arr = np.asarray(numeric_features, dtype=np.float32)

        feed = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "numeric_features": numeric_arr,
        }
        outputs = self.session.run(None, feed)
        scores: np.ndarray = np.asarray(outputs[0]).reshape(-1)
        return [float(s) for s in scores.tolist()]
