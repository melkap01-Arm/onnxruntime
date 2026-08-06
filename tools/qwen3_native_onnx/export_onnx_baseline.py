#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM


class LogitsOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def _dtype_str(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _is_float8(dtype: torch.dtype) -> bool:
    return _dtype_str(dtype).startswith("float8_")


def _dequantize_fp8_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: tuple[int, int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D FP8 weight, got shape {list(weight.shape)}")

    rows, cols = weight.shape
    block_rows, block_cols = block_size
    if rows % block_rows != 0 or cols % block_cols != 0:
        raise ValueError(
            f"Weight shape {list(weight.shape)} is not divisible by block size {list(block_size)}"
        )

    expected_scale_shape = (rows // block_rows, cols // block_cols)
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"Expected scale shape {list(expected_scale_shape)} for weight shape {list(weight.shape)}, "
            f"got {list(scale_inv.shape)}"
        )

    blocked_weight = weight.float().reshape(
        expected_scale_shape[0], block_rows, expected_scale_shape[1], block_cols
    )
    expanded_scale = scale_inv.float().unsqueeze(1).unsqueeze(3)
    return (blocked_weight * expanded_scale).reshape(weight.shape).to(output_dtype)


def _parse_baseline_dtype(s: str) -> torch.dtype:
    s = s.strip().lower()
    if s == "auto":
        raise ValueError("baseline dtype 'auto' must be resolved via _infer_baseline_dtype")
    if s in {"fp32", "float32", "f32"}:
        return torch.float32
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp16", "float16", "f16"}:
        return torch.float16
    raise ValueError(f"Unsupported --baseline-dtype: {s}")


def _infer_baseline_dtype(sd: dict[str, torch.Tensor]) -> torch.dtype:
    """
    Infer baseline dtype from the checkpoint.

    Picks the most common non-scale, non-fp8 floating dtype. Falls back to float32.
    """
    counts: dict[torch.dtype, int] = {}
    for name, t in sd.items():
        if "scale" in name.lower():
            continue
        if not t.is_floating_point():
            continue
        if _is_float8(t.dtype):
            continue
        counts[t.dtype] = counts.get(t.dtype, 0) + 1
    if not counts:
        return torch.float32
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _to_baseline_state_dict(
    sd: dict[str, torch.Tensor],
    baseline_dtype: torch.dtype,
    dequantize_fp8: bool = False,
    weight_block_size: tuple[int, int] = (128, 128),
) -> tuple[dict[str, torch.Tensor], dict]:
    """
    Prepare a state_dict for standard ONNX export.

    FP8 weights are either temporarily cast for the existing FP8 rewrite pipeline,
    or block-dequantized for a native floating-point model.
    """
    out: dict[str, torch.Tensor] = {}
    dropped = []
    converted = []

    for name, t in sd.items():
        lname = name.lower()
        # Drop scale tensors and other quantization auxiliaries.
        if "scale" in lname:
            dropped.append(name)
            continue

        if _is_float8(t.dtype):
            if dequantize_fp8:
                if not name.endswith(".weight"):
                    raise ValueError(f"Cannot locate scale for FP8 tensor: {name}")
                scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
                scale_inv = sd.get(scale_name)
                if scale_inv is None:
                    raise ValueError(f"Missing {scale_name} for FP8 weight {name}")
                out[name] = _dequantize_fp8_weight(t, scale_inv, weight_block_size, baseline_dtype)
            else:
                out[name] = t.to(dtype=baseline_dtype)
            converted.append((name, _dtype_str(t.dtype), _dtype_str(baseline_dtype)))
            continue

        # Preserve non-FP8 tensors as-is (bf16/fp16/float32) to keep native dtypes.
        out[name] = t

    meta = {
        "num_in": len(sd),
        "num_out": len(out),
        "num_dropped": len(dropped),
        "num_converted": len(converted),
        "baseline_dtype": _dtype_str(baseline_dtype),
        "fp8_conversion": "dequantized" if dequantize_fp8 else "temporary_cast",
        "weight_block_size": list(weight_block_size),
        "dropped_examples": dropped[:20],
        "converted_examples": converted[:20],
    }
    return out, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a baseline ONNX model (used as input to the FP8 rewrite).")
    parser.add_argument("--model-dir", default="models/Qwen3-0.6B-FP8", help="Local HF snapshot directory")
    parser.add_argument("--safetensors", default=None, help="Path to model.safetensors")
    parser.add_argument("--out-dir", default="onnx_baseline", help="Output directory for ONNX artifacts")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument("--seq", type=int, default=2, help="Dummy sequence length for export")
    parser.add_argument(
        "--baseline-dtype",
        default="auto",
        choices=["auto", "float32", "bfloat16", "float16", "fp32", "bf16", "fp16"],
        help="Baseline model dtype (FP8 weights are temporarily cast). Use auto to infer from checkpoint.",
    )
    parser.add_argument(
        "--dequantize-fp8",
        action="store_true",
        help="Dequantize FP8 weights with weight_scale_inv instead of temporarily casting them.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    st_path = Path(args.safetensors) if args.safetensors else (model_dir / "model.safetensors")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(args.baseline_dtype).strip().lower() == "auto":
        baseline_dtype = None
    else:
        baseline_dtype = _parse_baseline_dtype(str(args.baseline_dtype))

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")
    if not st_path.exists():
        raise FileNotFoundError(f"Missing safetensors: {st_path}")

    print(f"Loading config: {model_dir}")
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=False)
    config.use_cache = False
    quantization_config = getattr(config, "quantization_config", None) or {}
    weight_block_size = tuple(quantization_config.get("weight_block_size", (128, 128)))

    print(f"Loading safetensors: {st_path}")
    raw_sd = load_file(str(st_path), device="cpu")
    if baseline_dtype is None:
        baseline_dtype = _infer_baseline_dtype(raw_sd)
    baseline_sd, sd_meta = _to_baseline_state_dict(
        raw_sd,
        baseline_dtype=baseline_dtype,
        dequantize_fp8=args.dequantize_fp8,
        weight_block_size=weight_block_size,
    )

    meta_path = out_dir / "state_dict_baseline_meta.json"
    meta_path.write_text(json.dumps(sd_meta, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {meta_path}")
    print(
        f"State dict tensors: in={sd_meta['num_in']} out={sd_meta['num_out']} "
        f"dropped={sd_meta['num_dropped']} converted={sd_meta['num_converted']}"
    )

    print(f"Instantiating model from config (CPU, {sd_meta['baseline_dtype']})…")
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)

    # Load weights (ignore quantization aux keys we dropped).
    missing, unexpected = model.load_state_dict(baseline_sd, strict=False)
    print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    if unexpected:
        print("Unexpected (first 10):")
        for k in unexpected[:10]:
            print(f"  {k}")
    if missing:
        print("Missing (first 10):")
        for k in missing[:10]:
            print(f"  {k}")

    # Ensure weights are tied if config says so (helps dedup and correctness).
    try:
        model.tie_weights()
    except Exception:
        pass

    model.eval()
    model.to(dtype=baseline_dtype)

    wrapped = LogitsOnly(model)
    wrapped.eval()

    vocab = int(getattr(config, "vocab_size", 32000))
    seq = int(args.seq)
    input_ids = torch.randint(low=0, high=vocab, size=(1, seq), dtype=torch.int64)
    attention_mask = torch.ones((1, seq), dtype=torch.int64)

    onnx_path = out_dir / "model.onnx"
    print(f"Exporting ONNX to: {onnx_path}")

    export_kwargs = dict(
        f=str(onnx_path),
        args=(input_ids, attention_mask),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        opset_version=int(args.opset),
        do_constant_folding=False,
        dynamo=False,
        external_data=True,
        optimize=False,
    )

    torch.onnx.export(wrapped, **export_kwargs)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
