#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-0.6B-FP8}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-0.6B-FP8}"
VENV_DIR="${VENV_DIR:-.venv}"
OUT_DIR="${OUT_DIR:-onnx_native_fp32}"
EXPORT_SEQ="${EXPORT_SEQ:-2}"
KEEP_WORK="${KEEP_WORK:-0}"
WORK_ROOT="${WORK_ROOT:-}"

if [[ -e "$OUT_DIR/model.onnx" || -e "$OUT_DIR/model.onnx.data" ]]; then
  echo "ERROR: $OUT_DIR already contains model output; choose a new OUT_DIR." >&2
  exit 2
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [[ "${OFFLINE:-0}" != "1" ]]; then
  python -m pip install -U pip wheel setuptools
  if ! python -c "import torch" >/dev/null 2>&1; then
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
  fi
  python -m pip install -U numpy safetensors transformers onnx
fi

python - <<'PY'
import importlib.util

required = ["torch", "numpy", "safetensors", "transformers", "onnx", "huggingface_hub"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing Python packages: {missing}")
PY

if [[ ! -f "$MODEL_DIR/config.json" || ! -f "$MODEL_DIR/model.safetensors" ]]; then
  if [[ "${OFFLINE:-0}" == "1" ]]; then
    echo "ERROR: $MODEL_DIR is missing the checkpoint required for OFFLINE=1." >&2
    exit 2
  fi

  echo "== Download $MODEL_ID checkpoint =="
  python - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    allow_patterns=["config.json", "model.safetensors"],
)
PY
fi

created_work_root=0
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="$(mktemp -d /tmp/qwen_native_onnx.XXXXXX)"
  created_work_root=1
fi

if [[ "$created_work_root" == "1" && "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK_ROOT"' EXIT
fi

RAW_DIR="$WORK_ROOT/raw"
mkdir -p "$RAW_DIR" "$OUT_DIR"

echo "== Export native FP32 ONNX with dequantized checkpoint weights =="
python export_onnx_baseline.py \
  --model-dir "$MODEL_DIR" \
  --out-dir "$RAW_DIR" \
  --baseline-dtype float32 \
  --dequantize-fp8 \
  --seq "$EXPORT_SEQ"

echo "== Replace invalid complex casts with FP32 casts =="
python fix_complex_casts.py \
  --in-model "$RAW_DIR/model.onnx" \
  --out-model "$RAW_DIR/model.onnx" \
  --target-dtype float32

echo "== Pack initializers into model.onnx.data =="
python pack_onnx_external_data.py \
  --in-model "$RAW_DIR/model.onnx" \
  --out-dir "$OUT_DIR" \
  --data-file model.onnx.data

cp "$RAW_DIR/state_dict_baseline_meta.json" "$OUT_DIR/state_dict_baseline_meta.json"

echo "== Validate native ONNX graph =="
python - "$OUT_DIR/model.onnx" <<'PY'
import sys

import onnx

model_path = sys.argv[1]
model = onnx.load_model(model_path, load_external_data=False)
fp8_types = {
    onnx.TensorProto.FLOAT8E4M3FN,
    onnx.TensorProto.FLOAT8E4M3FNUZ,
    onnx.TensorProto.FLOAT8E5M2,
    onnx.TensorProto.FLOAT8E5M2FNUZ,
}
fp8_initializers = [initializer.name for initializer in model.graph.initializer if initializer.data_type in fp8_types]
fp8_nodes = [node.name for node in model.graph.node if node.op_type == "DynamicQuantMatMulFp8"]
matmul_count = sum(node.op_type == "MatMul" and node.domain in {"", "ai.onnx"} for node in model.graph.node)

if fp8_initializers:
    raise SystemExit(f"Native model still has FP8 initializers: {fp8_initializers[:5]}")
if fp8_nodes:
    raise SystemExit(f"Native model still has DynamicQuantMatMulFp8 nodes: {fp8_nodes[:5]}")
if matmul_count == 0:
    raise SystemExit("Native model has no standard ONNX MatMul nodes")

onnx.checker.check_model(model_path)
print(f"Validated native FP32 ONNX model with {matmul_count} MatMul nodes: {model_path}")
PY

if [[ "$created_work_root" == "1" && "$KEEP_WORK" == "1" ]]; then
  echo "Kept intermediate files in: $WORK_ROOT"
fi

echo "Native ONNX model: $OUT_DIR/model.onnx"
echo "External weights: $OUT_DIR/model.onnx.data"
