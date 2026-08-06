# Native Qwen3 ONNX with ONNX Runtime FP8 Fusion

This workflow converts the local `Qwen/Qwen3-0.6B-FP8` checkpoint into a standard FP32 ONNX model. The
ONNX file itself contains ordinary FP32 `MatMul` nodes; eligible nodes are converted to
`com.microsoft::DynamicQuantMatMulFp8` when the session initializes.

Run all commands from the ONNX Runtime repository unless a command explicitly changes directory.

## 1. Convert the checkpoint to native FP32 ONNX

The source checkpoint is expected at:

```text
tools/qwen3_native_onnx/models/Qwen3-0.6B-FP8/config.json
tools/qwen3_native_onnx/models/Qwen3-0.6B-FP8/model.safetensors
```

Generate the native model:

```bash
cd tools/qwen3_native_onnx
bash build_native_fp32_onnx.sh
cd ../..
```

The first invocation creates `.venv` and installs the required Python packages. Once those packages are
available, use `OFFLINE=1` to avoid network access:

```bash
cd tools/qwen3_native_onnx
OFFLINE=1 OUT_DIR=onnx_native_fp32_new bash build_native_fp32_onnx.sh
cd ../..
```

The exporter refuses to overwrite an existing model, so select a new `OUT_DIR` when necessary. The default
output is:

```text
tools/qwen3_native_onnx/onnx_native_fp32/model.onnx
tools/qwen3_native_onnx/onnx_native_fp32/model.onnx.data
tools/qwen3_native_onnx/onnx_native_fp32/state_dict_baseline_meta.json
```

The external data file must remain beside `model.onnx`. The generated graph has dynamic `batch` and `seq`
dimensions, standard ONNX `MatMul` nodes, no FP8 initializers, and no pre-generated
`DynamicQuantMatMulFp8` nodes. FP8 checkpoint weights are dequantized to FP32 during export. The checkpoint,
virtual environment, and generated model files are local test artifacts and should not be committed.

## 2. ONNX Runtime build requirement

ONNX Runtime must be built with `onnxruntime_USE_KLEIDIAI=ON`.
