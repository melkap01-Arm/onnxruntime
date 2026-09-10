# Native Qwen3 ONNX with ONNX Runtime FP8 Fusion

This workflow converts the local `Qwen/Qwen3-0.6B-FP8` checkpoint into a standard FP32 ONNX model. The
ONNX file itself contains ordinary FP32 `MatMul` nodes; eligible nodes are converted to
`com.microsoft::DynamicQuantMatMulFp8` when FP8 fusion is enabled and the session initializes.

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

## 3. Enable runtime FP8 fusion

The framework does not automatically choose between FP8 and INT8 based on hardware. The graph operator
determines the quantization path. For this workflow, enable the opt-in FP8 graph transformer with the
`session.enable_matmul_fp8_fusion` session configuration entry:

```text
FP32 MatMul
   |
   | session.enable_matmul_fp8_fusion=1
   v
com.microsoft::DynamicQuantMatMulFp8
   |
   +-- constant B is quantized to FP8 during session initialization
   +-- runtime A is dynamically quantized to FP8 for every inference
```

The setting is disabled by default because converting FP32 inputs and weights to FP8 changes numerical
behavior. It can be passed to `onnxruntime_perf_test` with `-C`:

```bash
<build-dir>/onnxruntime_perf_test \
  -e cpu \
  -o 99 \
  -C "session.enable_matmul_fp8_fusion|1" \
  -I \
  tools/qwen3_native_onnx/onnx_native_fp32/model.onnx
```

An ONNX operator is a graph operation identified by its domain and operator name. Standard operators use
the ONNX domain, while operators such as `com.microsoft::DynamicQuantMatMulFp8` use a vendor domain. An ONNX
model imports an operator-set (opset) version for each domain. Individual operator schemas receive new
versions only when their contract changes, so an operator's resolved schema version can be older than the
model's imported opset. For example, a model importing ONNX opset 18 resolves `MatMul` to schema version 13.

The transformer replaces a node only when all of the following conditions are satisfied:

- The node is a standard ONNX-domain `MatMul` whose resolved operator schema version is 1, 9, or 13, and it
  is assigned to the CPU Execution Provider.
- The node has exactly two inputs and one output, and A, B, and the output are FP32 tensors.
- B is a constant two-dimensional FP32 initializer with positive K and N dimensions.
- K and N are both divisible by the FP8 block size of 128.
- If the final dimension of A is known, it matches B's K dimension.

For a dynamic A shape such as `[batch, sequence, unknown]`, the transformer cannot determine whether the
unknown K dimension matches B, so it does not reject the node. Validation is deferred until inference, when
the actual A tensor is available. For example, B with shape `[1024, 3072]` accepts an A ending in 1024; an A
ending in any other value fails with a `MatMul` shape-mismatch error before quantization or GEMM execution.
An unknown dimension therefore defers the constraint; it does not remove it.

Nodes that do not meet every condition remain unchanged. The generated node does not specify an `fp8_type`
attribute, so [`DynamicQuantMatMulFp8`](../../onnxruntime/contrib_ops/cpu/quantization/dynamic_quant_matmul_fp8.cc)
uses its default, `FLOAT8E4M3FN`. Its CPU kernel computes block scales and quantizes A to that FP8 format at
runtime. The eligibility checks are implemented in
[`MatMulToDynamicQuantMatMulFp8Fusion`](../../onnxruntime/core/optimizer/matmul_to_dynamic_quant_matmul_fp8_fusion.cc),
and the session configuration entry is declared in
[the session config header](../../include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h).

The session option determines whether eligible MatMuls use FP8. CPU feature detection then chooses between the accelerated KleidiAI implementation and the generic MLAS FP8 fallback; it never changes the operation to INT8.

  - FP8 hardware + fusion disabled → original FP32 MatMul.
  - FP8 hardware + fusion enabled → KleidiAI FP8.
  - No FP8 hardware + fusion enabled → generic FP8 execution.

By comparison, INT8 quantisation is encoded in the model during quantization or export. The FP8 transformer changes an ordinary FP32 model’s numerical behavior at session initialization, so it requires explicit opt-in.
