#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx


def _set_tensor_elem_type(vi, elem_type: int, replacement_type: int) -> bool:
    if not vi.type.HasField("tensor_type"):
        return False
    tt = vi.type.tensor_type
    if not tt.HasField("elem_type"):
        return False
    if tt.elem_type != elem_type:
        return False
    tt.elem_type = replacement_type
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Rewrite invalid Cast(complex128) nodes in an ONNX model.")
    parser.add_argument("--in-model", required=True, help="Input ONNX path")
    parser.add_argument("--out-model", required=True, help="Output ONNX path")
    parser.add_argument(
        "--target-dtype",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Replacement type for Cast(complex128).",
    )
    args = parser.parse_args()

    in_path = Path(args.in_model)
    out_path = Path(args.out_model)
    # Cast nodes live in the graph protobuf; initializer bytes can remain external.
    model = onnx.load_model(str(in_path), load_external_data=False)
    replacement_type = {
        "bfloat16": onnx.TensorProto.BFLOAT16,
        "float32": onnx.TensorProto.FLOAT,
    }[args.target_dtype]

    rewritten = 0
    cast_outputs = set()
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to_attr = None
        for a in node.attribute:
            if a.name == "to":
                to_attr = a
                break
        if to_attr is None:
            continue
        if to_attr.i != onnx.TensorProto.COMPLEX128:
            continue
        to_attr.i = replacement_type
        rewritten += 1
        for out in node.output:
            cast_outputs.add(out)

    if rewritten:
        # Update value_info / graph outputs if they were annotated as complex128.
        for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
            if vi.name in cast_outputs:
                _set_tensor_elem_type(vi, onnx.TensorProto.COMPLEX128, replacement_type)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    print(f"Rewrote {rewritten} Cast node(s) to {args.target_dtype}. Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
