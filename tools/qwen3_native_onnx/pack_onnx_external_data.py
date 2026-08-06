#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-save an ONNX model so all external initializer data is stored in a single .data file."
    )
    parser.add_argument("--in-model", default="onnx_baseline/model.onnx", help="Input ONNX (may reference external data)")
    parser.add_argument("--out-dir", default="onnx_baseline_packed", help="Output directory")
    parser.add_argument("--data-file", default="model.onnx.data", help="External data filename to write")
    args = parser.parse_args()

    in_model = Path(args.in_model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "model.onnx"

    if not in_model.exists():
        raise FileNotFoundError(f"Missing input model: {in_model}")

    # Load with external data.
    model = onnx.load_model(str(in_model), load_external_data=True)

    # Save with a single external data file (keeps >2GB models shareable without protobuf limits).
    onnx.save_model(
        model,
        str(out_model),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=str(args.data_file),
        size_threshold=0,
        convert_attribute=False,
    )

    print(f"Wrote: {out_model}")
    print(f"Wrote: {out_dir / args.data_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
