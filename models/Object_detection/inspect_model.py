# save as: inspect_onnx_dtypes.py
# Usage:
#   python inspect_onnx_dtypes.py "C:\path\to\model.onnx"
#   python inspect_onnx_dtypes.py "C:\path\to\model.onnx" --show-cast
#   python inspect_onnx_dtypes.py "C:\path\to\model.onnx" --topk 30
# python inspect_model.py "C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp16.onnx" --show-cast

'''just for checking model is effective ?
mainly to answer: “Is my model really FP16 (or mixed precision), and where are dtype conversions happening?
'''

from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import argparse
import math

import onnx
from onnx import helper


def prod_int(dims) -> int:
    """Product of dims (handles empty dims as scalar -> 1)."""
    if dims is None or len(dims) == 0:
        return 1
    p = 1
    for d in dims:
        # dim can be 0 or unknown in some graphs; treat unknown/0 as 0 elements
        try:
            di = int(d)
        except Exception:
            return 0
        p *= di
    return p


def tensor_type_str(value_info) -> str:
    """Return human-readable type string for a ValueInfoProto."""
    try:
        t = value_info.type.tensor_type
        dt = t.elem_type
        return helper.tensor_dtype_to_string(dt)  # :contentReference[oaicite:2]{index=2}
    except Exception:
        return "UNKNOWN"


def inspect_initializers(model: onnx.ModelProto):
    """
    Weights/parameters in ONNX are typically stored in graph.initializer. :contentReference[oaicite:3]{index=3}
    We'll count dtype per tensor and per element.
    """
    dtype_tensor_count = Counter()
    dtype_elem_count = Counter()
    dtype_shapes_examples = defaultdict(list)

    for init in model.graph.initializer:
        dt_name = helper.tensor_dtype_to_string(init.data_type)  # :contentReference[oaicite:4]{index=4}
        dtype_tensor_count[dt_name] += 1
        n_elem = prod_int(init.dims)
        dtype_elem_count[dt_name] += n_elem
        if len(dtype_shapes_examples[dt_name]) < 3:
            dtype_shapes_examples[dt_name].append(list(init.dims))

    return dtype_tensor_count, dtype_elem_count, dtype_shapes_examples


def inspect_io(model: onnx.ModelProto):
    inputs = [(vi.name, tensor_type_str(vi)) for vi in model.graph.input]
    outputs = [(vi.name, tensor_type_str(vi)) for vi in model.graph.output]
    return inputs, outputs


def top_optypes(model: onnx.ModelProto, k: int = 15):
    c = Counter(n.op_type for n in model.graph.node)
    return c.most_common(k)


def show_cast_nodes(model: onnx.ModelProto, limit: int = 50):
    """
    Print Cast nodes and the dtype they cast to.
    In ONNX, Cast has attribute 'to' which is a TensorProto data_type enum.
    """
    shown = 0
    for n in model.graph.node:
        if n.op_type != "Cast":
            continue
        to_val = None
        for a in n.attribute:
            if a.name == "to":
                to_val = a.i
                break
        to_name = helper.tensor_dtype_to_string(int(to_val)) if to_val is not None else "UNKNOWN"
        print(f"  Cast node name={n.name!r} to={to_name} outputs={list(n.output)} inputs={list(n.input)}")
        shown += 1
        if shown >= limit:
            print(f"  ... (stopped after {limit} Cast nodes)")
            break
    if shown == 0:
        print("  (No Cast nodes found)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=str, help="Path to .onnx model")
    ap.add_argument("--show-cast", action="store_true", help="Print Cast nodes and target dtypes")
    ap.add_argument("--topk", type=int, default=15, help="Top-K op types to print")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {model_path}")

    print(f"Loading: {model_path}")
    model = onnx.load(str(model_path))  # ONNX Python API load :contentReference[oaicite:5]{index=5}

    # Optional sanity check (can be slow on very large models)
    try:
        onnx.checker.check_model(model)
        print("onnx.checker.check_model: OK")
    except Exception as e:
        print("onnx.checker.check_model: FAILED (still continuing)")
        print("  ", e)

    # ---- IO types ----
    inputs, outputs = inspect_io(model)
    print("\n== Model Inputs ==")
    for name, dt in inputs:
        print(f"  {name}: {dt}")
    print("\n== Model Outputs ==")
    for name, dt in outputs:
        print(f"  {name}: {dt}")

    # ---- Weights/initializers dtype counts ----
    tcount, ecount, examples = inspect_initializers(model)
    total_tensors = sum(tcount.values())
    total_elems = sum(ecount.values())

    print("\n== Initializer (Weights) dtype counts ==")
    print(f"  total initializer tensors: {total_tensors}")
    print(f"  total initializer elements: {total_elems}")

    # Print in descending element count (most important)
    for dt, elems in sorted(ecount.items(), key=lambda x: x[1], reverse=True):
        tensors = tcount[dt]
        pct = (elems / total_elems * 100.0) if total_elems else 0.0
        print(f"  {dt:10s}  tensors={tensors:6d}  elems={elems:12d}  ({pct:6.2f}%)  ex_shapes={examples[dt]}")

    # ---- Op types ----
    print(f"\n== Top {args.topk} op types ==")
    for op, n in top_optypes(model, args.topk):
        print(f"  {op:20s} {n}")

    # ---- Cast nodes (optional) ----
    if args.show_cast:
        print("\n== Cast nodes (first 50) ==")
        show_cast_nodes(model, limit=50)

    print("\nDone.")
    print("\nInterpretation tip:")
    print("  - If this is truly FP16, you should see most weight elements under FLOAT16.")
    print("  - If FLOAT dominates, your model is effectively FP32 (even if the filename says fp16).")


if __name__ == "__main__":
    main()
