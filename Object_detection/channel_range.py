import onnx
import numpy as np
from onnx import numpy_helper
from pathlib import Path

onnx_path = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp32.onnx")
model = onnx.load(str(onnx_path))

def tensor_stats(w: np.ndarray):
    w = w.astype(np.float32, copy=False)
    a = np.abs(w)
    return {
        "shape": w.shape,
        "min": float(w.min()),
        "max": float(w.max()),
        "maxabs": float(a.max()),
        "p99_abs": float(np.percentile(a, 99)),
        "p999_abs": float(np.percentile(a, 99.9)),
        "median_abs": float(np.median(a)),
        "outlier_ratio_maxabs_p99": float(a.max() / (np.percentile(a, 99) + 1e-12)),
    }

def per_out_channel_maxabs(w: np.ndarray):
    # Works for Conv weights (O, I, kH, kW) and Linear/Gemm weights (O, I)
    w = w.astype(np.float32, copy=False)
    if w.ndim not in (2, 4):
        return None
    O = w.shape[0]  # typical "out channels" / "out features"
    flat = np.abs(w).reshape(O, -1)
    return flat.max(axis=1)  # (O,)

for init in model.graph.initializer:
    w = numpy_helper.to_array(init)
    s = tensor_stats(w)

    ch = per_out_channel_maxabs(w)
    if ch is not None:
        s["per_channel_spread"] = float(ch.max() / (ch.min() + 1e-12))  # how different channels are
        s["per_channel_p50_maxabs"] = float(np.median(ch))
        s["per_channel_p95_maxabs"] = float(np.percentile(ch, 95))

    print(init.name, s)