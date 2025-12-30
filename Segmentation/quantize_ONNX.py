# save as: quantize_seg_ONNX.py
# Usage: python quantize_seg_ONNX.py

from pathlib import Path
import warnings

from ultralytics import YOLO

import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType


# ----------------- CONFIG -----------------
MODELS_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models")

PT_PATH = MODELS_DIR / "best_seg.pt"

ONNX_FP32 = MODELS_DIR / "best_seg_fp32.onnx"
ONNX_FP16 = MODELS_DIR / "best_seg_fp16.onnx"
ONNX_INT8 = MODELS_DIR / "best_seg_int8dyn_mm.onnx"  # dynamic INT8 (MatMul/Gemm only)

OPSET = 18
DYNAMIC = True

# If you see the same simplifier warning you saw before,
# set SIMPLIFY=False to silence it (export still works either way).
SIMPLIFY = False


# ----------------- HELPERS -----------------
def export_fp32_onnx(pt_path: Path, onnx_fp32_path: Path) -> Path:
    """Export .pt -> ONNX FP32 using Ultralytics."""
    print(f"[1/3] Exporting FP32 ONNX from: {pt_path}")
    model = YOLO(str(pt_path))

    exported = model.export(
        format="onnx",
        opset=OPSET,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        half=False,  # keep FP32 baseline
        imgsz=640,   # keep consistent export input (change if your seg model trained differently)
    )

    exported_path = Path(exported) if exported else pt_path.with_suffix(".onnx")
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics export did not produce ONNX. Expected: {exported_path}")

    # Copy/rename to our preferred name
    if exported_path.resolve() != onnx_fp32_path.resolve():
        onnx_fp32_path.write_bytes(exported_path.read_bytes())

    print(f"  Saved: {onnx_fp32_path} ({onnx_fp32_path.stat().st_size/1024/1024:.2f} MB)")
    return onnx_fp32_path


def convert_to_fp16(onnx_fp32_path: Path, onnx_fp16_path: Path) -> Path:
    """
    Convert FP32 ONNX -> FP16 ONNX (keeps IO as float32).
    Uses onnxconverter_common.float16.convert_float_to_float16. :contentReference[oaicite:2]{index=2}
    """
    print(f"[2/3] Converting to FP16: {onnx_fp32_path} -> {onnx_fp16_path}")
    m = onnx.load(str(onnx_fp32_path))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*will be truncated.*",
            category=UserWarning,
            module=r"onnxconverter_common\.float16",
        )
        try:
            m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True)
        except Exception as e:
            print(f"  FP16 conversion retry (disable_shape_infer=True) due to: {e}")
            m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)

    onnx.save(m_fp16, str(onnx_fp16_path))
    onnx.checker.check_model(onnx.load(str(onnx_fp16_path)))

    print(f"  Saved: {onnx_fp16_path} ({onnx_fp16_path.stat().st_size/1024/1024:.2f} MB)")
    return onnx_fp16_path


def quantize_int8_dynamic_mm(onnx_fp32_path: Path, onnx_int8_path: Path) -> Path:
    """
    Dynamic INT8 quantization (weights-only) for MatMul/Gemm.
    Dynamic quantization example & behavior are documented by ORT. :contentReference[oaicite:3]{index=3}
    """
    print(f"[3/3] INT8 dynamic quant (MatMul/Gemm): {onnx_fp32_path} -> {onnx_int8_path}")
    assert onnx_fp32_path.exists(), f"FP32 ONNX not found: {onnx_fp32_path}"

    # Try DepthAnything-style: quantize named MatMul/Gemm nodes only
    m = onnx.load(str(onnx_fp32_path))
    nodes = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm") and n.name]
    print("  MatMul/Gemm named nodes:", len(nodes))

    if len(nodes) == 0:
        print("  WARNING: No named MatMul/Gemm nodes. Falling back to op_types_to_quantize.")
        quantize_dynamic(
            model_input=str(onnx_fp32_path),
            model_output=str(onnx_int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
    else:
        quantize_dynamic(
            model_input=str(onnx_fp32_path),
            model_output=str(onnx_int8_path),
            weight_type=QuantType.QInt8,
            nodes_to_quantize=nodes,
        )

    # Sanity check
    qm = onnx.load(str(onnx_int8_path))
    onnx.checker.check_model(qm)

    print(f"  Saved: {onnx_int8_path} ({onnx_int8_path.stat().st_size/1024/1024:.2f} MB)")
    return onnx_int8_path


def main():
    assert PT_PATH.exists(), f"Model not found: {PT_PATH}"

    if not ONNX_FP32.exists():
        export_fp32_onnx(PT_PATH, ONNX_FP32)
    else:
        print("FP32 already exists:", ONNX_FP32)

    if not ONNX_FP16.exists():
        convert_to_fp16(ONNX_FP32, ONNX_FP16)
    else:
        print("FP16 already exists:", ONNX_FP16)

    if not ONNX_INT8.exists():
        quantize_int8_dynamic_mm(ONNX_FP32, ONNX_INT8)
    else:
        print("INT8 already exists:", ONNX_INT8)


if __name__ == "__main__":
    main()
