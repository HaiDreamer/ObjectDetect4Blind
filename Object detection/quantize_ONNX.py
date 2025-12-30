from pathlib import Path
import json

from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

# FP16 converter
import onnx
from onnxconverter_common import float16

'''
Source
    https://onnxruntime.ai/docs/performance/model-optimizations/float16.html
    https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
    https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html

Explain terminology
    Dynamic quantization does not require a calibration dataset
    opset:

'''

# ----------------- CONFIG -----------------
MODELS_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models")

PT_PATH = MODELS_DIR / "best-lan2.pt"

# Output ONNX files (in the same models folder)
ONNX_FP32 = MODELS_DIR / "best-lan2_fp32.onnx"
ONNX_FP16 = MODELS_DIR / "best-lan2_fp16.onnx"
ONNX_INT8 = MODELS_DIR / "best-lan2_int8dyn_mm.onnx"  # dynamic INT8 (MatMul/Gemm only)

OPSET = 18

# ----------------- HELPERS -----------------
def export_fp32_onnx(pt_path: Path, onnx_fp32_path: Path) -> Path:
    """Export .pt -> FP32 ONNX using Ultralytics. Renames output to desired filename if needed."""
    print(f"[1/4] Exporting FP32 ONNX from: {pt_path}")
    model = YOLO(str(pt_path))

    # Ultralytics docs: ONNX supports dynamic, opset, simplify, half, etc.
    # We explicitly keep half=False for FP32 baseline. :contentReference[oaicite:5]{index=5}
    exported = model.export(
        format="onnx",
        opset=OPSET,
        dynamic=True,
        simplify=True,
        half=False,
    )

    exported_path = Path(exported) if exported else None
    if exported_path is None or not exported_path.exists():
        # Fallback: Ultralytics usually exports next to weights with stem + ".onnx"
        exported_path = pt_path.with_suffix(".onnx")

    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics export did not produce an ONNX file. Expected: {exported_path}")

    # Rename/copy to our preferred filename
    if exported_path.resolve() != onnx_fp32_path.resolve():
        onnx_fp32_path.write_bytes(exported_path.read_bytes())
        print(f"  Renamed/copy to: {onnx_fp32_path}")
    else:
        print(f"  Saved: {onnx_fp32_path}")

    print(f"  FP32 size: {onnx_fp32_path.stat().st_size/1024/1024:.2f} MB")
    return onnx_fp32_path


def convert_to_fp16(onnx_fp32_path: Path, onnx_fp16_path: Path) -> Path:
    """Convert FP32 ONNX -> FP16 ONNX (keeps IO as float32 for easier integration)."""
    print(f"[2/4] Converting to FP16: {onnx_fp32_path} -> {onnx_fp16_path}")

    model = onnx.load(str(onnx_fp32_path))

    try:
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    except Exception as e:
        # Some models can fail shape inference; retry with disable_shape_infer
        print(f"FP16 conversion retry (disable_shape_infer=True) due to: {e}")
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)

    onnx.save(model_fp16, str(onnx_fp16_path))
    print(f"  FP16 size: {onnx_fp16_path.stat().st_size/1024/1024:.2f} MB")
    return onnx_fp16_path


def quantize_int8_dynamic_mm(onnx_fp32_path: Path, onnx_int8_path: Path) -> Path:
    """
    Dynamic INT8 quantization for MatMul/Gemm only (weights-only INT8),
    using the SAME 'nodes_to_quantize' style as the DepthAnything metric depth script(metric_onnx_convert_int8_outdoor).
    """
    assert onnx_fp32_path.exists(), f"FP32 ONNX not found: {onnx_fp32_path}"

    print(f"[INT8] Quantizing FP32 ONNX -> INT8 (dynamic quantization)...")
    print("       input :", onnx_fp32_path)
    print("       output:", onnx_int8_path)

    m = onnx.load(str(onnx_fp32_path))

    # Collect node names for MatMul/Gemm (DepthAnything style)
    nodes = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm") and n.name]
    print("Quantizing nodes:", len(nodes))

    if len(nodes) == 0:
        # Some exporters leave node names empty; fallback to op_types_to_quantize.
        print("WARNING: No named MatMul/Gemm nodes found. Falling back to op_types_to_quantize.")
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

    print(f"[INT8] Saved: {onnx_int8_path} ({onnx_int8_path.stat().st_size/1024/1024:.2f} MB)")
    return onnx_int8_path


# ----------------- MAIN -----------------
def main():
    assert PT_PATH.exists(), f"Model not found: {PT_PATH}"

    # Export FP32 if missing
    if not ONNX_FP32.exists():
        export_fp32_onnx(PT_PATH, ONNX_FP32)
    else:
        print("FP32 ONNX already exists:", ONNX_FP32)

    # Build FP16 from FP32
    if not ONNX_FP16.exists():
        convert_to_fp16(ONNX_FP32, ONNX_FP16)
    else:
        print("FP16 ONNX already exists:", ONNX_FP16)

    # Build INT8 dynamic (MatMul/Gemm only) from FP32
    if not ONNX_INT8.exists():
        quantize_int8_dynamic_mm(ONNX_FP32, ONNX_INT8)
    else:
        print("INT8 ONNX already exists:", ONNX_INT8)

if __name__ == "__main__":
    main()
