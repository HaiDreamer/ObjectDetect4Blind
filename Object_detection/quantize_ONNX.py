from pathlib import Path
import re

import onnx
import onnxruntime as ort
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16

'''quantize model original -> fp32 -> fp16 and int8 , but int8 model seems useless(u can ignore this qay of quantizing)'''

# ----------------- CONFIG -----------------
MODELS_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models")
PT_PATH = MODELS_DIR / "best-lan2.pt"

ONNX_FP32 = MODELS_DIR / "best-lan2_fp32.onnx"
ONNX_FP16 = MODELS_DIR / "best-lan2_fp16.onnx"     
ONNX_INT8 = MODELS_DIR / "best-lan2_int8dyn_mm.onnx"

OPSET = 18
DYNAMIC = True
SIMPLIFY = False   # avoid simplifier issues i saw
IMGSZ = 640

OUT  = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp16.onnx")
BAD_NODE_NAME = "/model.22/Cast_5"  # from error message, i have meet more than 10 time!!!

# ----------------- HELPERS -----------------
def export_fp32_onnx(pt_path: Path, onnx_fp32_path: Path) -> Path:
    """Export .pt -> ONNX FP32 using Ultralytics and copy/rename to desired filename."""
    print(f"[1/3] Exporting FP32 ONNX from: {pt_path}")
    model = YOLO(str(pt_path))

    exported = model.export(
        format="onnx",
        opset=OPSET,
        dynamic=DYNAMIC,
        simplify=SIMPLIFY,
        half=False,
        imgsz=IMGSZ,
    )

    exported_path = Path(exported) if exported else pt_path.with_suffix(".onnx")
    if not exported_path.exists():
        raise FileNotFoundError(f"Ultralytics export did not produce ONNX. Expected: {exported_path}")

    if exported_path.resolve() != onnx_fp32_path.resolve():
        onnx_fp32_path.write_bytes(exported_path.read_bytes())

    print(f"Saved: {onnx_fp32_path} ({onnx_fp32_path.stat().st_size/1024/1024:.2f} MB)")
    return onnx_fp32_path


def try_load(path: Path):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess


def convert_to_fp16(onnx_fp32_path: Path, onnx_fp16_path: Path) -> Path:
    """
    Convert FP32 ONNX -> FP16 ONNX
    Keep fp32 blocking problematic nodes if ORT cannot load the model.
    """
    assert onnx_fp32_path.exists(), f"FP32 ONNX not found: {onnx_fp32_path}"

    print(f"[2/3] Converting to FP16: {onnx_fp32_path} -> {onnx_fp16_path}")
    m = onnx.load(str(onnx_fp32_path))

    m16 = convert_float_to_float16(
        m,
        keep_io_types=True,          # keep model inputs/outputs float32 (easier for pipelines)
        disable_shape_infer=False
    )
    onnx.save(m16, str(OUT))

    print("Saved:", OUT)
    print("Testing load...")
    try:
        try_load(OUT)
        print("Loaded OK:", OUT)
    except Exception as e:
        print("Load failed after basic conversion:\n", e)

        print("\n[2] Retry: block the failing node so it stays float32 (mixed precision)")
        m16b = convert_float_to_float16(
            m,
            keep_io_types=True,
            node_block_list=[BAD_NODE_NAME],  # keep this node in FP32 when error 
            disable_shape_infer=True          # sometimes helps when types are already annotated 
        )
        onnx.save(m16b, str(OUT))
        print("Saved (blocked node):", OUT)
        print("Testing load again...")
        try_load(OUT)
        print("Loaded OK after blocking node:", OUT)


def quantize_int8_dynamic_mm(onnx_fp32_path: Path, onnx_int8_path: Path) -> Path:
    """
    Dynamic INT8 quantization for MatMul/Gemm only (weights-only INT8) for able to run in mobile app
    But it seems not work because YOLO model almost contain conv layers
    """
    assert onnx_fp32_path.exists(), f"FP32 ONNX not found: {onnx_fp32_path}"
    print(f"[3/3] INT8 dynamic quant (MatMul/Gemm): {onnx_fp32_path} -> {onnx_int8_path}")

    m = onnx.load(str(onnx_fp32_path))
    nodes = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm") and n.name]
    print("MatMul/Gemm named nodes:", len(nodes))

    if len(nodes) == 0:
        print("WARNING: No named MatMul/Gemm nodes. Falling back to op_types_to_quantize.")
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

    qm = onnx.load(str(onnx_int8_path))
    onnx.checker.check_model(qm)

    print(f"  Saved: {onnx_int8_path} ({onnx_int8_path.stat().st_size/1024/1024:.2f} MB)")
    return onnx_int8_path


# ----------------- MAIN -----------------
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
