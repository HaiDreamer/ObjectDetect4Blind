from pathlib import Path
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16

'''
FOR
    quantize model to FP32 then FP16, but int8 in here is nearly useless
'''

MODELS_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models")
PT_PATH = MODELS_DIR / "best_seg.pt"

ONNX_FP32 = MODELS_DIR / "best_seg_fp32.onnx"
ONNX_FP16 = MODELS_DIR / "best_seg_fp16.onnx"
ONNX_INT8 = MODELS_DIR / "best_seg_int8dyn_mm.onnx"

OPSET = 13
DYNAMIC = True
SIMPLIFY = False
IMGSZ = 640

BAD_NODE_NAME = "/model.22/Cast_5"  # may need to change for seg export

def export_fp32_onnx(pt_path: Path, onnx_fp32_path: Path) -> Path:
    print(f"[1/3] Exporting FP32 ONNX from: {pt_path}")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", opset=OPSET, dynamic=DYNAMIC, simplify=SIMPLIFY, half=False, imgsz=IMGSZ)
    exported_path = Path(exported) if exported else pt_path.with_suffix(".onnx")
    if not exported_path.exists():
        raise FileNotFoundError(f"Export failed, expected: {exported_path}")
    if exported_path.resolve() != onnx_fp32_path.resolve():
        onnx_fp32_path.write_bytes(exported_path.read_bytes())
    print("Saved:", onnx_fp32_path)
    return onnx_fp32_path

def try_load(path: Path, providers=None):
    providers = providers or ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)

def convert_to_fp16(onnx_fp32_path: Path, onnx_fp16_path: Path) -> Path:
    print(f"[2/3] Converting to FP16: {onnx_fp32_path} -> {onnx_fp16_path}")
    m = onnx.load(str(onnx_fp32_path))

    m16 = convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=False)
    onnx.save(m16, str(onnx_fp16_path))  

    print("Saved:", onnx_fp16_path)
    print("Testing load (CPU EP)...")
    try:
        try_load(onnx_fp16_path)
        print("Loaded OK:", onnx_fp16_path)
        return onnx_fp16_path
    except Exception as e:
        print("Load failed after conversion:\n", e)
        print("Retry with node_block_list (mixed precision)...")

        m16b = convert_float_to_float16(
            m,
            keep_io_types=True,
            node_block_list=[BAD_NODE_NAME],
            disable_shape_infer=True,
        )
        onnx.save(m16b, str(onnx_fp16_path))
        print("Saved (blocked node):", onnx_fp16_path)
        try_load(onnx_fp16_path)
        print("Loaded OK after blocking node:", onnx_fp16_path)
        return onnx_fp16_path

def quantize_int8_dynamic_mm(onnx_fp32_path: Path, onnx_int8_path: Path) -> Path:
    print(f"[3/3] INT8 dynamic quant (MatMul/Gemm): {onnx_fp32_path} -> {onnx_int8_path}")
    quantize_dynamic(
        model_input=str(onnx_fp32_path),
        model_output=str(onnx_int8_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    try_load(onnx_int8_path)
    print("Saved:", onnx_int8_path)
    return onnx_int8_path

def main():
    export_fp32_onnx(PT_PATH, ONNX_FP32)
    convert_to_fp16(ONNX_FP32, ONNX_FP16)        
    quantize_int8_dynamic_mm(ONNX_FP32, ONNX_INT8)

if __name__ == "__main__":
    main()
