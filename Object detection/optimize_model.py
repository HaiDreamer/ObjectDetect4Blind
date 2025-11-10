'''
Requirement: 
    - Sample data required (for recommended calibration) ~300-500 random img, no need label(only img)
        Priority from val setInclude normal cases and edge cases
        Need check mAP ?
    -> have record activation distribution
    - labeled set (images + ground-truth annotations)
    -> checking accuracy, re-train model for weights
    - QAT ?

Quantization implementation
    - Per-tensor vs per-channel ?
    - Symmetric vs asymmetric quantization ?
    - MinMax and Entropy method (need Percentile)

Calibration for ?
    - float tensor (weights and activations) get maped
    to 8 bit int, must choose range mapping to each tensor
    - Avoid poor ranges hurt classification and box regression

    ai-edge-litert>=1.2.0
'''

# this ONNX fail, too heavy data (267k KB >> 137 MB)
# ONNX files are often larger than PyTorch .pt

# save as: optimize_model_dynamic_mm.py
# Usage: python optimize_model_dynamic_mm.py

from pathlib import Path
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType
import json

# ----------------- CONFIG -----------------
MODELS_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models")
PT_PATH    = MODELS_DIR / "detect_best.pt"
ONNX_FP32  = MODELS_DIR / "detect_best.onnx"
ONNX_INT8  = MODELS_DIR / "detect_best_int8dyn_mm.onnx"  # <- quantized output here

IMAGE_PATH = Path(r"C:\Python\ObjectDetect4Blind\assets\demo01.jpg")
OUT_IMG    = Path(__file__).resolve().parent / "output"
RUN_NAME   = "run_quant_mm"

# ----------------- CHECKS -----------------
assert PT_PATH.exists(), f"Model not found: {PT_PATH}"
assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"

# ----------------- 1) .pt -> ONNX (if needed) -----------------
if not ONNX_FP32.exists():
    print("Exporting .pt to ONNX...")
    model_pt = YOLO(str(PT_PATH))
    # opset 13 is a safe default; dynamic=True keeps shape flexibility
    onnx_path_str = model_pt.export(format="onnx", opset=13, dynamic=True)
    print("ONNX saved:", onnx_path_str)
else:
    print("ONNX already exists:", ONNX_FP32)

# ----------------- 2) Dynamic quant (MatMul/Gemm only) -----------------
print("Applying dynamic quantization (MatMul/Gemm only)...")
quantize_dynamic(
    model_input=str(ONNX_FP32),
    model_output=str(ONNX_INT8),
    weight_type=QuantType.QInt8,              # weights-only INT8; no calibration
    op_types_to_quantize=["MatMul", "Gemm"]   # <-- avoid Conv -> no ConvInteger kernel issues
)
print("INT8 (dynamic) ONNX saved:", ONNX_INT8)

# ----------------- 3) Inference with quantized ONNX -----------------
print(f"Loading {ONNX_INT8} for ONNX Runtime inference via Ultralytics...")
model_q = YOLO(str(ONNX_INT8))

results = model_q.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    iou=0.7,
    save=True,          # save rendered image(s)
    save_txt=True,      # save YOLO-format txt predictions
    save_conf=True,     # include confidences in the txt files
    project=str(OUT_IMG),
    name=RUN_NAME,
    exist_ok=True
)
print("Saved renders to:", OUT_IMG / RUN_NAME)

# ----------------- 4) JSON output (same structure) -----------------
r = results[0]
ids   = [int(i) for i in r.boxes.cls.tolist()]
names = [model_q.names[i] for i in ids]
xyxy  = [list(map(float, b)) for b in r.boxes.xyxy.tolist()]
conf  = [float(c) for c in r.boxes.conf.tolist()]

payload = {
    "image": str(IMAGE_PATH),
    "detections": [
        {"class_id": ids[i], "class_name": names[i],
         "box_xyxy": xyxy[i], "score": conf[i]}
        for i in range(len(ids))
    ]
}

out_json = OUT_IMG / RUN_NAME / (IMAGE_PATH.stem + ".json")
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved JSON to:", out_json)
