from pathlib import Path
import random
import re

import numpy as np
import cv2
import onnx
import onnxruntime as ort
from ultralytics import YOLO

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

'''quantize model original -> fp32 -> int8 qdq static'''

# ----------------- CONFIG -----------------
MODELS_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models")
PT_PATH    = MODELS_DIR / "best-lan2.pt"

ONNX_FP32  = MODELS_DIR / "best-lan2_fp32.onnx"

# "standard" INT8 output (static calibrated)
ONNX_INT8_STATIC = MODELS_DIR / "best-lan2_int8_static_qdq.onnx"

DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\GroupProject_OD\data.yaml")

IMGSZ = 640
OPSET = 18
DYNAMIC = False       # strongly recommended for quantization stability
SIMPLIFY = False

# Calibration settings
N_CALIB_IMAGES = 300  
SEED = 0

# Quantization settings (CPU-friendly)
# For CPU EP, use activation=QUInt8, weight=QInt8 (U8S8).
ACTIVATION_TYPE = QuantType.QUInt8
WEIGHT_TYPE     = QuantType.QInt8
QFORMAT         = QuantFormat.QDQ

# Quantize YOLO-relevant ops
OP_TYPES_TO_QUANTIZE = ["Conv", "MatMul", "Gemm"]


# ----------------- UTILS -----------------
def export_fp32_onnx(pt_path: Path, onnx_fp32_path: Path) -> Path:
    print(f"[1/3] Export FP32 ONNX: {pt_path} -> {onnx_fp32_path}")
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
        raise FileNotFoundError(f"Ultralytics export did not produce ONNX: {exported_path}")

    if exported_path.resolve() != onnx_fp32_path.resolve():
        onnx_fp32_path.write_bytes(exported_path.read_bytes())

    print(f"  Saved: {onnx_fp32_path} ({onnx_fp32_path.stat().st_size/1024/1024:.2f} MB)")
    return onnx_fp32_path


def load_data_yaml_val_dir(yaml_path: Path) -> Path:
    """
    Tries to read 'val:' path from Ultralytics YAML.
    Uses PyYAML if present; otherwise falls back to regex.
    """
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")

    # Try PyYAML first (usually installed with Ultralytics)
    try:
        import yaml 
        data = yaml.safe_load(text)
        v = data.get("val", None)
        if v:
            return Path(v)
    except Exception:
        pass

    # Fallback regex: line that starts with "val:"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("val:"):
            return Path(s.split("val:", 1)[1].strip())

    raise ValueError(f"Could not parse 'val:' from {yaml_path}")


def list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for e in exts:
        files += list(folder.rglob(f"*{e}"))
    return files


def letterbox(im, new_shape=640, color=(114, 114, 114)):
    """Classic YOLO letterbox to keep aspect ratio."""
    shape = im.shape[:2]  # (h,w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w,h)

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im


def preprocess_yolo(im_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> NCHW float32 in [0,1], with letterbox."""
    im = letterbox(im_bgr, IMGSZ)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im.transpose(2, 0, 1)  # HWC -> CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    im = im[None, ...]  # add batch
    return im


class YoloCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths, input_name: str):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self._iter = iter(self.image_paths)

    def get_next(self):
        for p in self._iter:
            img = cv2.imread(str(p))
            if img is None:
                continue
            inp = preprocess_yolo(img)
            return {self.input_name: inp}
        return None

    def rewind(self):
        self._iter = iter(self.image_paths)


def try_load_ort(path: Path, providers=None):
    providers = providers or ["CPUExecutionProvider"]
    _ = ort.InferenceSession(str(path), providers=providers)


# ----------------- INT8 STATIC QUANT -----------------
def quantize_int8_static(fp32_path: Path, out_int8_path: Path, calib_images, prefer_trt=False):
    """
    prefer_trt=False (CPU): asymmetric activation + symmetric weights recommended.
    prefer_trt=True  (TensorRT): symmetric activation + symmetric weights required (usually).
    """
    print(f"[2/3] INT8 static quantization (QDQ): {fp32_path} -> {out_int8_path}")

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print("  Model input:", input_name, sess.get_inputs()[0].shape, sess.get_inputs()[0].type)

    reader = YoloCalibrationDataReader(calib_images, input_name=input_name)

    extra = {
        # CPU recommendation: asymmetric activations, symmetric weights
        "ActivationSymmetric": bool(prefer_trt),
        "WeightSymmetric": True,
    }

    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_int8_path),
        calibration_data_reader=reader,
        quant_format=QFORMAT,
        activation_type=(QuantType.QInt8 if prefer_trt else ACTIVATION_TYPE),
        weight_type=WEIGHT_TYPE,
        op_types_to_quantize=OP_TYPES_TO_QUANTIZE,
        per_channel=True,
        reduce_range=False,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=extra,
    )

    # ort load by cpu 
    try_load_ort(out_int8_path, providers=["CPUExecutionProvider"])
    print(f"ORT can load: {out_int8_path} ({out_int8_path.stat().st_size/1024/1024:.2f} MB)")


def main():
    if not PT_PATH.exists() and not ONNX_FP32.exists():
        raise FileNotFoundError(f"Need either PT or FP32 ONNX. Missing: {PT_PATH} and {ONNX_FP32}")

    if not ONNX_FP32.exists():
        export_fp32_onnx(PT_PATH, ONNX_FP32)
    else:
        print("FP32 exists:", ONNX_FP32)

    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"
    val_dir = load_data_yaml_val_dir(DATA_YAML)
    print("Validation images dir (for calibration):", val_dir)

    images = list_images(val_dir)
    if len(images) == 0:
        raise FileNotFoundError(f"No images found under: {val_dir}")

    random.seed(SEED)
    random.shuffle(images)
    calib = images[: min(N_CALIB_IMAGES, len(images))]
    print(f"Using {len(calib)} calibration images")

    # IMPORTANT: start with CPU-style INT8 (most common)
    quantize_int8_static(ONNX_FP32, ONNX_INT8_STATIC, calib, prefer_trt=False)

    print("[3/3] Done.")


if __name__ == "__main__":
    main()
