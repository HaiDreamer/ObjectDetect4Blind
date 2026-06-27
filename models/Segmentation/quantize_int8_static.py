from pathlib import Path
import random

import numpy as np
import cv2
import onnxruntime as ort
from ultralytics import YOLO

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
    CalibrationMethod,
)

'''
FOR: Quantize fp32 to int8 static qdq model

INPUT: original model
OUTPUT: int8 model
'''

# ----------------- CONFIG -----------------
MODELS_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models")
PT_PATH    = MODELS_DIR / "best_seg.pt"

ONNX_FP32  = MODELS_DIR / "best_seg_fp32.onnx"
ONNX_INT8_STATIC = MODELS_DIR / "best_seg_int8_static_qdq.onnx"

DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\data.yaml")

IMGSZ = 640
OPSET = 13  
DYNAMIC = False
SIMPLIFY = False

# Calibration
N_CALIB_IMAGES = 300
SEED = 0

# Quantization (CPU-friendly)
ACTIVATION_TYPE = QuantType.QUInt8
WEIGHT_TYPE     = QuantType.QInt8
QFORMAT         = QuantFormat.QDQ

# Quantize the heavy hitters
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
    Reads 'val:' from Ultralytics data.yaml.
    Resolves relative paths relative to yaml file location (important).
    """
    text = yaml_path.read_text(encoding="utf-8", errors="ignore")
    base = yaml_path.parent

    # Try PyYAML first
    try:
        import yaml
        data = yaml.safe_load(text)
        v = data.get("val", None)
        if v:
            p = Path(v)
            return (p if p.is_absolute() else (base / p)).resolve()
    except Exception:
        pass

    # Fallback: parse "val:" line
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("val:"):
            v = s.split("val:", 1)[1].strip()
            p = Path(v)
            return (p if p.is_absolute() else (base / p)).resolve()

    raise ValueError(f"Could not parse 'val:' from {yaml_path}")


def list_images(folder: Path):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = []
    for e in exts:
        files += list(folder.rglob(f"*{e}"))
    return files


def letterbox(im, new_shape=640, color=(114, 114, 114)):
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


def quantize_int8_static(fp32_path: Path, out_int8_path: Path, calib_images, prefer_trt=False):
    print(f"[2/3] INT8 static quantization (QDQ): {fp32_path} -> {out_int8_path}")

    sess = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print("  Model input:", input_name, sess.get_inputs()[0].shape, sess.get_inputs()[0].type)

    reader = YoloCalibrationDataReader(calib_images, input_name=input_name)

    extra = {
        "ActivationSymmetric": bool(prefer_trt),  # TRT usually wants symmetric activations
        "WeightSymmetric": True,                  # good default for weights
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

    quantize_int8_static(ONNX_FP32, ONNX_INT8_STATIC, calib, prefer_trt=False)
    print("[3/3] Done.")


if __name__ == "__main__":
    main()
