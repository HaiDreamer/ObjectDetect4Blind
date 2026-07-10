import os
import cv2
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Root
PROJECT_ROOT = Path(__file__).resolve()

while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "assets").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
ORIG_IMG = ASSETS_DIR / "demo03.jpg"

# Metric depth weight
METRIC_DEPTH_WEIGHTS = (
    r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
)

# Metric-depth outputs
METRIC_DEPTH_OUT_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_metric_depth")
METRIC_DEPTH_VIS_PNG = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}.png"
METRIC_DEPTH_RAW_NPY = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}_raw_depth_meter.npy"

# Inputs
PROJECT_ROOT = Path(__file__).resolve()
while PROJECT_ROOT != PROJECT_ROOT.parent:
    if (PROJECT_ROOT / "assets").exists():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
ORIG_IMG = ASSETS_DIR / "demo03.jpg"



def _run_metric_depth_onnx():
    '''metric depth via ONNX runtime'''
    if ort is None:
        raise RuntimeError("onnxruntime is not installed, cannot run ONNX metric depth backend.")

    onnx_path = METRIC_DEPTH_WEIGHTS
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"[METRIC_DEPTH_ONNX] ONNX model not found: {onnx_path}")

    print(f"[METRIC_DEPTH_ONNX] loading model: {onnx_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    img_bgr = cv2.imread(str(ORIG_IMG))
    if img_bgr is None:
        raise FileNotFoundError(f"[METRIC_DEPTH_ONNX] Original image not found: {ORIG_IMG}")
    H0, W0 = img_bgr.shape[:2]

    EXPORT_SIZE = 518
    bgr_resized = cv2.resize(img_bgr, (EXPORT_SIZE, EXPORT_SIZE), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, ...]

    out = sess.run([output_name], {input_name: x})[0]
    depth_small = np.squeeze(out).astype(np.float32)

    depth_map_m = cv2.resize(depth_small, (W0, H0), interpolation=cv2.INTER_LINEAR)
    depth_map_m = np.clip(depth_map_m, 1e-3, 80.0)

    METRIC_DEPTH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(METRIC_DEPTH_OUT_DIR), depth_map_m)
    print(f"[METRIC_DEPTH_ONNX] saved raw depth: {METRIC_DEPTH_RAW_NPY}")

    depth_norm = depth_map_m / 80.0
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_8u = (depth_norm * 255.0).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)

    if not cv2.imwrite(str(METRIC_DEPTH_VIS_PNG), depth_bgr):
        raise RuntimeError(f"[METRIC_DEPTH_ONNX] Failed to save depth PNG: {METRIC_DEPTH_VIS_PNG}")
    print(f"[METRIC_DEPTH_ONNX] saved vis PNG: {METRIC_DEPTH_VIS_PNG}")
