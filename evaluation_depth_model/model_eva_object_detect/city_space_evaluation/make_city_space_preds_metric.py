from pathlib import Path
import time
import cv2
import numpy as np
import sys
import json
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
CITYSCAPES_ROOT = Path(r"D:\ObjectDetection4Blind-pt2\CitySpace")
SPLIT           = "val"
EVAL_H, EVAL_W  = 256, 512
MAX_DEPTH       = 80.0

IMG_DIR    = CITYSCAPES_ROOT / "leftImg8bit" / SPLIT
DISP_DIR   = CITYSCAPES_ROOT / "disparity"   / SPLIT
CAMERA_DIR = CITYSCAPES_ROOT / "camera"      / SPLIT
GT_OUT_DIR   = CITYSCAPES_ROOT / "depth_gt"   / SPLIT  # output
PRED_OUT_DIR = CITYSCAPES_ROOT / "depth_pred" / SPLIT  # output2
GT_OUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
import torch
DEPTH_ANYTHING_REPO = Path(r"C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main").resolve()
METRIC_DIR = DEPTH_ANYTHING_REPO / "metric_depth"
if str(METRIC_DIR) not in sys.path:
    sys.path.insert(0, str(METRIC_DIR))
from depth_anything_v2.dpt import DepthAnythingV2         # type: ignore

CKPT   = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = DepthAnythingV2(encoder="vits", features=64,
                        out_channels=[48, 96, 192, 384], max_depth=MAX_DEPTH)
state = torch.load(str(CKPT), map_location="cpu")
if "model" in state:
    sd = state["model"]
elif "state_dict" in state:
    sd = state["state_dict"]
else:
    sd = state
model.load_state_dict(sd, strict=True)
model.to(DEVICE).eval()

@torch.inference_mode()
def infer_metric_depth(bgr: np.ndarray) -> np.ndarray:
    return model.infer_image(bgr, input_size=518).astype(np.float32, copy=False)

# ── GT loader ─────────────────────────────────────────────────────────────────
def load_depth_gt(disp_path: Path, cam_path: Path) -> np.ndarray:
    disparity = np.array(Image.open(disp_path)).astype(np.float32)

    with open(cam_path) as f:
        camera = json.load(f)
    baseline = camera["extrinsic"]["baseline"]
    focal    = camera["intrinsic"]["fx"]

    disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256.0
    depth_gt = np.zeros_like(disparity)
    mask = disparity > 0
    depth_gt[mask] = baseline * focal / disparity[mask]

    valid_mask = (depth_gt > 0).astype(np.uint8)
    valid_mask = np.array(
        Image.fromarray(valid_mask).resize((EVAL_W, EVAL_H), Image.NEAREST)
    )
    depth_gt = np.array(
        Image.fromarray(depth_gt).resize((EVAL_W, EVAL_H), Image.NEAREST)
    )
    depth_gt[valid_mask == 0] = 0  # restore invalid pixels to 0
    return depth_gt

# ── Inference loop ────────────────────────────────────────────────────────────
img_paths = sorted(IMG_DIR.rglob("*_leftImg8bit.png"))
assert img_paths, f"No images found in {IMG_DIR}"

print(f"Backend: PyTorch ({DEVICE})")
print(f"Found {len(img_paths)} val images")
print(f"GT   → {GT_OUT_DIR}")
print(f"Pred → {PRED_OUT_DIR}")

t0 = time.perf_counter()

for i, img_path in enumerate(img_paths, 1):
    city = img_path.parent.name
    base = img_path.name.replace("_leftImg8bit.png", "")

    disp_path = DISP_DIR   / city / f"{base}_disparity.png"
    cam_path  = CAMERA_DIR / city / f"{base}_camera.json"

    if not disp_path.exists() or not cam_path.exists():
        print(f"[WARN] Missing GT for {base}, skipping")
        continue

    # Load + resize image
    img_bgr = cv2.imread(str(img_path))
    img_bgr = cv2.resize(img_bgr, (EVAL_W, EVAL_H), interpolation=cv2.INTER_LINEAR)

    # Predict
    pred_m = infer_metric_depth(img_bgr)
    if pred_m.shape != (EVAL_H, EVAL_W):
        pred_m = cv2.resize(pred_m, (EVAL_W, EVAL_H), interpolation=cv2.INTER_LINEAR)
    pred_m = np.clip(pred_m, 1e-3, MAX_DEPTH)

    # Save GT as KITTI uint16 PNG (flat)
    gt_m   = load_depth_gt(disp_path, cam_path)
    gt_u16 = np.clip(np.rint(gt_m * 256.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(GT_OUT_DIR   / f"{base}_depth_gt.png"),   gt_u16)

    # Save pred as KITTI uint16 PNG (flat, same base name for eval script)
    pred_u16 = np.clip(np.rint(pred_m * 256.0), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(PRED_OUT_DIR / f"{base}_depth_pred.png"), pred_u16)

    if i % 25 == 0 or i == len(img_paths):
        print(f"{i}/{len(img_paths)} done")

elapsed = time.perf_counter() - t0
imgs = len(img_paths)
print(f"\nDone")
print(f"Total: {elapsed:.2f}s | Avg: {elapsed/max(imgs,1):.3f}s/img | Throughput: {imgs/elapsed if elapsed>0 else 0:.2f} img/s")