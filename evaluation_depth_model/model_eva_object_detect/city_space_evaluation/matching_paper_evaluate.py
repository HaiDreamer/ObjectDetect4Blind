from pathlib import Path
import cv2
import json
import numpy as np
from PIL import Image

"""
Evaluate depth on Cityscapes in METRIC DEPTH space (meters)
Avoids disparity conversion error amplification at far distances.

Run:
    python eval_cityscapes_metric.py

BUT: can not fair comparison
    Model                SOSD-Net (paper)        Depth Anything V2
    Output               Disparity [px]          Metric depth [meters]
    Training             Cityscapes (disparity)  Virtual KITTI
    Task                 Depth + Segmentation    Depth only
    Domain               Cityscapes urban        General / KITTI

=> Fine tune depth anything v2 model with CitySpace dataset ?
"""

# ── Config ────────────────────────────────────────────────────────────────────
CITYSCAPES_ROOT = Path(r"D:\ObjectDetection4Blind-pt2\CitySpace")
SPLIT           = "val"
EVAL_H, EVAL_W  = 256, 512
DMIN, DMAX      = 1e-3, 80.0

DISP_DIR   = CITYSCAPES_ROOT / "disparity"  / SPLIT
CAMERA_DIR = CITYSCAPES_ROOT / "camera"     / SPLIT
PRED_DIR   = CITYSCAPES_ROOT / "depth_pred" / SPLIT

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_pred_depth(pred_path: Path) -> np.ndarray:
    x = cv2.imread(str(pred_path), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(pred_path)
    if x.ndim != 2:
        x = x[..., 0]
    return x.astype(np.float32) / 256.0


def load_gt_depth_meters(disp_path: Path, cam_path: Path) -> np.ndarray:
    """GT disparity → metric depth in meters. Returns 0 where invalid."""
    raw = np.array(Image.open(disp_path)).astype(np.float32)

    with open(cam_path) as f:
        camera = json.load(f)
    baseline = camera["extrinsic"]["baseline"]
    fx       = camera["intrinsic"]["fx"]

    disp = np.zeros_like(raw)
    mask = raw > 0
    disp[mask] = (raw[mask] - 1.0) / 256.0

    depth = np.zeros_like(disp)
    valid = disp > 0
    depth[valid] = baseline * fx / disp[valid]

    # Resize to 256x512, preserve invalid mask
    valid_mask = (depth > 0).astype(np.uint8)
    depth      = np.array(Image.fromarray(depth).resize((EVAL_W, EVAL_H), Image.NEAREST))
    valid_mask = np.array(Image.fromarray(valid_mask).resize((EVAL_W, EVAL_H), Image.NEAREST))
    depth[valid_mask == 0] = 0

    return depth  # meters, 0 = invalid


def main():
    disp_paths = sorted(DISP_DIR.rglob("*_disparity.png"))
    assert disp_paths, f"No disparity PNGs found in {DISP_DIR}"

    # Per-image accumulators (same metrics as your eval_kitti_subset.py)
    accs    = []
    missing = 0

    for disp_path in disp_paths:
        city = disp_path.parent.name
        base = disp_path.name.replace("_disparity.png", "")

        pred_path = PRED_DIR / f"{base}_depth_pred.png"
        cam_path  = CAMERA_DIR / city / f"{base}_camera.json"

        if not pred_path.exists():
            missing += 1
            continue

        # GT in meters
        gt_m = load_gt_depth_meters(disp_path, cam_path)

        # Pred in meters
        pred_m = load_pred_depth(pred_path)
        if pred_m.shape != (EVAL_H, EVAL_W):
            pred_m = cv2.resize(pred_m, (EVAL_W, EVAL_H), interpolation=cv2.INTER_LINEAR)

        pred_m = np.clip(pred_m, DMIN, DMAX)

        # Valid pixels only
        valid = (gt_m > DMIN) & (gt_m <= DMAX)
        if valid.sum() == 0:
            continue

        g = gt_m[valid]
        p = pred_m[valid]

        thresh = np.maximum(p / g, g / p)
        d1     = (thresh < 1.25).mean()
        d2     = (thresh < 1.25 ** 2).mean()
        d3     = (thresh < 1.25 ** 3).mean()
        absrel = np.mean(np.abs(p - g) / g)
        sqrel  = np.mean(((p - g) ** 2) / g)
        rmse   = np.sqrt(np.mean((p - g) ** 2))
        mae    = np.mean(np.abs(p - g))
        rmselog= np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
        e      = np.log(p) - np.log(g)
        silog  = np.sqrt(np.mean(e ** 2) - np.mean(e) ** 2) * 100.0
        log10  = np.mean(np.abs(np.log10(p) - np.log10(g)))

        accs.append((d1, d2, d3, absrel, sqrel, rmse, mae, rmselog, silog, log10))

    if not accs:
        raise RuntimeError("No valid pairs found.")

    accs   = np.array(accs, dtype=np.float64)
    labels = ["d1", "d2", "d3", "AbsRel", "SqRel", "RMSE", "MAE", "RMSElog", "SILog", "log10"]

    print(f"\nImages evaluated : {len(disp_paths) - missing}")
    print(f"Missing          : {missing}")
    print(f"\n{', '.join(labels)}")
    print(', '.join(f"{accs[:, i].mean():.3f}" for i in range(accs.shape[1])))


if __name__ == "__main__":
    main()