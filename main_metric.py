from pathlib import Path
import subprocess
import threading
import cv2
import numpy as np
import time
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from utils.process_utils import _watch, _ensure_depth_size
from utils.distance_utils import _compute_box_distance, _nearest_sidewalk_distance
from utils.seg_utils import _load_seg_regions_from_border_txt
from utils.depth_onnx_utils import _run_metric_depth_onnx
from utils.draw_utils import putText_outline
from utils.config_loader import SEG_CLASS_NAMES, OD_CLASS_NAME, COLORS

r'''Fully run metric depth model + object detection model + image segmentation model
FOR WHAT?
    Input: RGB image
    Output: Combines all results on one depth image, computes distances, and save a PNG overlay and a JSON file with distances
'''

# =========================
# Paths and Python envs
# =========================
ROOT = Path(__file__).resolve().parent  # project root (this file is in src/)

# Scripts
YOLO_SCRIPT          = ROOT / "Object_detection" / "main.py"
METRIC_DEPTH_SCRIPT  = ROOT / "Depth-Anything-V2-main" / "metric_depth" / "run.py"
SEG_SCRIPT           = ROOT / "Segmentation" / "test_model.py"

# Python interpreters
with open(ROOT / "src" / "configs" / "config.json", "r", encoding="utf-8") as _cf:
    _APP_CFG = json.load(_cf)

PY_YOLO  = _APP_CFG["python_interpreters"]["yolo"]
PY_DEPTH = _APP_CFG["python_interpreters"]["depth"]
PY_SEG   = _APP_CFG["python_interpreters"]["seg"]

METRIC_DEPTH_WEIGHTS = _APP_CFG["metric_depth_weights"]
ORIG_IMG = ROOT / _APP_CFG["default_image"]

# YOLO labels
YOLO_LABELS_DIR = YOLO_SCRIPT.parent / "output" / "run1" / "labels"

# Metric-depth outputs
METRIC_DEPTH_OUT_DIR = ROOT / "output_metric_depth"
METRIC_DEPTH_VIS_PNG = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}.png"
METRIC_DEPTH_RAW_NPY = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}_raw_depth_meter.npy"

# Segmentation border txt
SEG_BORDER_TXT = ROOT / "Segmentation" / "output" / "mask_border.txt"

# Final overlay and JSON output
FINAL_OUT = ROOT / "output" / f"{ORIG_IMG.stem}_metric_depth_boxes_borders.png"
JSON_OUT  = ROOT / "output" / f"{ORIG_IMG.stem}_objects_distance.json"

# Colors (BGR)
COLOR_YOLO_BOX       = tuple(COLORS["yolo_box"])
COLOR_YOLO_TEXT      = tuple(COLORS["yolo_text"])
COLOR_SEG_BORDER     = tuple(COLORS["seg_border"])
COLOR_SIDEWALK_PT    = tuple(COLORS["sidewalk_pt"])
COLOR_SIDEWALK_TEXT  = tuple(COLORS["sidewalk_text"])

# =========================
# Helpers
# =========================
def _draw_seg_borders_on(depth_bgr, polys, *, color=(0, 0, 0), thickness=2):
    if not polys:
        return depth_bgr
    cv2.polylines(depth_bgr, polys, isClosed=True,
                  color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return depth_bgr


# Main pipeline
def run_parallel_and_overlay_metric(class_names: dict | None = None, seg_args: list[str] | None = None):

    # 1) YOLO
    p_yolo = subprocess.Popen(
        [PY_YOLO, str(YOLO_SCRIPT), "--image", str(ORIG_IMG)],
        cwd=str(YOLO_SCRIPT.parent)
    )

    # 2) Metric depth
    depth_ext = os.path.splitext(METRIC_DEPTH_WEIGHTS)[1].lower()

    if depth_ext == ".pth":
        metric_cmd = [
            PY_DEPTH, str(METRIC_DEPTH_SCRIPT),
            "--encoder", "vits",
            "--load-from", METRIC_DEPTH_WEIGHTS,
            "--max-depth", "80",
            "--img-path", str(ORIG_IMG),
            "--outdir", str(METRIC_DEPTH_OUT_DIR),
            "--pred-only",
            "--grayscale",
            "--save-numpy",
        ]
        p_depth = subprocess.Popen(metric_cmd, cwd=str(METRIC_DEPTH_SCRIPT.parent))
        t2 = threading.Thread(target=_watch, args=("METRIC_DEPTH", p_depth), daemon=True)
    else:
        def _depth_worker():
            print("[METRIC_DEPTH_ONNX] starting...")
            _run_metric_depth_onnx(ORIG_IMG, METRIC_DEPTH_WEIGHTS, METRIC_DEPTH_OUT_DIR, METRIC_DEPTH_RAW_NPY)
            print("[METRIC_DEPTH_ONNX] finished.")
        t2 = threading.Thread(target=_depth_worker, daemon=True)

    # 3) Segmentation
    seg_cmd = [
        PY_SEG, str(SEG_SCRIPT),
        "--image", str(ORIG_IMG),
        "--out-border", str(SEG_BORDER_TXT),
    ]
    if seg_args:
        seg_cmd.extend(seg_args)
    p_seg = subprocess.Popen(seg_cmd, cwd=str(ROOT))

    # 4) Wait for all
    t1 = threading.Thread(target=_watch, args=("YOLO", p_yolo), daemon=True)
    t3 = threading.Thread(target=_watch, args=("SEG", p_seg), daemon=True)
    for t in (t1, t2, t3):
        t.start()
    for t in (t1, t2, t3):
        t.join()

    # 5) Load original
    orig = cv2.imread(str(ORIG_IMG))
    if orig is None:
        raise FileNotFoundError(f"Original image not found: {ORIG_IMG}")
    H, W = orig.shape[:2]

    # 6) Load depth visualization
    if not METRIC_DEPTH_VIS_PNG.exists():
        raise FileNotFoundError(f"Metric depth PNG not found: {METRIC_DEPTH_VIS_PNG}")
    depth_bgr = cv2.imread(str(METRIC_DEPTH_VIS_PNG))
    if depth_bgr is None:
        raise FileNotFoundError(f"Failed to read PNG: {METRIC_DEPTH_VIS_PNG}")
    depth_bgr = _ensure_depth_size(depth_bgr, H, W)

    # 7) Load raw depth (meters)
    depth_map_m = None
    if METRIC_DEPTH_RAW_NPY.exists():
        depth_map_m = np.load(str(METRIC_DEPTH_RAW_NPY)).astype(np.float32)
        if depth_map_m.ndim == 3:
            depth_map_m = depth_map_m.squeeze()
        if depth_map_m.shape[:2] != (H, W):
            depth_map_m = cv2.resize(depth_map_m, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"[METRIC_DEPTH] raw depth .npy not found: {METRIC_DEPTH_RAW_NPY}")

    results: list[dict] = []

    # 8) YOLO boxes + distances
    label_file = YOLO_LABELS_DIR / f"{ORIG_IMG.stem}.txt"
    if not label_file.exists():
        print(f"[YOLO] label file not found: {label_file}")
    else:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for idx, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, ww, hh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None

            px, py = cx * W, cy * H
            pw, ph = ww * W, hh * H
            x1 = max(0, int(px - pw / 2))
            y1 = max(0, int(py - ph / 2))
            x2 = min(W - 1, int(px + pw / 2))
            y2 = min(H - 1, int(py + ph / 2))

            # Use OD_CLASS_NAME dictionary for object detection class names
            cls_name = OD_CLASS_NAME.get(cls, str(cls))
            label_txt = cls_name
            if conf is not None:
                label_txt += f" {conf:.2f}"

            dist = None
            if depth_map_m is not None:
                name_l = cls_name.lower()
                is_bottom_region = any(k in name_l for k in ("car", "bicycle", "truck", "motorbike", "motorcycle"))
                mode = "bottom" if is_bottom_region else "center"
                dist = _compute_box_distance(depth_map_m, x1, y1, x2, y2, frac=0.1, mode=mode)
                if dist is not None:
                    label_txt += f" {dist:.2f}m"

            cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), COLOR_YOLO_BOX, 2)
            cv2.putText(depth_bgr, label_txt, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YOLO_TEXT, 3, cv2.LINE_AA)

            results.append({
                "id": f"yolo_{idx}",
                "source": "yolo",
                "class_id": cls,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "distance_m": dist,
            })

    # 9) Segmentation regions + borders + (name/conf/distance) labels
    seg_regions = _load_seg_regions_from_border_txt(SEG_BORDER_TXT)
    seg_polys = [r["poly"] for r in seg_regions]

    depth_bgr = _draw_seg_borders_on(depth_bgr, seg_polys, color=COLOR_SEG_BORDER, thickness=2)

    if depth_map_m is not None and seg_regions:
        for idx, reg in enumerate(seg_regions):
            poly = reg["poly"]
            seg_name = reg["class_name"]
            seg_conf = float(reg["confidence"])
            seg_cls  = int(reg["class_id"])

            # Crop to the polygon's bounding box instead of scanning the
            # full frame: turns this from O(H*W) per region into O(bbox area).
            bx, by, bw, bh = cv2.boundingRect(poly)
            bx, by = max(0, bx), max(0, by)
            bw = min(bw, W - bx)
            bh = min(bh, H - by)

            if bw <= 0 or bh <= 0:
                d_min, x_min, y_min = None, None, None
            else:
                local_poly = poly - [bx, by]
                poly_mask = np.zeros((bh, bw), dtype=np.uint8)
                cv2.fillPoly(poly_mask, [local_poly], 1)

                # band_start_frac originally excluded the top 10% of the FULL
                # image; translate that same absolute row into the crop's
                # local coordinate frame so behavior is unchanged.
                global_band_row = 0.1 * H
                local_band_frac = float(np.clip((global_band_row - by) / bh, 0.0, 1.0))

                d_min, x_min, y_min = _nearest_sidewalk_distance(
                    depth_map_m[by:by + bh, bx:bx + bw],
                    poly_mask,
                    max_depth=80.0,
                    band_start_frac=local_band_frac,
                )

                if x_min is not None and y_min is not None:
                    x_min += bx
                    y_min += by

            # anchor point: nearest depth pixel if exists, else polygon centroid
            if x_min is None or y_min is None:
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    x_min = int(M["m10"] / M["m00"])
                    y_min = int(M["m01"] / M["m00"])
                else:
                    x_min, y_min = int(poly[0, 0, 0]), int(poly[0, 0, 1])

            x_min = int(np.clip(x_min, 0, W - 1))
            y_min = int(np.clip(y_min, 0, H - 1))

            if d_min is not None:
                cv2.circle(depth_bgr, (x_min, y_min), 4, COLOR_SIDEWALK_PT, -1)
                txt = f"{seg_name} {seg_conf:.2f} {d_min:.2f}m"
            else:
                txt = f"{seg_name} {seg_conf:.2f}"

            x_txt = int(np.clip(x_min + 6, 0, W - 1))
            y_txt = int(np.clip(y_min - 6, 0, H - 1))

            putText_outline(
                depth_bgr, txt, (x_txt, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                COLOR_SIDEWALK_TEXT, thickness=3
            )

            results.append({
                "id": f"seg_{idx}",
                "source": "segmentation",
                "class_id": seg_cls,
                "class_name": seg_name,
                "confidence": seg_conf,
                "polygon": poly.reshape(-1, 2).tolist(),
                "distance_m": d_min,
                "nearest_pixel": None if d_min is None else {"x": x_min, "y": y_min},
            })

    # 10) Save overlay
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(FINAL_OUT), depth_bgr):
        raise RuntimeError(f"Failed to save final overlay to {FINAL_OUT}")
    print(f"[FINAL] saved: {FINAL_OUT}")

    # 11) Save JSON
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[JSON] saved: {JSON_OUT}")


if __name__ == "__main__":
    _t0 = time.perf_counter()
    try:
        CLASS_NAMES = None
        SEG_ARGS = None
        run_parallel_and_overlay_metric(CLASS_NAMES, SEG_ARGS)
    finally:
        elapsed = time.perf_counter() - _t0
        print(f"[RUNTIME] total elapsed: {elapsed:.3f}s (~{elapsed/60:.2f} min)")