import json
from pathlib import Path
from datetime import datetime
from time import perf_counter

import numpy as np
from PIL import Image
import cv2


DET_JSON = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\bb_json_KITTI_val.json"
DEPTH_GT_DIR = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth"
OUT_JSON = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\bb_distance_json_KITTI_val_GT.json"

# =========================
# DISTANCE PARAMS
# =========================
MAX_DEPTH = 80.0
Q = 10.0
FRAC = 0.1
'''
frac: %region of ROI
        frac = sqrt(area_fraction)
            1% area -> 0.1
            10% area → sqrt(0.10)=0.316
            20% area → 0.447
            30% area → 0.548
            100% area → 1.0
'''
SUBSAMPLE = 1

# Confidence filtering
CONF_THR = 0.25
EXCLUDE_LOW_CONF = False
#   False: keep object in output, but don't evaluate distance (distance=None, excluded_low_conf=True)
#   True : drop object entirely from output

# ROI method
ROI_METHOD = "region"   # "region" or "pixel"
#   "pixel": 1 middle pixel (center) or bottom-center pixel (bottom)
#   "region": your ROI patch + percentile

# Mode selection (NO class-name keyword checks)
DEFAULT_MODE = "center"        # "center" or "bottom"
USE_DET_MODE_FIELD = False     # if True, read mode from detection dict field below
DET_MODE_FIELD = "roi_mode"    # expected values: "center" or "bottom"


# =========================
# Helpers
# =========================
def read_kitti_depth_png_to_meters(depth_png_path: Path) -> np.ndarray:
    """
    KITTI depth GT:
      - uint16 PNG
      - 0 invalid
      - meters = uint16 / 256.0
    """
    I = np.array(Image.open(depth_png_path), dtype=np.uint16)
    depth_m = I.astype(np.float32) / 256.0
    depth_m[I == 0] = np.nan
    return depth_m

def fast_percentile_1d(vals: np.ndarray, q: float) -> float | None:
    """Use np.partition for fast order statistic."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    k = int(round((q / 100.0) * (vals.size - 1)))
    k = max(0, min(vals.size - 1, k))
    return float(np.partition(vals, k)[k])

def _clamp_box_xyxy(x1, y1, x2, y2, H, W):
    '''Convert possibly-float bbox coords to integer pixel indices and Clamp them so they stay inside the image'''
    x1 = int(max(0, min(W - 1, np.floor(x1))))
    y1 = int(max(0, min(H - 1, np.floor(y1))))
    x2 = int(max(0, min(W, np.ceil(x2))))
    y2 = int(max(0, min(H, np.ceil(y2))))
    return x1, y1, x2, y2

def compute_box_distance_pixel(depth_m: np.ndarray, x1, y1, x2, y2, mode="center"):
    """
    1-pixel distance:
      - center: pixel at bbox center
      - bottom: pixel at bottom-center (y = y2-1)
    """
    H, W = depth_m.shape[:2]
    x1, y1, x2, y2 = _clamp_box_xyxy(x1, y1, x2, y2, H, W)
    if x2 <= x1 or y2 <= y1:
        return None, 0

    cx = (x1 + x2) // 2
    if mode == "bottom":
        cy = max(y1, min(H - 1, y2 - 1))   # bottom-center pixel
    else:
        cy = (y1 + y2) // 2

    d = depth_m[cy, cx]
    if not np.isfinite(d) or not (0.0 < d < MAX_DEPTH):
        return None, 0
    return float(d), 1

def compute_box_distance_region(depth_m: np.ndarray, x1, y1, x2, y2, frac=0.316, mode="center", q=10.0, subsample=1):
    '''
    default mode: center
    q: x% of lowest value of distance pixel array
    frac: %region of ROI
        frac = sqrt(area_fraction)
            10% area → sqrt(0.10)=0.316
            20% area → 0.447
            30% area → 0.548
            100% area → 1.0
    subsample: downsampling step used to speed up the distance calculation by using fewer depth pixels inside the ROI(region of interest)
        SUBSAMPLE = 1 → use every pixel (most accurate, slowest)
        SUBSAMPLE = 2 → use every 2nd row and every 2nd column → about 1/4 of pixels
        SUBSAMPLE = 4 → about 1/16 of pixels (faster, less accurate)
    '''
    H, W = depth_m.shape[:2]
    x1, y1, x2, y2 = _clamp_box_xyxy(x1, y1, x2, y2, H, W)
    if x2 <= x1 or y2 <= y1:
        return None, 0

    w = x2 - x1
    h = y2 - y1

    if mode == "bottom":
        ch = int(h * frac)
        if ch <= 0:
            return None, 0
        y_start = max(y1, y2 - ch)

        band_w = int(w * 0.5)  # fixed 50% width
        if band_w <= 0:
            return None, 0
        cx = (x1 + x2) // 2
        x_start = max(x1, cx - band_w // 2)
        x_end = min(x2, x_start + band_w)

        patch = depth_m[y_start:y2, x_start:x_end]
    else:
        cw = max(1, int(round(w * frac)))
        ch = max(1, int(round(h * frac)))

        if cw <= 0 or ch <= 0:
            return None, 0

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cx1 = max(0, cx - cw // 2)
        cy1 = max(0, cy - ch // 2)
        cx2 = min(W, cx1 + cw)
        cy2 = min(H, cy1 + ch)

        patch = depth_m[cy1:cy2, cx1:cx2]

    if patch.size == 0:
        return None, 0

    if subsample > 1:
        patch = patch[::subsample, ::subsample]

    valid = patch[np.isfinite(patch) & (patch > 0) & (patch < MAX_DEPTH)].reshape(-1)
    d = fast_percentile_1d(valid, q=q)
    return d, int(valid.size)

def find_depth_path(depth_dir: Path, file_name: str) -> Path | None:
    if not file_name:
        return None

    p = depth_dir / file_name
    if p.exists():
        return p

    if "_sync_image_" in file_name:
        depth_name = file_name.replace("_sync_image_", "_sync_groundtruth_depth_")
        p = depth_dir / depth_name
        if p.exists():
            return p

    stem = Path(file_name).stem
    if "_sync_image_" in stem:
        depth_stem = stem.replace("_sync_image_", "_sync_groundtruth_depth_")
        p = depth_dir / (depth_stem + ".png")
        if p.exists():
            return p

    return None

def _to_float_or_none(x):
    '''safely convert values like confidence into a float'''
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def main():
    det_path = Path(DET_JSON)
    depth_dir = Path(DEPTH_GT_DIR)
    out_path = Path(OUT_JSON)

    if not det_path.exists():
        raise FileNotFoundError(det_path)
    if not depth_dir.exists():
        raise FileNotFoundError(depth_dir)

    with det_path.open("r", encoding="utf-8") as f:
        det = json.load(f)

    out = {
        "task": "object_detection_distance_from_kitti_gt_depth",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "det_json": str(det_path),
            "depth_gt_dir": str(depth_dir),
            "kitti_depth_decode": "meters = uint16/256.0, 0=invalid",
        },
        "params": {
            "q": Q,
            "frac": FRAC,
            "subsample": SUBSAMPLE,
            "max_depth": MAX_DEPTH,
            "conf_thr": CONF_THR,
            "exclude_low_conf": EXCLUDE_LOW_CONF,
            "roi_method": ROI_METHOD,
            "default_mode": DEFAULT_MODE,
            "use_det_mode_field": USE_DET_MODE_FIELD,
            "det_mode_field": DET_MODE_FIELD,
        },
        "images": [],
    }

    missing_depth = 0
    total_objs = 0
    total_objs_with_dist = 0
    total_low_conf = 0

    eval_sec_total = 0.0
    eval_images = 0

    dist_sec_total = 0.0
    dist_objects = 0

    for im in det.get("images", []):
        file_name = im.get("file_name") or Path(im.get("file_path", "")).name
        if not file_name:
            continue

        depth_path = find_depth_path(depth_dir, file_name)
        if depth_path is None:
            missing_depth += 1
            depth_m = None
        else:
            depth_m = read_kitti_depth_png_to_meters(depth_path)

        if depth_m is not None:
            H, W = depth_m.shape
            if im.get("height") and im.get("width"):
                if (H, W) != (int(im["height"]), int(im["width"])):
                    depth_m = cv2.resize(
                        depth_m,
                        (int(im["width"]), int(im["height"])),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    H, W = depth_m.shape
        else:
            H = im.get("height")
            W = im.get("width")

        objs = []
        dets = im.get("detections", []) or []
        total_objs += len(dets)

        t0 = perf_counter() if depth_m is not None else None

        for i, d in enumerate(dets):
            cls_name = str(d.get("class_name", d.get("class_id", "unknown")))
            conf = _to_float_or_none(d.get("confidence", None))
            bbox = d.get("bbox_xyxy", None)

            low_conf = (conf is not None) and (conf < CONF_THR)
            if low_conf:
                total_low_conf += 1
                if EXCLUDE_LOW_CONF:
                    continue
                objs.append({
                    "id": f"det_{i}",
                    "class_id": d.get("class_id", None),
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox_xyxy": bbox,
                    "excluded_low_conf": True,
                    "distance_m": None,
                    "distance_detail": {"mode": None, "q": Q, "valid_px": 0, "roi_method": ROI_METHOD},
                })
                continue

            dist = None
            used_px = 0
            mode = DEFAULT_MODE

            if USE_DET_MODE_FIELD:
                m = d.get(DET_MODE_FIELD, None)
                if m in ("center", "bottom"):
                    mode = m

            if depth_m is not None and bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox

                t_obj = perf_counter()
                if ROI_METHOD == "pixel":
                    dist, used_px = compute_box_distance_pixel(depth_m, x1, y1, x2, y2, mode=mode)
                else:
                    dist, used_px = compute_box_distance_region(
                        depth_m, x1, y1, x2, y2,
                        frac=FRAC, mode=mode, q=Q, subsample=SUBSAMPLE
                    )
                dist_sec_total += (perf_counter() - t_obj)
                dist_objects += 1

            if dist is not None:
                total_objs_with_dist += 1

            objs.append({
                "id": f"det_{i}",
                "class_id": d.get("class_id", None),
                "class_name": cls_name,
                "confidence": conf,
                "bbox_xyxy": bbox,
                "excluded_low_conf": False,
                "distance_m": dist,
                "distance_detail": {"mode": mode, "q": Q, "valid_px": used_px, "roi_method": ROI_METHOD},
            })

        if t0 is not None:
            eval_images += 1
            eval_sec_total += (perf_counter() - t0)

        out["images"].append({
            "file_name": file_name,
            "depth_gt_path": str(depth_path) if depth_path else None,
            "width": W,
            "height": H,
            "objects": objs,
        })

    out["summary"] = {
        "image_count": len(out["images"]),
        "missing_depth_count": missing_depth,
        "total_objects_in_json": total_objs,
        "total_objects_low_conf": total_low_conf,
        "total_objects_with_distance": total_objs_with_dist,
    }

    avg_ms_per_bbox_distance_eval = (dist_sec_total / max(1, dist_objects)) * 1000.0

    out["timing"] = {
        "eval_seconds_total_after_depth_ready": eval_sec_total,
        "eval_images_counted": eval_images,
        "avg_eval_ms_per_image_after_depth_ready": (eval_sec_total / max(1, eval_images)) * 1000.0,

        "dist_seconds_total_bbox_to_roi_to_distance_eval": dist_sec_total,
        "dist_objects_counted": dist_objects,
        "avg_ms_per_bbox_distance_eval": avg_ms_per_bbox_distance_eval,

        "timer": "time.perf_counter",
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Saved:", out_path)
    print("Summary:", out["summary"])
    print("Timing:", out["timing"])


if __name__ == "__main__":
    main()
