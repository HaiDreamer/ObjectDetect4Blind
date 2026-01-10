import json
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image
import cv2

# =========================
# PATHS
# =========================
SEG_JSON = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\segment_json_KITTI_val.json"
DEPTH_GT_DIR = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth"
OUT_JSON = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\seg_distance_json_KITTI_val_GT.json"


# =========================
# MODE SWITCHES (edit these)
# =========================
# Distance method:
#   "quantile_band" -> Option B: pick 1 pixel by quantile in bottom band (stable)
#   "single_pixel"  -> pick exactly 1 fixed pixel (bottom band, median-x)
DISTANCE_MODE = "quantile_band"   # "quantile_band" or "single_pixel"

# If single_pixel lands on invalid depth, optionally fallback:
SINGLE_PIXEL_FALLBACK = "quantile_band"  # "none" or "quantile_band"


# =========================
# PARAMS (easy to adjust)
# =========================
MAX_DEPTH = 80.0
Q = 10.0                     # percentile (p5 nearer / more conservative than p10, but may lower accuracy)
SUBSAMPLE = 1                # >1 for speed (less accurate)
BOTTOM_BAND_FRAC = 0.1      # bottom X% of polygon height (0.10, 0.20, 0.30, 1.00...)

# Confidence filtering
CONF_THR = 0.25              # None to disable
EXCLUDE_LOW_CONF = False     # False: keep region but distance=None; True: drop region


# =========================
# Helpers
# =========================
def read_kitti_depth_png_to_meters(depth_png_path: Path) -> np.ndarray:
    """
    KITTI GT depth:
      - uint16 PNG
      - 0 invalid
      - meters = uint16 / 256.0
    """
    I = np.array(Image.open(depth_png_path), dtype=np.uint16)
    depth_m = I.astype(np.float32) / 256.0
    depth_m[I == 0] = np.nan
    return depth_m


def fast_percentile_1d(vals: np.ndarray, q: float) -> Optional[float]:
    """Fast percentile via np.partition (partial selection)."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    k = int(round((q / 100.0) * (vals.size - 1)))
    k = max(0, min(vals.size - 1, k))
    return float(np.partition(vals, k)[k])


def find_depth_path(depth_dir: Path, file_name: str) -> Optional[Path]:
    """
    RGB: ..._sync_image_0000000005_image_02.png
    GT : ..._sync_groundtruth_depth_0000000005_image_02.png
    """
    if not file_name:
        return None

    # 1) exact match (in case file_name already is GT)
    p = depth_dir / file_name
    if p.exists():
        return p

    # 2) KITTI cropped mapping
    if "_sync_image_" in file_name:
        depth_name = file_name.replace("_sync_image_", "_sync_groundtruth_depth_")
        p = depth_dir / depth_name
        if p.exists():
            return p

    # 3) jpg/jpeg -> png (rare but safe)
    stem = Path(file_name).stem
    if "_sync_image_" in stem:
        depth_stem = stem.replace("_sync_image_", "_sync_groundtruth_depth_")
        p = depth_dir / (depth_stem + ".png")
        if p.exists():
            return p

    return None


def _to_float_or_none(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def polygon_to_mask(poly_xy, H: int, W: int) -> np.ndarray:
    """
    Fill polygon area into a binary mask.
    Accepts:
      - [[x,y], [x,y], ...]
      - [x1,y1,x2,y2,...] (flat)
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    if not poly_xy:
        return mask

    # flat list -> Nx2
    if isinstance(poly_xy, (list, tuple)) and poly_xy and not isinstance(poly_xy[0], (list, tuple)):
        if len(poly_xy) < 6 or (len(poly_xy) % 2) != 0:
            return mask
        pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 2)
    else:
        if len(poly_xy) < 3:
            return mask
        pts = np.array(poly_xy, dtype=np.float32).reshape(-1, 2)

    pts_i32 = np.round(pts).astype(np.int32).reshape(-1, 1, 2)  # OpenCV polygon format :contentReference[oaicite:3]{index=3}
    cv2.fillPoly(mask, [pts_i32], 1)
    return mask


def band_y_bounds_from_mask(mask: np.ndarray, bottom_frac: float) -> Tuple[Optional[int], Optional[int]]:
    """
    bottom_frac=0.30 -> bottom 30% of polygon's own y-extent.
    Returns (y_start, y_max) or (None, None) if empty.
    """
    ys = np.nonzero(mask)[0]
    if ys.size == 0:
        return None, None

    y_min = int(ys.min())
    y_max = int(ys.max())
    poly_h = y_max - y_min + 1

    bottom_frac = float(np.clip(bottom_frac, 0.0, 1.0))
    band_h = max(1, int(round(poly_h * bottom_frac)))
    y_start = y_max - band_h + 1
    return y_start, y_max


def is_valid_depth(d: float, max_depth: float) -> bool:
    return np.isfinite(d) and (0.0 < d < max_depth)


def collect_valid_pixels_in_band(
    depth_m: np.ndarray,
    mask: np.ndarray,
    bottom_frac: float,
    subsample: int,
    max_depth: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Collect (ys, xs, vals) for valid depth pixels inside polygon AND inside bottom band.
    Fallback: if band has no valid pixels, use full polygon.
    """
    valid = (mask == 1) & np.isfinite(depth_m) & (depth_m > 0) & (depth_m < max_depth)
    if not np.any(valid):
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32), {
            "used_fallback_full_polygon": False,
            "band_y_start": None,
            "band_y_max": None,
        }

    y_start, y_max = band_y_bounds_from_mask(mask, bottom_frac)
    if y_start is None:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32), {
            "used_fallback_full_polygon": False,
            "band_y_start": None,
            "band_y_max": None,
        }

    band = np.zeros_like(valid, dtype=bool)
    band[y_start:y_max + 1, :] = True

    cond = valid & band
    used_fallback = False
    if not np.any(cond):
        cond = valid
        used_fallback = True

    ys, xs = np.where(cond)
    if ys.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32), {
            "used_fallback_full_polygon": used_fallback,
            "band_y_start": int(y_start),
            "band_y_max": int(y_max),
        }

    if subsample > 1:
        take = np.arange(0, ys.size, subsample, dtype=np.int64)
        ys, xs = ys[take], xs[take]

    vals = depth_m[ys, xs].astype(np.float32)
    keep = np.isfinite(vals)
    ys, xs, vals = ys[keep], xs[keep], vals[keep]

    return ys, xs, vals, {
        "used_fallback_full_polygon": used_fallback,
        "band_y_start": int(y_start),
        "band_y_max": int(y_max),
    }


def quantile_band_pick_1pixel(
    depth_m: np.ndarray,
    mask: np.ndarray,
    bottom_frac: float,
    q: float,
    subsample: int,
    max_depth: float
) -> Tuple[Optional[float], Optional[Dict[str, int]], int, Dict[str, Any]]:
    """
    Option B:
      - compute pQ in band
      - pick ONE pixel whose depth is closest to that pQ
    Returns (dist_m, pix_xy, used_px, detail)
    """
    ys, xs, vals, band_detail = collect_valid_pixels_in_band(depth_m, mask, bottom_frac, subsample, max_depth)
    if vals.size == 0:
        return None, None, 0, {"band": band_detail, "q": q}

    d_q = fast_percentile_1d(vals, q=q)  # np.partition :contentReference[oaicite:4]{index=4}
    if d_q is None:
        return None, None, int(vals.size), {"band": band_detail, "q": q}

    idx = int(np.argmin(np.abs(vals - d_q)))
    pix = {"x": int(xs[idx]), "y": int(ys[idx])}
    return float(d_q), pix, int(vals.size), {"band": band_detail, "q": q}


def single_pixel_pick(
    depth_m: np.ndarray,
    mask: np.ndarray,
    bottom_frac: float,
    max_depth: float
) -> Tuple[Optional[float], Optional[Dict[str, int]], Dict[str, Any]]:
    """
    Pick exactly ONE pixel, but ensure it's valid:
      - scan from bottom of band upwards
      - for each row: take x0 = median(mask pixels)
      - if depth at (y,x0) invalid -> pick nearest valid x on that row (still inside mask)
    """
    y_start, y_max = band_y_bounds_from_mask(mask, bottom_frac)
    if y_start is None:
        return None, None, {"reason": "empty_mask"}

    H, W = mask.shape[:2]
    y_start = max(0, min(H - 1, y_start))
    y_max   = max(0, min(H - 1, y_max))

    for y in range(y_max, y_start - 1, -1):
        xs_mask = np.where(mask[y] == 1)[0]
        if xs_mask.size == 0:
            continue

        x0 = int(np.median(xs_mask))

        # valid depths on this row *within the mask*
        row_depths = depth_m[y, xs_mask]
        ok = np.isfinite(row_depths) & (row_depths > 0) & (row_depths < max_depth)
        xs_valid = xs_mask[ok]

        if xs_valid.size == 0:
            continue  # try the next row up

        # choose valid x closest to the median-x target
        x = int(xs_valid[np.argmin(np.abs(xs_valid - x0))])
        d = float(depth_m[y, x])
        return d, {"x": x, "y": y}, {
            "reason": "ok",
            "x0_median": x0,
            "picked_nearest_valid_on_row": (x != x0),
            "valid_count_on_row": int(xs_valid.size),
        }

    return None, None, {"reason": "no_valid_depth_in_band_rows"}



# =========================
# Main
# =========================
def main():
    seg_path = Path(SEG_JSON)
    depth_dir = Path(DEPTH_GT_DIR)
    out_path = Path(OUT_JSON)

    if not seg_path.exists():
        raise FileNotFoundError(seg_path)
    if not depth_dir.exists():
        raise FileNotFoundError(depth_dir)

    with seg_path.open("r", encoding="utf-8") as f:
        seg = json.load(f)

    out = {
        "task": "segmentation_distance_from_kitti_gt_depth",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "seg_json": str(seg_path),
            "depth_gt_dir": str(depth_dir),
            "kitti_depth_decode": "meters = uint16/256.0, 0=invalid",
        },
        "params": {
            "distance_mode": DISTANCE_MODE,
            "single_pixel_fallback": SINGLE_PIXEL_FALLBACK,
            "q": Q,
            "subsample": SUBSAMPLE,
            "max_depth": MAX_DEPTH,
            "bottom_band_frac": BOTTOM_BAND_FRAC,
            "conf_thr": CONF_THR,
            "exclude_low_conf": EXCLUDE_LOW_CONF,
        },
        "images": [],
    }

    missing_depth = 0
    total_regions = 0
    regions_with_distance = 0
    low_conf_count = 0

    eval_sec_total = 0.0
    eval_images = 0
    eval_regions_attempted = 0

    for im in seg.get("images", []):
        file_name = im.get("file_name") or Path(im.get("file_path", "")).name
        if not file_name:
            continue

        depth_path = find_depth_path(depth_dir, file_name)
        if depth_path is None:
            missing_depth += 1
            depth_m = None
        else:
            depth_m = read_kitti_depth_png_to_meters(depth_path)

        # resize depth if needed
        if depth_m is not None and im.get("height") and im.get("width"):
            H0, W0 = depth_m.shape
            Ht, Wt = int(im["height"]), int(im["width"])
            if (H0, W0) != (Ht, Wt):
                depth_m = cv2.resize(depth_m, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        if depth_m is not None:
            H, W = depth_m.shape
        else:
            H = im.get("height")
            W = im.get("width")

        instances = im.get("instances", []) or []
        total_regions += len(instances)

        t0 = perf_counter() if depth_m is not None else None

        regions_out = []
        for i, inst in enumerate(instances):
            cls_name = str(inst.get("class_name", inst.get("class_id", "unknown")))
            conf = _to_float_or_none(inst.get("confidence", None))
            poly = inst.get("segmentation_xy") or []

            # confidence gating
            if CONF_THR is not None and conf is not None and conf < CONF_THR:
                low_conf_count += 1
                if EXCLUDE_LOW_CONF:
                    continue
                regions_out.append({
                    "id": f"seg_{i}",
                    "class_id": inst.get("class_id", None),
                    "class_name": cls_name,
                    "confidence": conf,
                    "excluded_low_conf": True,
                    "segmentation_xy": poly,
                    "distance_m": None,
                    "pixel_xy": None,
                    "distance_detail": {"reason": "low_conf"},
                })
                continue

            dist_m = None
            pix_xy = None
            detail = {}

            if depth_m is not None and poly:
                mask = polygon_to_mask(poly, H, W)

                if DISTANCE_MODE == "single_pixel":
                    d1, p1, d1_detail = single_pixel_pick(depth_m, mask, BOTTOM_BAND_FRAC, MAX_DEPTH)
                    dist_m, pix_xy = d1, p1
                    detail = {"mode": "single_pixel", **d1_detail}

                    if dist_m is None and SINGLE_PIXEL_FALLBACK == "quantile_band":
                        d2, p2, used_px, q_detail = quantile_band_pick_1pixel(
                            depth_m, mask, BOTTOM_BAND_FRAC, Q, SUBSAMPLE, MAX_DEPTH
                        )
                        dist_m, pix_xy = d2, p2
                        detail = {
                            "mode": "single_pixel",
                            "fallback": "quantile_band",
                            "fallback_used_px": used_px,
                            "fallback_detail": q_detail,
                            **detail,
                        }
                        eval_regions_attempted += 1

                else:
                    # default: quantile_band (Option B)
                    d2, p2, used_px, q_detail = quantile_band_pick_1pixel(
                        depth_m, mask, BOTTOM_BAND_FRAC, Q, SUBSAMPLE, MAX_DEPTH
                    )
                    dist_m, pix_xy = d2, p2
                    detail = {"mode": "quantile_band", "used_px": used_px, **q_detail}
                    eval_regions_attempted += 1

            if dist_m is not None:
                regions_with_distance += 1

            regions_out.append({
                "id": f"seg_{i}",
                "class_id": inst.get("class_id", None),
                "class_name": cls_name,
                "confidence": conf,
                "excluded_low_conf": False,
                "segmentation_xy": poly,
                "distance_m": dist_m,
                "pixel_xy": pix_xy,
                "distance_detail": {
                    "bottom_band_frac": BOTTOM_BAND_FRAC,
                    **detail,
                },
            })

        if t0 is not None:
            eval_images += 1
            eval_sec_total += (perf_counter() - t0)

        out["images"].append({
            "file_name": file_name,
            "depth_gt_path": str(depth_path) if depth_path else None,
            "width": W,
            "height": H,
            "regions": regions_out,
        })

    out["summary"] = {
        "image_count": len(out["images"]),
        "missing_depth_count": missing_depth,
        "total_regions_in_json": total_regions,
        "low_conf_count": low_conf_count,
        "regions_with_distance": regions_with_distance,
    }

    out["timing"] = {
        "eval_seconds_total_after_depth_ready": eval_sec_total,
        "eval_images_counted": eval_images,
        "eval_regions_attempted": eval_regions_attempted,
        "avg_eval_ms_per_image_after_depth_ready": (eval_sec_total / max(1, eval_images)) * 1000.0,
        "avg_eval_ms_per_region_attempted": (eval_sec_total / max(1, eval_regions_attempted)) * 1000.0,
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
