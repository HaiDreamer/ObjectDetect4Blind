from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np


# ========= CONFIG =========
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")

# INPUT: this is the JSON produced by your FIRST segmentation-GT script
SEG_GT_JSON = Path(r"C:\Python\ObjectDetect4Blind\seg_eva_detect\seg_distance_json_KITTI_val_GT.json")

# pred_metric_kitti_vkitti_vits_pruned1layer_torch_cpu, pred_metric_kitti_vkitti_vits_onnx_int8_cpu, 
#   pred_metric_kitti_vkitti_vits_onnx_azure (fp16 model), pred_metric_kitti_vkitti_vits_torch (original model)
PRED_DEPTH_DIR = ROOT / "pred_metric_kitti_vkitti_vits_torch"

# OUTPUT
OUT_JSON = ROOT / "seg_depth_with_pred.json"

MAX_DEPTH_M = 80.0

# If True: don't print per-image warnings; only print final summary
QUIET_MISSING = True


# ========= SEG DISTANCE PARAMS =========
# If the input JSON has out["params"], we will read these automatically.
# Otherwise these defaults are used.
DEFAULT_DISTANCE_MODE = "quantile_band"  # "quantile_band" or "single_pixel"
DEFAULT_SINGLE_PIXEL_FALLBACK = "quantile_band"  # "none" or "quantile_band"

DEFAULT_Q = 10.0
DEFAULT_SUBSAMPLE = 1
DEFAULT_BOTTOM_BAND_FRAC = 0.30


# ========= HELPERS (percentile, polygon mask, band logic) =========
def fast_percentile_1d(vals: np.ndarray, q: float) -> Optional[float]:
    """Fast percentile via np.partition."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    k = int(round((q / 100.0) * (vals.size - 1)))
    k = max(0, min(vals.size - 1, k))
    return float(np.partition(vals, k)[k])


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

    pts_i32 = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
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
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            {"used_fallback_full_polygon": False, "band_y_start": None, "band_y_max": None},
        )

    y_start, y_max = band_y_bounds_from_mask(mask, bottom_frac)
    if y_start is None:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            {"used_fallback_full_polygon": False, "band_y_start": None, "band_y_max": None},
        )

    band = np.zeros_like(valid, dtype=bool)
    band[y_start:y_max + 1, :] = True

    cond = valid & band
    used_fallback = False
    if not np.any(cond):
        cond = valid
        used_fallback = True

    ys, xs = np.where(cond)
    if ys.size == 0:
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            {"used_fallback_full_polygon": used_fallback, "band_y_start": int(y_start), "band_y_max": int(y_max)},
        )

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
    Option B (your GT seg logic):
      - compute pQ in band (or fallback full polygon)
      - pick ONE pixel whose depth is closest to that pQ
    Returns (dist_m, pix_xy, used_px, detail)
    """
    ys, xs, vals, band_detail = collect_valid_pixels_in_band(depth_m, mask, bottom_frac, subsample, max_depth)
    if vals.size == 0:
        return None, None, 0, {"band": band_detail, "q": q}

    d_q = fast_percentile_1d(vals, q=q)
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
    Pick exactly ONE fixed pixel:
      - find bottom band rows
      - take bottom-most row with any mask pixels
      - x = median of that row
      - distance = depth at that pixel (or None if invalid)
    """
    y_start, y_max = band_y_bounds_from_mask(mask, bottom_frac)
    if y_start is None:
        return None, None, {"reason": "empty_mask"}

    H, W = mask.shape[:2]
    y_start = max(0, min(H - 1, y_start))
    y_max = max(0, min(H - 1, y_max))

    for y in range(y_max, y_start - 1, -1):
        xs = np.where(mask[y] == 1)[0]
        if xs.size == 0:
            continue
        x = int(np.median(xs))
        d = float(depth_m[y, x])
        pix = {"x": int(x), "y": int(y)}
        if is_valid_depth(d, max_depth):
            return float(d), pix, {"reason": "ok"}
        return None, pix, {"reason": "invalid_depth_at_pixel"}

    return None, None, {"reason": "no_pixels_in_band_rows"}


# ========= PRED DEPTH LOADING (same strategy as your OBJ script) =========
def build_pred_index(pred_dir: Path):
    """
    Scan pred_dir once for fast lookup.
    Keys are filenames only (not full paths).
    """
    png_map = {}
    npy_map = {}
    for p in pred_dir.rglob("*.png"):
        png_map[p.name] = p
    for p in pred_dir.rglob("*.npy"):
        npy_map[p.name] = p
    return png_map, npy_map


def derive_candidates(img_entry: dict) -> list[str]:
    """
    Return candidate GT-style filenames we expect preds to be named after.
    Primary = depth_gt_path filename (best)
    Fallback = derived from file_name pattern
    """
    candidates: list[str] = []

    gt_path = img_entry.get("depth_gt_path")
    if gt_path:
        candidates.append(Path(gt_path).name)

    img_name = img_entry.get("file_name", "")
    parts = img_name.split("_image_")
    if len(parts) == 3:
        prefix, frame_str, cam_str = parts
        candidates.append(f"{prefix}_groundtruth_depth_{frame_str}_image_{cam_str}")

    # dedupe while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_pred_depth_for_entry(
    img_entry: dict, png_index: dict, npy_index: dict
) -> tuple[np.ndarray | None, str | None]:
    """
    Try candidates:
      - <gt_stem>_pred_m.npy (meters float32) preferred
      - <gt_name>.png (uint16/256.0) fallback
    Returns (depth_map_m or None, chosen_path or None)
    """
    tried = []
    for gt_name in derive_candidates(img_entry):
        gt_stem = Path(gt_name).stem
        npy_name = f"{gt_stem}_pred_m.npy"

        tried.append(npy_name)
        p_npy = npy_index.get(npy_name)
        if p_npy is not None and p_npy.exists():
            depth_m = np.load(str(p_npy)).astype(np.float32)
            if depth_m.ndim == 3:
                depth_m = depth_m.squeeze()
            return depth_m, str(p_npy)

        tried.append(gt_name)
        p_png = png_index.get(gt_name)
        if p_png is not None and p_png.exists():
            im = cv2.imread(str(p_png), cv2.IMREAD_UNCHANGED)
            if im is None:
                return None, None
            if im.ndim == 3:
                im = im[..., 0]
            # KITTI-style uint16 depth -> meters
            return (im.astype(np.float32) / 256.0), str(p_png)

    if not QUIET_MISSING:
        print(f"[WARN] no pred depth found for {img_entry.get('file_name')} (tried {len(tried)} names)")
    return None, None


# ========= MAIN =========
def main():
    assert SEG_GT_JSON.exists(), f"Missing SEG GT JSON: {SEG_GT_JSON}"
    assert PRED_DEPTH_DIR.exists(), f"Missing PRED_DEPTH_DIR: {PRED_DEPTH_DIR}"

    with SEG_GT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise RuntimeError("Unexpected JSON structure: expected a dict with key 'images'.")

    # Pull params from GT-json if available (so you match exactly what produced GT distances)
    p = data.get("params", {}) if isinstance(data.get("params", {}), dict) else {}

    DISTANCE_MODE = str(p.get("distance_mode", DEFAULT_DISTANCE_MODE))
    SINGLE_PIXEL_FALLBACK = str(p.get("single_pixel_fallback", DEFAULT_SINGLE_PIXEL_FALLBACK))
    Q = float(p.get("q", DEFAULT_Q))
    SUBSAMPLE = int(p.get("subsample", DEFAULT_SUBSAMPLE))
    BOTTOM_BAND_FRAC = float(p.get("bottom_band_frac", DEFAULT_BOTTOM_BAND_FRAC))
    MAX_DEPTH = float(p.get("max_depth", MAX_DEPTH_M))

    images = data["images"]
    print(f"Loaded {len(images)} images from {SEG_GT_JSON}")
    print(f"Using distance params: mode={DISTANCE_MODE}, fallback={SINGLE_PIXEL_FALLBACK}, "
          f"Q={Q}, subsample={SUBSAMPLE}, bottom_frac={BOTTOM_BAND_FRAC}, max_depth={MAX_DEPTH}")

    png_index, npy_index = build_pred_index(PRED_DEPTH_DIR)
    print(f"Indexed preds: {len(png_index)} png, {len(npy_index)} npy")

    out = dict(data)
    out["pred_depth_dir"] = str(PRED_DEPTH_DIR)
    out["pred_eval_generated_at"] = datetime.now().isoformat(timespec="seconds")
    out["pred_eval_params"] = {
        "distance_mode": DISTANCE_MODE,
        "single_pixel_fallback": SINGLE_PIXEL_FALLBACK,
        "q": Q,
        "subsample": SUBSAMPLE,
        "bottom_band_frac": BOTTOM_BAND_FRAC,
        "max_depth": MAX_DEPTH,
    }

    missing_pred = 0
    used_pred = 0
    total_regions = 0
    regions_with_gt = 0
    regions_with_pred = 0

    new_images = []

    for idx, img_entry in enumerate(images, 1):
        new_img = dict(img_entry)
        regions = img_entry.get("regions", []) or []
        total_regions += len(regions)

        depth_pred, pred_path = load_pred_depth_for_entry(img_entry, png_index, npy_index)
        new_img["pred_depth_path"] = pred_path

        # choose H/W
        H = int(new_img.get("height")) if new_img.get("height") is not None else None
        W = int(new_img.get("width")) if new_img.get("width") is not None else None

        if depth_pred is None:
            missing_pred += 1
        else:
            used_pred += 1
            if H is None or W is None:
                H, W = depth_pred.shape[:2]
                new_img["height"] = int(H)
                new_img["width"] = int(W)
            # resize pred to json-recorded size if needed
            if depth_pred.shape[:2] != (H, W):
                depth_pred = cv2.resize(depth_pred, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_pred = np.clip(depth_pred.astype(np.float32), 0.0, MAX_DEPTH)

        new_regions = []
        for r in regions:
            new_r = dict(r)

            gt_dist = r.get("distance_m", None)
            if gt_dist is not None:
                regions_with_gt += 1

            poly = r.get("segmentation_xy") or []
            excluded_low_conf = bool(r.get("excluded_low_conf", False))

            pred_dist = None
            pred_pix = None
            pred_detail: Dict[str, Any] = {}

            if excluded_low_conf:
                pred_detail = {"reason": "excluded_low_conf"}
            elif depth_pred is None or gt_dist is None or not poly or H is None or W is None:
                pred_detail = {"reason": "missing_pred_or_gt_or_poly"}
            else:
                mask = polygon_to_mask(poly, H, W)

                if DISTANCE_MODE == "single_pixel":
                    d1, p1, d1_detail = single_pixel_pick(depth_pred, mask, BOTTOM_BAND_FRAC, MAX_DEPTH)
                    pred_dist, pred_pix = d1, p1
                    pred_detail = {"mode": "single_pixel", **d1_detail}

                    if pred_dist is None and SINGLE_PIXEL_FALLBACK == "quantile_band":
                        d2, p2, used_px, q_detail = quantile_band_pick_1pixel(
                            depth_pred, mask, BOTTOM_BAND_FRAC, Q, SUBSAMPLE, MAX_DEPTH
                        )
                        pred_dist, pred_pix = d2, p2
                        pred_detail = {
                            "mode": "single_pixel",
                            "fallback": "quantile_band",
                            "fallback_used_px": used_px,
                            "fallback_detail": q_detail,
                            **pred_detail,
                        }
                else:
                    d2, p2, used_px, q_detail = quantile_band_pick_1pixel(
                        depth_pred, mask, BOTTOM_BAND_FRAC, Q, SUBSAMPLE, MAX_DEPTH
                    )
                    pred_dist, pred_pix = d2, p2
                    pred_detail = {"mode": "quantile_band", "used_px": used_px, **q_detail}

            new_r["pred_distance_m"] = float(pred_dist) if pred_dist is not None else None
            new_r["pred_pixel_xy"] = pred_pix
            new_r["pred_distance_detail"] = {"bottom_band_frac": BOTTOM_BAND_FRAC, **pred_detail}

            if (gt_dist is not None) and (pred_dist is not None):
                regions_with_pred += 1
                new_r["wrongly_distance_m"] = float(float(gt_dist) - float(pred_dist))  # gt - pred
            else:
                new_r["wrongly_distance_m"] = None

            new_regions.append(new_r)

        new_img["regions"] = new_regions
        new_images.append(new_img)

        if idx % 50 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)}")

    out["images"] = new_images
    out["pred_eval_summary"] = {
        "image_count": len(new_images),
        "pred_available_images": used_pred,
        "pred_missing_images": missing_pred,
        "total_regions": total_regions,
        "regions_with_gt_distance": regions_with_gt,
        "regions_with_pred_distance": regions_with_pred,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUT_JSON}")
    print("Summary:", out["pred_eval_summary"])


if __name__ == "__main__":
    main()
