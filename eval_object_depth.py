from pathlib import Path
import json
import cv2
import numpy as np

"""
Compare per-object distance between:
- GT object distance (from KITTI GT depth)      -> gt_distance_m
- Predicted object distance (from metric model) -> pred_distance_m

INPUT:
  - obj_depth_gt.json   (created by kitti_object_depth_gt.py code)
  - predicted depth maps in:
        pred_metric_kitti_vkitti_vits/
        * uint16 KITTI PNGs:   <gt_name>.png
        * (optional) float32:  <gt_name>_pred_m.npy

OUTPUT:
  - obj_depth_err.json  with per-object errors
  - basic stats printed per eval_category
"""

# ----------------------------
# CONFIG
# ----------------------------
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")

OBJ_GT_JSON = ROOT / "obj_depth_gt.json"

# where make_kitti_preds_metric.py wrote its outputs
PRED_DIR = ROOT / "pred_metric_kitti_vkitti_vits"

OUT_ERR_JSON = ROOT / "obj_depth_err.json"

MAX_DEPTH_M = 80.0


# ----------------------------
# Helpers
# ----------------------------
def load_pred_depth_for_image(image_name: str) -> np.ndarray | None:
    """
    Given an 'image' filename from obj_depth_gt.json, derive the corresponding
    GT-style name and then locate the prediction files.

    Image name example:
      2011_09_26_drive_0002_sync_image_0000000005_image_02.png

    GT name:
      2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png

    Prediction files (produced by make_kitti_preds_metric.py):
      PRED_DIR / GT_name                  (uint16 PNG)
      PRED_DIR / (GT_stem + "_pred_m.npy") (optional float32 meters)
    """
    # Recover GT-style name from image name using the same split rule as before
    parts = image_name.split("_image_")
    if len(parts) != 3:
        print(f"[WARN] unexpected image filename pattern: {image_name}")
        return None

    prefix, frame_str, cam_str = parts
    gt_name = f"{prefix}_groundtruth_depth_{frame_str}_image_{cam_str}"   # e.g. ..._groundtruth_depth_0000000005_image_02.png

    png_path = PRED_DIR / gt_name
    npy_path = PRED_DIR / (Path(gt_name).stem + "_pred_m.npy")

    depth_m = None

    if npy_path.exists():
        # preferred: exact float32 prediction (meters)
        depth_m = np.load(str(npy_path)).astype(np.float32)
        # if someone saved with extra channel dim, squeeze
        if depth_m.ndim == 3:
            depth_m = depth_m.squeeze()
    elif png_path.exists():
        # fallback: KITTI-style uint16 PNG: depth[m] = value / 256.0
        im = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] failed to read pred PNG: {png_path}")
            return None
        if im.ndim == 3:
            im = im[..., 0]
        depth_m = im.astype(np.float32) / 256.0
    else:
        print(f"[WARN] no prediction file found for image {image_name} (expected {png_path} or {npy_path})")
        return None

    return depth_m


def box_distance(depth_map_m: np.ndarray,
                 x1: int, y1: int, x2: int, y2: int,
                 mode: str, frac: float = 0.3) -> float | None:
    """
    Same logic as box_distance_from_gt, but generic.
    We want identical behavior for GT and prediction.
    """
    H, W = depth_map_m.shape[:2]

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    if mode == "bottom":
        band_h = int(h * frac)
        if band_h <= 0:
            return None
        y_start = max(y1, y2 - band_h)
        patch = depth_map_m[y_start:y2, x1:x2]
    else:  # "center"
        cw = int(w * frac)
        ch = int(h * frac)
        if cw <= 0 or ch <= 0:
            return None
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cx1 = max(0, cx - cw // 2)
        cy1 = max(0, cy - ch // 2)
        cx2 = min(W, cx1 + cw)
        cy2 = min(H, cy1 + ch)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        patch = depth_map_m[cy1:cy2, cx1:cx2]

    if patch.size == 0:
        return None

    valid = patch[(patch > 0.0) & np.isfinite(patch) & (patch < MAX_DEPTH_M)]
    if valid.size == 0:
        return None

    return float(np.median(valid))


def mode_for_category(eval_category: str) -> str:
    """
    Use same policy as GT script:
      - bottom for vehicles / cyclists
      - center for person
    """
    if eval_category in ("Car", "Truck", "LargeVeh", "LargeVehicle", "Cyclist", "Cyclist/Bicycle"):
        return "bottom"
    return "center"


# ----------------------------
# MAIN
# ----------------------------
def main():
    assert OBJ_GT_JSON.exists(), f"Missing GT JSON: {OBJ_GT_JSON}"
    with open(OBJ_GT_JSON, "r", encoding="utf-8") as f:
        objs = json.load(f)

    print(f"Loaded {len(objs)} object GT entries from {OBJ_GT_JSON}")

    # group by image so we don't reload depth for each object
    objs_by_image = {}
    for o in objs:
        img = o["image"]
        objs_by_image.setdefault(img, []).append(o)

    out_entries = []

    # stats containers: {eval_category: [errors...]}
    stats = {}

    for idx, (img_name, obj_list) in enumerate(objs_by_image.items(), 1):
        print(f"[{idx}/{len(objs_by_image)}] Image: {img_name} with {len(obj_list)} objects")

        pred_depth = load_pred_depth_for_image(img_name)
        if pred_depth is None:
            print(f"  [WARN] skipping all objects for {img_name} (no prediction)")
            continue

        H, W = pred_depth.shape[:2]

        for o in obj_list:
            x1, y1, x2, y2 = o["bbox_xyxy"]
            eval_cat = o.get("eval_category", "Unknown")
            gt_dist = o.get("gt_distance_m", None)

            if gt_dist is None:
                # nothing to compare
                continue

            mode = mode_for_category(eval_cat)

            pred_dist = box_distance(pred_depth, x1, y1, x2, y2, mode=mode, frac=0.3)

            if pred_dist is None:
                # could not get a valid predicted depth in this box
                out = {
                    **o,
                    "pred_distance_m": None,
                    "abs_error_m": None,
                    "rel_error": None
                }
            else:
                err = pred_dist - gt_dist
                abs_err = abs(err)
                rel_err = abs_err / gt_dist if gt_dist > 1e-6 else None

                out = {
                    **o,
                    "pred_distance_m": float(pred_dist),
                    "abs_error_m": float(abs_err),
                    "rel_error": float(rel_err) if rel_err is not None else None
                }

                # accumulate stats by category
                if rel_err is not None:
                    stats.setdefault(eval_cat, []).append(abs_err)

            out_entries.append(out)

    # save extended JSON
    with open(OUT_ERR_JSON, "w", encoding="utf-8") as f:
        json.dump(out_entries, f, indent=2)
    print(f"\nSaved object distance errors to: {OUT_ERR_JSON}")

    # print simple stats
    print("\nPer-class |abs distance error| (meters):")
    for cat, errs in stats.items():
        if not errs:
            continue
        arr = np.array(errs, dtype=np.float32)
        mean_err = float(arr.mean())
        med_err = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        print(f"  {cat:15s}  N={len(errs):5d}  mean={mean_err:.2f}  median={med_err:.2f}  p90={p90:.2f}")


if __name__ == "__main__":
    main()
