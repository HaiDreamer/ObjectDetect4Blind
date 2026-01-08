from pathlib import Path
import json
import numpy as np
from collections import defaultdict

"""
INPUT (seg_depth_with_pred.json):
  {
    ...,
    "images": [
      {
        "file_name": "...png",
        "regions": [
          {
            "class_name": ...,
            "distance_m": ... ,        # GT
            "pred_distance_m": ...,    # Pred
            "excluded_low_conf": ... , # optional
          }, ...
        ]
      }, ...
    ]
  }

OUTPUT (printed):
  Per-class error stats, similar to your object detection report.

RESULT 
original model
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
crosswalk                          2      1.594      1.594        1.594        1.594      2.984      1.727      12.58      12.58
sidewalk                          68      0.575      0.574        0.572        0.574      0.394      0.628       8.85       8.89
stairs                            31      2.367      1.117        0.925        0.810     13.332      3.651      12.94       9.98
tree line                         42      0.677      0.574       -0.206        0.011      0.867      0.931      14.77       6.75
--------------------------------------------------------------------------------------------------------------------------------
ALL                              143      1.008      0.626        0.434        0.536      3.374      1.837      11.53       8.83

pruned1layer
========== PER-CLASS ERROR STATS (SEGMENTATION) ==========
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
sidewalk                          27      0.331      0.304        0.246        0.265      0.147      0.383       5.18       5.25
stairs                             7      1.533      0.730       -1.533       -0.730      4.415      2.101      11.58      11.01
--------------------------------------------------------------------------------------------------------------------------------
ALL                               34      0.579      0.372       -0.120        0.190      1.025      1.013       6.50       5.70

onnx int8 model
========== PER-CLASS ERROR STATS (SEGMENTATION) ==========
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
crosswalk                          2      0.548      0.548        0.235        0.235      0.356      0.597       4.33       4.33
sidewalk                          68      0.233      0.202       -0.045       -0.054      0.083      0.287       3.73       3.07
stairs                            31      2.528      0.942       -0.295       -0.355     15.065      3.881      13.88       6.74
tree line                         42      0.800      0.655       -0.689       -0.655      0.958      0.979      16.61       9.70
--------------------------------------------------------------------------------------------------------------------------------
ALL                              143      0.902      0.395       -0.284       -0.207      3.592      1.895       9.72       5.42

onnx fp16 model
========== PER-CLASS ERROR STATS (SEGMENTATION) ==========
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
crosswalk                          2      0.401      0.401        0.155        0.155      0.185      0.430       3.16       3.16
sidewalk                          68      0.225      0.216        0.042        0.017      0.076      0.276       3.57       3.53
stairs                            31      2.657      1.164       -0.332       -0.252     17.055      4.130      14.57       7.54
tree line                         42      0.696      0.466       -0.563       -0.455      0.774      0.880      14.87       7.21
--------------------------------------------------------------------------------------------------------------------------------
ALL                              143      0.893      0.365       -0.215       -0.090      3.963      1.991       9.27       4.83
"""

# ====== CONFIG ======
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
# seg_depth_with_pred_origin (original model), seg_depth_with_pred_pruned1layer.json, seg_depth_with_pred_onnx_int8_cpu.json
#   seg_depth_with_pred_onnx_fp16_cpu.json
JSON_PATH = ROOT / "seg_depth_with_pred_onnx_fp16_cpu.json"

# If True, ignore regions marked as excluded_low_conf
SKIP_EXCLUDED_LOW_CONF = True

# Relative error denominator threshold
GT_EPS = 1e-6


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def pick_class_name(r: dict) -> str:
    # prefer class_name, fallback to eval_category (if you ever add it), else class_id, else Unknown
    cat = r.get("class_name") or r.get("eval_category") or r.get("class_id") or "Unknown"
    return str(cat)


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise RuntimeError("Expected dict JSON with top-level key 'images'.")

    images = data["images"]
    print(f"Total images in JSON: {len(images)}")

    # per class -> lists
    per_class_abs     = defaultdict(list)
    per_class_signed  = defaultdict(list)
    per_class_rel     = defaultdict(list)
    per_class_sq_err  = defaultdict(list)

    n_total_regions = 0
    n_skipped_missing = 0
    n_skipped_excluded = 0

    for img in images:
        regions = img.get("regions", []) or []
        for r in regions:
            n_total_regions += 1

            if SKIP_EXCLUDED_LOW_CONF and bool(r.get("excluded_low_conf", False)):
                n_skipped_excluded += 1
                continue

            gt = safe_float(r.get("distance_m"))
            pred = safe_float(r.get("pred_distance_m"))
            cat = pick_class_name(r)

            if gt is None or pred is None:
                n_skipped_missing += 1
                continue

            err = gt - pred
            abs_err = abs(err)
            sq_err = err * err

            per_class_abs[cat].append(abs_err)
            per_class_signed[cat].append(err)
            per_class_sq_err[cat].append(sq_err)

            if gt > GT_EPS:
                per_class_rel[cat].append(100.0 * abs_err / gt)

    used = sum(len(v) for v in per_class_abs.values())
    print(f"Total regions in JSON:          {n_total_regions}")
    print(f"Valid regions used:            {used}")
    print(f"Skipped (missing gt/pred):     {n_skipped_missing}")
    print(f"Skipped (excluded_low_conf):   {n_skipped_excluded}")

    print("\n========== PER-CLASS ERROR STATS (SEGMENTATION) ==========")
    header = (
        f"{'Class':28s} "
        f"{'N':>7s} "
        f"{'mean|e|':>10s} "
        f"{'med|e|':>10s} "
        f"{'mean(gt-p)':>12s} "
        f"{'med(gt-p)':>12s} "
        f"{'MSE':>10s} "
        f"{'RMSE':>10s} "
        f"{'meanRel%':>10s} "
        f"{'medRel%':>10s}"
    )
    print(header)
    print("-" * len(header))

    # (optional) also compute global "ALL" row
    all_abs = []
    all_signed = []
    all_sq = []
    all_rel = []

    for cat in sorted(per_class_abs.keys()):
        abs_list    = np.asarray(per_class_abs[cat], dtype=np.float32)
        signed_list = np.asarray(per_class_signed[cat], dtype=np.float32)
        sq_list     = np.asarray(per_class_sq_err[cat], dtype=np.float32)
        rel_list    = np.asarray(per_class_rel.get(cat, []), dtype=np.float32)

        N = abs_list.size
        mean_abs = float(abs_list.mean())
        med_abs  = float(np.median(abs_list))
        mean_signed = float(signed_list.mean())
        med_signed  = float(np.median(signed_list))

        mean_sq = float(sq_list.mean()) if sq_list.size else float("nan")
        rmse = float(np.sqrt(mean_sq)) if np.isfinite(mean_sq) else float("nan")

        mean_rel = float(rel_list.mean()) if rel_list.size else float("nan")
        med_rel  = float(np.median(rel_list)) if rel_list.size else float("nan")

        print(
            f"{cat:28s} "
            f"{N:7d} "
            f"{mean_abs:10.3f} "
            f"{med_abs:10.3f} "
            f"{mean_signed:12.3f} "
            f"{med_signed:12.3f} "
            f"{mean_sq:10.3f} "
            f"{rmse:10.3f} "
            f"{mean_rel:10.2f} "
            f"{med_rel:10.2f}"
        )

        all_abs.append(abs_list)
        all_signed.append(signed_list)
        all_sq.append(sq_list)
        if rel_list.size:
            all_rel.append(rel_list)

    if all_abs:
        abs_all = np.concatenate(all_abs)
        signed_all = np.concatenate(all_signed)
        sq_all = np.concatenate(all_sq)
        rel_all = np.concatenate(all_rel) if all_rel else np.asarray([], dtype=np.float32)

        N = abs_all.size
        mean_abs = float(abs_all.mean())
        med_abs = float(np.median(abs_all))
        mean_signed = float(signed_all.mean())
        med_signed = float(np.median(signed_all))
        mse = float(sq_all.mean())
        rmse = float(np.sqrt(mse))
        mean_rel = float(rel_all.mean()) if rel_all.size else float("nan")
        med_rel = float(np.median(rel_all)) if rel_all.size else float("nan")

        print("-" * len(header))
        print(
            f"{'ALL':28s} "
            f"{N:7d} "
            f"{mean_abs:10.3f} "
            f"{med_abs:10.3f} "
            f"{mean_signed:12.3f} "
            f"{med_signed:12.3f} "
            f"{mse:10.3f} "
            f"{rmse:10.3f} "
            f"{mean_rel:10.2f} "
            f"{med_rel:10.2f}"
        )


if __name__ == "__main__":
    main()
