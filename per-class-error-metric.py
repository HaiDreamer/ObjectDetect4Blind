from pathlib import Path
import json
import numpy as np
from collections import defaultdict

"""
INPUT
    - Per-class distance error analysis for obj_depth_with_pred.json
    - Each object in JSON is expected to have at least:
        - eval_category
        - gt_distance_m
        - ground_distance_predict

For each eval_category, we compute:
    - N (number of objects)
    - mean / median absolute error (|gt - pred|)      [meters]
    - mean / median signed error (gt - pred)          [meters]
    - mean / median relative error (%)                [ 100 * |gt - pred| / gt ]
    - mean squared error (MSE)                        [ (gt - pred)^2, meters^2 ]
    - root mean squared error (RMSE)                  [ sqrt(MSE), meters ]
"""

# ====== CONFIG ======
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
JSON_PATH = ROOT / "obj_depth_with_pred.json"


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total objects in JSON: {len(data)}")

    # Containers: per class -> list of values
    per_class_abs     = defaultdict(list)   # |gt - pred|
    per_class_signed  = defaultdict(list)   # (gt - pred)
    per_class_rel     = defaultdict(list)   # 100 * |gt - pred| / gt
    per_class_sq_err  = defaultdict(list)   # (gt - pred)^2

    n_skipped_missing = 0

    for o in data:
        gt = o.get("gt_distance_m")
        pred = o.get("ground_distance_predict")
        cat = o.get("eval_category", "Unknown")

        # Skip objects with invalid distances
        if gt is None or pred is None:
            n_skipped_missing += 1
            continue

        err = gt - pred
        abs_err = abs(err)
        sq_err = err * err

        # Avoid divide-by-zero for relative error
        if gt > 1e-6:
            rel_err = 100.0 * abs_err / gt  # percent
        else:
            rel_err = None

        per_class_abs[cat].append(abs_err)
        per_class_signed[cat].append(err)
        per_class_sq_err[cat].append(sq_err)
        if rel_err is not None:
            per_class_rel[cat].append(rel_err)

    print(f"\nValid objects used: {sum(len(v) for v in per_class_abs.values())}")
    print(f"Skipped due to missing gt/pred: {n_skipped_missing}")

    # Pretty print per-class stats
    print("\n========== PER-CLASS ERROR STATS ==========")
    header = (
        f"{'Class':20s} "
        f"{'N':>6s} "
        f"{'mean|e|':>10s} "
        f"{'med|e|':>10s} "
        f"{'mean(gt-p)':>10s} "
        f"{'med(gt-p)':>10s} "
        f"{'MSE':>10s} "
        f"{'RMSE':>10s} "
        f"{'meanRel%':>10s} "
        f"{'medRel%':>10s}"
    )
    print(header)
    print("-" * len(header))

    for cat in sorted(per_class_abs.keys()):
        abs_list    = np.array(per_class_abs[cat],        dtype=np.float32)
        signed_list = np.array(per_class_signed[cat],     dtype=np.float32)
        rel_list    = np.array(per_class_rel.get(cat, []), dtype=np.float32)
        sq_list     = np.array(per_class_sq_err[cat],     dtype=np.float32)

        N = len(abs_list)
        mean_abs = float(abs_list.mean())
        med_abs  = float(np.median(abs_list))
        mean_signed = float(signed_list.mean())
        med_signed  = float(np.median(signed_list))

        if sq_list.size > 0:
            mean_sq = float(sq_list.mean())
            # Your line here:
            rmse = float(np.sqrt(mean_sq)) if np.isfinite(mean_sq) else float("nan")
        else:
            mean_sq = float("nan")
            rmse    = float("nan")

        if rel_list.size > 0:
            mean_rel = float(rel_list.mean())
            med_rel  = float(np.median(rel_list))
        else:
            mean_rel = float("nan")
            med_rel  = float("nan")

        print(
            f"{cat:20s} "
            f"{N:6d} "
            f"{mean_abs:10.3f} "
            f"{med_abs:10.3f} "
            f"{mean_signed:10.3f} "
            f"{med_signed:10.3f} "
            f"{mean_sq:10.3f} "
            f"{rmse:10.3f} "
            f"{mean_rel:10.2f} "
            f"{med_rel:10.2f}"
        )


if __name__ == "__main__":
    main()
