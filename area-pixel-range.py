from pathlib import Path
import json
import numpy as np
from collections import defaultdict

"""
Per-class + per-size error analysis for obj_depth_with_pred.json

For each eval_category:

1) Collect bbox areas (w*h) for all objects in that class (with valid gt/pred).
2) Compute area percentiles:
       A1 = 33rd percentile  (small / medium boundary)
       A2 = 66th percentile  (medium / large boundary)
   Also compute:
       min_area, max_area  per class.
3) Define size bins per class:
       small  : area < A1
       medium : A1 <= area < A2
       large  : area >= A2
4) For each (class, size_bin), compute:
       - N (number of objects)
       - mean / median absolute error  |gt - pred|   [m]
       - mean / median signed error    (gt - pred)   [m]
       - mean / median relative error  100*|gt-p|/gt [%]
"""

# ====== CONFIG ======
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
JSON_PATH = ROOT / "obj_depth_with_pred.json"


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total objects in JSON: {len(data)}")

    # 1) Collect areas per class (only for objects with valid boxes and valid gt/pred)
    areas_per_class = defaultdict(list)

    for o in data:
        gt = o.get("gt_distance_m")
        pred = o.get("ground_distance_predict")
        if gt is None or pred is None:
            continue

        x1, y1, x2, y2 = o["bbox_xyxy"]
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        area = w * h
        cat = o.get("eval_category", "Unknown")
        areas_per_class[cat].append(area)

    # 2) Compute A1, A2, min, max per class (33rd and 66th percentiles + min/max)
    thresh_per_class = {}   # cat -> (min_area, max_area, A1, A2)

    print("\nPer-class area statistics (pixel^2):")
    for cat, areas in areas_per_class.items():
        arr = np.array(areas, dtype=np.float32)
        min_area = float(arr.min())
        max_area = float(arr.max())
        if arr.size < 3:
            # not enough samples to define three bins sensibly
            A1 = A2 = None
        else:
            A1 = float(np.percentile(arr, 33))
            A2 = float(np.percentile(arr, 66))
        thresh_per_class[cat] = (min_area, max_area, A1, A2)

        print(
            f"  {cat:20s}: "
            f"min={min_area:.1f}, max={max_area:.1f}, "
            f"A1(33%)={A1}, A2(66%)={A2}"
        )

    # 3) Prepare containers for error stats: stats[(cat, size_bin)] -> lists
    stats = defaultdict(lambda: {"abs": [], "signed": [], "rel": []})

    n_skipped_missing = 0
    n_skipped_bad_box = 0

    for o in data:
        gt = o.get("gt_distance_m")
        pred = o.get("ground_distance_predict")
        if gt is None or pred is None:
            n_skipped_missing += 1
            continue

        x1, y1, x2, y2 = o["bbox_xyxy"]
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            n_skipped_bad_box += 1
            continue

        area = w * h
        cat = o.get("eval_category", "Unknown")

        if cat not in thresh_per_class:
            continue

        min_area, max_area, A1, A2 = thresh_per_class[cat]
        if A1 is None or A2 is None:
            # not enough data to define bins for this class
            continue

        # 4) Assign size bin based on per-class thresholds
        if area < A1:
            size_bin = "small"
        elif area < A2:
            size_bin = "medium"
        else:
            size_bin = "large"

        err = gt - pred
        abs_err = abs(err)
        if gt > 1e-6:
            rel_err = 100.0 * abs_err / gt
        else:
            rel_err = None

        key = (cat, size_bin)
        stats[key]["abs"].append(abs_err)
        stats[key]["signed"].append(err)
        if rel_err is not None:
            stats[key]["rel"].append(rel_err)

    print(f"\nSkipped due to missing gt/pred: {n_skipped_missing}")
    print(f"Skipped due to invalid bbox (w<=0 or h<=0): {n_skipped_bad_box}")

    # 5) Print per-class + per-size-bin stats
    print("\n========== PER-CLASS + PER-SIZE ERROR STATS ==========")
    header = (
        f"{'SizeBin':10s} "
        f"{'N':>6s} "
        f"{'mean|e|':>10s} "
        f"{'med|e|':>10s} "
        f"{'mean(gt-p)':>10s} "
        f"{'med(gt-p)':>10s} "
        f"{'meanRel%':>10s} "
        f"{'medRel%':>10s}"
    )

    total_used = 0
    classes = sorted(areas_per_class.keys())

    for cat in classes:
        min_area, max_area, A1, A2 = thresh_per_class[cat]
        if A1 is None or A2 is None:
            print(f"\nClass: {cat} (not enough samples to define small/medium/large bins)")
            print(f"  min area = {min_area:.1f}, max area = {max_area:.1f}")
            continue

        print(f"\nClass: {cat}")
        print(
            f"  Area stats: min={min_area:.1f}, max={max_area:.1f}, "
            f"A1(33%)={A1:.1f}, A2(66%)={A2:.1f}"
        )
        print(header)
        print("-" * len(header))

        for size_bin in ["small", "medium", "large"]:
            key = (cat, size_bin)
            d = stats.get(key, None)
            if d is None:
                N = 0
                mean_abs = med_abs = mean_signed = med_signed = float("nan")
                mean_rel = med_rel = float("nan")
            else:
                abs_list = np.array(d["abs"], dtype=np.float32)
                signed_list = np.array(d["signed"], dtype=np.float32)
                rel_list = np.array(d["rel"], dtype=np.float32)
                N = len(abs_list)
                total_used += N

                if N == 0:
                    mean_abs = med_abs = mean_signed = med_signed = float("nan")
                    mean_rel = med_rel = float("nan")
                else:
                    mean_abs = float(abs_list.mean())
                    med_abs  = float(np.median(abs_list))
                    mean_signed = float(signed_list.mean())
                    med_signed  = float(np.median(signed_list))
                    if rel_list.size > 0:
                        mean_rel = float(rel_list.mean())
                        med_rel  = float(np.median(rel_list))
                    else:
                        mean_rel = med_rel = float("nan")

            print(
                f"{size_bin:10s} "
                f"{N:6d} "
                f"{mean_abs:10.3f} "
                f"{med_abs:10.3f} "
                f"{mean_signed:10.3f} "
                f"{med_signed:10.3f} "
                f"{mean_rel:10.2f} "
                f"{med_rel:10.2f}"
            )

    print(f"\nTotal objects used across all (class, size_bin): {total_used}")


if __name__ == "__main__":
    main()
