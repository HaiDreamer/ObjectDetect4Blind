from pathlib import Path
import json
import numpy as np
from collections import defaultdict

"""
Per-class + per-distance error analysis for obj_depth_with_pred.json

Distance bins (based on GT distance in meters):
    [0, 10), [10, 20), [20, 40), [40, 80]

For each (eval_category, distance bin) we compute:
    - N (number of objects)
    - mean / median absolute error |gt - pred|     [m]
    - mean / median signed error (gt - pred)       [m]
    - mean / median relative error (%)             [100 * |gt - pred| / gt]
"""

# ====== CONFIG ======
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
JSON_PATH = ROOT / "obj_depth_with_pred.json"

# GT-distance bins: 0–10, 10–20, 20–40, 40–80
BINS = [0.0, 10.0, 20.0, 40.0, 80.0]


def bin_label(lo: float, hi: float) -> str:
    """Human-readable label for a distance bin."""
    # last bin we treat as closed on right for printing
    if hi == BINS[-1]:
        return f"[{lo:.0f}, {hi:.0f}]"
    return f"[{lo:.0f}, {hi:.0f})"


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total objects in JSON: {len(data)}")

    # stats[(class, (lo, hi))] -> {"abs": [...], "signed": [...], "rel": [...]}
    stats = defaultdict(lambda: {"abs": [], "signed": [], "rel": [], "sq": []})

    n_skipped_missing = 0
    n_skipped_outside = 0

    for o in data:
        gt = o.get("gt_distance_m")
        pred = o.get("ground_distance_predict")
        cat = o.get("eval_category", "Unknown")

        # Skip invalid distances
        if gt is None or pred is None:
            n_skipped_missing += 1
            continue

        # assign to a distance bin based on GT
        placed = False
        for i in range(len(BINS) - 1):
            lo, hi = BINS[i], BINS[i + 1]

            # last bin: include hi as well (<= hi) to catch gt == 80
            in_bin = (lo <= gt < hi) or (hi == BINS[-1] and lo <= gt <= hi)
            if not in_bin:
                continue

            err = gt - pred
            abs_err = abs(err)
            sq_err = err * err

            if gt > 1e-6:
                rel_err = 100.0 * abs_err / gt
            else:
                rel_err = None

            key = (cat, (lo, hi))
            stats[key]["abs"].append(abs_err)
            stats[key]["signed"].append(err)
            stats[key]["sq"].append(sq_err) 
            if rel_err is not None:
                stats[key]["rel"].append(rel_err)

            placed = True
            break

        if not placed:
            n_skipped_outside += 1

    print(f"\nSkipped due to missing gt/pred: {n_skipped_missing}")
    print(f"Skipped because GT distance out of all bins: {n_skipped_outside}")

    # collect all classes present
    classes = sorted({cat for (cat, _bin) in stats.keys()})

    print("\n========== PER-CLASS + PER-DISTANCE ERROR STATS ==========")
    header = (
        f"{'GT bin (m)':12s} "
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

    total_used = 0

    for cat in classes:
        print(f"\nClass: {cat}")
        print(header)
        print("-" * len(header))

        for i in range(len(BINS) - 1):
            lo, hi = BINS[i], BINS[i + 1]
            key = (cat, (lo, hi))
            d = stats.get(key, None)

            if d is None:
                N = 0
                mean_abs = med_abs = mean_signed = med_signed = float("nan")
                mean_rel = med_rel = float("nan")
                mse = rmse = float("nan") 
            else:
                abs_list = np.array(d["abs"], dtype=np.float32)
                signed_list = np.array(d["signed"], dtype=np.float32)
                rel_list = np.array(d["rel"], dtype=np.float32)
                sq_list     = np.array(d["sq"],     dtype=np.float32)

                N = len(abs_list)
                total_used += N

                if N == 0:
                    mean_abs = med_abs = mean_signed = med_signed = float("nan")
                    mean_rel = med_rel = float("nan")
                    mse = rmse = float("nan") 
                else:
                    mean_abs = float(abs_list.mean())
                    med_abs  = float(np.median(abs_list))
                    mean_signed = float(signed_list.mean())
                    med_signed  = float(np.median(signed_list))
                    mse  = float(sq_list.mean())
                    rmse = float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
                    if rel_list.size > 0:
                        mean_rel = float(rel_list.mean())
                        med_rel  = float(np.median(rel_list))
                    else:
                        mean_rel = med_rel = float("nan")

            print(
                f"{bin_label(lo, hi):12s} "
                f"{N:6d} "
                f"{mean_abs:10.3f} "
                f"{med_abs:10.3f} "
                f"{mean_signed:10.3f} "
                f"{med_signed:10.3f} "
                f"{mse:10.3f} "
                f"{rmse:10.3f} " 
                f"{mean_rel:10.2f} "
                f"{med_rel:10.2f}"
            )

    print(f"\nTotal (class,bin) objects used (counted with repetition across bins): {total_used}")


if __name__ == "__main__":
    main()
