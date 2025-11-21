from pathlib import Path
import json
import numpy as np
from collections import defaultdict

"""
Per-distance error analysis for obj_depth_with_pred.json

Distance bins (based on GT distance in meters):
    [0, 10), [10, 20), [20, 40), [40, 80]

For each bin we compute:
    - N (number of objects)
    - mean / median absolute error |gt - pred|     [m]
    - mean / median signed error (gt - pred)       [m]
    - mean / median relative error (%)             [ 100 * |gt - pred| / gt ]
"""

# ====== CONFIG ======
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
JSON_PATH = ROOT / "obj_depth_with_pred.json"

# Define GT-distance bins (closed on left, open on right, except last you can treat as closed)
BINS = [0.0, 10.0, 20.0, 40.0, 80.0]   # 0–10, 10–20, 20–40, 40–80


def bin_name(lo: float, hi: float) -> str:
    return f"[{lo:.0f}, {hi:.0f})"


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total objects in JSON: {len(data)}")

    # Prepare containers per bin
    # key: (lo, hi) -> dict of lists: abs errors, signed errors, rel errors
    bin_stats = {}
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        bin_stats[(lo, hi)] = {
            "abs": [],
            "signed": [],
            "rel": []
        }

    n_skipped_missing = 0
    n_skipped_outside = 0

    for o in data:
        gt = o.get("gt_distance_m")
        pred = o.get("ground_distance_predict")

        # Skip if invalid distances
        if gt is None or pred is None:
            n_skipped_missing += 1
            continue

        # Choose bin based on GT distance
        placed = False
        for lo, hi in bin_stats.keys():
            # last bin treat hi as inclusive to catch gt == 80
            if lo <= gt < hi or (hi == BINS[-1] and lo <= gt <= hi):
                err = gt - pred
                abs_err = abs(err)
                # avoid div-by-zero
                if gt > 1e-6:
                    rel_err = 100.0 * abs_err / gt
                else:
                    rel_err = None

                bin_stats[(lo, hi)]["abs"].append(abs_err)
                bin_stats[(lo, hi)]["signed"].append(err)
                if rel_err is not None:
                    bin_stats[(lo, hi)]["rel"].append(rel_err)

                placed = True
                break

        if not placed:
            n_skipped_outside += 1

    print(f"\nSkipped due to missing gt/pred: {n_skipped_missing}")
    print(f"Skipped because GT distance out of all bins: {n_skipped_outside}")

    # ===== Print per-bin stats =====
    header = (
        f"{'GT bin (m)':12s} "
        f"{'N':>6s} "
        f"{'mean|e|':>10s} "
        f"{'med|e|':>10s} "
        f"{'mean(gt-p)':>10s} "
        f"{'med(gt-p)':>10s} "
        f"{'meanRel%':>10s} "
        f"{'medRel%':>10s}"
    )
    print("\n========== PER-DISTANCE ERROR STATS ==========")
    print(header)
    print("-" * len(header))

    total_used = 0

    for (lo, hi), d in bin_stats.items():
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
            f"{bin_name(lo, hi):12s} "
            f"{N:6d} "
            f"{mean_abs:10.3f} "
            f"{med_abs:10.3f} "
            f"{mean_signed:10.3f} "
            f"{med_signed:10.3f} "
            f"{mean_rel:10.2f} "
            f"{med_rel:10.2f}"
        )

    print(f"\nTotal objects used across bins: {total_used}")


if __name__ == "__main__":
    main()
