from pathlib import Path
import json
import numpy as np

"""
INPUT: seg_depth_with_pred.json (dict with "images"->"regions")

Distance bins are based on GT distance (meters):
  [0,10), [10,20), [20,40), [40,80]  (last bin includes 80)

RESULT
original model
========== PER-DISTANCE ERROR STATS (SEGMENTATION) ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)         112      0.609      0.563       0.255       0.501      0.572      0.756      11.07       8.75
[10, 20)         23      2.086      1.115       1.029       0.995     11.282      3.359      14.29       8.29
[20, 40)          6      3.035      3.338       0.172       0.921     12.042      3.470       9.81      10.40
[40, 80]          2      4.891      4.891       4.408       4.408     43.357      6.585      10.73      10.73

pruned1layer model
========== PER-DISTANCE ERROR STATS (SEGMENTATION) ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)          31      0.347      0.304       0.155       0.221      0.160      0.400       5.33       5.25
[10, 20)          3      2.970      3.445      -2.970      -3.445      9.969      3.157      18.56      18.20
[20, 40)          0        nan        nan         nan         nan        nan        nan        nan        nan
[40, 80]          0        nan        nan         nan         nan        nan        nan        nan        nan

int8 model
========== PER-DISTANCE ERROR STATS (SEGMENTATION) ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)         112      0.439      0.298      -0.283      -0.193      0.396      0.630       8.58       4.33
[10, 20)         23      2.233      0.856      -0.248      -0.263     12.625      3.553      15.02       6.74
[20, 40)          6      2.964      1.801      -1.355      -0.740     18.493      4.300       9.98       7.09
[40, 80]          2      5.284      5.284       2.452       2.452     33.930      5.825      11.76      11.76

fp16 model
========== PER-DISTANCE ERROR STATS (SEGMENTATION) ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)         112      0.400      0.309      -0.188      -0.082      0.333      0.577       7.85       4.67
[10, 20)         23      2.366      1.016      -0.269      -0.151     14.277      3.778      15.79       8.18
[20, 40)          6      2.747      1.550      -1.476      -0.447     19.120      4.373       9.30       5.77
[40, 80]          2      6.007      6.007       2.665       2.665     43.181      6.571      13.38      13.38
"""

# ====== CONFIG ======
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")

# seg_depth_with_pred_origin (original model), seg_depth_with_pred_pruned1layer.json, seg_depth_with_pred_onnx_int8_cpu.json
#   seg_depth_with_pred_onnx_fp16_cpu.json
JSON_PATH = ROOT / "seg_depth_with_pred_onnx_fp16_cpu.json"

BINS = [0.0, 10.0, 20.0, 40.0, 80.0]  # last bin includes 80

# If True, ignore regions marked as excluded_low_conf
SKIP_EXCLUDED_LOW_CONF = True

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


def bin_label(lo, hi):
    if hi == BINS[-1]:
        return f"[{lo:.0f}, {hi:.0f}]"
    return f"[{lo:.0f}, {hi:.0f})"


def pick_bin(gt):
    # bins: [lo,hi) except last [lo,hi]
    for i in range(len(BINS) - 1):
        lo, hi = BINS[i], BINS[i + 1]
        if hi == BINS[-1]:
            if lo <= gt <= hi:
                return (lo, hi)
        else:
            if lo <= gt < hi:
                return (lo, hi)
    return None


def iter_regions(data: dict):
    for img in data.get("images", []):
        for r in (img.get("regions", []) or []):
            yield r


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise RuntimeError("Expected dict JSON with top-level key 'images'.")

    # Prepare bin containers (order matters; dict preserves insertion order in Python 3.7+)
    bin_pairs = [(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
    stats = {bp: {"abs": [], "signed": [], "rel": [], "sq": []} for bp in bin_pairs}

    total_regions = 0
    used = 0
    skipped_missing = 0
    skipped_excluded = 0
    skipped_bad_gt = 0
    skipped_outside = 0

    for r in iter_regions(data):
        total_regions += 1

        if SKIP_EXCLUDED_LOW_CONF and bool(r.get("excluded_low_conf", False)):
            skipped_excluded += 1
            continue

        gt = safe_float(r.get("distance_m"))
        pred = safe_float(r.get("pred_distance_m"))

        if gt is None or pred is None:
            skipped_missing += 1
            continue
        if gt <= GT_EPS:
            skipped_bad_gt += 1
            continue

        bp = pick_bin(gt)
        if bp is None:
            skipped_outside += 1
            continue

        err = gt - pred
        abs_err = abs(err)
        sq_err = err * err
        rel_err = 100.0 * abs_err / gt

        stats[bp]["abs"].append(abs_err)
        stats[bp]["signed"].append(err)
        stats[bp]["sq"].append(sq_err)
        stats[bp]["rel"].append(rel_err)

        used += 1

    print(f"Images in JSON: {len(data.get('images', []))}")
    print(f"Total regions: {total_regions}")
    print(f"Valid regions used: {used}")
    print(f"Skipped (excluded_low_conf): {skipped_excluded}")
    print(f"Skipped (missing gt/pred): {skipped_missing}")
    print(f"Skipped (gt <= 0): {skipped_bad_gt}")
    print(f"Skipped (gt outside bins): {skipped_outside}")

    header = (
        f"{'GT bin (m)':12s} "
        f"{'N':>6s} "
        f"{'mean|e|':>10s} "
        f"{'med|e|':>10s} "
        f"{'mean(gt-p)':>11s} "
        f"{'med(gt-p)':>11s} "
        f"{'MSE':>10s} "
        f"{'RMSE':>10s} "
        f"{'meanRel%':>10s} "
        f"{'medRel%':>10s}"
    )
    print("\n========== PER-DISTANCE ERROR STATS (SEGMENTATION) ==========")
    print(header)
    print("-" * len(header))

    for bp in bin_pairs:
        lo, hi = bp
        abs_list = np.array(stats[bp]["abs"], dtype=np.float32)
        signed_list = np.array(stats[bp]["signed"], dtype=np.float32)
        rel_list = np.array(stats[bp]["rel"], dtype=np.float32)
        sq_list = np.array(stats[bp]["sq"], dtype=np.float32)

        N = abs_list.size
        if N == 0:
            mean_abs = med_abs = mean_signed = med_signed = float("nan")
            mean_sq = rmse = float("nan")
            mean_rel = med_rel = float("nan")
        else:
            mean_abs = float(abs_list.mean())
            med_abs = float(np.median(abs_list))
            mean_signed = float(signed_list.mean())
            med_signed = float(np.median(signed_list))
            mean_sq = float(sq_list.mean())
            rmse = float(np.sqrt(mean_sq)) if np.isfinite(mean_sq) else float("nan")
            mean_rel = float(rel_list.mean())
            med_rel = float(np.median(rel_list))

        print(
            f"{bin_label(lo, hi):12s} "
            f"{N:6d} "
            f"{mean_abs:10.3f} "
            f"{med_abs:10.3f} "
            f"{mean_signed:11.3f} "
            f"{med_signed:11.3f} "
            f"{mean_sq:10.3f} "
            f"{rmse:10.3f} "
            f"{mean_rel:10.2f} "
            f"{med_rel:10.2f}"
        )


if __name__ == "__main__":
    main()
