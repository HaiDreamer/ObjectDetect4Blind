from pathlib import Path
import json
import numpy as np

"""
INPUT: obj_depth_with_pred.json (dict with "images"->"detections")

Distance bins are based on GT distance (meters):
  [0,10), [10,20), [20,40), [40,80]  (last bin includes 80)

RESULT (pruned1layer)
========== PER-DISTANCE ERROR STATS ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)        1000      1.992      1.571      -1.979      -1.571      6.828      2.613      32.93      25.10
[10, 20)       1130      3.047      2.650      -2.742      -2.552     13.775      3.711      21.82      18.78
[20, 40)        944      2.957      2.249      -0.497      -0.959     15.933      3.992      11.06       8.23
[40, 80]        131      9.790      7.461       9.413       7.445    170.915     13.073      18.32      15.90

RESULT (int8 onnx model)
========== PER-DISTANCE ERROR STATS ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)        1000      0.811      0.536      -0.659      -0.464      1.708      1.307      12.87       8.32
[10, 20)       1130      2.031      1.460      -1.409      -1.268      7.943      2.818      14.10      10.59
[20, 40)        944      4.410      3.747      -2.042      -2.517     32.068      5.663      16.03      14.18
[40, 80]        131      9.536      7.994       2.123       0.972    163.360     12.781      18.91      16.52

RESULT (fp16 onnx model)
========== PER-DISTANCE ERROR STATS ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)        1000      0.748      0.466      -0.563      -0.354      1.569      1.252      11.98       7.08
[10, 20)       1130      1.950      1.349      -1.321      -1.119      7.925      2.815      13.53       9.53
[20, 40)        944      4.156      3.435      -1.886      -2.262     29.462      5.428      15.11      13.05
[40, 80]        131      8.893      7.072       2.111       0.564    150.628     12.273      17.46      15.27

RESULT (original model)
========== PER-DISTANCE ERROR STATS ==========
GT bin (m)        N    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
-------------------------------------------------------------------------------------------------------------
[0, 10)        1000      0.719      0.433      -0.293      -0.087      1.807      1.344      11.62       6.76
[10, 20)       1130      1.589      0.972       0.012       0.334      6.542      2.558      11.09       6.50
[20, 40)        944      3.029      2.162       1.264       1.137     18.378      4.287      11.10       7.86
[40, 80]        131      7.296      4.836       5.771       4.042    110.817     10.527      14.03       9.67
"""

ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
# obj_depth_with_pred_pruned1layer.json, obj_depth_with_pred_onnx_int8_cpu.json, 
#   obj_depth_with_pred_onnx_fp16_cpu.json, obj_depth_with_pred_origin.json
JSON_PATH = ROOT / "obj_depth_with_pred_origin.json"

BINS = [0.0, 10.0, 20.0, 40.0, 80.0]  # last bin includes 80


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


def iter_detections(data: dict):
    for img in data.get("images", []):
        for det in (img.get("detections", []) or []):
            yield det


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise RuntimeError("Expected dict JSON with top-level key 'images'.")

    # Prepare bin containers (order matters; dict order is guaranteed in modern Python) :contentReference[oaicite:0]{index=0}
    bin_pairs = [(BINS[i], BINS[i + 1]) for i in range(len(BINS) - 1)]
    stats = {bp: {"abs": [], "signed": [], "rel": [], "sq": []} for bp in bin_pairs}

    total_dets = 0
    used = 0
    skipped_missing = 0
    skipped_bad_gt = 0
    skipped_outside = 0

    for det in iter_detections(data):
        total_dets += 1

        gt = safe_float(det.get("gt_distance_m"))
        pred = safe_float(det.get("ground_distance_predict"))

        if gt is None or pred is None:
            skipped_missing += 1
            continue
        if gt <= 1e-6:
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
    print(f"Total detections: {total_dets}")
    print(f"Valid detections used: {used}")
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
    print("\n========== PER-DISTANCE ERROR STATS ==========")
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
