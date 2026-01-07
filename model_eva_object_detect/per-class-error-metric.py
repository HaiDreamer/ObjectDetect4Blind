from pathlib import Path
import json
import numpy as np
from collections import defaultdict

"""
INPUT
  obj_depth_with_pred.json (dict)
    {
      ...,
      "images": [
        {
          "file_name": "...png",
          "detections": [
            {
              "eval_category": ...,
              "gt_distance_m": ...,
              "ground_distance_predict": ...
            }, ...
          ]
        }, ...
      ]
    }

OUTPUT (printed)
  Per-class stats:
    - N
    - mean/median abs error |gt-pred|
    - mean/median signed error (gt-pred)
    - MSE / RMSE
    - mean/median relative error (%)

RESULT(pruned1layer)
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
Car                               95      3.170      2.484       -0.617       -1.782     23.686      4.867      21.60      20.60
Cyclist                           35      1.084      0.925       -0.931       -0.716      1.827      1.351      14.42      12.02
Person                            53      4.377      4.291       -4.377       -4.291     20.999      4.582      53.21      48.49
electric pole                      1      3.261      3.261       -3.261       -3.261     10.635      3.261      22.99      22.99
pedestrian crossing sign           2      5.416      5.416       -5.416       -5.416     29.447      5.426      52.94      52.94
tree                              19      3.938      3.038       -2.020       -2.822     22.212      4.713      27.03      24.55

int8 onnx model
========== PER-CLASS ERROR STATS ==========
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
Car                             2639      2.596      1.284       -1.144       -0.753     19.842      4.454      12.31       9.76
Cyclist                           98      1.210      1.003       -1.137       -0.989      2.475      1.573      13.88      12.24
LargeVeh                          37      2.296      1.478       -2.259       -1.478     10.529      3.245      17.76      18.01
Person                           174      3.014      2.629       -2.452       -2.461     13.278      3.644      29.17      28.10
Truck                             33      1.513      0.860       -0.623       -0.687      5.118      2.262      12.79      12.30
electric pole                     14      2.945      2.231       -2.932       -2.231     12.334      3.512      26.41      24.51
motocycle                         17      0.752      0.406       -0.111        0.052      1.296      1.139      10.35       4.96
pedestrian crossing sign          17      3.086      2.379       -2.903       -2.379     13.104      3.620      26.13      18.14
tree                             176      4.454      3.217       -0.834       -2.032     36.722      6.060      30.85      27.34

fp16 onnx model
========== PER-CLASS ERROR STATS ==========
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
Car                             2639      2.427      1.145       -1.013       -0.637     18.317      4.280      11.44       8.66
Cyclist                           98      1.196      1.001       -1.110       -0.922      2.570      1.603      13.61      11.36
LargeVeh                          37      2.155      1.434       -2.103       -1.434      9.121      3.020      16.81      17.20
Person                           174      2.973      2.518       -2.443       -2.406     13.138      3.625      28.85      26.85
Truck                             33      1.581      0.857       -0.582       -0.747      5.442      2.333      13.11      11.30
electric pole                     14      2.516      2.019       -2.516       -2.019      9.096      3.016      22.86      22.51
motocycle                         17      0.788      0.374       -0.202       -0.017      1.436      1.198      10.69       4.26
pedestrian crossing sign          17      3.252      2.879       -3.051       -2.879     14.481      3.805      26.02      17.95
tree                             176      4.351      3.010       -0.938       -1.710     35.661      5.972      30.06      23.66

original model
========== PER-CLASS ERROR STATS ==========
Class                              N    mean|e|     med|e|   mean(gt-p)    med(gt-p)        MSE       RMSE   meanRel%    medRel%
--------------------------------------------------------------------------------------------------------------------------------
Car                             2639      1.837      0.860        0.897        0.411     11.947      3.456       9.19       6.33
Cyclist                           98      0.690      0.497       -0.338       -0.188      0.878      0.937       7.92       5.69
LargeVeh                          37      0.799      0.414       -0.455       -0.308      1.514      1.230       8.38       6.22
Person                           174      2.154      1.624       -1.699       -1.577      8.853      2.975      21.10      18.95
Truck                             33      1.571      1.017        0.749       -0.112      6.569      2.563      11.85       9.20
electric pole                     14      2.751      1.863       -2.535       -1.863     14.116      3.757      24.87      18.89
motocycle                         17      1.011      0.567        0.114        0.384      1.884      1.373      15.28       7.09
pedestrian crossing sign          17      1.929      1.064       -0.579       -0.395      7.793      2.792      17.83      11.28
tree                             176      4.950      4.160       -1.892       -2.989     41.343      6.430      34.91      31.47
"""

# ====== CONFIG ======
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
# obj_depth_with_pred_pruned1layer.json, obj_depth_with_pred_onnx_int8_cpu.json, 
#   obj_depth_with_pred_onnx_fp16_cpu.json, obj_depth_with_pred_origin.json
JSON_PATH = ROOT / "obj_depth_with_pred_origin.json"


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

    n_total_dets = 0
    n_skipped_missing = 0

    for img in images:
        dets = img.get("detections", []) or []
        for det in dets:
            n_total_dets += 1

            gt = safe_float(det.get("gt_distance_m"))
            pred = safe_float(det.get("ground_distance_predict"))

            # prefer eval_category, fallback to class_name, else "Unknown"
            cat = det.get("eval_category") or det.get("class_name") or "Unknown"
            cat = str(cat)

            if gt is None or pred is None:
                n_skipped_missing += 1
                continue

            err = gt - pred
            abs_err = abs(err)
            sq_err = err * err

            per_class_abs[cat].append(abs_err)
            per_class_signed[cat].append(err)
            per_class_sq_err[cat].append(sq_err)

            if gt > 1e-6:
                per_class_rel[cat].append(100.0 * abs_err / gt)

    used = sum(len(v) for v in per_class_abs.values())
    print(f"Total detections in JSON: {n_total_dets}")
    print(f"Valid detections used:   {used}")
    print(f"Skipped (missing gt/pred): {n_skipped_missing}")

    print("\n========== PER-CLASS ERROR STATS ==========")
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


if __name__ == "__main__":
    main()
