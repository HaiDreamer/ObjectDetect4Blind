from pathlib import Path
import json
import numpy as np

"""
Sanity-check obj_depth_with_pred.json

- Counts total objects
- Filters to entries with valid gt_distance_m and ground_distance_predict
- Computes:
    * N_valid
    * mean / median signed error (gt - pred)
    * mean / median absolute error
    * min / max absolute error
    * basic GT / pred distance stats
"""

# CONFIG
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")
JSON_PATH = ROOT / "obj_depth_with_pred.json"


def main():
    assert JSON_PATH.exists(), f"JSON not found: {JSON_PATH}"

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total objects in JSON: {len(data)}")

    # Collect valid pairs
    gt_list = []
    pred_list = []
    wrong_list = []  # gt - pred
    abs_list = []

    n_missing_gt = 0
    n_missing_pred = 0

    for o in data:
        gt = o.get("gt_distance_m")
        pred = o.get("ground_distance_predict")

        if gt is None:
            n_missing_gt += 1
            continue
        if pred is None:
            n_missing_pred += 1
            continue

        e = gt - pred
        gt_list.append(gt)
        pred_list.append(pred)
        wrong_list.append(e)
        abs_list.append(abs(e))

    if not gt_list:
        print("No valid (gt, pred) pairs found. Check your pipeline.")
        return

    gt_arr   = np.array(gt_list,   dtype=np.float32)
    pred_arr = np.array(pred_list, dtype=np.float32)
    err_arr  = np.array(wrong_list, dtype=np.float32)
    abs_arr  = np.array(abs_list,  dtype=np.float32)

    print("\n==== SANITY CHECK RESULTS ====")
    print(f"Valid (gt, pred) pairs: {len(gt_arr)}")
    print(f"Missing gt_distance_m: {n_missing_gt}")
    print(f"Missing ground_distance_predict: {n_missing_pred}")

    # Basic distance stats
    print("\nGT distance (m):")
    print(f"  mean:   {gt_arr.mean():.3f}")
    print(f"  median: {np.median(gt_arr):.3f}")
    print(f"  min:    {gt_arr.min():.3f}")
    print(f"  max:    {gt_arr.max():.3f}")

    print("\nPred distance (m):")
    print(f"  mean:   {pred_arr.mean():.3f}")
    print(f"  median: {np.median(pred_arr):.3f}")
    print(f"  min:    {pred_arr.min():.3f}")
    print(f"  max:    {pred_arr.max():.3f}")

    # Error stats
    print("\nSigned error (gt - pred) (m):")
    print(f"  mean:   {err_arr.mean():.3f}")
    print(f"  median: {np.median(err_arr):.3f}")
    print(f"  min:    {err_arr.min():.3f}")
    print(f"  max:    {err_arr.max():.3f}")

    print("\nAbsolute error |gt - pred| (m):")
    print(f"  mean:   {abs_arr.mean():.3f}")
    print(f"  median: {np.median(abs_arr):.3f}")
    print(f"  min:    {abs_arr.min():.3f}")
    print(f"  max:    {abs_arr.max():.3f}")

    worst = max(
        (o for o in data if o["gt_distance_m"] is not None and o["ground_distance_predict"] is not None),
        key=lambda o: abs(o["gt_distance_m"] - o["ground_distance_predict"])
    )
    print(worst)


if __name__ == "__main__":
    main()
