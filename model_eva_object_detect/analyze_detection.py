import json
from pathlib import Path
from collections import Counter
import math

'''sanity check

OUTPUT
=== Detections by eval_category ===
  2661  Car
   201  tree
   182  Person
    99  Cyclist
    37  LargeVeh
    36  pedestrian crossing sign
    33  Truck
    18  motocycle
    15  electric pole

'''

JSON_PATH = Path(r"C:\Python\ObjectDetect4Blind\model_eva_object_detect\bb_json_KITTI_val_with_gt_dist.json")

def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None

def main():
    if not JSON_PATH.exists():
        raise FileNotFoundError(JSON_PATH)

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle either dict-root (your current format) or list-root (older scripts)
    if isinstance(data, dict) and "images" in data:
        images = data["images"]
    elif isinstance(data, list):
        # list of detections (fallback)
        images = [{"file_name": None, "detections": data}]
    else:
        raise ValueError("Unknown JSON structure. Expected dict with key 'images' or a list.")

    total_images = len(images)
    per_image_counts = []
    class_counter = Counter()
    eval_counter = Counter()

    total_dets = 0
    missing_dist = 0
    dist_vals = []

    for img in images:
        dets = img.get("detections", []) or []
        per_image_counts.append((img.get("file_name"), len(dets)))
        total_dets += len(dets)

        for d in dets:
            class_counter[d.get("class_name", "UNKNOWN")] += 1
            eval_counter[d.get("eval_category", "UNKNOWN")] += 1

            dist = safe_float(d.get("gt_distance_m"))
            if dist is None:
                missing_dist += 1
            else:
                dist_vals.append(dist)

    zero_det_imgs = [name for name, c in per_image_counts if c == 0]

    print("\n=== SUMMARY ===")
    print("JSON:", JSON_PATH)
    print("Images:", total_images)
    print("Total detections:", total_dets)
    print("Images with 0 detections:", len(zero_det_imgs))

    print("\n=== TOP 10 images by detection count ===")
    for name, c in sorted(per_image_counts, key=lambda x: x[1], reverse=True)[:10]:
        print(f"{c:4d}  {name}")

    print("\n=== Detections by class_name (Top 20) ===")
    for k, v in class_counter.most_common(20):
        print(f"{v:6d}  {k}")

    print("\n=== Detections by eval_category ===")
    for k, v in eval_counter.most_common():
        print(f"{v:6d}  {k}")

    print("\n=== GT distance availability ===")
    print("Distances present:", len(dist_vals))
    print("Distances missing:", missing_dist)
    if dist_vals:
        dist_sorted = sorted(dist_vals)
        def pct(p):
            idx = int(round((p/100)*(len(dist_sorted)-1)))
            return dist_sorted[max(0, min(len(dist_sorted)-1, idx))]
        print(f"Distance min/median/p10: {dist_sorted[0]:.3f} / {pct(50):.3f} / {pct(10):.3f} (m)")

    if zero_det_imgs:
        print("\n=== First 20 images with 0 detections ===")
        for n in zero_det_imgs[:20]:
            print(n)

if __name__ == "__main__":
    main()
