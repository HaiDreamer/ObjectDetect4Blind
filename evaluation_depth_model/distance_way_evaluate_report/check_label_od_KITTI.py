import json
from collections import Counter
from pathlib import Path

JSON_PATH = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\bb_json_KITTI_val.json"
CONF_MIN = 0.4  # e.g. 0.25 to count only detections with conf>=0.25, or None for all

'''
INPUT: bb_json_KITTI_val.json

OUTPUT
File: C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\bb_json_KITTI_val.json
Images: 1000
Images with >=1 detection: 871
Total detections: 3282
Avg detections/image: 3.282

Detections by class:
  car: 2661
  tree: 201
  person: 182
  bicycle: 99
  bus: 37
  pedestrian crossing sign: 36
  truck: 33
  motocycle: 18
  electric pole: 15
  
'''

def main():
    p = Path(JSON_PATH)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    total_images = len(images)

    total_objs = 0
    images_with_obj = 0
    per_class = Counter()

    for im in images:
        dets = im.get("detections", []) or []
        if CONF_MIN is not None:
            dets = [d for d in dets if float(d.get("confidence", 0.0)) >= CONF_MIN]

        if len(dets) > 0:
            images_with_obj += 1

        total_objs += len(dets)

        for d in dets:
            # Prefer class_name, fallback to class_id
            name = d.get("class_name")
            if not name:
                name = str(d.get("class_id", "unknown"))
            per_class[name] += 1

    print(f"File: {p}")
    print(f"Images: {total_images}")
    print(f"Images with >=1 detection: {images_with_obj}")
    print(f"Total detections: {total_objs}")
    print(f"Avg detections/image: {total_objs/total_images:.3f}" if total_images else "Avg detections/image: N/A")
    print("\nDetections by class:")
    for k, v in per_class.most_common():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
