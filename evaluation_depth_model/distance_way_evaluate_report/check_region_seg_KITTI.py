import json
from collections import Counter
from pathlib import Path

'''
INPUT
    segment_json_KITTI_val.json

OUTPUT
    File: C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\segment_json_KITTI_val.json
    Images: 1000
    Images with >=1 region: 136
    Total regions: 145
    Avg regions/image: 0.145
    
    Regions by class:
    sidewalk: 68
    tree line: 42
    stairs: 32
    crosswalk: 3

'''

JSON_PATH = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\segment_json_KITTI_val.json"
CONF_MIN = None  # e.g. 0.25, None for all

def main():
    p = Path(JSON_PATH)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    total_images = len(images)

    total_regions = 0
    images_with_region = 0
    per_class = Counter()

    for im in images:
        inst = im.get("instances", []) or []
        if CONF_MIN is not None:      
            inst = [d for d in inst if float(d.get("confidence", 0.0)) >= CONF_MIN]     # take confidence score, if null -> =0, then take this instance if > min required confidence

        if len(inst) > 0:
            images_with_region += 1

        total_regions += len(inst)

        # counting instance qualified
        for d in inst:
            name = d.get("class_name")
            if not name:
                name = str(d.get("class_id", "unknown"))
            per_class[name] += 1

    print(f"File: {p}")
    print(f"Images: {total_images}")
    print(f"Images with >=1 region: {images_with_region}")
    print(f"Total regions: {total_regions}")
    print(f"Avg regions/image: {total_regions/total_images:.3f}" if total_images else "Avg regions/image: N/A")
    print("\nRegions by class:")
    for k, v in per_class.most_common():        # print each class label have how much instances
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
