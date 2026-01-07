import json
from pathlib import Path

DET_JSON = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\bb_json_KITTI_val.json"
DEPTH_GT_DIR = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth"

det = json.load(open(DET_JSON, "r", encoding="utf-8"))
depth_dir = Path(DEPTH_GT_DIR)

print("Example JSON file_name:")
for i in range(3):
    print(" ", det["images"][i].get("file_name"))

print("\nExample depth files:")
for p in list(depth_dir.glob("*.png"))[:3]:
    print(" ", p.name)

print("\nDepth png count:", len(list(depth_dir.glob("*.png"))))
