from pathlib import Path
import json
import cv2
import numpy as np
from ultralytics import YOLO

'''
HOW TO RUN
- just press run ?!

ALGORITHM
- use ground truth folder with origin image -> "relative true" distance of each object 

NOTE
- Person and Person in bicycle/moto has slightly different distance (different way evaluating -> different result!)


'''

# =========================
# CONFIG
# =========================
IMG_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\image")
GT_DIR  = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth")

YOLO_WEIGHTS = r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\yolov8m.pt"

OUT_IMG_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\obj_depth_vis")
OUT_JSON    = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\obj_depth_gt.json")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

MAX_DEPTH_M = 80.0  # typical KITTI cap


# =========================
# HELPER: load KITTI GT depth as meters
# KITTI convention: depth_m = uint16 / 256.0, 0 = invalid
# =========================
def load_kitti_depth_meters(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 3:
        im = im[..., 0]
    depth_m = im.astype(np.float32) / 256.0
    return depth_m


# =========================
# HELPER: compute per-box GT distance from depth map
# =========================
def box_distance_from_gt(depth_map_m: np.ndarray,
                         x1: int, y1: int, x2: int, y2: int,
                         mode: str = "bottom",
                         frac: float = 0.5) -> float | None:
    """
    Compute a robust distance for an object inside bbox using KITTI GT depth.

    mode:
        - "bottom": use bottom frac of bbox height (for vehicles, bikes)
        - "center": use central frac x frac region (for persons)
    frac:
        fraction of bbox size to use vertically (and horizontally for "center").
    """
    H, W = depth_map_m.shape[:2]

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    if mode == "bottom":
        # bottom band of the box: lower frac of height
        band_h = int(h * frac)
        if band_h <= 0:
            return None
        y_start = max(y1, y2 - band_h)
        patch = depth_map_m[y_start:y2, x1:x2]

    else:  # "center"
        cw = int(w * frac)
        ch = int(h * frac)
        if cw <= 0 or ch <= 0:
            return None
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cx1 = max(0, cx - cw // 2)
        cy1 = max(0, cy - ch // 2)
        cx2 = min(W, cx1 + cw)
        cy2 = min(H, cy1 + ch)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        patch = depth_map_m[cy1:cy2, cx1:cx2]

    if patch.size == 0:
        return None

    valid = patch[(patch > 0.0) & np.isfinite(patch) & (patch < MAX_DEPTH_M)]
    if valid.size == 0:
        return None

    # robust against outliers: median depth
    dist = float(np.median(valid))
    return dist


# =========================
# CLASS MAPPING
# YOLO (COCO) IDs -> names
# =========================
# COCO / YOLO class names (0..79); we only need a subset:
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# We want to map COCO classes to final evaluation categories:
#   Person
#   Cyclist/Bicycle      (bicycle + motorcycle)
#   Car (Car + Van)      (COCO "car" covers car/van type)
#   Truck
#   Large vehicle (Bus)  (proxy for Tram)
def map_coco_to_eval(cat_name: str) -> str | None:
    cat = cat_name.lower()
    if cat == "person":
        return "Person"
    if cat in ("bicycle", "motorcycle"):
        return "Cyclist"
    if cat == "car":
        return "Car"
    if cat == "truck":
        return "Truck"
    if cat == "bus":
        return "LargeVeh"
    # ignore anything else
    return None


# =========================
# MAIN
# =========================
def main():
    # Load YOLOv8 model
    model = YOLO(YOLO_WEIGHTS)

    all_results = []

    img_paths = sorted(IMG_DIR.glob("*.png"))
    if not img_paths:
        raise RuntimeError(f"No PNG images found in {IMG_DIR}")

    for img_idx, img_path in enumerate(img_paths, 1):
        print(f"[{img_idx}/{len(img_paths)}] Processing {img_path.name}")

        # Map image file to GT depth file (KITTI naming pattern)
        name = img_path.name
        parts = name.split("_image_")
        if len(parts) != 3:
            print(f"[WARN] unexpected filename pattern: {name}")
            continue

        prefix, frame_str, cam_str = parts
        gt_name = f"{prefix}_groundtruth_depth_{frame_str}_image_{cam_str}"
        gt_path = GT_DIR / gt_name

        if not gt_path.exists():
            print(f"  [WARN] GT depth not found for {name} -> {gt_name}, skipping image.")
            continue

        # Load RGB + GT depth
        rgb = cv2.imread(str(img_path))
        if rgb is None:
            print(f"  [WARN] Failed to read image {img_path}, skipping.")
            continue
        H, W = rgb.shape[:2]

        depth_gt = load_kitti_depth_meters(gt_path)
        if depth_gt.shape[:2] != (H, W):
            depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)

        # Run YOLO detection
        yolo_res = model(str(img_path), verbose=False)[0]

        # For drawing
        vis = rgb.copy()

        img_objects = []

        for det_id, box in enumerate(yolo_res.boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            # Map to COCO name and then to our eval category
            if cls_id < 0 or cls_id >= len(COCO_NAMES):
                continue
            coco_name = COCO_NAMES[cls_id]
            eval_cat = map_coco_to_eval(coco_name)
            if eval_cat is None:
                continue  # skip non-target classes

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Choose depth mode: bottom for vehicles, center for people/cyclists
            if eval_cat in ("Car", "Truck", "LargeVehicle", "Cyclist/Bicycle"):
                mode = "bottom"
            else:
                mode = "center"

            dist_gt = box_distance_from_gt(depth_gt, x1, y1, x2, y2, mode=mode, frac=0.3)

            # Draw on visualization
            color = (0, 255, 0)  # green
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{eval_cat}"
            if dist_gt is not None:
                label += f" {dist_gt:.1f}m"
            cv2.putText(vis, label, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            obj_info = {
                "image": img_path.name,
                "det_id": det_id,
                "yolo_class_id": cls_id,
                "yolo_class_name": coco_name,
                "eval_category": eval_cat,
                "confidence": conf,
                "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                "gt_distance_m": dist_gt
            }
            img_objects.append(obj_info)
            all_results.append(obj_info)

        # Save visualization image
        out_img = OUT_IMG_DIR / img_path.name
        cv2.imwrite(str(out_img), vis)
        print(f"  Saved overlay: {out_img} with {len(img_objects)} objects")

    # Save JSON with all object GT distances
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all GT object distances to: {OUT_JSON}")


if __name__ == "__main__":
    main()
