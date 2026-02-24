import json
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

'''
OUTPUT: json file with ground truth depth distance + object detection for our label
'''

# PATHS
MODEL_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2.pt"
IMAGE_DIR  = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\image"
GT_DIR     = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth"

# Output JSON
OUT_JSON   = r"C:\Python\ObjectDetect4Blind\model_eva_object_detect\distance_way_evaluate_report\bb_json_KITTI_val_with_gt_dist.json"

# Output image
OUT_IMG_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\obj_depth_vis")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Override class names (ONLY if your model class order matches this list)
OVERRIDE_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "tree", "perdestrian_crossing_sign", "electric_pole"
]
USE_OVERRIDE_NAMES = True

# Inference settings
CONF_THRES = 0.25
IOU_THRES  = 0.7
IMGSZ      = 640
DEVICE     = None  # "cpu" or 0 for GPU; None = auto

# Distance settings 
MAX_DEPTH_M = 80.0
BOX_FRAC    = 0.3
BOX_Q       = 10.0
BOX_SUBSAMP = 1


def get_model_names(model) -> list[str]:
    """Ultralytics model.names can be a dict {id: name} or a list."""
    names = model.names
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def xyxy_to_xywh(x1, y1, x2, y2):
    return [x1, y1, (x2 - x1), (y2 - y1)]


# KITTI GT depth convention: depth_m = uint16 / 256.0, 0 = invalid
def load_kitti_depth_meters(path: Path) -> np.ndarray:
    depth_png = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_png is None:
        raise FileNotFoundError(path)
    if depth_png.ndim == 3:
        depth_png = depth_png[..., 0]
    depth_m = depth_png.astype(np.float32) / 256.0
    return depth_m


def _fast_percentile_1d(vals: np.ndarray, q: float) -> float | None:
    '''take k% smallest value of array'''
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    k = int(round((q / 100.0) * (vals.size - 1)))
    k = max(0, min(vals.size - 1, k))
    return float(np.partition(vals, k)[k])


def compute_box_distance(
    depth_map_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *,
    frac: float = BOX_FRAC,
    mode: str = "center",      # "center" or "bottom"
    q: float = BOX_Q,
    subsample: int = BOX_SUBSAMP,
) -> float | None:
    """
      - "center": central frac x frac
      - "bottom": bottom frac of height + central 50% width band
      - distance = low percentile (p10 by default)
      - x2,y2 treated as slice end
    """
    # clamping bb, bb must be inside img
    H, W = depth_map_m.shape[:2]
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    # fallback case 
    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1

    # fallback case
    if w <= 0 or h <= 0:
        return None

    if mode == "bottom":
        ch = int(h * frac)
        if ch <= 0:
            return None
        y_start = max(y1, y2 - ch)

        center_band_width = int(w * 0.5)
        if center_band_width <= 0:
            return None
        cx = (x1 + x2) // 2
        x_start = max(x1, cx - center_band_width // 2)
        x_end = min(x2, x_start + center_band_width)
        if x_end <= x_start:
            return None

        patch = depth_map_m[y_start:y2, x_start:x_end]
    else:
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

    if patch.size == 0:     # fallback case
        return None
    if subsample > 1:
        patch = patch[::subsample, ::subsample]

    # take valid value, reshape(-1) for ensure valid is 1D array
    valid = patch[(patch > 0.0) & np.isfinite(patch) & (patch < MAX_DEPTH_M)].reshape(-1)   
    return _fast_percentile_1d(valid, q=q)


def kitti_gt_path_from_image_name(img_name: str, gt_dir: Path) -> Path | None:
    parts = img_name.split("_image_")
    if len(parts) != 3:
        return None
    prefix, frame_str, cam_str = parts
    gt_name = f"{prefix}_groundtruth_depth_{frame_str}_image_{cam_str}"
    return gt_dir / gt_name


def is_bottom_region(class_name: str) -> bool:
    n = class_name.lower()
    return any(k in n for k in ("car", "bicycle", "motorcycle", "truck", "bus"))


def map_eval_category(class_name: str) -> str:
    n = class_name.lower()
    if n == "person":
        return "Person"
    if n in ("bicycle", "motorcycle"):
        return "Cyclist"
    if n == "car":
        return "Car"
    if n == "truck":
        return "Truck"
    if n == "bus":
        return "LargeVeh"
    return class_name


def main():
    image_dir = Path(IMAGE_DIR)
    gt_dir = Path(GT_DIR)

    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {image_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT_DIR not found: {gt_dir}")

    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise RuntimeError(f"No images found under: {image_dir}")

    model = YOLO(MODEL_PATH)

    model_names = get_model_names(model)
    class_names = OVERRIDE_CLASS_NAMES if USE_OVERRIDE_NAMES else model_names

    print("Model class names:", model_names)
    if USE_OVERRIDE_NAMES:
        print("Override class names:", OVERRIDE_CLASS_NAMES)

    out = {
        "model_path": str(MODEL_PATH),
        "source_dir": str(image_dir),
        "gt_dir": str(gt_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_count": len(image_paths),
        "class_names": class_names,
        "distance_method": {
            "box_frac": BOX_FRAC,
            "percentile_q": BOX_Q,
            "subsample": BOX_SUBSAMP,
            "vehicle_mode": "bottom",
            "other_mode": "center",
            "max_depth_m": MAX_DEPTH_M,
        },
        "images": []
    }

    for idx, img_path in enumerate(image_paths, 1):
        vis = cv2.imread(str(img_path))  # BGR color not RGB
        with Image.open(img_path) as im:
            width, height = im.size

        gt_path = kitti_gt_path_from_image_name(img_path.name, gt_dir)
        if gt_path is None or not gt_path.exists():
            depth_gt = None
            gt_path_str = None
        else:
            depth_gt = load_kitti_depth_meters(gt_path)
            if depth_gt.shape[:2] != (height, width):
                depth_gt = cv2.resize(depth_gt, (width, height), interpolation=cv2.INTER_NEAREST)
            depth_gt = np.clip(depth_gt, 0.0, MAX_DEPTH_M)
            gt_path_str = str(gt_path)

        results = model.predict(
            source=str(img_path),
            conf=CONF_THRES,
            iou=IOU_THRES,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False
        )

        r = results[0]
        dets = []

        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy()

            for (x1f, y1f, x2f, y2f), conf, cls_id in zip(boxes_xyxy, confs, clss):
                cls_id_int = int(cls_id)
                name = class_names[cls_id_int] if 0 <= cls_id_int < len(class_names) else str(cls_id_int)

                # Convert to ints for slicing 
                x1 = int(np.floor(x1f))
                y1 = int(np.floor(y1f))
                x2 = int(np.ceil(x2f))
                y2 = int(np.ceil(y2f))

                mode = "bottom" if is_bottom_region(name) else "center"

                gt_distance_m = None
                if depth_gt is not None:
                    gt_distance_m = compute_box_distance(
                        depth_gt, x1, y1, x2, y2,
                        frac=BOX_FRAC,
                        mode=mode,
                        q=BOX_Q,
                        subsample=BOX_SUBSAMP
                    )

                dets.append({
                    "class_id": cls_id_int,
                    "class_name": name,
                    "eval_category": map_eval_category(name),
                    "confidence": float(conf),
                    "bbox_xyxy": [float(x1f), float(y1f), float(x2f), float(y2f)],
                    "bbox_xywh": xyxy_to_xywh(float(x1f), float(y1f), float(x2f), float(y2f)),
                    "bbox_xyxy_int": [int(x1), int(y1), int(x2), int(y2)],
                    "distance_mode": mode,
                    "gt_distance_m": None if gt_distance_m is None else float(gt_distance_m),
                })

                # DRAW PER DETECTION
                if vis is not None:
                    H_vis, W_vis = vis.shape[:2]

                    # clamp 
                    x1c = max(0, min(W_vis - 1, x1))
                    y1c = max(0, min(H_vis - 1, y1))
                    x2c = max(0, min(W_vis, x2))
                    y2c = max(0, min(H_vis, y2))

                    # for drawing, OpenCV wants bottom-right corner inside image
                    x2d = max(x1c, x2c - 1)
                    y2d = max(y1c, y2c - 1)

                    cv2.rectangle(vis, (x1c, y1c), (x2d, y2d), (0, 255, 0), 2)

                    label = f"{name} {float(conf):.2f}"
                    if gt_distance_m is not None:
                        label += f" {gt_distance_m:.2f}m"

                    cv2.putText(vis, label, (x1c, max(0, y1c - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out["images"].append({
            "file_name": img_path.name,
            "file_path": str(img_path),
            "width": width,
            "height": height,
            "gt_depth_path": gt_path_str,
            "detections": dets
        })

        # SAVE ONCE PER IMAGE (after all boxes drawn)
        if vis is not None:
            out_img_path = OUT_IMG_DIR / img_path.name
            cv2.imwrite(str(out_img_path), vis)

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)}")

    os.makedirs(Path(OUT_JSON).parent, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {OUT_JSON}")
    print(f"Saved overlays to: {OUT_IMG_DIR}")


if __name__ == "__main__":
    main()
