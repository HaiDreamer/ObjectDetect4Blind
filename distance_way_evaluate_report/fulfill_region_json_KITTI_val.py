import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
from ultralytics import YOLO


# --- USER PATHS ---
MODEL_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg.pt"
IMAGE_DIR  = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\image"
OUT_JSON   = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\segment_json_KITTI_val.json"

# Your model labels (must match class-id order in the model)
OVERRIDE_CLASS_NAMES = ["crosswalk", "tree line", "sidewalk", "stairs"]
USE_OVERRIDE_NAMES = True  # set False if you want to use model.names instead

# Inference settings
CONF_THRES = 0.25
IOU_THRES  = 0.7
IMGSZ      = 640
DEVICE     = None  # e.g. "cpu" or 0 for GPU; None=auto


def get_model_names(model) -> list[str]:
    names = model.names
    # model.names can be dict {id:name} or list
    if isinstance(names, dict):
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def xyxy_to_xywh(x1, y1, x2, y2):
    return [x1, y1, (x2 - x1), (y2 - y1)]


def to_list_2d(arr, decimals=2):
    """arr: (M,2) -> list[[x,y],...] with rounding to reduce JSON size"""
    if arr is None:
        return []
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return []
    if decimals is not None:
        a = np.round(a, decimals)
    return a.tolist()


def main():
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {image_dir}")

    model = YOLO(MODEL_PATH)

    model_names = get_model_names(model)
    class_names = OVERRIDE_CLASS_NAMES if USE_OVERRIDE_NAMES else model_names

    print("Model class names:", model_names)
    if USE_OVERRIDE_NAMES:
        print("Using override class names:", class_names)
        if len(model_names) != len(class_names):
            print("WARNING: override class count != model class count. "
                  "Make sure class-id order matches your training!")

    out = {
        "task": "segmentation",
        "model_path": str(MODEL_PATH),
        "source_dir": str(image_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "class_names": class_names,
        "images": []
    }

    # stream=True => generator of Results (memory efficient for folder inference) :contentReference[oaicite:3]{index=3}
    results = model.predict(
        source=str(image_dir),
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMGSZ,
        device=DEVICE,
        stream=True,
        verbose=False
    )

    count = 0
    for r in results:
        count += 1

        # r.path and r.orig_shape come with Results object :contentReference[oaicite:4]{index=4}
        file_path = str(r.path)
        h, w = int(r.orig_shape[0]), int(r.orig_shape[1])

        instances = []

        has_boxes = (r.boxes is not None) and (len(r.boxes) > 0)
        has_masks = (r.masks is not None)

        if has_boxes and has_masks:
            # Boxes: xyxy/conf/cls :contentReference[oaicite:5]{index=5}
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy()

            # Masks: xy (pixel polygons), xyn (normalized polygons) :contentReference[oaicite:6]{index=6}
            polys_xy  = r.masks.xy
            polys_xyn = r.masks.xyn

            n = min(len(boxes_xyxy), len(polys_xy), len(polys_xyn))

            for i in range(n):
                x1, y1, x2, y2 = [float(v) for v in boxes_xyxy[i]]
                conf = float(confs[i])
                cls_id = int(clss[i])

                name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)

                instances.append({
                    "class_id": cls_id,
                    "class_name": name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "bbox_xywh": xyxy_to_xywh(x1, y1, x2, y2),
                    # region polygons (mask boundary)
                    "segmentation_xy":  to_list_2d(polys_xy[i], decimals=2),   # pixel coords
                    "segmentation_xyn": to_list_2d(polys_xyn[i], decimals=4),  # normalized coords
                })

        out["images"].append({
            "file_name": Path(file_path).name,
            "file_path": file_path,
            "width": w,
            "height": h,
            "instances": instances
        })

        if count % 50 == 0:
            print(f"Processed {count} images...")

    out["image_count"] = count

    os.makedirs(Path(OUT_JSON).parent, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
