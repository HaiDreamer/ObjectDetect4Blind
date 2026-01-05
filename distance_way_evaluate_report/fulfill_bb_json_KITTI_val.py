import json
import os
from pathlib import Path
from datetime import datetime

from PIL import Image
from ultralytics import YOLO  # pip install ultralytics


# --- USER PATHS ---
MODEL_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2.pt"
IMAGE_DIR  = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\image"

# Output JSON 
OUT_JSON   = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\bb_json_KITTI_val.json"

# Optional: override class names (ONLY if your model class order matches this list)
OVERRIDE_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "bus", "truck",
    "traffic light", "tree", "perdestrian_crossing_sign", "electric_pole"
]
USE_OVERRIDE_NAMES = False  # set True if you want to force the above names

# Inference settings
CONF_THRES = 0.25
IOU_THRES  = 0.7
IMGSZ      = 640
DEVICE     = None  # e.g. "cpu" or 0 for GPU; None = auto


def get_model_names(model) -> list[str]:
    """
    Ultralytics model.names can be a dict {id: name} or a list.
    """
    names = model.names
    if isinstance(names, dict):
        # ensure sorted by class id
        return [names[i] for i in sorted(names.keys())]
    return list(names)


def xyxy_to_xywh(x1, y1, x2, y2):
    return [x1, y1, (x2 - x1), (y2 - y1)]


def main():
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR not found: {image_dir}")

    # Collect images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])
    if not image_paths:
        raise RuntimeError(f"No images found under: {image_dir}")

    # Load model once
    model = YOLO(MODEL_PATH)

    # Determine class names to write
    model_names = get_model_names(model)
    class_names = OVERRIDE_CLASS_NAMES if USE_OVERRIDE_NAMES else model_names

    # (Recommended) sanity check
    print("Model class names:", model_names)
    if USE_OVERRIDE_NAMES:
        print("Override class names:", OVERRIDE_CLASS_NAMES)

    out = {
        "model_path": str(MODEL_PATH),
        "source_dir": str(image_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "image_count": len(image_paths),
        "class_names": class_names,
        "images": []
    }

    for idx, img_path in enumerate(image_paths, 1):
        # Read size (no need to load full image into numpy)
        with Image.open(img_path) as im:
            width, height = im.size

        # Run inference on this image
        # Ultralytics Results: r.boxes.xyxy, r.boxes.conf, r.boxes.cls :contentReference[oaicite:2]{index=2}
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

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, clss):
                cls_id_int = int(cls_id)
                name = class_names[cls_id_int] if 0 <= cls_id_int < len(class_names) else str(cls_id_int)

                # Keep a reasonable number of decimals
                x1f, y1f, x2f, y2f = [float(v) for v in (x1, y1, x2, y2)]
                dets.append({
                    "class_id": cls_id_int,
                    "class_name": name,
                    "confidence": float(conf),
                    "bbox_xyxy": [x1f, y1f, x2f, y2f],               # [xmin, ymin, xmax, ymax]
                    "bbox_xywh": xyxy_to_xywh(x1f, y1f, x2f, y2f),  # [xmin, ymin, w, h]
                })

        out["images"].append({
            "file_name": img_path.name,
            "file_path": str(img_path),
            "width": width,
            "height": height,
            "detections": dets
        })

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"Processed {idx}/{len(image_paths)}")

    # Write JSON
    os.makedirs(Path(OUT_JSON).parent, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
