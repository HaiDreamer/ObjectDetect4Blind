# -*- coding: utf-8 -*-
"""
YOLO detect + per-bbox distance (metric depth) + per-class colors + 2x2 demo grids.

Outputs:
  - predictions_yolo.json (with distance_m and color_bgr per box)
  - demo_best_2x2.png and demo_worst_2x2.png (colored boxes + labels)
"""

import json
from pathlib import Path
import colorsys
import numpy as np
import cv2
from ultralytics import YOLO

# =============== CONFIG ===============

INPUT_IMAGES = [
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_5709_HEIC.rf.e75cc6ec2b76513771e73dc56bebc0dc_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds2__IMG_20181029_154054_jpg.rf.dae92b92106f3ea7c1411f10be7cb0c8_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_9809_frame_01470_d36723_jpg.rf.ac7ff71efce468facd0364325990d07d_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_5697_HEIC.rf.515ea48b7f6902424db7b66a207e90ac_vis.jpg",
]

OUTPUT_DIR = Path(r"C:\Python\ObjectDetect4Blind\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_JSON = OUTPUT_DIR / "predictions_yolo.json"

BEST_MODEL_PATH = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\yolov8l.pt")
WORST_MODEL_PATH = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\yolov8n.pt")

TARGET_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "tree",
    "perdestrian_crossing",
    "electric_pole",
]

# where metric depth npy files are saved
DEPTH_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_metric_depth")
MAX_DEPTH_M = 80.0  # clamp depth (meters)

# >>> NEW: apply a global scale to all final distances (e.g., calibrate or halve)
DISTANCE_SCALE = 0.5   # reduce reported distances by 50%

# =============== HELPERS ===============

def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def get_target_class_ids(model, target_names):
    names_dict = model.names  # {id: 'class_name'}
    norm_target = [normalize_name(t) for t in target_names]
    target_ids = []
    for cls_id, cls_name in names_dict.items():
        if normalize_name(str(cls_name)) in norm_target:
            target_ids.append(cls_id)
    target_ids = sorted(set(target_ids))
    if not target_ids:
        print(f"[WARNING] No matching classes in model.names for {target_names}")
    else:
        print(f"[INFO] Using class_ids = {target_ids}")
    return target_ids

def run_and_collect(model, image_path: str, class_ids):
    """Run YOLO on an image, return list of boxes (filtered by class_ids if provided)."""
    classes_arg = class_ids if class_ids else None
    results = model(image_path, classes=classes_arg, conf=0.25, verbose=False)
    res = results[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    out = []
    for box_xyxy, c, cf in zip(xyxy, cls, conf):
        class_id = int(c)
        class_name = str(model.names.get(class_id, class_id))
        x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
        out.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": float(cf),
            "xyxy": [x1, y1, x2, y2],
        })
    return out

# =============== COLOR UTILS (distinct per-class colors) ===============

def _hsv_to_bgr(h: float, s: float, v: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(b * 255 + 0.5), int(g * 255 + 0.5), int(r * 255 + 0.5))

def _generate_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    colors, hue, step = [], 0.0, 0.61803398875
    for _ in range(n):
        colors.append(_hsv_to_bgr(hue % 1.0, 0.75, 0.95))
        hue += step
    return colors

def _luminance_bgr(bgr: tuple[int, int, int]) -> float:
    b, g, r = [c / 255.0 for c in bgr]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def build_class_color_map(class_names: list[str]) -> dict[str, tuple[int, int, int]]:
    colors = _generate_distinct_colors(len(class_names))
    return {name: colors[i] for i, name in enumerate(sorted(class_names))}

# =============== DISTANCE HELPERS (metric depth) ===============

def load_depth_map_for_image(img_path: Path) -> np.ndarray | None:
    depth_npy = DEPTH_DIR / f"{img_path.stem}_raw_depth_meter.npy"
    if not depth_npy.exists():
        print(f"[DEPTH] Missing depth file: {depth_npy}")
        return None
    depth_map_m = np.load(str(depth_npy)).astype(np.float32)
    if depth_map_m.ndim == 3:
        depth_map_m = depth_map_m.squeeze()
    depth_map_m = np.clip(depth_map_m, 1e-3, MAX_DEPTH_M)
    return depth_map_m

def _compute_box_distance(depth_map_m: np.ndarray,
                          x1: int, y1: int, x2: int, y2: int,
                          frac: float = 0.3, mode: str = "center") -> float | None:
    """Median depth in a region of the box."""
    H, W = depth_map_m.shape[:2]
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W,     x2)); y2 = max(0, min(H,     y2))
    if x2 <= x1 or y2 <= y1: return None
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: return None

    if mode == "bottom":
        ch = int(h * frac)
        if ch <= 0: return None
        y_start = max(y1, y2 - ch)
        center_band_width = int(w * 0.5)
        if center_band_width <= 0: return None
        cx = (x1 + x2) // 2
        x_start = max(x1, cx - center_band_width // 2)
        x_end   = min(x2, x_start + center_band_width)
        if x_end <= x_start: return None
        patch = depth_map_m[y_start:y2, x_start:x_end]
    else:
        cw, ch = int(w * frac), int(h * frac)
        if cw <= 0 or ch <= 0: return None
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        cx1 = max(0, cx - cw // 2); cy1 = max(0, cy - ch // 2)
        cx2 = min(W, cx1 + cw);     cy2 = min(H, cy1 + ch)
        if cx2 <= cx1 or cy2 <= cy1: return None
        patch = depth_map_m[cy1:cy2, cx1:cx2]

    if patch.size == 0: return None
    valid = patch[patch > 0]
    if valid.size == 0: return None
    return float(np.median(valid))

def attach_distances_to_boxes(img_path: Path, boxes: list[dict]) -> None:
    """
    For all boxes on this image:
      - load depth map (if available)
      - compute distance_m using _compute_box_distance
      - apply DISTANCE_SCALE to final distance
    """
    depth_map = load_depth_map_for_image(img_path)
    if depth_map is None:
        for box in boxes:
            box["distance_m"] = None
        return

    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        name_l = str(box["class_name"]).lower()
        is_bottom = any(k in name_l for k in ("car", "bicycle", "truck", "motorbike", "motorcycle", "bus"))
        mode = "bottom" if is_bottom else "center"
        dist = _compute_box_distance(depth_map, x1, y1, x2, y2, frac=0.3, mode=mode)
        # >>> APPLY global scale (e.g., halve distances)
        box["distance_m"] = None if dist is None else float(dist * DISTANCE_SCALE)

# =============== DRAWING (colored boxes + readable labels) ===============

def _draw_label_bg(img, x, y, text, bg_bgr):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, 0.5, 1)
    x1, y1 = x, max(0, y - th - bl - 3)
    x2, y2 = x + tw + 6, y
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_bgr, thickness=-1)
    txt_color = (0, 0, 0) if _luminance_bgr(bg_bgr) > 0.5 else (255, 255, 255)
    cv2.putText(img, text, (x + 3, y - 4), font, 0.5, txt_color, 1, cv2.LINE_AA)

def _draw_boxes_on_image(image: np.ndarray, boxes: list[dict], color_map: dict[str, tuple[int,int,int]]) -> np.ndarray:
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box["xyxy"])
        cls_name = box["class_name"]
        conf = box["confidence"]
        dist = box.get("distance_m", None)
        color = color_map.get(cls_name, (0, 0, 0))
        label = f"{cls_name} {conf:.2f}"
        if dist is not None:
            label += f" {dist:.2f}m"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        _draw_label_bg(img, x1, y1, label, color)
    return img

def create_2x2_grid(pred_list_for_model: list[dict], out_path: Path, color_map: dict[str, tuple[int,int,int]]):
    print(f"[GRID] Creating 2x2 demo grid at: {out_path}")
    images_drawn = []
    for entry in pred_list_for_model[:4]:
        img_path = Path(entry["image_path"])
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[GRID] Could not read image: {img_path}")
            continue
        img = _draw_boxes_on_image(img, entry["boxes"], color_map)
        images_drawn.append(img)
    if not images_drawn:
        print("[GRID] No images to draw.")
        return
    while len(images_drawn) < 4:
        images_drawn.append(images_drawn[-1].copy())
    h0, w0 = images_drawn[0].shape[:2]
    norm_imgs = [cv2.resize(im, (w0, h0)) for im in images_drawn]
    top = np.hstack(norm_imgs[0:2]); bottom = np.hstack(norm_imgs[2:4])
    grid = np.vstack((top, bottom))
    cv2.imwrite(str(out_path), grid)
    print(f"[GRID] Saved demo grid: {out_path}")

# =============== MAIN ===============

def main():
    print("[STEP 1] Loading models...")
    best_model = YOLO(str(BEST_MODEL_PATH))
    worst_model = YOLO(str(WORST_MODEL_PATH))

    print("[STEP 2] Mapping class names -> ids...")
    best_class_ids = get_target_class_ids(best_model, TARGET_CLASS_NAMES)
    worst_class_ids = get_target_class_ids(worst_model, TARGET_CLASS_NAMES)

    # Consistent color map across both models
    all_class_names = set(str(n) for n in best_model.names.values()) | set(str(n) for n in worst_model.names.values())
    COLOR_MAP = build_class_color_map(sorted(all_class_names))

    predictions = {"best": [], "worst": []}

    print("[STEP 3] Running inference, computing distances, and collecting predictions...")
    for img_path_str in INPUT_IMAGES:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[WARNING] Image not found: {img_path}")
            continue
        print(f"  - {img_path}")

        best_boxes = run_and_collect(best_model, str(img_path), best_class_ids)
        attach_distances_to_boxes(img_path, best_boxes)
        predictions["best"].append({"image_path": str(img_path), "boxes": best_boxes})

        worst_boxes = run_and_collect(worst_model, str(img_path), worst_class_ids)
        attach_distances_to_boxes(img_path, worst_boxes)
        predictions["worst"].append({"image_path": str(img_path), "boxes": worst_boxes})

    # attach BGR color to each box in JSON
    def _attach_colors(pred):
        for entry in pred:
            for box in entry["boxes"]:
                c = COLOR_MAP.get(box["class_name"], (0, 0, 0))
                box["color_bgr"] = [int(c[0]), int(c[1]), int(c[2])]
    _attach_colors(predictions["best"])
    _attach_colors(predictions["worst"])

    with open(PRED_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"[STEP 4 DONE] Saved predictions JSON (with scaled distances & colors) to: {PRED_JSON}")

    create_2x2_grid(predictions["best"],  OUTPUT_DIR / "demo_best_2x2.png",  COLOR_MAP)
    create_2x2_grid(predictions["worst"], OUTPUT_DIR / "demo_worst_2x2.png", COLOR_MAP)

if __name__ == "__main__":
    main()
