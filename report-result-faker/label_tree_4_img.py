# -*- coding: utf-8 -*-
"""
Trees JSON + 2x2 visualization

- Reads 4 images + 4 YOLO detection .txt (cls cx cy w h [conf])
- Loads metric-depth npy per image:
    DEPTH_DIR / f"{Path(image).stem}_raw_depth_meter.npy"
  (also tries stem without '_vis')
- Builds one JSON with all tree boxes:
    class_name="tree", confidence~[0.70,0.90], distance_m scaled by 0.5x
- Saves per-image overlays with boxes + labels
- Saves a 2x2 grid (square tiles with padding, no distortion)

Adjust DISTANCE_SCALE or TILE_SIZE as you like.
"""

from pathlib import Path
import json
import hashlib
import numpy as np
import cv2

# ---------------- CONFIG ----------------
INPUT_IMAGES = [
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_5709_HEIC.rf.e75cc6ec2b76513771e73dc56bebc0dc_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds2__IMG_20181029_154054_jpg.rf.dae92b92106f3ea7c1411f10be7cb0c8_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_9809_frame_01470_d36723_jpg.rf.ac7ff71efce468facd0364325990d07d_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_5697_HEIC.rf.515ea48b7f6902424db7b66a207e90ac_vis.jpg",
]

# NOTE: your original list missed a comma between item 2 and 3; fixed below.
LABELS = [
    r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\Tree_detect\labels\836592ef-ds1__ds1__IMG_5709_HEIC.rf.e75cc6ec2b76513771e73dc56bebc0dc_vis.txt",
    r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\Tree_detect\labels\f707fa15-ds1__ds2__IMG_20181029_154054_jpg.rf.dae92b92106f3ea7c1411f10be7cb0c8_vis.txt",
    r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\Tree_detect\labels\9501ce59-ds1__ds1__IMG_9809_frame_0147_9TYwLgS.rf.ac7ff71efce468facd0364325990d07d_vis.txt",
    r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\Tree_detect\labels\be5d6072-ds1__ds1__IMG_5697_HEIC.rf.515ea48b7f6902424db7b66a207e90ac_vis.txt",
]

# If you know the class id for "tree", set it here (e.g., [0]).
# If labels contain only trees, leave as None.
TREE_CLASS_IDS = None

DEPTH_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_metric_depth")
MAX_DEPTH_M = 80.0
DISTANCE_SCALE = 0.5          # halve distances
CONF_MIN, CONF_MAX = 0.70, 0.90

OUT_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_tree_json")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUT_DIR / "trees_with_distance.json"

# Visualization
BOX_COLOR = (0, 200, 0)       # BGR, green for trees
FONT = cv2.FONT_HERSHEY_SIMPLEX
TILE_SIZE = 640               # each tile in the 2x2 grid will be TILE_SIZE x TILE_SIZE
GRID_PATH = OUT_DIR / "trees_demo_2x2.png"


# --------------- HELPERS ----------------
def _stable_confidence(image_path: Path, box_xyxy: tuple[float, float, float, float]) -> float:
    """
    Stable pseudo-random confidence in [CONF_MIN, CONF_MAX] per box.
    """
    x1, y1, x2, y2 = box_xyxy
    key = f"{image_path.stem}|{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}".encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    frac = int(h[:12], 16) / float(0xFFFFFFFFFFFF)
    val = CONF_MIN + (CONF_MAX - CONF_MIN) * frac
    return round(val, 3)


def _load_depth_map(img_path: Path) -> np.ndarray | None:
    """
    Load depth map in meters; tries both <stem> and <stem without '_vis'>.
    """
    stem = img_path.stem
    candidates = [DEPTH_DIR / f"{stem}_raw_depth_meter.npy"]
    if stem.endswith("_vis"):
        candidates.append(DEPTH_DIR / f"{stem[:-4]}_raw_depth_meter.npy")

    for p in candidates:
        if p.exists():
            arr = np.load(str(p)).astype(np.float32)
            return np.clip(arr.squeeze(), 1e-3, MAX_DEPTH_M)
    print(f"[DEPTH] Missing depth .npy for: {img_path.name}")
    return None


def _compute_box_distance(depth_map: np.ndarray, xyxy: tuple[int, int, int, int],
                          frac: float = 0.3, mode: str = "center") -> float | None:
    """
    Median depth in a subregion of the bbox.
    For trees (tall objects), 'center' typically works well.
    """
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W,     int(x2)))
    y2 = max(0, min(H,     int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None

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
        patch = depth_map[y_start:y2, x_start:x_end]
    else:
        cw, ch = int(w * frac), int(h * frac)
        if cw <= 0 or ch <= 0: return None
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        cx1 = max(0, cx - cw // 2); cy1 = max(0, cy - ch // 2)
        cx2 = min(W, cx1 + cw);     cy2 = min(H, cy1 + ch)
        if cx2 <= cx1 or cy2 <= cy1: return None
        patch = depth_map[cy1:cy2, cx1:cx2]

    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _parse_yolo_det_file(label_file: Path, img_w: int, img_h: int,
                         keep_class_ids: list[int] | None) -> list[tuple[float, float, float, float]]:
    """
    Parse YOLO detection .txt -> list of pixel xyxy boxes.
    Accepts lines: cls cx cy w h [conf]  (all normalized 0..1).
    """
    boxes = []
    if not label_file.exists():
        print(f"[LABEL] Not found: {label_file}")
        return boxes

    with open(label_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
        except ValueError:
            continue
        if keep_class_ids is not None and cls not in keep_class_ids:
            continue

        try:
            cx, cy, w, h = map(float, parts[1:5])
        except Exception:
            continue

        px, py = cx * img_w, cy * img_h
        pw, ph = w * img_w, h * img_h
        x1 = max(0.0, px - pw / 2.0)
        y1 = max(0.0, py - ph / 2.0)
        x2 = min(float(img_w - 1), px + pw / 2.0)
        y2 = min(float(img_h - 1), py + ph / 2.0)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def _draw_label_with_bg(img, x, y, text, bg_bgr):
    (tw, th), bl = cv2.getTextSize(text, FONT, 0.6, 1)
    x1, y1 = x, max(0, y - th - bl - 4)
    x2, y2 = x + tw + 8, y
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_bgr, -1)
    # contrast-aware text color
    b, g, r = bg_bgr
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    txt_color = (0, 0, 0) if lum > 0.5 else (255, 255, 255)
    cv2.putText(img, text, (x + 4, y - 4), FONT, 0.6, txt_color, 1, cv2.LINE_AA)


def _overlay_boxes(img: np.ndarray, boxes_xyxy: list[tuple[float, float, float, float]],
                   confidences: list[float], distances: list[float | None]) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2), conf, dist in zip(boxes_xyxy, confidences, distances):
        x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), BOX_COLOR, 2)
        label = f"tree {conf:.2f}"
        if dist is not None:
            label += f" {dist:.2f}m"
        _draw_label_with_bg(out, x1i, y1i, label, BOX_COLOR)
    return out


def _resize_to_square_with_padding(img: np.ndarray, size: int, pad_color=(0, 0, 0)) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)


def _make_2x2_grid(images: list[np.ndarray], tile_size: int) -> np.ndarray:
    tiles = [ _resize_to_square_with_padding(im, tile_size) for im in images ]
    while len(tiles) < 4:
        tiles.append(tiles[-1].copy())
    top = np.hstack(tiles[0:2])
    bot = np.hstack(tiles[2:4])
    return np.vstack((top, bot))


# ---------------- MAIN ------------------
def main():
    results = {"images": []}
    viz_tiles = []

    for img_path_str, lbl_path_str in zip(INPUT_IMAGES, LABELS):
        img_path = Path(img_path_str)
        lbl_path = Path(lbl_path_str)

        if not img_path.exists():
            print(f"[IMAGE] Missing: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[IMAGE] Could not read: {img_path}")
            continue
        H, W = img.shape[:2]

        # Parse labels -> boxes
        boxes_xyxy = _parse_yolo_det_file(lbl_path, W, H, TREE_CLASS_IDS)

        # Load depth map
        depth = _load_depth_map(img_path)

        image_entry = {
            "image_path": str(img_path),
            "label_path": str(lbl_path),
            "boxes": []
        }

        # Compute distance + confidence + build overlay vectors
        dists = []
        confs = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            if depth is not None:
                dist = _compute_box_distance(depth, (int(x1), int(y1), int(x2), int(y2)),
                                             frac=0.3, mode="center")
                dist = None if dist is None else float(dist * DISTANCE_SCALE)
            else:
                dist = None

            conf = _stable_confidence(img_path, (x1, y1, x2, y2))
            dists.append(dist)
            confs.append(conf)

            image_entry["boxes"].append({
                "class_name": "tree",
                "confidence": conf,
                "distance_m": dist,
                "xyxy": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2), 2), round(float(y2), 2)]
            })

        results["images"].append(image_entry)

        # Per-image overlay (optional but handy)
        overlay = _overlay_boxes(img, boxes_xyxy, confs, dists)
        out_img_path = OUT_DIR / f"{img_path.stem}_tree_overlay.png"
        cv2.imwrite(str(out_img_path), overlay)
        viz_tiles.append(overlay)
        print(f"[OVERLAY] Saved {out_img_path}")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] Saved JSON: {OUTPUT_JSON}")

    # 2x2 grid
    if viz_tiles:
        grid = _make_2x2_grid(viz_tiles[:4], TILE_SIZE)
        cv2.imwrite(str(GRID_PATH), grid)
        print(f"[GRID] Saved 2x2 visualization: {GRID_PATH}")
    else:
        print("[GRID] No tiles to render.")

if __name__ == "__main__":
    main()
