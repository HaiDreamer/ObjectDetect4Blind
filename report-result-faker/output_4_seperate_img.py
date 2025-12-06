# -*- coding: utf-8 -*-
"""
Side-by-side "Labels vs Predictions" canvas with BOTH segmentation polygons and
object-detection bounding boxes. Each tile is letterboxed to a SQUARE size.

Left  (2×2): seg + boxes (NO text)
Right (2×2): seg + boxes + class + confidence + distance (from JSONs)

Also reads a second JSON containing additional "tree" detections and overlays
their labels on the Predictions side.

Outputs:
  <OUTPUT_DIR>/labels_vs_predictions_square_2x2.png
"""

from pathlib import Path
import json
import cv2
import numpy as np
import re
from typing import List, Dict, Any, Tuple

# =========================
# CONFIG (adjust paths)
# =========================
SEG_OUT_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_seg_distance")
DET_JSON    = Path(r"C:\Python\ObjectDetect4Blind\output\predictions_yolo.json")
DET_GROUP   = "best"   # "best" or "worst"

PRED_OUT_DIR = SEG_OUT_DIR / "predictions_single"   # <--- NEW: folder for 4 images

# Extra detections JSON (the additional file you provided)
# Change this path to where you saved that JSON.
TREE_JSON   = Path(r"C:\Python\ObjectDetect4Blind\output_tree_json\trees_with_distance.json")

# Use explicit list so order matches your example
SEG_JSONS: List[str] = [
    str(SEG_OUT_DIR / "ds1__ds1__IMG_5709_HEIC.rf.e75cc6ec2b76513771e73dc56bebc0dc_vis_segments.json"),
    str(SEG_OUT_DIR / "ds1__ds2__IMG_20181029_154054_jpg.rf.dae92b92106f3ea7c1411f10be7cb0c8_vis_segments.json"),
    str(SEG_OUT_DIR / "ds1__ds1__IMG_9809_frame_01470_d36723_jpg.rf.ac7ff71efce468facd0364325990d07d_vis_segments.json"),
    str(SEG_OUT_DIR / "ds1__ds1__IMG_5697_HEIC.rf.515ea48b7f6902424db7b66a207e90ac_vis_segments.json"),
]


# Square tile size
TILE = 640  # each tile becomes TILE x TILE

# Visuals
FILL_ALPHA = 0.35
OUT_THICK  = 2
DIVIDER_W  = 24
FOOTER_H   = 80

# =========================
# Helpers
# =========================
def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _norm_stem(p: Path) -> str:
    """Match stems robustly: strip a trailing '_vis' if present."""
    s = p.stem
    return re.sub(r"_vis$", "", s)

def _resize_letterbox_square(img: np.ndarray, size: int, pad_color=(0,0,0)) -> np.ndarray:
    """Resize preserving aspect, then pad with borders to make a 'size x size' square."""
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    nw, nh = max(1, int(round(w*s))), max(1, int(round(h*s)))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    x0, y0 = (size - nw)//2, (size - nh)//2
    canvas[y0:y0+nh, x0:x0+nw] = rs
    return canvas

def _polygon_to_bbox(poly_pts: np.ndarray) -> Tuple[int,int,int,int]:
    pts = poly_pts.reshape(-1, 2)
    return int(np.min(pts[:,0])), int(np.min(pts[:,1])), int(np.max(pts[:,0])), int(np.max(pts[:,1]))

def _draw_seg_geometry(img: np.ndarray, segments: List[Dict[str, Any]]) -> np.ndarray:
    """Draw filled polygons + outlines (no text)."""
    overlay = img.copy()
    for seg in segments:
        color = tuple(int(c) for c in seg.get("color_bgr", (0,0,0)))
        poly  = np.array(seg["polygon"], dtype=np.int32).reshape(-1,1,2)
        cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(overlay, [poly], True, color, OUT_THICK, cv2.LINE_AA)
    out = img.copy()
    cv2.addWeighted(overlay, FILL_ALPHA, out, 1.0-FILL_ALPHA, 0, out)
    return out

def _draw_boxes_geometry(img: np.ndarray, boxes: List[Dict[str,Any]]) -> np.ndarray:
    """Draw detection rectangles (no text)."""
    out = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b["xyxy"])
        color = tuple(int(c) for c in b.get("color_bgr", (255,255,255)))
        cv2.rectangle(out, (x1,y1), (x2,y2), color, OUT_THICK)
    return out

def _put_label(img: np.ndarray, text: str, anchor: Tuple[int,int], bg_bgr, scale: float, thick: int):
    """Solid bg + contrast text."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = anchor
    cv2.rectangle(img, (x-4, y-th-bl-6), (x+tw+4, y+2), bg_bgr, -1)
    b,g,r = bg_bgr
    lum = (0.2126*r + 0.7152*g + 0.0722*b) / 255.0
    txt_col = (0,0,0) if lum > 0.5 else (255,255,255)
    cv2.putText(img, text, (x, y-6), font, scale, txt_col, thick, cv2.LINE_AA)

def _draw_predictions(img: np.ndarray, segments: List[Dict[str,Any]], boxes: List[Dict[str,Any]]) -> np.ndarray:
    """Draw seg + box geometry, then overlay text (name/conf/distance) for both."""
    H, W = img.shape[:2]
    base = _draw_seg_geometry(img, segments)
    base = _draw_boxes_geometry(base, boxes)

    # scale text to image height so it looks consistent after square letterbox
    seg_scale = max(0.55, 0.0011 * H)
    seg_thick = max(2, int(round(0.0020 * H)))
    box_scale = seg_scale
    box_thick = seg_thick

    # SEGMENT labels (class, conf, median distance)
    for s in segments:
        color = tuple(int(c) for c in s.get("color_bgr", (0,0,0)))
        cls   = str(s.get("class_name",""))
        conf  = s.get("confidence", None)
        med   = (s.get("stats") or {}).get("median_m", None)
        parts = [cls]
        if conf is not None: parts.append(f"{float(conf):.2f}")
        if med  is not None: parts.append(f"{float(med):.2f}m")
        label = " ".join(parts)

        poly = np.array(s["polygon"], dtype=np.int32).reshape(-1,1,2)
        cx, cy = poly.reshape(-1,2).mean(axis=0).astype(int)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, seg_scale, seg_thick)
        x = int(max(0, min(W - tw - 8, cx - tw//2)))
        y = int(max(th + bl + 8, min(H - 8, cy)))
        _put_label(base, label, (x,y), color, seg_scale, seg_thick)

    # BOX labels (class, conf, distance)
    for b in boxes:
        color = tuple(int(c) for c in b.get("color_bgr", (255,255,255)))
        cls   = str(b.get("class_name",""))
        conf  = b.get("confidence", None)
        dist  = b.get("distance_m", None)
        parts = [cls]
        if conf is not None: parts.append(f"{float(conf):.2f}")
        if dist is not None: parts.append(f"{float(dist):.2f}m")
        label = " ".join(parts)

        x1, y1, x2, y2 = map(int, b["xyxy"])
        lx, ly = x1 + 4, max(y1, 12 + 6 + 4)
        _put_label(base, label, (lx, ly), color, box_scale, box_thick)

    return base

def _grid_2x2_square(images: List[np.ndarray], size: int) -> np.ndarray:
    imgs = images[:4]
    while len(imgs) < 4:
        imgs.append(imgs[-1].copy())
    tiles = [_resize_letterbox_square(im, size) for im in imgs]
    top = np.hstack(tiles[:2])
    bot = np.hstack(tiles[2:4])
    return np.vstack((top, bot))

def _compose_canvas(left_grid: np.ndarray, right_grid: np.ndarray) -> np.ndarray:
    Lh, Lw = left_grid.shape[:2]
    Rh, Rw = right_grid.shape[:2]
    assert Lh == Rh, "grids should have same height"
    H = Lh
    W = Lw + DIVIDER_W + Rw
    body = np.full((H, W, 3), (0,0,0), dtype=np.uint8)
    body[:, 0:Lw] = left_grid
    body[:, Lw+DIVIDER_W:Lw+DIVIDER_W+Rw] = right_grid

    canvas = np.full((H + FOOTER_H, W, 3), (255,255,255), dtype=np.uint8)
    canvas[:H, :W] = body

    # footer titles
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 1.0, 3
    # left
    tL = "Labels"
    (twL, thL), bl = cv2.getTextSize(tL, font, scale, thick)
    xL = (Lw - twL)//2
    y  = H + (FOOTER_H + thL)//2
    cv2.putText(canvas, tL, (xL, y), font, scale, (0,0,0), thick, cv2.LINE_AA)
    # right
    tR = "Predictions"
    (twR, thR), bl = cv2.getTextSize(tR, font, scale, thick)
    xR = Lw + DIVIDER_W + (Rw - twR)//2
    cv2.putText(canvas, tR, (xR, y), font, scale, (0,0,0), thick, cv2.LINE_AA)
    return canvas

# =========================
# Load detections (map by image stem)
# =========================
def _load_detections(det_json: Path, group: str) -> Dict[str, List[Dict[str,Any]]]:
    """Load base detections in your YOLO-style JSON {group: [{image_path, boxes: [...]}, ...]}."""
    if not det_json.exists():
        print(f"[WARN] Detection JSON not found: {det_json}")
        return {}
    jd = _read_json(det_json)
    lst = jd.get(group, [])
    by_stem: Dict[str, List[Dict[str,Any]]] = {}
    for entry in lst:
        ip = Path(entry.get("image_path", ""))
        stem = _norm_stem(ip)
        by_stem[stem] = entry.get("boxes", []) or []
    return by_stem

def _load_tree_detections(tree_json: Path) -> Dict[str, List[Dict[str,Any]]]:
    """
    Load the additional tree detections JSON:
    {
      "images": [
        {"image_path": "...", "label_path": "...", "boxes": [ { "class_name": "tree", "confidence": ..., "distance_m": ..., "xyxy": [...] }, ... ]},
        ...
      ]
    }
    """
    if not tree_json.exists():
        print(f"[WARN] Tree detection JSON not found: {tree_json}")
        return {}
    jd = _read_json(tree_json)
    imgs = jd.get("images", [])
    by_stem: Dict[str, List[Dict[str,Any]]] = {}
    for item in imgs:
        ip_str = item.get("image_path") or ""
        if not ip_str:
            continue
        stem = _norm_stem(Path(ip_str))
        boxes = item.get("boxes", []) or []
        # ensure sensible defaults and a distinct color for trees
        for b in boxes:
            if not b.get("class_name"):
                b["class_name"] = "tree"
            if "color_bgr" not in b and str(b.get("class_name","")).lower() == "tree":
                b["color_bgr"] = (40, 180, 40)  # green-ish
        by_stem[stem] = boxes
    return by_stem

# =========================
# Main
# =========================
def main():
    det_by_stem  = _load_detections(DET_JSON, DET_GROUP)
    tree_by_stem = _load_tree_detections(TREE_JSON)

    PRED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, seg_path in enumerate(SEG_JSONS):
        sd = _read_json(Path(seg_path))
        img_path = Path(sd["image_path"])
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read {img_path}")
            continue

        segments = sd.get("segments", []) or []
        stem = _norm_stem(img_path)

        # Merge YOLO boxes + tree boxes for this image
        base_boxes = det_by_stem.get(stem, []) or []
        tree_boxes = tree_by_stem.get(stem, []) or []
        boxes      = list(base_boxes) + list(tree_boxes)

        # RIGHT: geometry + text for seg and boxes (this is the "Predictions" view)
        right_img = _draw_predictions(img, segments, boxes)

        # 🔹 If you want them SQUARE like before (letterboxed), keep this:
        right_tile = _resize_letterbox_square(right_img, TILE)

        # 🔹 If you prefer original aspect ratio, use `right_img` instead of `right_tile`
        out_name = f"{idx+1:02d}_{stem}_prediction.png"
        out_path = PRED_OUT_DIR / out_name
        cv2.imwrite(str(out_path), right_tile)
        print(f"[OK] Saved {out_path}")

    print("[DONE] Generated individual prediction images.")


if __name__ == "__main__":
    main()
