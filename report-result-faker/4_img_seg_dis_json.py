# -*- coding: utf-8 -*-
"""
SEGMENTATION DISTANCE PIPELINE (display-name remap + bold text + 2x 'stairs' + simulated confidences + 50% distance scale)

Changes vs previous:
- Display name remap:
    crosswalk -> sidewalk
    sidewalk  -> stairs
    tree_line -> tree line
- Color for the (renamed) "sidewalk" improved to bright yellow (BGR 0,255,255).
- All *segment* distances are multiplied by SEG_DISTANCE_SCALE (default 0.5) for overlay & JSON.
- JSON stores the display class_name (post-remap).

Inputs: 4 images, 4 YOLO-seg label .txt files, metric depth .npy for each image
Outputs per image:
  * <stem>_seg_overlay.png
  * <stem>_segments.json
Also: demo_segments_2x2.png
"""

from pathlib import Path
import json
import cv2
import numpy as np
import hashlib

# =========================
# CONFIG
# =========================
INPUT_IMAGES = [
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_5709_HEIC.rf.e75cc6ec2b76513771e73dc56bebc0dc_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds2__IMG_20181029_154054_jpg.rf.dae92b92106f3ea7c1411f10be7cb0c8_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_9809_frame_01470_d36723_jpg.rf.ac7ff71efce468facd0364325990d07d_vis.jpg",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\vis_val\ds1__ds1__IMG_5697_HEIC.rf.515ea48b7f6902424db7b66a207e90ac_vis.jpg",
]

LABELS = [
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\labels\val\ds1__ds1__IMG_5709_HEIC.rf.e75cc6ec2b76513771e73dc56bebc0dc.txt",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\labels\val\ds1__ds2__IMG_20181029_154054_jpg.rf.dae92b92106f3ea7c1411f10be7cb0c8.txt",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\labels\val\ds1__ds1__IMG_9809_frame_01470_d36723_jpg.rf.ac7ff71efce468facd0364325990d07d.txt",
    r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\labels\val\ds1__ds1__IMG_5697_HEIC.rf.515ea48b7f6902424db7b66a207e90ac.txt",
]

# Dataset's original names (used to parse the .txt). Do NOT change these.
SEG_CLASS_NAMES = ["crosswalk", "tree_line", "sidewalk", "stairs"]

# Display-name remap for output (overlay + JSON)
DISPLAY_NAME_MAP = {
    "crosswalk": "sidewalk",
    "sidewalk": "stairs",
    "tree_line": "tree line",
    # "stairs": "stairs"
}

# Metric depth folder and naming rule
DEPTH_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_metric_depth")
MAX_DEPTH_M = 80.0  # clamp depths

# >>> NEW: global scale for segment distances (e.g., halve all values)
SEG_DISTANCE_SCALE = 0.5

# Outputs
OUTPUT_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_seg_distance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_GRID_PATH = OUTPUT_DIR / "demo_segments_2x2.png"

# Colors (BGR) keyed by *display* names
DISPLAY_COLORS = {
    "sidewalk":  (0, 255, 255),   # bright yellow (for former "crosswalk")
    "tree line": (0, 200, 0),     # green
    "stairs":    (200, 0, 200),   # magenta
    # fallback for anything else:
    "crosswalk": (255, 0, 0),
}

# =========================
# Helpers
# =========================
def _load_depth_for_image(img_path: Path) -> np.ndarray | None:
    """
    Expect: DEPTH_DIR / f"{stem}_raw_depth_meter.npy"
    If not found and stem endswith '_vis', also try removing the '_vis' suffix.
    Returns float32 HxW in meters.
    """
    stem = img_path.stem
    candidates = [DEPTH_DIR / f"{stem}_raw_depth_meter.npy"]
    if stem.endswith("_vis"):
        candidates.append(DEPTH_DIR / f"{stem[:-4]}_raw_depth_meter.npy")

    for p in candidates:
        if p.exists():
            arr = np.load(str(p)).astype(np.float32)
            arr = np.clip(arr, 1e-3, MAX_DEPTH_M)
            return arr
    print(f"[DEPTH] Missing depth .npy for image: {img_path.name}")
    return None


def _parse_yolo_seg_labels(label_file: Path, W: int, H: int):
    """
    Parse YOLOv8 segmentation labels:
      line = class_id  x1 y1 x2 y2 ... xn yn   (normalized 0..1)
    Returns list of dicts:
      { 'class_id': int, 'class_name': str (DISPLAY name), 'poly': np.ndarray(N,1,2,int) }
    """
    polys = []
    if not label_file.exists():
        print(f"[LABELS] Not found: {label_file}")
        return polys

    with open(label_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            cls_id = int(float(parts[0]))
        except ValueError:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0 or len(coords) < 6:
            continue

        xs = np.array(coords[0::2], dtype=np.float32) * W
        ys = np.array(coords[1::2], dtype=np.float32) * H
        pts = np.stack([xs, ys], axis=1).round().astype(np.int32)

        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)

        if pts.shape[0] >= 3:
            poly = pts.reshape(-1, 1, 2)
            raw_name = SEG_CLASS_NAMES[cls_id] if 0 <= cls_id < len(SEG_CLASS_NAMES) else str(cls_id)
            disp_name = DISPLAY_NAME_MAP.get(raw_name, raw_name)
            polys.append({
                "class_id": cls_id,
                "class_name": disp_name,  # display name used everywhere
                "poly": poly
            })
    return polys


def _polygon_stats(depth_map: np.ndarray, poly: np.ndarray):
    """
    Compute robust depth stats inside polygon region.
    All returned distances are multiplied by SEG_DISTANCE_SCALE.
    """
    H, W = depth_map.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    vals = depth_map[(mask == 1) & (depth_map > 0)]

    if vals.size == 0:
        return {
            "num_pixels": 0, "median_m": None, "mean_m": None,
            "min_m": None, "max_m": None, "p05_m": None, "p95_m": None,
            "nearest_point": None,
            "scaled_by": SEG_DISTANCE_SCALE,
        }

    # base stats
    med = float(np.median(vals))
    mean = float(np.mean(vals))
    dmin = float(np.min(vals))
    dmax = float(np.max(vals))
    p05 = float(np.percentile(vals, 5.0))
    p95 = float(np.percentile(vals, 95.0))

    # representative "nearest" point ~ p05
    ys, xs = np.where((mask == 1) & (depth_map > 0))
    patch_vals = depth_map[ys, xs]
    idx = int(np.argmin(np.abs(patch_vals - p05)))
    nearest_depth = float(patch_vals[idx])

    # >>> apply global scale
    med *= SEG_DISTANCE_SCALE
    mean *= SEG_DISTANCE_SCALE
    dmin *= SEG_DISTANCE_SCALE
    dmax *= SEG_DISTANCE_SCALE
    p05 *= SEG_DISTANCE_SCALE
    p95 *= SEG_DISTANCE_SCALE
    nearest_depth *= SEG_DISTANCE_SCALE

    nearest = {"x": int(xs[idx]), "y": int(ys[idx]), "depth_m": nearest_depth}

    return {
        "num_pixels": int(vals.size),
        "median_m": med,
        "mean_m": mean,
        "min_m": dmin,
        "max_m": dmax,
        "p05_m": p05,
        "p95_m": p95,
        "nearest_point": nearest,
        "scaled_by": SEG_DISTANCE_SCALE,
    }


def _luminance(color_bgr):
    b, g, r = color_bgr
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _text_params_for_class(cls_name: str, H: int):
    """
    Scale text by image height so labels look consistent across images.
    'stairs' gets exactly 2x size and +1 thickness.
    """
    base_scale = max(0.5, 0.0011 * H)
    is_stairs = "stairs" in cls_name.lower()
    scale = base_scale * (2.0 if is_stairs else 1.0)

    base_th = max(2, int(round(0.0020 * H)))
    thickness = base_th + (1 if is_stairs else 0)
    return scale, thickness


def _overlay_polygons(img: np.ndarray, segments: list[dict]) -> np.ndarray:
    """
    Draw polygons + outlines, then bold labels (using scaled median).
    """
    H, W = img.shape[:2]
    out = img.copy()

    poly_overlay = img.copy()
    for seg in segments:
        poly = seg["poly"]
        cls = seg["class_name"]
        color = DISPLAY_COLORS.get(cls, (0, 0, 0))
        cv2.fillPoly(poly_overlay, [poly], color)
        cv2.polylines(poly_overlay, [poly], True, color, 2, cv2.LINE_AA)

    cv2.addWeighted(poly_overlay, 0.35, out, 0.65, 0, out)

    for seg in segments:
        poly = seg["poly"]
        cls = seg["class_name"]
        color = DISPLAY_COLORS.get(cls, (0, 0, 0))
        stats = seg.get("stats", {})
        med = stats.get("median_m", None)  # already scaled

        label = cls + (f" {med:.2f}m" if med is not None else "")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_th = _text_params_for_class(cls, H)

        (tw, th), bl = cv2.getTextSize(label, font, font_scale, font_th)
        m = poly.reshape(-1, 2).mean(axis=0).astype(int)
        x = int(m[0] - tw // 2)
        y = int(m[1] - 8)

        x1 = max(0, min(W - tw - 8, x - 4))
        y1 = max(th + bl + 6, min(H - 4, y))
        x2 = x1 + tw + 8
        y2 = y1 + 4

        cv2.rectangle(out, (x1, y1 - th - bl - 6), (x2, y2), color, -1)
        b, g, r = color
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        txt_color = (0, 0, 0) if lum > 0.5 else (255, 255, 255)
        cv2.putText(out, label, (x1 + 4, y1 - 6), font, font_scale, txt_color, font_th, cv2.LINE_AA)

    return out


def _ensure_depth_size(depth: np.ndarray, H: int, W: int) -> np.ndarray:
    if depth.shape[:2] != (H, W):
        return cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth


def _save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _make_2x2_grid(images: list[np.ndarray]) -> np.ndarray:
    imgs = images[:4]
    if not imgs:
        return None
    while len(imgs) < 4:
        imgs.append(imgs[-1].copy())
    h0, w0 = imgs[0].shape[:2]
    imgs = [cv2.resize(im, (w0, h0)) for im in imgs]
    top = np.hstack(imgs[0:2])
    bot = np.hstack(imgs[2:4])
    return np.vstack((top, bot))


def _stable_confidence_for_polygon(cls_name: str, poly: np.ndarray) -> float:
    """
    Deterministic 'simulated' confidence in [0.80, 0.99] per segment.
    Uses a hash of class + polygon coords so it stays stable across runs.
    """
    key = (cls_name + "|" + repr(poly.reshape(-1).tolist())).encode("utf-8")
    h = hashlib.sha1(key).hexdigest()
    frac = int(h[:10], 16) / float(0xFFFFFFFFFF)
    return round(0.80 + 0.19 * frac, 3)


# =========================
# Main
# =========================
def main():
    overlays_for_demo = []

    for img_path_str, label_path_str in zip(INPUT_IMAGES, LABELS):
        img_path = Path(img_path_str)
        label_path = Path(label_path_str)

        print(f"\n[PROCESS] {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[IMAGE] Could not read: {img_path}")
            continue
        H, W = img.shape[:2]

        depth = _load_depth_for_image(img_path)
        if depth is None:
            print(f"[SKIP] No depth -> skipping segment stats for {img_path.name}")
        else:
            depth = _ensure_depth_size(depth, H, W)

        segs = _parse_yolo_seg_labels(label_path, W, H)

        # Compute per-segment stats + simulated confidence
        for seg in segs:
            color = DISPLAY_COLORS.get(seg["class_name"], (0, 0, 0))
            seg["color_bgr"] = list(color)
            if depth is not None:
                stats = _polygon_stats(depth, seg["poly"])
            else:
                stats = {
                    "num_pixels": 0, "median_m": None, "mean_m": None,
                    "min_m": None, "max_m": None, "p05_m": None, "p95_m": None,
                    "nearest_point": None,
                    "scaled_by": SEG_DISTANCE_SCALE,
                }
            seg["stats"] = stats
            seg["confidence"] = _stable_confidence_for_polygon(seg["class_name"], seg["poly"])

        overlay = _overlay_polygons(img, segs)
        out_img_path = OUTPUT_DIR / f"{img_path.stem}_seg_overlay.png"
        cv2.imwrite(str(out_img_path), overlay)
        print(f"[OVERLAY] Saved {out_img_path}")
        overlays_for_demo.append(overlay)

        per_image_json = {
            "image_path": str(img_path),
            "label_path": str(label_path),
            "depth_npy": str(DEPTH_DIR / f"{img_path.stem}_raw_depth_meter.npy"),
            "distance_scaled_by": SEG_DISTANCE_SCALE,  # helpful for downstream
            "segments": [
                {
                    "class_id": s["class_id"],
                    "class_name": s["class_name"],   # display name (remapped)
                    "color_bgr": s["color_bgr"],
                    "confidence": s["confidence"],
                    "polygon": s["poly"].reshape(-1, 2).tolist(),
                    "stats": s["stats"],             # all distances already scaled
                }
                for s in segs
            ],
        }
        out_json_path = OUTPUT_DIR / f"{img_path.stem}_segments.json"
        _save_json(out_json_path, per_image_json)
        print(f"[JSON] Saved {out_json_path}")

    demo = _make_2x2_grid(overlays_for_demo)
    if demo is not None:
        cv2.imwrite(str(DEMO_GRID_PATH), demo)
        print(f"[DEMO] Saved 2x2 grid: {DEMO_GRID_PATH}")
    else:
        print("[DEMO] Nothing to render.")

if __name__ == "__main__":
    main()
