from pathlib import Path
import subprocess
import threading
import cv2
import numpy as np
import time
import json
import os

try:
    import onnxruntime as ort
except ImportError:
    ort = None

r'''Fully run metric depth model + object detection model + image segmentation model
FOR WHAT?
    Input: RGB image
    Output: Combines all results on one depth image, computes distances, and save a PNG overlay and a JSON file with distances
'''

# =========================
# Paths and Python envs
# =========================
ROOT = Path(__file__).resolve().parent

# Scripts
YOLO_SCRIPT          = ROOT / "Object_detection" / "main.py"
METRIC_DEPTH_SCRIPT  = ROOT / "Depth-Anything-V2-main" / "metric_depth" / "run.py"
SEG_SCRIPT           = ROOT / "Segmentation" / "test_model.py"

SEG_CLASS_NAMES = {0: 'Stairs', 1: 'crosswalk', 2: 'sidewalk', 3: 'tree-lined'}
OD_CLASS_NAME = {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'electric pole', 4: 'motocycle', 5: 'pedestrian crossing sign', 6: 'person', 7: 'tree', 8: 'truck'}

# Python interpreters
PY_YOLO   = r"C:\Python\miniconda\envs\tensor_test\python.exe"
PY_DEPTH  = r"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe"
PY_SEG    = PY_YOLO

# Metric depth weight
METRIC_DEPTH_WEIGHTS = (
    r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
)

# =========================
# Inputs / Outputs
# =========================
ORIG_IMG = ROOT / "assets" / "demo03.jpg"

# YOLO labels
YOLO_LABELS_DIR = YOLO_SCRIPT.parent / "output" / "run1" / "labels"

# Metric-depth outputs
METRIC_DEPTH_OUT_DIR = Path(r"C:\Python\ObjectDetect4Blind\output_metric_depth")
METRIC_DEPTH_VIS_PNG = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}.png"
METRIC_DEPTH_RAW_NPY = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}_raw_depth_meter.npy"

# Segmentation border txt
SEG_BORDER_TXT = ROOT / "Segmentation" / "output" / "mask_border.txt"

# Final overlay and JSON output
FINAL_OUT = ROOT / "output" / f"{ORIG_IMG.stem}_metric_depth_boxes_borders.png"
JSON_OUT  = ROOT / "output" / f"{ORIG_IMG.stem}_objects_distance.json"

# Intermediate visualization outputs for pipeline steps
STEP1_DEPTH_ONLY = ROOT / "output" / f"{ORIG_IMG.stem}_step1_depth_only.png"
STEP2_DEPTH_OD   = ROOT / "output" / f"{ORIG_IMG.stem}_step2_depth_with_objects.png"
STEP3_DEPTH_SEG  = ROOT / "output" / f"{ORIG_IMG.stem}_step3_depth_with_segmentation.png"

# Colors (BGR)
COLOR_YOLO_BOX       = (0, 0, 0)
COLOR_YOLO_TEXT      = (0, 0, 0)
COLOR_SEG_BORDER     = (0, 0, 0)
COLOR_SIDEWALK_PT    = (0, 0, 0)
COLOR_SIDEWALK_TEXT  = (0, 0, 0)

# =========================
# Helpers
# =========================
def _watch(name: str, proc: subprocess.Popen):
    rc = proc.wait()
    print(f"[{name}] finished with exit code {rc}")


def _ensure_depth_size(depth_bgr, H, W):
    if depth_bgr is None:
        return None
    if (depth_bgr.shape[0], depth_bgr.shape[1]) != (H, W):
        depth_bgr = cv2.resize(depth_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth_bgr


def _draw_seg_borders_on(depth_bgr, polys, *, color=(0, 0, 0), thickness=2):
    if not polys:
        return depth_bgr
    cv2.polylines(depth_bgr, polys, isClosed=True,
                  color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return depth_bgr


def _fast_percentile_1d(vals: np.ndarray, q: float) -> float | None:
    if vals is None:
        return None
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    k = int(round((q / 100.0) * (vals.size - 1)))
    k = max(0, min(vals.size - 1, k))
    return float(np.partition(vals, k)[k])


def _compute_box_distance(
    depth_map_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    frac: float = 0.5,
    mode: str = "center",
    q: float = 10.0,
    subsample: int = 1,
) -> float | None:
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

    if patch.size == 0:
        return None

    if subsample > 1:
        patch = patch[::subsample, ::subsample]

    valid = patch[(patch > 0) & np.isfinite(patch)].reshape(-1)
    return _fast_percentile_1d(valid, q=q)


def _nearest_sidewalk_distance(
    depth_map_m: np.ndarray,
    sidewalk_mask: np.ndarray,
    max_depth: float = 80.0,
    band_start_frac: float = 0.1,
    q: float = 10.0,
    subsample: int = 1,
):
    assert depth_map_m.shape == sidewalk_mask.shape, "Depth and mask must have same size"
    H, W = depth_map_m.shape

    base_cond = (
        (sidewalk_mask == 1) &
        (depth_map_m > 0) &
        (depth_map_m < max_depth) &
        np.isfinite(depth_map_m)
    )
    if not np.any(base_cond):
        return None, None, None

    y_band_start = int(H * band_start_frac)
    band_mask = np.zeros_like(sidewalk_mask, dtype=bool)
    band_mask[y_band_start:H, :] = True

    cond = base_cond & band_mask
    if not np.any(cond):
        cond = base_cond

    ys, xs = np.where(cond)

    if subsample > 1 and ys.size > 0:
        take = np.arange(0, ys.size, subsample, dtype=np.int64)
        ys = ys[take]
        xs = xs[take]

    vals = depth_map_m[ys, xs].astype(np.float32)
    finite = np.isfinite(vals)
    vals = vals[finite]
    ys = ys[finite]
    xs = xs[finite]

    d_q = _fast_percentile_1d(vals, q=q)
    if d_q is None:
        return None, None, None

    idx = int(np.argmin(np.abs(vals - d_q)))
    return float(d_q), int(xs[idx]), int(ys[idx])


def _load_seg_regions_from_border_txt(border_txt_path: Path):
    """
    Expects lines like:
      <cls_id> <conf> x1 y1 x2 y2 x3 y3 ...
    """
    if not border_txt_path.exists():
        print(f"[SEG] border file not found: {border_txt_path}")
        return []

    regions = []
    with open(border_txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            vals = ln.split()
            if len(vals) < 2 + 6:
                continue

            cls_id = int(float(vals[0]))
            conf = float(vals[1])

            coords = list(map(float, vals[2:]))
            if len(coords) % 2 != 0:
                continue

            pts = []
            it = iter(coords)
            for x, y in zip(it, it):
                pts.append([int(round(x)), int(round(y))])

            if len(pts) < 3:
                continue

            poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            cls_name = SEG_CLASS_NAMES.get(cls_id, str(cls_id))
            regions.append({
                "poly": poly,
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf
            })

    return regions


# =========================
# Metric depth via ONNX Runtime
# =========================
def _run_metric_depth_onnx():
    if ort is None:
        raise RuntimeError("onnxruntime is not installed, cannot run ONNX metric depth backend.")

    onnx_path = METRIC_DEPTH_WEIGHTS
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"[METRIC_DEPTH_ONNX] ONNX model not found: {onnx_path}")

    print(f"[METRIC_DEPTH_ONNX] loading model: {onnx_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    img_bgr = cv2.imread(str(ORIG_IMG))
    if img_bgr is None:
        raise FileNotFoundError(f"[METRIC_DEPTH_ONNX] Original image not found: {ORIG_IMG}")
    H0, W0 = img_bgr.shape[:2]

    EXPORT_SIZE = 518
    bgr_resized = cv2.resize(img_bgr, (EXPORT_SIZE, EXPORT_SIZE), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, ...]

    out = sess.run([output_name], {input_name: x})[0]
    depth_small = np.squeeze(out).astype(np.float32)

    depth_map_m = cv2.resize(depth_small, (W0, H0), interpolation=cv2.INTER_LINEAR)
    depth_map_m = np.clip(depth_map_m, 1e-3, 80.0)

    METRIC_DEPTH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(METRIC_DEPTH_RAW_NPY), depth_map_m)
    print(f"[METRIC_DEPTH_ONNX] saved raw depth: {METRIC_DEPTH_RAW_NPY}")

    depth_norm = depth_map_m / 80.0
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_8u = (depth_norm * 255.0).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)

    if not cv2.imwrite(str(METRIC_DEPTH_VIS_PNG), depth_bgr):
        raise RuntimeError(f"[METRIC_DEPTH_ONNX] Failed to save depth PNG: {METRIC_DEPTH_VIS_PNG}")
    print(f"[METRIC_DEPTH_ONNX] saved vis PNG: {METRIC_DEPTH_VIS_PNG}")


def putText_outline(img, text, org, fontFace, fontScale, color, thickness,
                    outline_color=(255,255,255), outline_thickness=None):
    # draw outline first, then main text on top (thickness controls "bold")
    if outline_thickness is None:
        outline_thickness = thickness + 4
    cv2.putText(img, text, org, fontFace, fontScale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, fontFace, fontScale, color, thickness, cv2.LINE_AA)


# Main pipeline
def run_parallel_and_overlay_metric(class_names: dict | None = None, seg_args: list[str] | None = None):

    # 1) YOLO
    p_yolo = subprocess.Popen(
        [PY_YOLO, str(YOLO_SCRIPT), "--image", str(ORIG_IMG)],
        cwd=str(YOLO_SCRIPT.parent)
    )

    # 2) Metric depth
    depth_ext = os.path.splitext(METRIC_DEPTH_WEIGHTS)[1].lower()

    if depth_ext == ".pth":
        metric_cmd = [
            PY_DEPTH, str(METRIC_DEPTH_SCRIPT),
            "--encoder", "vits",
            "--load-from", METRIC_DEPTH_WEIGHTS,
            "--max-depth", "80",
            "--img-path", str(ORIG_IMG),
            "--outdir", str(METRIC_DEPTH_OUT_DIR),
            "--pred-only",
            "--save-numpy",
        ]
        p_depth = subprocess.Popen(metric_cmd, cwd=str(METRIC_DEPTH_SCRIPT.parent))
        t2 = threading.Thread(target=_watch, args=("METRIC_DEPTH", p_depth), daemon=True)
    else:
        def _depth_worker():
            print("[METRIC_DEPTH_ONNX] starting...")
            _run_metric_depth_onnx()
            print("[METRIC_DEPTH_ONNX] finished.")
        t2 = threading.Thread(target=_depth_worker, daemon=True)

    # 3) Segmentation
    seg_cmd = [
        PY_SEG, str(SEG_SCRIPT),
        "--image", str(ORIG_IMG),
        "--out-border", str(SEG_BORDER_TXT),
    ]
    if seg_args:
        seg_cmd.extend(seg_args)
    p_seg = subprocess.Popen(seg_cmd, cwd=str(ROOT))

    # 4) Wait for all
    t1 = threading.Thread(target=_watch, args=("YOLO", p_yolo), daemon=True)
    t3 = threading.Thread(target=_watch, args=("SEG", p_seg), daemon=True)
    for t in (t1, t2, t3):
        t.start()
    for t in (t1, t2, t3):
        t.join()

    # 5) Load original
    orig = cv2.imread(str(ORIG_IMG))
    if orig is None:
        raise FileNotFoundError(f"Original image not found: {ORIG_IMG}")
    H, W = orig.shape[:2]

    # 6) Load depth visualization
    if not METRIC_DEPTH_VIS_PNG.exists():
        raise FileNotFoundError(f"Metric depth PNG not found: {METRIC_DEPTH_VIS_PNG}")
    depth_bgr = cv2.imread(str(METRIC_DEPTH_VIS_PNG))
    if depth_bgr is None:
        raise FileNotFoundError(f"Failed to read PNG: {METRIC_DEPTH_VIS_PNG}")
    depth_bgr = _ensure_depth_size(depth_bgr, H, W)

    # 7) Load raw depth (meters)
    depth_map_m = None
    if METRIC_DEPTH_RAW_NPY.exists():
        depth_map_m = np.load(str(METRIC_DEPTH_RAW_NPY)).astype(np.float32)
        if depth_map_m.ndim == 3:
            depth_map_m = depth_map_m.squeeze()
        if depth_map_m.shape[:2] != (H, W):
            depth_map_m = cv2.resize(depth_map_m, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        print(f"[METRIC_DEPTH] raw depth .npy not found: {METRIC_DEPTH_RAW_NPY}")

    results: list[dict] = []

    # 8) YOLO boxes + distances
    label_file = YOLO_LABELS_DIR / f"{ORIG_IMG.stem}.txt"
    if not label_file.exists():
        print(f"[YOLO] label file not found: {label_file}")
    else:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for idx, ln in enumerate(lines):
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, ww, hh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None

            px, py = cx * W, cy * H
            pw, ph = ww * W, hh * H
            x1 = max(0, int(px - pw / 2))
            y1 = max(0, int(py - ph / 2))
            x2 = min(W - 1, int(px + pw / 2))
            y2 = min(H - 1, int(py + ph / 2))

            # Use OD_CLASS_NAME dictionary for object detection class names
            cls_name = OD_CLASS_NAME.get(cls, str(cls))
            label_txt = cls_name
            if conf is not None:
                label_txt += f" {conf:.2f}"

            dist = None
            if depth_map_m is not None:
                name_l = cls_name.lower()
                is_bottom_region = any(k in name_l for k in ("car", "bicycle", "truck", "motorbike", "motorcycle"))
                mode = "bottom" if is_bottom_region else "center"
                dist = _compute_box_distance(depth_map_m, x1, y1, x2, y2, frac=0.1, mode=mode)
                if dist is not None:
                    label_txt += f" {dist:.2f}m"

            cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), COLOR_YOLO_BOX, 2)
            cv2.putText(depth_bgr, label_txt, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_YOLO_TEXT, 3, cv2.LINE_AA)

            results.append({
                "id": f"yolo_{idx}",
                "source": "yolo",
                "class_id": cls,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "distance_m": dist,
            })

    # 9) Segmentation regions + borders + (name/conf/distance) labels
    seg_regions = _load_seg_regions_from_border_txt(SEG_BORDER_TXT)
    seg_polys = [r["poly"] for r in seg_regions]

    depth_bgr = _draw_seg_borders_on(depth_bgr, seg_polys, color=COLOR_SEG_BORDER, thickness=2)

    if depth_map_m is not None and seg_regions:
        for idx, reg in enumerate(seg_regions):
            poly = reg["poly"]
            seg_name = reg["class_name"]
            seg_conf = float(reg["confidence"])
            seg_cls  = int(reg["class_id"])

            poly_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly], 1)

            d_min, x_min, y_min = _nearest_sidewalk_distance(
                depth_map_m,
                poly_mask,
                max_depth=80.0,
                band_start_frac=0.1,
            )

            # anchor point: nearest depth pixel if exists, else polygon centroid
            if x_min is None or y_min is None:
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    x_min = int(M["m10"] / M["m00"])
                    y_min = int(M["m01"] / M["m00"])
                else:
                    x_min, y_min = int(poly[0, 0, 0]), int(poly[0, 0, 1])

            x_min = int(np.clip(x_min, 0, W - 1))
            y_min = int(np.clip(y_min, 0, H - 1))

            if d_min is not None:
                cv2.circle(depth_bgr, (x_min, y_min), 4, COLOR_SIDEWALK_PT, -1)
                txt = f"{seg_name} {seg_conf:.2f} {d_min:.2f}m"
            else:
                txt = f"{seg_name} {seg_conf:.2f}"

            x_txt = int(np.clip(x_min + 6, 0, W - 1))
            y_txt = int(np.clip(y_min - 6, 0, H - 1))

            putText_outline(
                depth_bgr, txt, (x_txt, y_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                COLOR_SIDEWALK_TEXT, thickness=3
            )

            results.append({
                "id": f"seg_{idx}",
                "source": "segmentation",
                "class_id": seg_cls,
                "class_name": seg_name,
                "confidence": seg_conf,
                "polygon": poly.reshape(-1, 2).tolist(),
                "distance_m": d_min,
                "nearest_pixel": None if d_min is None else {"x": x_min, "y": y_min},
            })

    # 10) Save overlay
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(FINAL_OUT), depth_bgr):
        raise RuntimeError(f"Failed to save final overlay to {FINAL_OUT}")
    print(f"[FINAL] saved: {FINAL_OUT}")

    # 11) Save JSON
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[JSON] saved: {JSON_OUT}")


if __name__ == "__main__":
    _t0 = time.perf_counter()
    try:
        CLASS_NAMES = None
        SEG_ARGS = None
        run_parallel_and_overlay_metric(CLASS_NAMES, SEG_ARGS)
    finally:
        elapsed = time.perf_counter() - _t0
        print(f"[RUNTIME] total elapsed: {elapsed:.3f}s (~{elapsed/60:.2f} min)")