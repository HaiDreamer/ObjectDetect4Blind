from pathlib import Path
import subprocess
import threading
import cv2
import numpy as np
import time
import os
import json

try:
    import onnxruntime as ort  # used only if we run metric depth via ONNX
except ImportError:
    ort = None

# ---------------------
# PATH & ENVIRONMENT
# ---------------------
ROOT = Path(r"C:\Python\ObjectDetect4Blind")

YOLO_SCRIPT   = ROOT / "Object detection" / "main.py"
# Use metric depth script instead of relative depth script
METRIC_DEPTH_SCRIPT  = ROOT / "Depth-Anything-V2-main" / "metric_depth" / "run.py"
SEG_SCRIPT    = ROOT / "Segmentation" / "test_model.py"

PY_YOLO   = r"C:\Python\miniconda\envs\tensor_test\python.exe"
PY_DEPTH  = r"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe"
PY_SEG    = PY_YOLO   

# Metric-depth weights
# NOTE:
#   - If this points to *.pth  → use original metric_depth/run.py (PyTorch backend)
#   - If this points to *.onnx or *.ort → use ONNX Runtime backend defined below
METRIC_DEPTH_WEIGHTS = (
    r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
    # r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits_fp16.onnx"
    # r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits_fp16.with_runtime_opt.ort"
)

# Colors (BGR) – same style as in MAIN.py
COLOR_YOLO_BOX       = (0, 0, 0)   # bounding boxes
COLOR_YOLO_TEXT      = (0, 0, 0)   # object labels
COLOR_SEG_BORDER     = (0, 0, 0)   # segmentation outlines
COLOR_SIDEWALK_PT    = (0, 0, 0)   # point for nearest sidewalk
COLOR_SIDEWALK_TEXT  = (0, 0, 0)   # text for sidewalk distance


def _watch(name: str, proc: subprocess.Popen):
    '''observe child process, print message'''
    rc = proc.wait()
    print(f"[{name}] finished with exit code {rc}")


def _ensure_depth_size(depth_bgr, H, W):
    '''resize image if depth img size != original img size'''
    if depth_bgr is None:
        return None
    if (depth_bgr.shape[0], depth_bgr.shape[1]) != (H, W):
        depth_bgr = cv2.resize(depth_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth_bgr


def _compute_box_distance(
    depth_map_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    frac: float = 0.5,
    mode: str = "center",   # "center" or "bottom"
) -> float | None:
    """
    Compute distance to a bounding box using only a danger-relevant region and
    a robust statistic (median).

    mode:
      - "center": central frac of the box (for tall objects: human, traffic light, tree, pole)
    y1  +-----------------------------------------+
        |                                         |
        |                (ignored)                |
        |             ^^^^^^^^^^^^^^^^            |
        |             ^   CENTER     ^            |
        |             ^  SAMPLED     ^            |
        |             ^   REGION     ^            |
        |             ^^^^^^^^^^^^^^^^            |
        |                                         |
        |                                         |
    y2  +-----------------------------------------+
        x1                                        x2

    - "bottom": lower frac of the box (for ground-contact objects: car, bicycle, truck, motorbike)
    y1  +--------------------------------------+
        |                                      |
        |                (ignored)             |
        |                                      |
        +--------------------------------------+
        |                                      |  <- bottom half (sampled)
        |      BOTTOM SAMPLED REGION           |
    y2  +--------------------------------------+
        x1                               x2
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
        # Use a horizontal strip at the bottom of the box:
        #   full width [x1:x2], bottom frac of height.
        ch = int(h * frac)
        if ch <= 0:
            return None
        y_start = max(y1, y2 - ch)
        patch = depth_map_m[y_start:y2, x1:x2]
    else:  # "center" (default)
        # Use central frac of the box (both width and height).
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

    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    return float(np.median(valid))


def _compute_poly_distance(depth_map_m: np.ndarray, poly: np.ndarray) -> float | None:
    '''Distance for segmentation mask'''
    H, W = depth_map_m.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)

    valid = depth_map_m[(mask == 1) & (depth_map_m > 0)]
    if valid.size == 0:
        return None
    return float(valid.mean())


def _nearest_sidewalk_distance(
    depth_map_m: np.ndarray,
    sidewalk_mask: np.ndarray,
    max_depth: float = 80.0,
    band_start_frac: float = 0.5,
    percentile: float = 5.0
):
    """
    Find the nearest sidewalk point using:
      - only a lower vertical band of the image
      - a low percentile (e.g. 5th) instead of raw min for robustness

    band_start_frac: fraction of height from which to start (0..1).
                     0.5 -> bottom half of the image only.
    percentile: percentile of depths to use (e.g. 5.0 = 5th percentile).
    """
    assert depth_map_m.shape == sidewalk_mask.shape, "Depth and mask must have same size"

    H, W = depth_map_m.shape

    # base condition: mask == 1, valid depth, within max_depth
    base_cond = (sidewalk_mask == 1) & (depth_map_m > 0) & (depth_map_m < max_depth)
    if not np.any(base_cond):
        return None, None, None

    # restrict to lower band
    y_band_start = int(H * band_start_frac)
    band_mask = np.zeros_like(sidewalk_mask, dtype=bool)
    band_mask[y_band_start:H, :] = True

    cond = base_cond & band_mask
    if not np.any(cond):
        # fallback: if nothing in the band, use full mask as before
        cond = base_cond

    ys, xs = np.where(cond)
    vals = depth_map_m[ys, xs]
    if vals.size == 0:
        return None, None, None

    # use low percentile instead of raw min for robustness
    d_target = float(np.percentile(vals, percentile))

    # pick pixel whose depth is closest to that percentile value
    idx = np.argmin(np.abs(vals - d_target))
    y_min = int(ys[idx])
    x_min = int(xs[idx])

    return d_target, x_min, y_min


def _load_seg_polys_from_border_txt(border_txt_path: Path, W: int, H: int):
    '''Load segmentation polygons from border.txt'''
    if not border_txt_path.exists():
        print(f"[SEG] border file not found: {border_txt_path}")
        return []

    with open(border_txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    seg_polys = []
    for ln in lines:
        vals = ln.split()
        if len(vals) < 4 or len(vals) % 2 != 0:
            continue
        pts = []
        it = iter(map(float, vals))
        for x, y in zip(it, it):
            pts.append([int(round(x)), int(round(y))])
        if len(pts) >= 3:
            poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            seg_polys.append(poly)

    return seg_polys


def _draw_seg_borders_on(depth_bgr, polys, *, color=(255, 255, 255), thickness=2):
    '''draw segmentation border to output img, using polys loaded from border text file'''
    if not polys:
        return depth_bgr
    cv2.polylines(
        depth_bgr,
        polys,
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )
    return depth_bgr


def _run_metric_depth_onnx(img_path: Path, vis_png: Path, raw_npy: Path):
    """
    Runs metric depth model via ONNX Runtime instead of metric_depth/run.py, and
    writes:
        - vis_png : colored depth visualization
        - raw_npy : raw float32 depth in meters
    """
    if ort is None:
        raise RuntimeError("onnxruntime is not installed, cannot run ONNX metric depth backend.")

    onnx_path = METRIC_DEPTH_WEIGHTS
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"[METRIC_DEPTH_ONNX] ONNX model not found: {onnx_path}")

    print(f"[METRIC_DEPTH_ONNX] loading model: {onnx_path}")
    providers = ort.get_available_providers()
    print(f"[METRIC_DEPTH_ONNX] providers: {providers}")
    sess = ort.InferenceSession(onnx_path, providers=providers)

    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"[METRIC_DEPTH_ONNX] Original image not found: {img_path}")
    H0, W0 = img_bgr.shape[:2]

    EXPORT_SIZE = 518
    bgr_resized = cv2.resize(img_bgr, (EXPORT_SIZE, EXPORT_SIZE), interpolation=cv2.INTER_LINEAR)

    # Preprocess BGR -> RGB, normalize
    rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = x.transpose(2, 0, 1)[None, ...]

    # Run inference
    out = sess.run([output_name], {input_name: x})[0]
    depth_small = np.squeeze(out).astype(np.float32)

    # Resize depth back to original image resolution
    depth_map_m = cv2.resize(
        depth_small,
        (W0, H0),
        interpolation=cv2.INTER_LINEAR
    )

    # Clamp to valid metric range
    depth_map_m = np.clip(depth_map_m, 1e-3, 80.0)

    vis_png.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(raw_npy), depth_map_m)
    print(f"[METRIC_DEPTH_ONNX] saved raw depth: {raw_npy}")

    depth_norm = depth_map_m / 80.0
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_8u = (depth_norm * 255.0).astype(np.uint8)
    depth_bgr = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)

    if not cv2.imwrite(str(vis_png), depth_bgr):
        raise RuntimeError(f"[METRIC_DEPTH_ONNX] Failed to save depth PNG: {vis_png}")
    print(f"[METRIC_DEPTH_ONNX] saved vis PNG: {vis_png}")


def run_full_pipeline_for_image(image_path: Path, class_names: dict | None = None, seg_args: list[str] | None = None) -> Path:
    '''run parallel 3 models depth + seg + obj detect (metric-depth version)'''
    image_path = Path(image_path).resolve()
    stem = image_path.stem

    YOLO_LABELS_DIR = YOLO_SCRIPT.parent / "output" / "run1" / "labels"

    # metric depth outputs for this particular image
    METRIC_DEPTH_OUT_DIR = ROOT / "output_metric_depth"
    METRIC_DEPTH_OUT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DEPTH_VIS_PNG = METRIC_DEPTH_OUT_DIR / f"{stem}.png"
    METRIC_DEPTH_RAW_NPY = METRIC_DEPTH_OUT_DIR / f"{stem}_raw_depth_meter.npy"

    SEG_OUT_DIR     = ROOT / "Segmentation" / "output"
    SEG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEG_BORDER_TXT  = SEG_OUT_DIR / f"{stem}_border.txt"

    FINAL_OUT_DIR   = ROOT / "output"
    FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUT       = FINAL_OUT_DIR / f"{stem}_metric_depth_boxes_borders.png"
    JSON_OUT        = FINAL_OUT_DIR / f"{stem}_objects_distance.json"

    print(f"[PIPELINE] image: {image_path}")
    print(f"[PIPELINE] YOLO labels dir: {YOLO_LABELS_DIR}")
    print(f"[PIPELINE] metric depth PNG: {METRIC_DEPTH_VIS_PNG}")
    print(f"[PIPELINE] metric depth raw npy: {METRIC_DEPTH_RAW_NPY}")
    print(f"[PIPELINE] seg border txt: {SEG_BORDER_TXT}")
    print(f"[PIPELINE] final out: {FINAL_OUT}")
    print(f"[PIPELINE] json out: {JSON_OUT}")

    # 1) run 3 external scripts in parallel
    p_yolo = subprocess.Popen(
        [PY_YOLO, str(YOLO_SCRIPT), "--image", str(image_path)],
        cwd=str(YOLO_SCRIPT.parent),
    )

    # Decide metric depth backend based on METRIC_DEPTH_WEIGHTS extension
    depth_ext = os.path.splitext(METRIC_DEPTH_WEIGHTS)[1].lower()
    threads = []

    if depth_ext == ".pth":
        metric_cmd = [
            PY_DEPTH, str(METRIC_DEPTH_SCRIPT),
            "--encoder", "vits",
            "--load-from", METRIC_DEPTH_WEIGHTS,
            "--max-depth", "80",
            "--img-path", str(image_path),
            "--outdir", str(METRIC_DEPTH_OUT_DIR),
            "--pred-only",
            "--save-numpy",
        ]
        p_depth = subprocess.Popen(metric_cmd, cwd=str(METRIC_DEPTH_SCRIPT.parent))
        threads.append(threading.Thread(target=_watch, args=("METRIC_DEPTH", p_depth), daemon=True))
    else:
        # ONNX/ORT backend
        def _depth_worker():
            print("[METRIC_DEPTH_ONNX] starting...")
            _run_metric_depth_onnx(image_path, METRIC_DEPTH_VIS_PNG, METRIC_DEPTH_RAW_NPY)
            print("[METRIC_DEPTH_ONNX] finished.")
        threads.append(threading.Thread(target=_depth_worker, daemon=True))

    seg_cmd = [
        PY_SEG,
        str(SEG_SCRIPT),
        "--image",
        str(image_path),
        "--out-border",
        str(SEG_BORDER_TXT),
    ]
    if seg_args:
        seg_cmd.extend(seg_args)

    p_seg = subprocess.Popen(seg_cmd, cwd=str(ROOT))

    t0 = time.perf_counter()
    threads.append(threading.Thread(target=_watch, args=("YOLO", p_yolo), daemon=True))
    threads.append(threading.Thread(target=_watch, args=("SEG", p_seg), daemon=True))

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    print(f"[PIPELINE] total external processes time: {elapsed:.3f} s (~{elapsed/60:.2f} min)")

    # 3) load original
    orig = cv2.imread(str(image_path))
    if orig is None:
        raise FileNotFoundError(f"Original image not found: {image_path}")
    H, W = orig.shape[:2]

    # 4) load depth visualization (metric)
    if not METRIC_DEPTH_VIS_PNG.exists():
        raise FileNotFoundError(f"Metric depth PNG not found: {METRIC_DEPTH_VIS_PNG}")
    depth_bgr = cv2.imread(str(METRIC_DEPTH_VIS_PNG))
    if depth_bgr is None:
        raise FileNotFoundError(f"Failed to read PNG: {METRIC_DEPTH_VIS_PNG}")
    depth_bgr = _ensure_depth_size(depth_bgr, H, W)

    # 5) load raw metric depth (meters)
    depth_map_m = None
    if METRIC_DEPTH_RAW_NPY.exists():
        depth_map_m = np.load(str(METRIC_DEPTH_RAW_NPY)).astype(np.float32)
        if depth_map_m.ndim == 3:
            depth_map_m = depth_map_m.squeeze()
        if depth_map_m.shape[:2] != (H, W):
            depth_map_m = cv2.resize(
                depth_map_m,
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )
    else:
        print(f"[METRIC_DEPTH] raw depth .npy not found: {METRIC_DEPTH_RAW_NPY}")

    results: list[dict] = []

    # 6) YOLO boxes + distances (danger-zone median)
    label_file = YOLO_LABELS_DIR / f"{stem}.txt"
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

            cls_name = class_names.get(cls, str(cls)) if class_names else str(cls)
            label_txt = cls_name
            if conf is not None:
                label_txt += f" {conf:.2f}"

            dist = None
            if depth_map_m is not None:
                name_l = cls_name.lower()

                # Objects where the dangerous part is at the bottom of the box
                is_bottom_region = any(
                    k in name_l
                    for k in ("car", "bicycle", "truck", "motorbike")
                )

                # Human / traffic light / tree / electric pole: use middle (center) of bbox
                mode = "bottom" if is_bottom_region else "center"

                # frac controls how tall the sampled region is (0.3 = 30% of bbox height)
                dist = _compute_box_distance(depth_map_m, x1, y1, x2, y2, frac=0.3, mode=mode)

                if dist is not None:
                    label_txt += f" {dist:.2f}m"

            cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), COLOR_YOLO_BOX, 2)
            cv2.putText(
                depth_bgr,
                label_txt,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_YOLO_TEXT,
                1,
                cv2.LINE_AA,
            )

            results.append({
                "id": f"yolo_{idx}",
                "source": "yolo",
                "class_id": cls,
                "class_name": cls_name,
                "confidence": conf,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "distance_m": dist,
            })

    # 7) segmentation borders + distances
    seg_polys = _load_seg_polys_from_border_txt(SEG_BORDER_TXT, W, H)
    depth_bgr = _draw_seg_borders_on(depth_bgr, seg_polys,
                                     color=COLOR_SEG_BORDER, thickness=2)

    if depth_map_m is not None and seg_polys:
        # per-polygon distances
        for idx, poly in enumerate(seg_polys):
            dist = _compute_poly_distance(depth_map_m, poly)
            results.append({
                "id": f"seg_{idx}",
                "source": "segmentation",
                "polygon": poly.reshape(-1, 2).tolist(),
                "distance_m": dist,
            })

        # build sidewalk mask and get nearest sidewalk distance
        sidewalk_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(sidewalk_mask, seg_polys, 1)

        d_min, x_min, y_min = _nearest_sidewalk_distance(
            depth_map_m,
            sidewalk_mask,
            max_depth=80.0,
            band_start_frac=0.5,   # bottom half
            percentile=5.0         # 5th percentile depth
        )

        if d_min is not None:
            print(f"[SIDEWALK] nearest sidewalk: {d_min:.2f} m at ({x_min}, {y_min})")

            # mark nearest sidewalk point
            cv2.circle(depth_bgr, (x_min, y_min), 5, COLOR_SIDEWALK_PT, -1)

            # put distance text above sidewalk mask (above nearest point)
            text = f"{d_min:.2f} m"
            text_x = max(0, x_min - 40)
            text_y = max(10, y_min - 10)

            cv2.putText(depth_bgr, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        COLOR_SIDEWALK_TEXT, 2, cv2.LINE_AA)

            results.append({
                "id": "sidewalk_nearest_point",
                "source": "sidewalk",
                "pixel": {"x": x_min, "y": y_min},
                "distance_m": d_min,
            })
        else:
            print("[SIDEWALK] no valid sidewalk point found")

    # 8) save final overlay
    if not cv2.imwrite(str(FINAL_OUT), depth_bgr):
        raise RuntimeError(f"Failed to save final overlay to {FINAL_OUT}")
    print(f"[PIPELINE] final overlay saved to: {FINAL_OUT}")

    # 9) save JSON distances
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[PIPELINE] JSON saved to: {JSON_OUT}")

    return FINAL_OUT
