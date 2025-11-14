from pathlib import Path
import subprocess, threading, sys
import cv2
import numpy as np
import time

# =========================
# Paths and Python envs
# =========================
ROOT = Path(__file__).resolve().parent

# Scripts
YOLO_SCRIPT          = ROOT / "Object detection" / "main.py"
METRIC_DEPTH_SCRIPT  = ROOT / "Depth-Anything-V2-main" / "metric_depth" / "run.py"
SEG_SCRIPT           = ROOT / "Segmentation" / "test_model.py"

# Python interpreters
PY_YOLO   = r"C:\Python\miniconda\envs\tensor_test\python.exe"
PY_DEPTH  = r"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe"
PY_SEG    = PY_YOLO

# =========================
# Inputs / Outputs
# =========================
ORIG_IMG         = ROOT / "assets" / "demo01.jpg"

# YOLO labels (same as your original script)
YOLO_LABELS_DIR  = YOLO_SCRIPT.parent / "output" / "run1" / "labels"

# Metric-depth outputs
METRIC_DEPTH_OUT_DIR   = Path(r"C:\Python\ObjectDetect4Blind\output_metric_depth")
METRIC_DEPTH_VIS_PNG   = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}.png"
METRIC_DEPTH_RAW_NPY   = METRIC_DEPTH_OUT_DIR / f"{ORIG_IMG.stem}_metric_depth.npy"

# Segmentation border txt (same as before)
SEG_BORDER_TXT   = ROOT / "Segmentation" / "output" / "mask_border.txt"

# Final overlay
FINAL_OUT        = ROOT / "output" / f"{ORIG_IMG.stem}_metric_depth_boxes_borders.png"


# =========================
# Helpers
# =========================
def _watch(name: str, proc: subprocess.Popen):
    """Wait for the child process to exit and log its rc."""
    rc = proc.wait()
    print(f"[{name}] finished with exit code {rc}")


def _ensure_depth_size(depth_bgr, H, W):
    if depth_bgr is None:
        return None
    if (depth_bgr.shape[0], depth_bgr.shape[1]) != (H, W):
        depth_bgr = cv2.resize(depth_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth_bgr


def _draw_yolo_boxes_on(depth_bgr, labels_dir: Path, stem: str, W: int, H: int,
                        class_names: dict | None = None):
    """Draw YOLO-format labels (class cx cy w h [conf]) on depth image."""
    label_file = labels_dir / f"{stem}.txt"
    if not label_file.exists():
        print(f"[YOLO] label file not found: {label_file}")
        return depth_bgr

    with open(label_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, ww, hh = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None

        px, py = cx * W, cy * H
        pw, ph = ww * W, hh * H
        x1 = max(0, int(px - pw / 2)); y1 = max(0, int(py - ph / 2))
        x2 = min(W - 1, int(px + pw / 2)); y2 = min(0 + H - 1, int(py + ph / 2))

        label = (class_names.get(cls, str(cls)) if class_names else str(cls))
        if conf is not None:
            label = f"{label} {conf:.2f}"

        cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(depth_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return depth_bgr


def _draw_seg_borders_on(depth_bgr, border_txt_path: Path, W: int, H: int, *,
                         normalized=False, color=(255, 255, 255), thickness=2):
    """Draw segmentation border polylines from .txt (x1 y1 x2 y2 ...) onto depth image."""
    if not border_txt_path.exists():
        print(f"[SEG] border file not found: {border_txt_path}")
        return depth_bgr

    with open(border_txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    polys = []
    for ln in lines:
        vals = ln.split()
        if len(vals) < 4 or len(vals) % 2 != 0:
            continue
        pts = []
        it = iter(map(float, vals))
        if normalized:
            for x, y in zip(it, it):
                pts.append([int(round(x * W)), int(round(y * H))])
        else:
            for x, y in zip(it, it):
                pts.append([int(round(x)), int(round(y))])

        if len(pts) >= 2:
            poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            polys.append(poly)

    if polys:
        cv2.polylines(depth_bgr, polys, isClosed=True,
                      color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return depth_bgr


def _compute_box_distance(depth_map_m: np.ndarray,
                          x1: int, y1: int, x2: int, y2: int) -> float | None:
    """
    Example: compute an average distance (in meters) inside a bounding box
    from a metric depth map (H x W, values in meters).
    Returns None if box is invalid or depth has no valid pixels.
    """
    H, W = depth_map_m.shape[:2]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    patch = depth_map_m[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    # Mask out zeros / negatives if they mean invalid depth in your output
    valid = patch[patch > 0]
    if valid.size == 0:
        return None

    return float(valid.mean())


# =========================
# Core pipeline
# =========================
def run_parallel_and_overlay_metric(class_names: dict | None = None,
                                    seg_args: list[str] | None = None):
    """
    Launch in parallel:
      - YOLO object detection
      - Metric depth estimation
      - Segmentation (optional)

    Then:
      - load metric depth visualization PNG
      - overlay YOLO boxes + seg borders
      - (optionally) compute & print distance per object using metric depth .npy
    """

    # 1) Start YOLO
    p_yolo = subprocess.Popen(
        [PY_YOLO, str(YOLO_SCRIPT), "--image", str(ORIG_IMG)],
        cwd=str(YOLO_SCRIPT.parent)
    )

    # 2) Start Metric Depth
    # Adjust arguments to match your metric_depth\run.py
    metric_cmd = [
        PY_DEPTH, str(METRIC_DEPTH_SCRIPT),
        "--encoder", "vits",
        "--load-from", r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth",
        "--max-depth", "80",
        "--img-path", str(ORIG_IMG),
        "--outdir", str(METRIC_DEPTH_OUT_DIR),
        "--pred-only",
        "--save-numpy", 
    ]
    p_depth = subprocess.Popen(metric_cmd, cwd=str(METRIC_DEPTH_SCRIPT.parent))

    # 3) Start Segmentation
    seg_cmd = [PY_SEG, str(SEG_SCRIPT)]
    if seg_args:
        seg_cmd.extend(seg_args)
    p_seg = subprocess.Popen(seg_cmd, cwd=str(ROOT))

    # 4) Wait for all with watcher threads
    t1 = threading.Thread(target=_watch, args=("YOLO", "p_yolo"), daemon=True)
    t2 = threading.Thread(target=_watch, args=("METRIC_DEPTH", p_depth), daemon=True)
    t3 = threading.Thread(target=_watch, args=("SEG", p_seg), daemon=True)
    for t in (t1, t2, t3):
        t.start()
    for t in (t1, t2, t3):
        t.join()

    # 5) Load original image to get size
    orig = cv2.imread(str(ORIG_IMG))
    if orig is None:
        raise FileNotFoundError(f"Original image not found: {ORIG_IMG}")
    H, W = orig.shape[:2]

    # 6) Load metric depth visualization PNG
    depth_bgr = cv2.imread(str(METRIC_DEPTH_VIS_PNG))
    if depth_bgr is None:
        raise FileNotFoundError(f"Metric depth PNG not found: {METRIC_DEPTH_VIS_PNG}")
    depth_bgr = _ensure_depth_size(depth_bgr, H, W)

    # 7) (Optional) load metric depth raw map
    depth_map_m = None
    if METRIC_DEPTH_RAW_NPY.exists():
        depth_map_m = np.load(str(METRIC_DEPTH_RAW_NPY))
        # ensure shape (H, W)
        if depth_map_m.shape[:2] != (H, W):
            depth_map_m = cv2.resize(
                depth_map_m.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            )

    # 8) Overlay YOLO boxes, optionally annotating distance
    label_file = YOLO_LABELS_DIR / f"{ORIG_IMG.stem}.txt"
    if not label_file.exists():
        print(f"[YOLO] label file not found: {label_file}")
    else:
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for ln in lines:
            parts = ln.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, ww, hh = map(float, parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None

            px, py = cx * W, cy * H
            pw, ph = ww * W, hh * H
            x1 = max(0, int(px - pw / 2)); y1 = max(0, int(py - ph / 2))
            x2 = min(W - 1, int(px + pw / 2)); y2 = min(H - 1, int(py + ph / 2))

            # Build label
            label = (class_names.get(cls, str(cls)) if class_names else str(cls))
            if conf is not None:
                label = f"{label} {conf:.2f}"

            # If metric depth map exists, compute average distance inside box
            if depth_map_m is not None:
                dist = _compute_box_distance(depth_map_m, x1, y1, x2, y2)
                if dist is not None:
                    # Append distance in meters
                    label += f" {dist:.2f}m"

            # Draw on depth image
            cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(depth_bgr, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # 9) Overlay segmentation borders
    depth_bgr = _draw_seg_borders_on(depth_bgr, SEG_BORDER_TXT, W, H,
                                     normalized=False, color=(255, 255, 255), thickness=2)

    # 10) Save final result
    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(FINAL_OUT), depth_bgr):
        raise RuntimeError(f"Failed to save final overlay to {FINAL_OUT}")
    print(f"[FINAL] saved: {FINAL_OUT}")


if __name__ == "__main__":
    _t0 = time.perf_counter()
    try:
        CLASS_NAMES = None  # e.g., {0: "person", 1: "car", ...}
        SEG_ARGS = None     # e.g., ["--something", "value"]
        run_parallel_and_overlay_metric(CLASS_NAMES, SEG_ARGS)
    finally:
        elapsed = time.perf_counter() - _t0
        print(f"[RUNTIME] total elapsed: {elapsed:.3f}s (~{elapsed/60:.2f} min)")
