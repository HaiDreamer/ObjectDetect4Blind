from pathlib import Path
import subprocess, threading, sys, json
import cv2
import numpy as np
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from utils.process_utils import _watch, _ensure_depth_size

'''
Run demo 1 image fully object detect + segment + evaluate distance per bb and region from segment
'''

# Paths and Python environments
ROOT = Path(__file__).resolve().parent
YOLO_SCRIPT   = ROOT / "Object_detection" / "main.py"
DEPTH_SCRIPT  = ROOT / "Depth-Anything-V2-main" / "run.py"
SEG_SCRIPT    = ROOT / "Segmentation" / "test_model.py"         

with open(ROOT / "src" / "configs" / "config.json", "r", encoding="utf-8") as _cf:
    _APP_CFG = json.load(_cf)

PY_YOLO  = _APP_CFG["python_interpreters"]["yolo"]
PY_DEPTH = _APP_CFG["python_interpreters"]["depth"]
PY_SEG   = _APP_CFG["python_interpreters"]["seg"]

# Inputs/outputs
ORIG_IMG   = ROOT / "assets" / "demo01.jpg"                                          
YOLO_LABELS_DIR = YOLO_SCRIPT.parent / "output" / "run1" / "labels" 
DEPTH_OUT_PNG   = DEPTH_SCRIPT.parent / "depth_vis" / f"{ORIG_IMG.stem}.png"
# SEG_BORDER_TXT  = ROOT / "Segmentation" / "output" / "mask_border.txt"
# Use per-image border file so it matches test_model.py --out-border
SEG_BORDER_TXT  = ROOT / "Segmentation" / "output" / f"{ORIG_IMG.stem}_border.txt"
FINAL_OUT       = ROOT / "output" / f"{ORIG_IMG.stem}_depth_boxes_borders.png"


# Helpers
def _draw_yolo_boxes_on(depth_bgr, labels_dir: Path, stem: str, W: int, H: int, class_names: dict | None = None):
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
        x2 = min(W - 1, int(px + pw / 2)); y2 = min(H - 1, int(py + ph / 2))

        label = (class_names.get(cls, str(cls)) if class_names else str(cls))
        if conf is not None:
            label = f"{label} {conf:.2f}"

        cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(depth_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return depth_bgr

def _draw_seg_borders_on(depth_bgr, border_txt_path: Path, W: int, H: int, *, normalized=False,
                         color=(255, 255, 255), thickness=2):
    """Draw segmentation border polylines from .txt (x1 y1 x2 y2 ...) onto depth image."""
    if not border_txt_path.exists():
        print(f"[SEG] border file not found: {border_txt_path}")
        return depth_bgr

    with open(border_txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    polys = []
    for ln in lines:
        vals = ln.split()
        # Line format is "class_id conf x1 y1 x2 y2 ..." — skip the first
        # two fields, otherwise class_id/conf get read in as a phantom
        # (x, y) vertex near the image origin.
        if len(vals) < 2 + 6:
            continue
        coords = vals[2:]
        if len(coords) % 2 != 0:
            continue
        pts = []
        it = iter(map(float, coords))
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
        # Draw many polylines at once
        cv2.polylines(depth_bgr, polys, isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return depth_bgr

def run_parallel_and_overlay(class_names: dict | None = None, seg_args: list[str] | None = None):
    """
    Launch 3 procs in parallel:
      - Object detection (YOLO)
      - Depth estimation (Depth Anything V2)
      - Segmentation (test_model.py)
    Then overlay YOLO boxes + seg borders on depth PNG.
    """
    # Object detection
    p_yolo = subprocess.Popen(
        [PY_YOLO, str(YOLO_SCRIPT), "--image", str(ORIG_IMG)],
        cwd=str(YOLO_SCRIPT.parent)
    )
    # Depth estimation
    image_path = ORIG_IMG  # Path(r"C:\Python\ObjectDetect4Blind\assets\demo01.jpg")
    p_depth = subprocess.Popen(
        [
            PY_DEPTH, "-u", str(DEPTH_SCRIPT),
            "--encoder", "vits",
            "--precision", "int8",
            "--img-path", str(image_path),
            "--outdir", "depth_vis",
            "--pred-only",
        ],
        cwd=str(DEPTH_SCRIPT.parent)
    )
    # Image segmentation
    # seg_cmd = [PY_SEG, str(SEG_SCRIPT)]
    # if seg_args:
    #     seg_cmd.extend(seg_args)
    # p_seg = subprocess.Popen(seg_cmd, cwd=str(ROOT))
    seg_cmd = [
        PY_SEG,
        str(SEG_SCRIPT),
        "--image",
        str(ORIG_IMG),
        "--out-border",
        str(SEG_BORDER_TXT),
    ]
    if seg_args:
        seg_cmd.extend(seg_args)
    p_seg = subprocess.Popen(seg_cmd, cwd=str(ROOT))

    # wait with lightweight watcher threads (join waits until thread finishes)
    t1 = threading.Thread(target=_watch, args=("YOLO", p_yolo), daemon=True)
    t2 = threading.Thread(target=_watch, args=("DEPTH", p_depth), daemon=True)
    t3 = threading.Thread(target=_watch, args=("SEG", p_seg), daemon=True)
    for t in (t1, t2, t3):
        t.start()
    for t in (t1, t2, t3):
        t.join()

    # Load images and draw overlays
    orig = cv2.imread(str(ORIG_IMG))
    if orig is None:
        raise FileNotFoundError(f"Original image not found: {ORIG_IMG}")
    H, W = orig.shape[:2]

    depth_bgr = cv2.imread(str(DEPTH_OUT_PNG))
    if depth_bgr is None:
        raise FileNotFoundError(f"Depth PNG not found: {DEPTH_OUT_PNG}")
    depth_bgr = _ensure_depth_size(depth_bgr, H, W)

    # YOLO boxes
    depth_bgr = _draw_yolo_boxes_on(depth_bgr, YOLO_LABELS_DIR, ORIG_IMG.stem, W, H, class_names=class_names)
    # Segmentation border
    depth_bgr = _draw_seg_borders_on(depth_bgr, SEG_BORDER_TXT, W, H, normalized=False,
                                     color=(255, 255, 255), thickness=2)

    FINAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(FINAL_OUT), depth_bgr):
        raise RuntimeError(f"Failed to save final overlay to {FINAL_OUT}")
    print(f"[FINAL] saved: {FINAL_OUT}")

if __name__ == "__main__":
    # Calculate run time
    _t0 = time.perf_counter()
    try:
        # Optionally pass human-readable names for class ids (0:"person", etc.)
        CLASS_NAMES = None
        # If your segmentation script accepts args, add them here:
        SEG_ARGS = None  # e.g., ["--something", "value"]
        run_parallel_and_overlay(CLASS_NAMES, SEG_ARGS)
    finally:
        elapsed = time.perf_counter() - _t0
        print(f"[RUNTIME] total elapsed: {elapsed:.3f}s (~{elapsed/60:.2f} min)")