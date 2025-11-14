# pipeline.py
from pathlib import Path
import subprocess
import threading
import cv2
import numpy as np
import time

ROOT = Path(r"C:\Python\ObjectDetect4Blind")

YOLO_SCRIPT   = ROOT / "Object detection" / "main.py"
DEPTH_SCRIPT  = ROOT / "Depth-Anything-V2-main" / "run.py"
SEG_SCRIPT    = ROOT / "Segmentation" / "test_model.py"

PY_YOLO   = r"C:\Python\miniconda\envs\tensor_test\python.exe"
PY_DEPTH  = r"C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe"
PY_SEG    = PY_YOLO   # use same env as YOLO for segmentation


def _watch(name: str, proc: subprocess.Popen):
    rc = proc.wait()
    print(f"[{name}] finished with exit code {rc}")


def _ensure_depth_size(depth_bgr, H, W):
    if (depth_bgr.shape[0], depth_bgr.shape[1]) != (H, W):
        depth_bgr = cv2.resize(depth_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth_bgr


def _draw_yolo_boxes_on(depth_bgr, labels_dir: Path, stem: str, W: int, H: int,
                        class_names: dict | None = None):
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

        x1 = max(0, int(px - pw / 2))
        y1 = max(0, int(py - ph / 2))
        x2 = min(W - 1, int(px + pw / 2))
        y2 = min(H - 1, int(py + ph / 2))

        label = class_names.get(cls, str(cls)) if class_names else str(cls)
        if conf is not None:
            label = f"{label} {conf:.2f}"

        cv2.rectangle(depth_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            depth_bgr,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return depth_bgr


def _draw_seg_borders_on(depth_bgr, border_txt_path: Path, W: int, H: int,
                         *, normalized=False, color=(255, 255, 255), thickness=2):
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
        cv2.polylines(
            depth_bgr,
            polys,
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    return depth_bgr


def run_full_pipeline_for_image(
    image_path: Path,
    class_names: dict | None = None,
    seg_args: list[str] | None = None,
) -> Path:
    image_path = Path(image_path).resolve()
    stem = image_path.stem

    YOLO_LABELS_DIR = YOLO_SCRIPT.parent / "output" / "run1" / "labels"
    DEPTH_OUT_PNG   = DEPTH_SCRIPT.parent / "depth_vis" / f"{stem}.png"

    SEG_OUT_DIR     = ROOT / "Segmentation" / "output"
    SEG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEG_BORDER_TXT  = SEG_OUT_DIR / f"{stem}_border.txt"

    FINAL_OUT_DIR   = ROOT / "output"
    FINAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUT       = FINAL_OUT_DIR / f"{stem}_depth_boxes_borders.png"

    print(f"[PIPELINE] image: {image_path}")
    print(f"[PIPELINE] YOLO labels dir: {YOLO_LABELS_DIR}")
    print(f"[PIPELINE] depth PNG: {DEPTH_OUT_PNG}")
    print(f"[PIPELINE] seg border txt: {SEG_BORDER_TXT}")
    print(f"[PIPELINE] final out: {FINAL_OUT}")

    # 1) run 3 external scripts in parallel
    p_yolo = subprocess.Popen(
        [PY_YOLO, str(YOLO_SCRIPT), "--image", str(image_path)],
        cwd=str(YOLO_SCRIPT.parent),
    )

    p_depth = subprocess.Popen(
        [
            PY_DEPTH, "-u", str(DEPTH_SCRIPT),
            "--encoder", "vits",
            "--precision", "int8",
            "--img-path", str(image_path),
            "--outdir", "depth_vis",
            "--pred-only",
        ],
        cwd=str(DEPTH_SCRIPT.parent),
    )

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
    threads = [
        threading.Thread(target=_watch, args=("YOLO", p_yolo), daemon=True),
        threading.Thread(target=_watch, args=("DEPTH", p_depth), daemon=True),
        threading.Thread(target=_watch, args=("SEG", p_seg), daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    print(f"[PIPELINE] total external processes time: {elapsed:.3f} s (~{elapsed/60:.2f} min)")

    # 3) load images
    orig = cv2.imread(str(image_path))
    if orig is None:
        raise FileNotFoundError(f"Original image not found: {image_path}")
    H, W = orig.shape[:2]

    depth_bgr = cv2.imread(str(DEPTH_OUT_PNG))
    if depth_bgr is None:
        raise FileNotFoundError(f"Depth PNG not found: {DEPTH_OUT_PNG}")
    depth_bgr = _ensure_depth_size(depth_bgr, H, W)

    # 4) overlay YOLO boxes
    depth_bgr = _draw_yolo_boxes_on(
        depth_bgr,
        YOLO_LABELS_DIR,
        stem,
        W,
        H,
        class_names=class_names,
    )

    # 5) overlay segmentation borders
    depth_bgr = _draw_seg_borders_on(
        depth_bgr,
        SEG_BORDER_TXT,
        W,
        H,
        normalized=False,
        color=(255, 255, 255),
        thickness=2,
    )

    if not cv2.imwrite(str(FINAL_OUT), depth_bgr):
        raise RuntimeError(f"Failed to save final overlay to {FINAL_OUT}")

    print(f"[PIPELINE] final overlay saved to: {FINAL_OUT}")
    return FINAL_OUT
