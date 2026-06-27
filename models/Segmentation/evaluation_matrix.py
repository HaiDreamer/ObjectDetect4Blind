# eval_seg_confusion_all_objects.py
from pathlib import Path
import json
import csv
import time
import traceback
import shutil

import numpy as np
import yaml
from ultralytics import YOLO


r"""
SEGMENTATION + CONFUSION MATRIX EXPORT

Confusion matrix format matches Ultralytics plot:
- Rows = Predicted
- Columns = True
- Last row/col = background (for detect/segment validators)

Outputs (per model) under:
  eval_output/<run_name>/
    confusion/
      confusion_full_counts.csv
      confusion_full_normalized.csv
      confusion_full_normalized_list.json   
      confusion_matrix.png
      confusion_matrix_normalized.png
    visuals_copy/
      val_batch*_labels.jpg
      val_batch*_pred.jpg
      *_curve.png
      results*.png
"""

# ----------------- CONFIG -----------------
DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\data.yaml")

MODELS = [
    Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_int8_static_qdq.onnx"),
]

TASK = "segment"
IMGSZ = 640
BATCH = 1
DEVICE = "cpu"
WORKERS = 0

# MUST be True: otherwise validator won't compute/store confusion matrix plots
PLOTS = True

OUT_DIR = Path(__file__).resolve().parent / "eval_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- LABEL CONVERSION (YOUR LOGIC) -----------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def bbox_to_rect_poly_line(parts):
    """
    parts: [cls, cx, cy, w, h] (+ optional track_id)
    Converts bbox to a 4-corner rectangle polygon:
      cls x1 y1 x2 y1 x2 y2 x1 y2
    """
    cls = parts[0]

    # Support optional track_id
    track_id = parts[5] if len(parts) >= 6 else None

    cx, cy, w, h = map(float, parts[1:5])
    x1, y1 = clamp01(cx - w / 2), clamp01(cy - h / 2)
    x2, y2 = clamp01(cx + w / 2), clamp01(cy + h / 2)

    poly = [x1, y1, x2, y1, x2, y2, x1, y2]
    line = cls + " " + " ".join(f"{p:.6f}" for p in poly)

    if track_id is not None:
        line += f" {track_id}"
    return line


def convert_labels_dir(lbl_dir: Path):
    changed_files = 0
    changed_lines = 0

    for txt in lbl_dir.rglob("*.txt"):
        lines = [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue

        new_lines = []
        changed = False

        for line in lines:
            parts = line.split()
            # bbox detection labels are usually 5 tokens (or 6 with optional track_id)
            if len(parts) in (5, 6):
                new_lines.append(bbox_to_rect_poly_line(parts))
                changed = True
                changed_lines += 1
            else:
                # already polygon segmentation (variable length)
                new_lines.append(line)

        if changed:
            shutil.copy2(txt, str(txt) + ".bak")  # backup
            txt.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            changed_files += 1

    return changed_files, changed_lines


def dataset_root_from_yaml(data_yaml: Path) -> Path:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = cfg.get("path", None)
    if root:
        root = Path(root)
        if not root.is_absolute():
            root = (data_yaml.parent / root).resolve()
        return root
    return data_yaml.parent.resolve()


# ----------------- METRICS EXTRACTOR (SEG) -----------------
def extract_metrics(val_result):
    """
    Extract common segmentation metrics (box + mask) in a version-tolerant way.
    """
    out = {}

    if isinstance(val_result, dict):
        out.update(val_result)
        return out

    if hasattr(val_result, "results_dict"):
        try:
            rd = val_result.results_dict
            if isinstance(rd, dict):
                out.update(rd)
        except Exception:
            pass

    if hasattr(val_result, "fitness"):
        try:
            out["fitness"] = float(val_result.fitness)
        except Exception:
            out["fitness"] = val_result.fitness

    if hasattr(val_result, "speed"):
        try:
            sp = val_result.speed
            if isinstance(sp, dict):
                for k, v in sp.items():
                    out[f"speed.{k}"] = float(v) if isinstance(v, (int, float)) else v
        except Exception:
            pass

    box = getattr(val_result, "box", None)
    seg = getattr(val_result, "seg", None)

    def grab_metric(obj, prefix):
        if obj is None:
            return
        for k in ("mp", "mr", "map50", "map75", "map"):
            if hasattr(obj, k):
                try:
                    out[f"{prefix}.{k}"] = float(getattr(obj, k))
                except Exception:
                    out[f"{prefix}.{k}"] = getattr(obj, k)

    grab_metric(box, "box")
    grab_metric(seg, "mask")

    if hasattr(val_result, "save_dir"):
        try:
            out["save_dir"] = str(val_result.save_dir)
        except Exception:
            pass

    if not out:
        out["raw"] = str(val_result)

    return out


# ----------------- CONFUSION EXPORT -----------------
def sanitize_run_name(p: Path) -> str:
    name = p.stem
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, "_")
    return name


def write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["predicted\\true"] + labels
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + [matrix[i, j] for j in range(len(labels))])


def export_confusions(val_result, run_dir: Path) -> dict:
    """
    Exports:
      - counts CSV
      - normalized CSV (Ultralytics-style: column-normalize)
      - normalized list-of-lists JSON (rounded to 2 decimals)
      - confusion_matrix.png and confusion_matrix_normalized.png via cm.plot()
    """
    cm = getattr(val_result, "confusion_matrix", None)
    if cm is None:
        raise RuntimeError("No confusion_matrix found. Ensure model.val(..., plots=True).")

    names = list(getattr(cm, "names", {}).values())
    nc = len(names)

    mat_full = np.array(getattr(cm, "matrix", None))
    if mat_full is None or mat_full.size == 0:
        raise RuntimeError("Confusion matrix is empty. Ensure plots=True.")

    labels_full = names + ["background"]

    conf_dir = run_dir / "confusion"
    conf_dir.mkdir(parents=True, exist_ok=True)

    # raw counts
    counts_csv = conf_dir / "confusion_full_counts.csv"
    write_matrix_csv(counts_csv, labels_full, mat_full.astype(int))

    # normalized like Ultralytics plot: normalize each TRUE column
    mat_norm = mat_full / (mat_full.sum(0, keepdims=True) + 1e-9)
    norm_csv = conf_dir / "confusion_full_normalized.csv"
    write_matrix_csv(norm_csv, labels_full, np.round(mat_norm, 6))

    # Make list-of-lists like your screenshot (2 decimals; optionally clamp tiny values to 0.00)
    mat_norm_display = mat_norm.copy()
    mat_norm_display[mat_norm_display < 0.005] = 0.0  # optional: mimics "0.00" dominance in plot
    mat_norm_display = np.round(mat_norm_display, 2)

    norm_list_json = conf_dir / "confusion_full_normalized_list.json"
    norm_list_json.write_text(
        json.dumps(mat_norm_display.tolist(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # save Ultralytics plot images
    cm.plot(normalize=False, save_dir=str(conf_dir))
    cm.plot(normalize=True, save_dir=str(conf_dir))

    return {
        "confusion_dir": str(conf_dir),
        "counts_csv": str(counts_csv),
        "normalized_csv": str(norm_csv),
        "normalized_list_json": str(norm_list_json),
        "img_counts": str(conf_dir / "confusion_matrix.png"),
        "img_normalized": str(conf_dir / "confusion_matrix_normalized.png"),
    }


def copy_val_visuals(run_dir: Path) -> Path:
    """
    Copy Ultralytics-produced visuals (val_batch images + curves + results) into run_dir/visuals_copy.
    """
    dst = run_dir / "visuals_copy"
    dst.mkdir(parents=True, exist_ok=True)

    patterns = [
        "val_batch*_labels.jpg",
        "val_batch*_pred.jpg",
        "*_curve.png",
        "confusion_matrix*.png",
        "results*.png",
    ]

    copied = 0
    # direct run_dir
    for pat in patterns:
        for p in run_dir.glob(pat):
            shutil.copy2(p, dst / p.name)
            copied += 1

    # some versions use nested folders
    for sub in ["plots", "val"]:
        subdir = run_dir / sub
        if subdir.exists() and subdir.is_dir():
            for pat in patterns:
                for p in subdir.glob(pat):
                    shutil.copy2(p, dst / p.name)
                    copied += 1

    (dst / "_copied_count.txt").write_text(str(copied), encoding="utf-8")
    return dst


# ----------------- MAIN -----------------
def main():
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    # Convert bbox labels -> rectangle polygon labels (if needed)
    dataset_root = dataset_root_from_yaml(DATA_YAML)
    val_labels_dir = dataset_root / "labels" / "val"
    if val_labels_dir.exists():
        f, l = convert_labels_dir(val_labels_dir)
        print(f"[labels/val] converted files: {f}, bbox-lines converted: {l}")
    else:
        print("[WARN] labels/val not found:", val_labels_dir)

    summary = []

    for model_path in MODELS:
        row = {"model": str(model_path), "ok": False}

        if not model_path.exists():
            row["error"] = f"Model not found: {model_path}"
            summary.append(row)
            continue

        try:
            run_name = sanitize_run_name(model_path)
            print("\n==============================")
            print("Validating:", model_path)
            print("Run name:", run_name)
            print("==============================")

            model = YOLO(str(model_path), task=TASK)

            t0 = time.time()
            val_result = model.val(
                data=str(DATA_YAML),
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                split="val",
                plots=PLOTS,          # IMPORTANT
                project=str(OUT_DIR),
                name=run_name,
                verbose=True,
                workers=WORKERS,
            )
            dt = time.time() - t0

            row.update(extract_metrics(val_result))
            row["val_time_sec"] = round(dt, 3)
            row["ok"] = True

            run_dir = Path(getattr(val_result, "save_dir", OUT_DIR / run_name))
            row["run_dir"] = str(run_dir)

            # Export confusion matrix (Pred x True + background)
            conf_info = export_confusions(val_result, run_dir)
            row.update({f"conf.{k}": v for k, v in conf_info.items()})

            # Copy visuals for convenience
            row["visuals_dir"] = str(copy_val_visuals(run_dir))

        except Exception as e:
            row["error"] = str(e)
            row["traceback"] = traceback.format_exc()

        summary.append(row)

    # Save global summary
    out_json = OUT_DIR / "val_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved JSON:", out_json)

    keys = set()
    for r in summary:
        keys.update(r.keys())
    keys = ["model", "ok"] + sorted(k for k in keys if k not in ("model", "ok"))

    out_csv = OUT_DIR / "val_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    print("Saved CSV:", out_csv)


if __name__ == "__main__":
    main()