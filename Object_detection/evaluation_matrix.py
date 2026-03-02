# eval_confusion_all_objects.py
from pathlib import Path
import json
import csv
import time
import traceback
import shutil

import numpy as np
from ultralytics import YOLO


# ----------------- CONFIG -----------------
DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\GroupProject_OD\data.yaml")

MODELS = [
    Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8_static_qdq.onnx")
]

TASK = "detect"
IMGSZ = 640
BATCH = 1
DEVICE = "cpu"
WORKERS = 0

# MUST be True if you want Ultralytics to compute & save confusion matrix + val_batch images
PLOTS = True

OUT_DIR = Path(__file__).resolve().parent / "eval_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- HELPERS -----------------
def sanitize_run_name(p: Path) -> str:
    name = p.stem
    for ch in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        name = name.replace(ch, "_")
    return name


def extract_metrics(val_result):
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

    box = getattr(val_result, "box", None)
    if box is not None:
        for k in ("mp", "mr", "map50", "map75", "map"):
            if hasattr(box, k):
                try:
                    out[f"box.{k}"] = float(getattr(box, k))
                except Exception:
                    out[f"box.{k}"] = getattr(box, k)

    if hasattr(val_result, "save_dir"):
        try:
            out["save_dir"] = str(val_result.save_dir)
        except Exception:
            pass

    if not out:
        out["raw"] = str(val_result)
    return out


def write_matrix_csv(path: Path, labels: list[str], matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["predicted\\true"] + labels
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, row_label in enumerate(labels):
            w.writerow([row_label] + [matrix[i, j] for j in range(len(labels))])


def plot_matrix_png(matrix: np.ndarray, labels: list[str], title: str, out_path: Path, normalize: bool):
    # Minimal matplotlib plot (no seaborn) similar to Ultralytics
    import matplotlib.pyplot as plt

    mat = matrix.astype(np.float64)
    if normalize:
        mat = mat / (mat.sum(0, keepdims=True) + 1e-9)  # column-normalize (True class) like Ultralytics

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, interpolation="none")

    ax.set_title(title + (" Normalized" if normalize else ""), pad=20)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha="center")
    ax.set_yticklabels(labels)

    # annotate if small enough
    if len(labels) <= 30:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mat[i, j]
                if normalize:
                    txt = f"{val:.2f}"
                else:
                    txt = str(int(val))
                ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def export_confusions(val_result, run_dir: Path) -> dict:
    cm = getattr(val_result, "confusion_matrix", None)
    if cm is None:
        raise RuntimeError("No confusion_matrix found. Ensure model.val(plots=True) ran successfully.")

    names = list(getattr(cm, "names", {}).values())
    nc = len(names)

    # Detection confusion matrix shape is (nc+1, nc+1) with last index = background
    mat_full = np.array(getattr(cm, "matrix", None))
    if mat_full is None or mat_full.size == 0:
        raise RuntimeError("Confusion matrix is empty. Ensure plots=True.")

    labels_full = names + ["background"]
    labels_obj = names  # object-only

    out = {}
    conf_dir = run_dir / "confusion"
    conf_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) FULL (objects + background) CSV ---
    full_counts_csv = conf_dir / "confusion_full_counts.csv"
    write_matrix_csv(full_counts_csv, labels_full, mat_full.astype(int))

    full_norm_csv = conf_dir / "confusion_full_normalized.csv"
    mat_full_norm = mat_full / (mat_full.sum(0, keepdims=True) + 1e-9)
    write_matrix_csv(full_norm_csv, labels_full, np.round(mat_full_norm, 6))

    # --- 2) FULL images saved via Ultralytics method (guaranteed filenames) ---
    # cm.plot() creates:
    # - confusion_matrix.png
    # - confusion_matrix_normalized.png
    cm.plot(normalize=False, save_dir=str(conf_dir))
    cm.plot(normalize=True, save_dir=str(conf_dir))

    # --- 3) OBJECTS-ONLY (exclude background row/col) ---
    mat_obj = mat_full[:nc, :nc]

    obj_counts_csv = conf_dir / "confusion_objects_only_counts.csv"
    write_matrix_csv(obj_counts_csv, labels_obj, mat_obj.astype(int))

    obj_norm_csv = conf_dir / "confusion_objects_only_normalized.csv"
    mat_obj_norm = mat_obj / (mat_obj.sum(0, keepdims=True) + 1e-9)
    write_matrix_csv(obj_norm_csv, labels_obj, np.round(mat_obj_norm, 6))

    # OBJECTS-ONLY images
    plot_matrix_png(mat_obj, labels_obj, "Confusion Matrix Objects Only", conf_dir / "confusion_objects_only.png", normalize=False)
    plot_matrix_png(mat_obj, labels_obj, "Confusion Matrix Objects Only", conf_dir / "confusion_objects_only_normalized.png", normalize=True)

    # Per-class TP/FP/FN (object classes only; FP/FN include background effects via full matrix)
    per_class = []
    for c in range(nc):
        tp = float(mat_full[c, c])
        fp = float(mat_full[c, :].sum() - tp)  # predicted=c but true!=c (includes true=background)
        fn = float(mat_full[:, c].sum() - tp)  # true=c but predicted!=c (includes predicted=background)
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = (2 * prec * rec) / (prec + rec + 1e-9)

        per_class.append({
            "class_id": c,
            "class_name": names[c],
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
        })

    per_class_csv = conf_dir / "per_class_from_confusion.csv"
    with per_class_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0].keys()) if per_class else [])
        w.writeheader()
        for r in per_class:
            w.writerow(r)

    out.update({
        "confusion_dir": str(conf_dir),
        "full_counts_csv": str(full_counts_csv),
        "full_norm_csv": str(full_norm_csv),
        "obj_counts_csv": str(obj_counts_csv),
        "obj_norm_csv": str(obj_norm_csv),
        "per_class_csv": str(per_class_csv),
        "full_img": str(conf_dir / "confusion_matrix.png"),
        "full_img_norm": str(conf_dir / "confusion_matrix_normalized.png"),
        "obj_img": str(conf_dir / "confusion_objects_only.png"),
        "obj_img_norm": str(conf_dir / "confusion_objects_only_normalized.png"),
    })
    return out


def copy_val_visuals(run_dir: Path) -> Path:
    """
    Copy Ultralytics-produced visuals (val_batch images + curves + confusion images) into run_dir/visuals_copy
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
    for pat in patterns:
        for p in run_dir.glob(pat):
            shutil.copy2(p, dst / p.name)
            copied += 1

    # Also check nested "plots" folders in some versions
    for sub in ["plots", "val", ""]:
        subdir = run_dir / sub
        if subdir.exists() and subdir.is_dir():
            for pat in patterns:
                for p in subdir.glob(pat):
                    shutil.copy2(p, dst / p.name)
                    copied += 1

    (dst / "_copied_count.txt").write_text(str(copied), encoding="utf-8")
    return dst


def main():
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    summary = []

    for model_path in MODELS:
        row = {"model": str(model_path), "ok": False}

        if not model_path.exists():
            row["error"] = f"Model not found: {model_path}"
            summary.append(row)
            continue

        try:
            run_name = sanitize_run_name(model_path)
            print("Validating:", model_path)
            print("Run name:", run_name)

            model = YOLO(str(model_path), task=TASK)

            t0 = time.time()
            val_result = model.val(
                data=str(DATA_YAML),
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                split="val",
                plots=PLOTS,          # REQUIRED for confusion matrix + val_batch visuals
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

            # Export confusion matrices + images
            conf_info = export_confusions(val_result, run_dir)
            row.update({f"conf.{k}": v for k, v in conf_info.items()})

            # Copy val visuals into one folder (optional convenience)
            visuals_dir = copy_val_visuals(run_dir)
            row["visuals_dir"] = str(visuals_dir)

        except Exception as e:
            row["error"] = str(e)
            row["traceback"] = traceback.format_exc()

        summary.append(row)

    # Save global summary JSON/CSV
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