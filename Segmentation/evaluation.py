from pathlib import Path
import json
import csv
import time
import traceback
from ultralytics import YOLO

'''
OUTPUT
'''

# ----------------- CONFIG -----------------
DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\data.yaml")

MODELS = [
    # Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg.pt"),
    # Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp32.onnx"),
    # Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp16.onnx"),
    # Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_int8_static_qdq.onnx"),
    Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp32.onnx"),
]

TASK = "segment"    # segmentation task (Ultralytics uses 'segment')
IMGSZ = 640
BATCH = 1
DEVICE = "cpu"      # change to 0 if using GPU + proper runtime/provider

OUT_DIR = Path(__file__).resolve().parent / "eval_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------- METRICS EXTRACTOR -----------------
def extract_metrics(val_result):
    """
    Tries to extract common segmentation metrics.

    Ultralytics segmentation metrics keys include both:
      - Box metrics:  metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B)
      - Mask metrics: metrics/precision(M), metrics/recall(M), metrics/mAP50(M), metrics/mAP50-95(M)
    :contentReference[oaicite:2]{index=2}
    """
    out = {}

    # Case 1: dict already
    if isinstance(val_result, dict):
        out.update(val_result)
        return out

    # Case 2: has results_dict (common in Ultralytics metrics objects)
    if hasattr(val_result, "results_dict"):
        try:
            rd = val_result.results_dict
            if isinstance(rd, dict):
                out.update(rd)
        except Exception:
            pass

    # Add fitness if available
    if hasattr(val_result, "fitness"):
        try:
            out["fitness"] = float(val_result.fitness)
        except Exception:
            out["fitness"] = val_result.fitness

    # Add speed if available (some versions store it in metrics object)
    if hasattr(val_result, "speed"):
        try:
            sp = val_result.speed
            if isinstance(sp, dict):
                for k, v in sp.items():
                    out[f"speed.{k}"] = float(v) if isinstance(v, (int, float)) else v
        except Exception:
            pass

    # Fallback: pull from .box and .seg (SegmentMetrics has both) :contentReference[oaicite:3]{index=3}
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

    # If still nothing worked, store string
    if not out:
        out["raw"] = str(val_result)

    return out


def main():
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    summary = []

    for model_path in MODELS:
        row = {"model": str(model_path), "ok": False}

        if not Path(model_path).exists():
            row["error"] = f"Model not found: {model_path}"
            summary.append(row)
            continue

        try:
            print("\n==============================")
            print("Validating:", model_path)
            print("==============================")

            model = YOLO(str(model_path), task=TASK)

            # Run validation + measure wall time
            t0 = time.time()
            val_result = model.val(
                data=str(DATA_YAML),
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                split="val",
                plots=False,
                verbose=True,
            )
            dt = time.time() - t0

            metrics = extract_metrics(val_result)
            row.update(metrics)
            row["val_time_sec"] = round(dt, 3)
            row["ok"] = True

        except Exception as e:
            row["error"] = str(e)
            row["traceback"] = traceback.format_exc()

        summary.append(row)

    # Save JSON
    out_json = OUT_DIR / "val_summary.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nSaved JSON:", out_json)

    # Save CSV (flatten keys)
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
