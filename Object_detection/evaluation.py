from pathlib import Path
import json
import csv
import time
import traceback
from ultralytics import YOLO

'''
FOR: checking accuracy of model

RESULT
model,ok,error,fitness,metrics/mAP50(B),metrics/mAP50-95(B),metrics/precision(B),metrics/recall(B),traceback,val_time_sec
C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp32.onnx,True,,0.5451066511344781,0.7239038738003981,0.5451066511344781,0.7412172402367192,0.6972859594105918,,839.858

C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp16.onnx
        Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1817/1817 2.2it/s 13:32
                all       1817      12444      0.741      0.697      0.724      0.545
            bicycle        186        307      0.783      0.704       0.74      0.488
                bus         53         81      0.864      0.782      0.829      0.632
                car        910       2460      0.819      0.803      0.845      0.722
        electric pole        382        408      0.679      0.564      0.594      0.354
            motocycle       1024       3460      0.779      0.836      0.823      0.617
pedestrian crossing sign        413        587      0.916      0.944      0.954       0.83
                person       1105       3753       0.84      0.757      0.832      0.637
                tree        536       1296      0.495      0.472      0.447      0.243
                truck         72         92      0.499      0.411      0.451      0.379
    Speed: 0.4ms preprocess, 430.3ms inference, 0.0ms loss, 0.6ms postprocess per image
C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8dyn_mm.onnx,True,,0.5451066511344781,0.7239038738003981,0.5451066511344781,0.7412172402367192,0.6972859594105918,,808.698
    This model is useless, pls ignore
C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8_static_qdq.onnx
                Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 1817/1817 5.0it/s 6:05
                   all       1817      12444      0.755      0.677      0.717      0.528
               bicycle        186        307      0.803      0.675      0.726      0.477
                   bus         53         81      0.879      0.728      0.831       0.63
                   car        910       2460      0.824      0.794      0.839      0.696
         electric pole        382        408      0.687      0.527      0.589      0.338
             motocycle       1024       3460      0.781      0.827      0.809      0.581
pedestrian crossing sign        413        587      0.941      0.935      0.955       0.82
                person       1105       3753      0.854      0.741       0.82      0.602
                  tree        536       1296      0.517      0.451      0.432      0.228
                 truck         72         92      0.512      0.413      0.457      0.383
Speed: 0.4ms preprocess, 187.0ms inference, 0.0ms loss, 0.6ms postprocess per image
'''

# ----------------- CONFIG -----------------
DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\GroupProject_OD\data.yaml")

MODELS = [
    #Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp32.onnx"),
    #Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp16.onnx"),
    #Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8dyn_mm.onnx"),         # useless
    Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8_static_qdq.onnx")
]

TASK = "detect"     # dataset names indicate detection
IMGSZ = 640
BATCH = 1           # keep small for stability on Windows/CPU
DEVICE = "cpu"      # change to 0 if have CUDA + onnxruntime-gpu (or a GPU EP)

OUT_DIR = Path(__file__).resolve().parent / "eval_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# METRICS EXTRACTOR
def extract_metrics(val_result):
    """
    - extract common detection metrics
    - Ultralytics val returns a metrics object (or dict) depending on version/task. 
    """
    out = {}

    # Case 1: dict
    if isinstance(val_result, dict):
        # already stats dict
        out.update(val_result)
        return out

    # Case 2: has results_dict
    if hasattr(val_result, "results_dict"):
        try:
            rd = val_result.results_dict
            if isinstance(rd, dict):
                out.update(rd)
                return out
        except Exception:
            pass

    # Case 3: typical detection object with .box (may vary by version)
    box = getattr(val_result, "box", None)
    if box is not None:
        for k in ("mp", "mr", "map50", "map75", "map"):
            if hasattr(box, k):
                try:
                    out[f"box.{k}"] = float(getattr(box, k))
                except Exception:
                    out[f"box.{k}"] = getattr(box, k)

    # If nothing worked, store string
    if not out:
        out["raw"] = str(val_result)

    return out


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
            print("Validating:", model_path)

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

            # Extract metrics + mark success
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
