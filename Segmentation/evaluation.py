from pathlib import Path
import json
import csv
import time
import traceback
from ultralytics import YOLO
import shutil
import yaml

r'''
Note: No more error should be occurs

FOR: Checking accuracy with onnx model (onnx model is fine for mobile app)

PIPELINE
    model.val(task="segment") evaluates the model's predicted instance masks against the ground-truth segmentation masks from your validation labels

OUTPUT
    model,ok,box.map,box.map50,box.map75,box.mp,box.mr,fitness,mask.map,mask.map50,mask.map75,mask.mp,mask.mr,metrics/mAP50(B),metrics/mAP50(M),metrics/mAP50-95(B),metrics/mAP50-95(M),metrics/precision(B),metrics/precision(M),metrics/recall(B),metrics/recall(M),speed.inference,speed.loss,speed.postprocess,speed.preprocess,val_time_sec
    C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp32.onnx,True,0.6461789502900329,0.7768278285093966,0.6622442895405956,0.8401071845677001,0.7159244826497447,1.2473796070616139,0.6012006567715809,0.7798513884262136,0.6202990661503389,0.8437078704271864,0.7230267917590061,0.7768278285093966,0.7798513884262136,0.6461789502900329,0.6012006567715809,0.8401071845677001,0.8437078704271864,0.7159244826497447,0.7230267917590061,187.12310740331873,0.0011334491863566004,1.0423357357192047,0.41995619120976285,304.91
                   all       1486       2259       0.84      0.716      0.777      0.646      0.844      0.723       0.78      0.601
                Stairs        334        390      0.941       0.86      0.908      0.866      0.941       0.86      0.908      0.852
             crosswalk        287        343      0.948      0.739      0.849      0.697      0.948      0.741      0.855      0.673
              sidewalk        879       1185      0.828      0.764      0.832      0.735      0.823       0.76      0.818      0.628
            tree-lined        158        341      0.643      0.501      0.517      0.287      0.663      0.531      0.538      0.253
    model,ok,box.map,box.map50,box.map75,box.mp,box.mr,fitness,mask.map,mask.map50,mask.map75,mask.mp,mask.mr,metrics/mAP50(B),metrics/mAP50(M),metrics/mAP50-95(B),metrics/mAP50-95(M),metrics/precision(B),metrics/precision(M),metrics/recall(B),metrics/recall(M),speed.inference,speed.loss,speed.postprocess,speed.preprocess,val_time_sec
    C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp16.onnx,True,0.6459245180093143,0.776737314311576,0.6625565320515399,0.8403163766443402,0.7159383076756011,1.2476230444995058,0.6016985264901915,0.7796687426334827,0.6221291772174977,0.8437167157016282,0.7234782494017277,0.776737314311576,0.7796687426334827,0.6459245180093143,0.6016985264901915,0.8403163766443402,0.8437167157016282,0.7159383076756011,0.7234782494017277,211.067969042151,0.0011664191649507225,1.0548125186930053,0.43884314070825503,336.033
                   all       1486       2259       0.84      0.716      0.777      0.646      0.844      0.723       0.78      0.602
                Stairs        334        390      0.941       0.86      0.908      0.866      0.941       0.86      0.908      0.852
             crosswalk        287        343      0.948      0.739       0.85      0.697      0.947      0.741      0.855      0.673
              sidewalk        879       1185      0.829      0.764      0.832      0.734      0.823      0.761      0.819      0.629
            tree-lined        158        341      0.644      0.501      0.517      0.287      0.663      0.532      0.536      0.253
    model,ok,box.map,box.map50,box.map75,box.mp,box.mr,fitness,mask.map,mask.map50,mask.map75,mask.mp,mask.mr,metrics/mAP50(B),metrics/mAP50(M),metrics/mAP50-95(B),metrics/mAP50-95(M),metrics/precision(B),metrics/precision(M),metrics/recall(B),metrics/recall(M),speed.inference,speed.loss,speed.postprocess,speed.preprocess,val_time_sec
    C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_int8_static_qdq.onnx,True,0.6156278179467233,0.7655939371064144,0.6307589711712641,0.8281804827387338,0.7209576103229316,1.1908582430560295,0.5752304251093061,0.7587151455839897,0.5917628581703065,0.8223754885641443,0.7159899311177687,0.7655939371064144,0.7587151455839897,0.6156278179467233,0.5752304251093061,0.8281804827387338,0.8223754885641443,0.7209576103229316,0.7159899311177687,111.93960894796682,0.001229409725075615,1.190674295999902,0.4158555813607525,190.143
                    all       1486       2259      0.828      0.721      0.766      0.616      0.822      0.716      0.759      0.575
                Stairs        334        390      0.941      0.857      0.901      0.845      0.937      0.854      0.896      0.834
                crosswalk        287        343       0.93       0.74      0.835      0.665      0.927      0.737      0.834      0.643
                sidewalk        879       1185      0.804      0.772      0.809       0.69      0.799      0.767      0.798      0.594
                tree-lined        158        341      0.637      0.515      0.517      0.261      0.626      0.506      0.507      0.231
'''

# ----------------- CONFIG -----------------
DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\data.yaml")

MODELS = [
    # Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg.pt"),
    # Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp32.onnx"),
    Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp16.onnx"),
    #Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_int8_static_qdq.onnx"),
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

    poly = [x1, y1,  x2, y1,  x2, y2,  x1, y2]
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


def main():
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    # --- Convert bbox labels -> rectangle polygon labels (if needed)
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
