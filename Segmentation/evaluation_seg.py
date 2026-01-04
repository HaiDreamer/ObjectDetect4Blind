from ultralytics import YOLO
import json
from pathlib import Path

"""
MODELS
    C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg.pt
    C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp16.onnx
    C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp32.onnx
    C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_int8_static_qdq.onnx
OUTPUT


"""

model = Path(r"")
metrics = model.val(data="/storage/student5/blind/GroupProject_Seg/data.yaml",
                    split="val", imgsz=640, batch=1, device=1, verbose=True)

out = {
    "map50": float(metrics.box.map50),
    "map": float(metrics.box.map),
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "map_per_class": [float(x) for x in metrics.box.maps],  # mAP theo class
}
with open("metrics.json", "w") as f:
    json.dump(out, f, indent=2)
print(out)
