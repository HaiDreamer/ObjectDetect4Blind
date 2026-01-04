from ultralytics import YOLO
import json

model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="data.yaml", split="val", imgsz=640, verbose=True)

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
