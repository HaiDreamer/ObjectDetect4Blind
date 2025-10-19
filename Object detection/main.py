from pathlib import Path
from ultralytics import YOLO
import json

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "assets" / "demo01.jpg"
MODEL_PATH = ROOT / "models" / "detect_best.pt"
OUT_IMG = ROOT / "output"
RUN_NAME = "run1"

assert IMAGE_PATH.exists(), f"Image not found: {IMAGE_PATH}"
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    iou=0.7,
    save=True,          # save rendered image(s)
    save_txt=True,      # save YOLO-format txt predictions
    save_conf=True,     # include confidences in the txt files
    project=str(OUT_IMG),
    name=RUN_NAME,
    exist_ok=True
)

print("Saved to:", OUT_IMG / RUN_NAME)

# Serialize detections form image to JSON
r = results[0]
ids = [int(i) for i in r.boxes.cls.tolist()]
names = [model.names[i] for i in ids]
xyxy = [list(map(float, b)) for b in r.boxes.xyxy.tolist()]
conf = [float(c) for c in r.boxes.conf.tolist()]

payload = {
    "image": str(IMAGE_PATH),
    "detections": [
        {"class_id": ids[i], "class_name": names[i],
         "box_xyxy": xyxy[i], "score": conf[i]}
        for i in range(len(ids))
    ]
}

out_json = OUT_IMG / RUN_NAME / (IMAGE_PATH.stem + ".json")
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
# print("Saved:", out_json)
