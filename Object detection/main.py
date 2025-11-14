# Object detection/main.py
from pathlib import Path
import argparse
from ultralytics import YOLO
import json

ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE_PATH = Path(r"C:\Python\ObjectDetect4Blind\assets\demo01.jpg")
MODEL_PATH = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\detect_best.pt")

OUT_IMG = ROOT / "output"
RUN_NAME = "run1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        default=str(DEFAULT_IMAGE_PATH),
        help="Path to input image",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()

    assert image_path.exists(), f"Image not found: {image_path}"
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"

    model = YOLO(str(MODEL_PATH))

    results = model.predict(
        source=str(image_path),
        conf=0.25,
        iou=0.7,
        save=True,          # save rendered image(s)
        save_txt=True,      # save YOLO-format txt predictions
        save_conf=True,     # include confidences
        project=str(OUT_IMG),
        name=RUN_NAME,
        exist_ok=True
    )

    print("Saved to:", OUT_IMG / RUN_NAME)

    r = results[0]
    ids = [int(i) for i in r.boxes.cls.tolist()]
    names = [model.names[i] for i in ids]
    xyxy = [list(map(float, b)) for b in r.boxes.xyxy.tolist()]
    conf = [float(c) for c in r.boxes.conf.tolist()]

    payload = {
        "image": str(image_path),
        "detections": [
            {
                "class_id": ids[i],
                "class_name": names[i],
                "box_xyxy": xyxy[i],
                "score": conf[i],
            }
            for i in range(len(ids))
        ],
    }

    out_json = OUT_IMG / RUN_NAME / (image_path.stem + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
