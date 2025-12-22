from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

DATA_YAML = r"D:\OD\GroupProject_OD-20251219T104740Z-3-001\GroupProject_OD\data.yaml"
ROOT = Path(r"D:\OD\GroupProject_OD-20251219T104740Z-3-001\GroupProject_OD")

SPLIT = "val"  # "train" or "val" or "test"
IMAGES_DIR = ROOT / "images" / SPLIT
LABELS_DIR = ROOT / "labels" / SPLIT
OUT_DIR    = ROOT / "viz_gt" / SPLIT
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# How many images to process
# -----------------------
MAX_IMAGES = 10   # None = process ALL (default)

# -----------------------
# Load class names from data.yaml
# -----------------------
with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data.get("names", [])
if isinstance(names, dict):
    class_names = [names[i] for i in sorted(names.keys())]
else:
    class_names = list(names)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return x1, y1, x2, y2

# -----------------------
# Main loop
# -----------------------
if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

image_paths = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])

if MAX_IMAGES is not None:
    image_paths = image_paths[:MAX_IMAGES]

if not image_paths:
    raise FileNotFoundError(f"No images found in: {IMAGES_DIR}")

# ✅ FIX: initialize counters before using them
count_drawn = 0
count_skipped_no_label = 0
count_bad_images = 0

for img_path in tqdm(image_paths, desc=f"Annotating {SPLIT}", total=len(image_paths)):
    label_path = LABELS_DIR / f"{img_path.stem}.txt"

    img = cv2.imread(str(img_path))
    if img is None:
        count_bad_images += 1
        continue

    h, w = img.shape[:2]

    if not label_path.exists():
        count_skipped_no_label += 1
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        cv2.putText(img, cls_name, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = OUT_DIR / img_path.name
    cv2.imwrite(str(out_path), img)
    count_drawn += 1

print(f"Saved annotated images to: {OUT_DIR}")
print(f"Annotated (had labels): {count_drawn}")
print(f"Skipped (no label file): {count_skipped_no_label}")
print(f"Bad/unreadable images: {count_bad_images}")
