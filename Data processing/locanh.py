import os
from pathlib import Path

# Path
image_dir = Path("D:\Hieu\B3\Group Project\Dataset\pedestrian sign.v1i.yolov8\images")  # img folder
label_dir = Path("D:\Hieu\B3\Group Project\Dataset\pedestrian sign.v1i.yolov8\labels")  # label folder (.txt)


image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic"}

# File list of all label
label_stems = {p.stem for p in label_dir.glob("*.txt")}

print(f"Number of label: {len(label_stems)}")

deleted = 0
kept = 0

for img_path in image_dir.iterdir():
    if not img_path.is_file():
        continue
    if img_path.suffix.lower() not in image_exts:
        continue

    # file name without extension
    stem = img_path.stem  

    # If no corresponding label -> delete image
    if stem not in label_stems:
        print(f"Delete image without label: {img_path}")
        img_path.unlink()   # delete file
        deleted += 1
    else:
        kept += 1

print(f"Deleted {deleted} images without label.")
print(f"Remaining {kept} images with corresponding labels.")