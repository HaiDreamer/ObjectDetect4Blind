from pathlib import Path

'''
Images without labels: 751
Labels without images: 92
Example images without labels: 'ds1__ds1__ds1__IMG_8157_heic.rf.f552d4ddcbcc056802efca350585c4d6', 'ds1__ds1__ds1__IMG_8159_heic.rf.78586ec863f172ec2289714bd8b0f081', 'ds1__ds1__ds1__IMG_8159_heic.rf.9e1256613b56fde527609b0f18359691', 'ds1__ds1__ds1__IMG_8161_heic.rf.2ccca98fc598e480872d921bb7f3ed67', 'ds1__ds1__ds1__IMG_8161_heic.rf.87c4b588951259fe6c94acbde0084c58', 'ds1__ds1__ds1__IMG_8162_heic.rf.2ef387132a1f6cc7c3dfde134b28a347
'''

ROOT = Path(r"D:\OD\GroupProject_OD-20251219T104740Z-3-001\GroupProject_OD")
SPLIT = "train"
IMAGES_DIR = ROOT / "images" / SPLIT
LABELS_DIR = ROOT / "labels" / SPLIT

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_stems = {p.stem for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXTS}
label_stems = {p.stem for p in LABELS_DIR.glob("*.txt")}

missing_label = sorted(image_stems - label_stems)
missing_image = sorted(label_stems - image_stems)

print("Images without labels:", len(missing_label))
print("Labels without images:", len(missing_image))

# show a few examples
print("Example images without labels:", missing_label[:20])
print("Example labels without images:", missing_image[:20])
