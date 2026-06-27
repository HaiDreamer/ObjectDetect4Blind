from pathlib import Path
import hashlib
from collections import Counter, defaultdict
import yaml
from PIL import Image

'''checking if dataset is valid'''

DATA_YAML = Path(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\GroupProject_OD\data.yaml")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def resolve_split_path(data: dict, key: str) -> Path:
    """
    Supports absolute paths (like YAML) or relative paths resolved from `path:`.
    Ultralytics data.yaml commonly uses `path:` + relative split paths.
    """
    base = Path(data.get("path", "")).expanduser()
    p = Path(data[key]).expanduser()
    return p if p.is_absolute() else (base / p)


def list_images(val_path: Path) -> list[Path]:
    # val can be a directory or a .txt list of images in YOLO setups
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        imgs = []
        for line in val_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                imgs.append(Path(line))
        return imgs

    if val_path.is_dir():
        imgs = []
        for p in val_path.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                imgs.append(p)
        return sorted(imgs)

    raise FileNotFoundError(f"val path is not a dir or .txt list: {val_path}")


def infer_labels_dir(val_images_path: Path) -> Path:
    """
    Typical Ultralytics/YOLO layout:
      .../images/val  -> .../labels/val
    """
    parts = list(val_images_path.parts)
    # Replace the last occurrence of "images" with "labels"
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    # Fallback: sibling folder named "labels"
    return val_images_path.parent.parent / "labels" / val_images_path.name


def parse_label_file(label_path: Path, nc: int):
    """
    YOLO label: class x_center y_center width height (normalized 0..1)
    nc: number of classes
    """
    errors = []
    boxes = []
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return boxes, errors  # empty label file is allowed (no objects)

    for li, line in enumerate(txt.splitlines(), start=1):
        s = line.strip().split()
        if len(s) != 5:
            errors.append(f"{label_path.name}: line {li} -> expected 5 fields, got {len(s)}")
            continue
        try:
            cls = int(s[0])
            x, y, w, h = map(float, s[1:])
        except Exception:
            errors.append(f"{label_path.name}: line {li} -> parse error")
            continue

        # class range
        if cls < 0 or cls >= nc:
            errors.append(f"{label_path.name}: line {li} -> class {cls} out of [0,{nc-1}]")

        # normalized range checks
        for name, v in (("x", x), ("y", y), ("w", w), ("h", h)):
            if not (0.0 <= v <= 1.0):
                errors.append(f"{label_path.name}: line {li} -> {name}={v} out of [0,1]")

        # positive box size
        if w <= 0 or h <= 0:
            errors.append(f"{label_path.name}: line {li} -> non-positive w/h (w={w}, h={h})")

        boxes.append((cls, x, y, w, h))

    return boxes, errors


def main():
    assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"

    data = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    nc = int(data["nc"])

    val_images_path = resolve_split_path(data, "val")
    val_images = list_images(val_images_path)

    labels_dir = infer_labels_dir(val_images_path)

    print("Val images path:", val_images_path)
    print("Inferred labels dir:", labels_dir)
    print("Val images found:", len(val_images))
    print("nc:", nc)
    print("-" * 60)

    missing_label = 0
    corrupt_images = 0
    label_parse_errors = []
    class_counts = Counter()
    empty_labels = 0

    manifest = []

    for img_path in val_images:
        # image readability check
        try:
            with Image.open(img_path) as im:
                im.verify()  # quick corruption check
            # need reopen after verify for size (optional)
        except Exception:
            corrupt_images += 1
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_label += 1
            continue

        boxes, errors = parse_label_file(label_path, nc=nc)
        if errors:
            label_parse_errors.extend(errors)

        if len(boxes) == 0:
            empty_labels += 1

        for cls in boxes:
            class_counts[cls] += 1

    print("Corrupt/unreadable images:", corrupt_images)
    print("Images missing label file:", missing_label)
    print("Empty label files (0 objects):", empty_labels)
    print("Label parse / range errors:", len(label_parse_errors))

    if label_parse_errors:
        print("\nFirst 20 label issues:")
        for e in label_parse_errors[:20]:
            print("  -", e)

    print("\nPer-class instance counts (by class id):")
    for k in range(nc):
        print(f"  {k}: {class_counts.get(k, 0)}")


if __name__ == "__main__":
    main()
