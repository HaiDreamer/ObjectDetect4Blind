from pathlib import Path

IMG_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\images\val")
LBL_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\labels\val")
PROJECT_ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg")  # for deleting *.cache

FIX_MISSING_LABEL_FILES = True  # creates empty .txt for images without labels
DELETE_CACHE_FILES = False       # no deletes *.cache so Ultralytics rebuilds

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

def main():
    assert IMG_DIR.exists(), f"Missing IMG_DIR: {IMG_DIR}"
    LBL_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    bad = []

    images = [p for p in IMG_DIR.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print("Found images:", len(images))

    for im in images:
        lb = LBL_DIR / (im.stem + ".txt")

        if not lb.exists():
            missing.append(str(lb))
            if FIX_MISSING_LABEL_FILES:
                lb.write_text("", encoding="utf-8")
            continue

        text = lb.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        for ln, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()

            # class + at least 3 points (6 coords), even coords count
            if len(parts) < 1 + 6 or (len(parts) - 1) % 2 != 0:
                bad.append((str(lb), ln, "bad coord count", line))
                continue

            if not is_float(parts[0]):
                bad.append((str(lb), ln, "class not numeric", line))
                continue

            coords = parts[1:]
            if not all(is_float(c) for c in coords):
                bad.append((str(lb), ln, "non-numeric coord", line))
                continue

            coords_f = list(map(float, coords))
            if any((c < 0.0 or c > 1.0) for c in coords_f):
                bad.append((str(lb), ln, "coord outside [0,1]", line))
                continue

    print("\nMissing label files:", len(missing))
    if missing and not FIX_MISSING_LABEL_FILES:
        print("Example missing:", missing[0])

    print("Bad label lines:", len(bad))
    for item in bad[:30]:
        print("BAD:", item)

    if DELETE_CACHE_FILES:
        caches = list(PROJECT_ROOT.rglob("*.cache"))
        for c in caches:
            try:
                c.unlink()
            except Exception as e:
                print("Could not delete", c, "->", e)
        print("\nDeleted cache files:", len(caches))

    if bad:
        print("\nFix the BAD label lines above (those files will crash validation).")
    else:
        print("\nLabels look OK now. Re-run evaluation.")

if __name__ == "__main__":
    main()
