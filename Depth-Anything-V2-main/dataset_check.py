from pathlib import Path
from typing import Optional 
import argparse, random, cv2, numpy as np

'''
Purpose: test train/val image kitti dataset(for depth estimation)
How to run
cd C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main
python dataset_check.py --root "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped" --shuffle (has set as default)

NEED TO CHECK FILE LOCATION AGAIN!!!!!!
'''

def is_img(p: Path):
    return p.suffix.lower() in {".png", ".jpg", ".jpeg"}

def find_depth_for_image(img_path: Path, dep_dir: Path) -> Optional[Path]:
    # 1) KITTI val_selection_cropped uses identical names across folders
    same = dep_dir / img_path.name.replace(img_path.suffix, ".png")
    if same.exists():
        return same

    # 2) Fallbacks for odd mirrors
    stem = img_path.stem
    # try matching by trailing frame id and camera (########## + 02/03)
    import re
    m = re.search(r'_(\d{10})_image_(0[23])$', stem)
    if m:
        frame, cam = m.groups()
        for p in dep_dir.glob(f"*{frame}_image_{cam}.png"):
            return p

    # last resort: any file sharing the same stem prefix
    for p in dep_dir.glob(f"{stem}*.png"):
        return p
    return None

def check_depth(depth_path: Path):
    if not depth_path.exists():
        return False, "missing"
    im = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return False, "unreadable"
    if im.dtype != np.uint16:
        return True, "warn:not_uint16"  # ok if you use float elsewhere; just a warning
    return True, "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--shuffle", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "image"
    dep_dir = root / "groundtruth_depth"
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    print("depth dir exists:", dep_dir.exists())
    print("#depth png:", len(list(dep_dir.glob("*.png"))))
    print("img_dir:", img_dir)
    print("dep_dir:", dep_dir)
    # show one example file from each (if present)
    try:
        print("sample image:", next(p for p in img_dir.rglob("*") if p.is_file()))
    except StopIteration:
        print("[WARN] No files under image/")
    try:
        print("sample depth:", next(p for p in dep_dir.rglob("*") if p.is_file()))
    except StopIteration:
        print("[WARN] No files under groundtruth_depth/")

    imgs = sorted([p for p in img_dir.rglob("*") if p.is_file() and is_img(p)])

    pairs = []
    bad = 0
    for img in imgs:
        dpath = find_depth_for_image(img, dep_dir)
        if dpath is None:
            print(f"[MISS] {img.name}")
            bad += 1
            continue

        ok, msg = check_depth(dpath)
        if not ok:
            print(f"[MISS] {img.name} -> {dpath.name} ({msg})")
            bad += 1
            continue
        if msg != "ok":
            print(f"[WARN] depth not uint16: {dpath.name}")

        pairs.append(img.relative_to(img_dir).as_posix())

    if args.shuffle:
        random.shuffle(pairs)
    cut = int(len(pairs) * args.train_ratio)
    train, val = pairs[:cut], pairs[cut:]

    (split_dir / "train.txt").write_text("\n".join(train), encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val), encoding="utf-8")

    print(f"Found {len(imgs)} images; usable {len(pairs)}; missing {bad}")
    print(f"Wrote {split_dir/'train.txt'} ({len(train)} lines)")
    print(f"Wrote {split_dir/'val.txt'}   ({len(val)} lines)")

if __name__ == "__main__":
    main()
