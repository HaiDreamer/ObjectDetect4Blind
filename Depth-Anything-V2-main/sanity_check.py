from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (contains image/, groundtruth_depth/)")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--shuffle", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "image"
    dep_dir = root / "groundtruth_depth"
    split_dir = root / "splits"