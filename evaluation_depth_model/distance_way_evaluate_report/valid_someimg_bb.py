import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

'''run
python valid_someimg_bb.py --json "C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\bb_json_KITTI_val.json" --out  "C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\annotated" --max-images 10 --min-conf 0.25 --only-with-dets --shuffle


'''

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def color_for_class(cls_id: int):
    # deterministic "random" color per class id
    rng = random.Random(cls_id * 99991 + 7)
    return (rng.randint(30, 255), rng.randint(30, 255), rng.randint(30, 255))


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    # Pillow >= 8 has textbbox; fallback if needed
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return (r - l, b - t)
    except Exception:
        return draw.textsize(text, font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="Path to bb_json_*.json created by your script")
    ap.add_argument("--out", required=True, help="Output folder for annotated images")
    ap.add_argument("--max-images", type=int, default=50, help="How many images to export")
    ap.add_argument("--min-conf", type=float, default=0.25, help="Only draw detections >= this confidence")
    ap.add_argument("--only-with-dets", action="store_true", help="Skip images with 0 detections")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle before picking first --max-images")
    args = ap.parse_args()

    json_path = Path(args.json)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    if args.only_with_dets:
        images = [im for im in images if im.get("detections")]

    if args.shuffle:
        random.shuffle(images)

    font = ImageFont.load_default()

    n = 0
    for imrec in images:
        if n >= args.max_images:
            break

        img_path = Path(imrec["file_path"])
        if not img_path.exists():
            # try relative to JSON location
            alt = json_path.parent / img_path.name
            if alt.exists():
                img_path = alt
            else:
                print(f"[WARN] Missing image: {imrec['file_path']}")
                continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # thickness scaled a bit by image size
        thickness = max(2, int(round(min(img.width, img.height) / 300)))

        for det in imrec.get("detections", []):
            conf = float(det.get("confidence", 0.0))
            if conf < args.min_conf:
                continue

            cls_id = int(det.get("class_id", -1))
            cls_name = str(det.get("class_name", cls_id))
            x1, y1, x2, y2 = det["bbox_xyxy"]  # [xmin, ymin, xmax, ymax]

            # round + clamp to image bounds
            x1 = clamp(int(round(x1)), 0, img.width - 1)
            y1 = clamp(int(round(y1)), 0, img.height - 1)
            x2 = clamp(int(round(x2)), 0, img.width - 1)
            y2 = clamp(int(round(y2)), 0, img.height - 1)

            color = color_for_class(cls_id)

            # draw rectangle (Pillow uses [x0,y0,x1,y1]) :contentReference[oaicite:1]{index=1}
            draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

            # label
            label = f"{cls_name} {conf:.2f}"
            tw, th = text_size(draw, label, font)

            # put label above box when possible
            tx = x1
            ty = y1 - (th + 6)
            if ty < 0:
                ty = y1 + 2

            # label background
            draw.rectangle([tx, ty, tx + tw + 6, ty + th + 6], fill=color)
            draw.text((tx + 3, ty + 3), label, fill=(0, 0, 0), font=font)

        out_path = out_dir / f"{img_path.stem}_bbox{img_path.suffix}"
        img.save(out_path)
        n += 1
        print(f"[OK] {out_path}")

    print(f"Done. Wrote {n} images to: {out_dir}")


if __name__ == "__main__":
    main()
