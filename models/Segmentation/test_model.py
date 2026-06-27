from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best.pt")
OUT_IMG_DIR = ROOT / "output"


def _load_rgb(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _save_mask_png(mask_hw: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_hw.astype(np.uint8), mode="L").save(out_path)
    return out_path


def _results_to_semantic_mask(res, out_h: int, out_w: int, class_whitelist: list[int] | None = None) -> np.ndarray:
    # No predictions -> all black
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    m = res.masks.data.cpu().numpy()  # (N, hm, wm)
    cls = res.boxes.cls.cpu().numpy().astype(int)
    conf = res.boxes.conf.cpu().numpy()

    if class_whitelist is not None:
        keep = np.isin(cls, np.asarray(class_whitelist, dtype=int))
        if keep.sum() == 0:
            return np.zeros((out_h, out_w), dtype=np.uint8)
        m = m[keep]
        conf = conf[keep]

    m = m > 0.5

    hm, wm = m.shape[1], m.shape[2]
    if (hm, wm) != (out_h, out_w):
        m_rs = []
        for i in range(m.shape[0]):
            m_rs.append(
                cv2.resize(m[i].astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            )
        m = np.stack(m_rs, axis=0)

    # merge all instances into one binary mask (for debug/visualization)
    order = np.argsort(conf)
    bin_mask = np.zeros((out_h, out_w), dtype=bool)
    for i in order:
        bin_mask |= m[i]

    return (bin_mask.astype(np.uint8) * 255)


def save_instances_border_txt(
    res,
    out_txt_path: Path,
    out_h: int,
    out_w: int,
    simplify_eps_ratio: float = 0.002,
) -> Path:
    """
    Writes one line per instance:
      <cls_id> <conf> x1 y1 x2 y2 x3 y3 ...
    """
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        out_txt_path.write_text("", encoding="utf-8")
        return out_txt_path

    masks = (res.masks.data > 0.5).cpu().numpy()          # (N, hm, wm)
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)     # (N,)
    confs = res.boxes.conf.cpu().numpy().astype(float)    # (N,)

    hm, wm = masks.shape[1], masks.shape[2]

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for i in range(masks.shape[0]):
            m = masks[i].astype(np.uint8) * 255

            # resize mask to original image size
            if (hm, wm) != (out_h, out_w):
                m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

            # find contour on binary mask (white object on black background) :contentReference[oaicite:1]{index=1}
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            # pick largest contour (more stable)
            cnt = max(contours, key=cv2.contourArea)

            # approximate contour to polygon (approxPolyDP uses epsilon * arc length) :contentReference[oaicite:2]{index=2}
            peri = cv2.arcLength(cnt, True)
            eps = simplify_eps_ratio * peri
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

            if approx.shape[0] < 3:
                continue

            coords = " ".join(f"{int(x)} {int(y)}" for x, y in approx)
            f.write(f"{cls_ids[i]} {confs[i]:.6f} {coords}\n")

    return out_txt_path


def save_border_txt_from_mask(
    mask_path: Path,
    out_txt_path: Path,
    simplify_eps_ratio: float = 0.002,
) -> Path:
    """
    Old format (debug): only polygon coords.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask image: {mask_path}")

    if mask.ndim == 2:
        gray = mask
    elif mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    H, W = gray.shape[:2]
    bin_u8 = (gray > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(bin_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    lines = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        eps = simplify_eps_ratio * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        pts = approx.reshape(-1, 2).astype(np.int32)
        if pts.shape[0] >= 3:
            lines.append(" ".join(f"{int(x)} {int(y)}" for x, y in pts))

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return out_txt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out-border", type=str, required=True, help="Path to output border txt (instances: cls conf poly)")
    parser.add_argument("--save-merged-debug", action="store_true", help="Also save merged mask border debug txt")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    out_border_path = Path(args.out_border).resolve()

    print(f"[SEG] image: {image_path}")
    print(f"[SEG] out border txt (instances): {out_border_path}")

    img_rgb = _load_rgb(image_path)
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))
    # model.names gives the class mapping at runtime :contentReference[oaicite:3]{index=3}
    print("[SEG] model.names =", model.names)

    results = model.predict(img_rgb, conf=0.1, iou=0.7, device=device, verbose=False)
    res = results[0]

    H, W = img_rgb.shape[:2]

    # 1) Write instance polygons WITH class/conf (this is what your main pipeline expects)
    save_instances_border_txt(res, out_border_path, H, W, simplify_eps_ratio=0.002)
    print(f"[SEG] saved instance border txt: {out_border_path}")

    # 2) Save merged mask png (optional debug/visualization)
    mask_hw = _results_to_semantic_mask(res, H, W)
    stem = image_path.stem
    mask_img_path = OUT_IMG_DIR / f"{stem}_mask.png"
    _save_mask_png(mask_hw, mask_img_path)
    print(f"[SEG] saved merged mask: {mask_img_path}")

    # 3) OPTIONAL: save old-style merged-border txt to a DIFFERENT file (so we don't overwrite instance txt)
    if args.save_merged_debug:
        merged_txt = out_border_path.with_name(out_border_path.stem + "_merged.txt")
        save_border_txt_from_mask(mask_img_path, merged_txt, simplify_eps_ratio=0.002)
        print(f"[SEG] saved merged border debug txt: {merged_txt}")


if __name__ == "__main__":
    main()
