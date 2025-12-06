from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\segment_best.pt")
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

    m = res.masks.data.cpu().numpy()
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

    order = np.argsort(conf)
    bin_mask = np.zeros((out_h, out_w), dtype=bool)
    for i in order:
        bin_mask |= m[i]

    return (bin_mask.astype(np.uint8) * 255)


def save_border_txt_from_mask(
    mask_path: Path,
    out_txt_path: Path,
    simplify_eps_ratio: float = 0.002,
    normalize: bool = False
) -> Path:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask image: {mask_path}")

    if mask.ndim == 2:
        gray = mask
    elif mask.ndim == 3:
        c = mask.shape[2]
        if c == 1:
            gray = mask[:, :, 0]
        elif c == 3:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        elif c == 4:
            gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"Unsupported channel count: {c}")
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
        pts = approx.reshape(-1, 2).astype(np.float32)
        if normalize:
            pts[:, 0] /= W
            pts[:, 1] /= H
            line = " ".join(f"{p:.6f}" for p in pts.flatten())
        else:
            line = " ".join(str(int(v)) for v in pts.flatten())
        if line:
            lines.append(line)

    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    return out_txt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out-border", type=str, required=True, help="Path to output border txt")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    out_border_path = Path(args.out_border).resolve()

    print(f"[SEG] image: {image_path}")
    print(f"[SEG] out border txt: {out_border_path}")

    img_rgb = _load_rgb(image_path)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH))

    results = model.predict(img_rgb, conf=0.1, iou=0.7, device=device, verbose=False)
    res = results[0]

    H, W = img_rgb.shape[:2]
    mask_hw = _results_to_semantic_mask(res, H, W)

    stem = image_path.stem
    mask_img_path = OUT_IMG_DIR / f"{stem}_mask.png"
    _save_mask_png(mask_hw, mask_img_path)

    save_border_txt_from_mask(
        mask_img_path,
        out_border_path,
        simplify_eps_ratio=0.002,
        normalize=False
    )

    print(f"[SEG] saved mask: {mask_img_path}")
    print(f"[SEG] saved border txt: {out_border_path}")


if __name__ == "__main__":
    main()
