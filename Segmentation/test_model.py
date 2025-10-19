from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "assets" / "demo01.jpg"
MODEL_PATH = ROOT / "models" / "segment_best.pt"   # YOLO model
OUT_IMG = ROOT / "output"

def _load_rgb(path: Path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def _save_mask_png(mask_hw: np.ndarray, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mask.png"
    Image.fromarray(mask_hw.astype(np.uint8), mode="L").save(out_path)
    return out_path

def _results_to_semantic_mask(res, out_h: int, out_w: int, class_whitelist: list[int] | None = None) -> np.ndarray:
    """
    Build a binary mask (H,W) where segmented pixels are 255 (white) and background is 0 (black).
    If class_whitelist is provided, only those class IDs are considered.
    """
    # No predictions -> all black
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        return np.zeros((out_h, out_w), dtype=np.uint8)

    # Instance masks and per-instance class IDs/confidences
    m = res.masks.data.cpu().numpy()               
    cls = res.boxes.cls.cpu().numpy().astype(int)    
    conf = res.boxes.conf.cpu().numpy()             

    # Optionally keep only certain classes (e.g., persons, etc.)
    if class_whitelist is not None:
        keep = np.isin(cls, np.asarray(class_whitelist, dtype=int))
        if keep.sum() == 0:
            return np.zeros((out_h, out_w), dtype=np.uint8)
        m = m[keep]
        conf = conf[keep]

    # Ensure boolean masks (threshold if they are floats)
    m = m > 0.5

    # Resize each instance mask to the desired output size (nearest preserves labels)
    hm, wm = m.shape[1], m.shape[2]
    if (hm, wm) != (out_h, out_w):
        m_rs = []
        for i in range(m.shape[0]):
            m_rs.append(
                cv2.resize(m[i].astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            )
        m = np.stack(m_rs, axis=0)

    # Combine instances into a single binary mask:
    # draw low→high confidence so the last op corresponds to highest conf (though we only need "any")
    order = np.argsort(conf)  # ascending
    bin_mask = np.zeros((out_h, out_w), dtype=bool)
    for i in order:
        bin_mask |= m[i]

    # Convert to 0/255 uint8 for a white mask
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

    # SAFE GRAYSCALE CONVERSION
    if mask.ndim == 2:
        # already single-channel (H, W)
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

    # Treat any non-zero as foreground (works for class-ID or binary masks)
    bin_u8 = (gray > 0).astype(np.uint8) * 255

    # Contours of white-on-black binary image
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
    img_rgb = _load_rgb(IMAGE_PATH)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH)) 

    results = model.predict(img_rgb, conf=0.1, iou=0.7, device=device, verbose=False)
    res = results[0]

    # Convert instance masks -> semantic class-id mask
    H, W = img_rgb.shape[:2]
    mask_hw = _results_to_semantic_mask(res, H, W)

    _save_mask_png(mask_hw, OUT_IMG) 

    mask_img = ROOT / "output" / "mask.png"         
    out_txt  = ROOT / "output" / "mask_border.txt"
    save_border_txt_from_mask(mask_img, out_txt, simplify_eps_ratio=0.002, normalize=False)
    # print(f"[BORDER] saved coords: {out_txt}")            # Debug 

if __name__ == "__main__":
    main()
