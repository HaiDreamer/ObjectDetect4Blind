from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO

'''
usage: 
python segment_visualize.py --image "C:\Python\ObjectDetect4Blind\assets\demo03.jpg" --out-border "C:\Python\ObjectDetect4Blind\Segmentation\output\mask_border.txt"
'''

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best.pt")
OUT_IMG_DIR = ROOT / "output"

# Class names mapping
SEG_CLASS_NAMES = {0: 'Stairs', 1: 'crosswalk', 2: 'sidewalk', 3: 'tree-lined'}

# Colors for visualization (BGR format)
CLASS_COLORS = {
    0: (255, 0, 0),      # Stairs - Blue
    1: (0, 255, 255),    # crosswalk - Yellow
    2: (0, 255, 0),      # sidewalk - Green
    3: (255, 0, 255),    # tree-lined - Magenta
}


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

            # find contour on binary mask
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            # pick largest contour
            cnt = max(contours, key=cv2.contourArea)

            # approximate contour to polygon
            peri = cv2.arcLength(cnt, True)
            eps = simplify_eps_ratio * peri
            approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

            if approx.shape[0] < 3:
                continue

            coords = " ".join(f"{int(x)} {int(y)}" for x, y in approx)
            f.write(f"{cls_ids[i]} {confs[i]:.6f} {coords}\n")

    return out_txt_path


def visualize_segmentation(
    img_bgr: np.ndarray,
    res,
    out_path: Path,
    simplify_eps_ratio: float = 0.002,
) -> Path:
    """
    Create visualization with colored masks, borders, class names and confidence scores
    """
    if res.masks is None or res.boxes is None or len(res.boxes) == 0:
        print("[SEG] No detections to visualize")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img_bgr)
        return out_path

    H, W = img_bgr.shape[:2]
    vis = img_bgr.copy()
    
    masks = (res.masks.data > 0.5).cpu().numpy()          # (N, hm, wm)
    cls_ids = res.boxes.cls.cpu().numpy().astype(int)     # (N,)
    confs = res.boxes.conf.cpu().numpy().astype(float)    # (N,)
    
    hm, wm = masks.shape[1], masks.shape[2]
    
    # Create overlay for semi-transparent masks
    overlay = vis.copy()
    
    for i in range(masks.shape[0]):
        cls_id = cls_ids[i]
        conf = confs[i]
        
        # Get class info
        cls_name = SEG_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        
        # Get mask
        m = masks[i].astype(np.uint8) * 255
        
        # Resize mask to original image size
        if (hm, wm) != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Apply colored mask to overlay
        mask_bool = m > 127
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array(color, dtype=np.uint8) * 0.5
        
        # Find contour for border
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        
        # Pick largest contour
        cnt = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        eps = simplify_eps_ratio * peri
        approx = cv2.approxPolyDP(cnt, eps, True)
        
        # Draw border
        cv2.polylines(vis, [approx], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_AA)
        
        # Find position for text label (centroid)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Fallback to first point
            cx, cy = int(approx[0, 0, 0]), int(approx[0, 0, 1])
        
        # Prepare text
        text = f"{cls_name} {conf:.2f}"
        
        # Draw text with outline for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw white outline
        cv2.putText(vis, text, (cx - text_w//2, cy), font, font_scale, (255, 255, 255), thickness + 4, cv2.LINE_AA)
        # Draw colored text
        cv2.putText(vis, text, (cx - text_w//2, cy), font, font_scale, color, thickness, cv2.LINE_AA)
    
    # Blend overlay with original
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
    
    # Save visualization
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"[SEG] Saved visualization: {out_path}")
    
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--out-border", type=str, required=True, help="Path to output border txt (instances: cls conf poly)")
    parser.add_argument("--save-vis", action="store_true", help="Save visualization with class names and confidence")
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    out_border_path = Path(args.out_border).resolve()

    print(f"[SEG] image: {image_path}")
    print(f"[SEG] out border txt (instances): {out_border_path}")

    img_rgb = _load_rgb(image_path)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(str(MODEL_PATH))
    print("[SEG] model.names =", model.names)

    results = model.predict(img_rgb, conf=0.1, iou=0.7, device=device, verbose=False)
    res = results[0]

    H, W = img_rgb.shape[:2]

    # 1) Write instance polygons WITH class/conf
    save_instances_border_txt(res, out_border_path, H, W, simplify_eps_ratio=0.002)
    print(f"[SEG] saved instance border txt: {out_border_path}")

    # 2) Save merged mask png
    mask_hw = _results_to_semantic_mask(res, H, W)
    stem = image_path.stem
    mask_img_path = OUT_IMG_DIR / f"{stem}_mask.png"
    _save_mask_png(mask_hw, mask_img_path)
    print(f"[SEG] saved merged mask: {mask_img_path}")

    # 3) Save visualization with class names and confidence (if requested)
    if args.save_vis:
        vis_path = OUT_IMG_DIR / f"{stem}_segmentation_vis.png"
        visualize_segmentation(img_bgr, res, vis_path, simplify_eps_ratio=0.002)


if __name__ == "__main__":
    main()