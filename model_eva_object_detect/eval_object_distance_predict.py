from pathlib import Path
import json
import cv2
import numpy as np
from datetime import datetime

# CONFIG
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")

OBJ_GT_JSON    = Path(r"C:\Python\ObjectDetect4Blind\model_eva_object_detect\bb_json_KITTI_val_with_gt_dist.json")
# pred_metric_kitti_vkitti_vits_pruned1layer_torch_cpu, pred_metric_kitti_vkitti_vits_onnx_int8_cpu, 
#   pred_metric_kitti_vkitti_vits_onnx_azure (fp16 model), pred_metric_kitti_vkitti_vits_torch (original model)
PRED_DEPTH_DIR = ROOT / "pred_metric_kitti_vkitti_vits_pruned1layer_torch_cpu"
OUT_ERR_JSON   = ROOT / "obj_depth_with_pred_pruned1layer.json"

MAX_DEPTH_M  = 80.0
BOX_FRAC     = 0.3
BOX_Q        = 10.0   # p10
BOX_SUBSAMP  = 1

# If True: don't print per-image warnings; only print final summary
QUIET_MISSING = True


def _fast_percentile_1d(vals: np.ndarray, q: float) -> float | None:
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    k = int(round((q / 100.0) * (vals.size - 1)))
    k = max(0, min(vals.size - 1, k))
    return float(np.partition(vals, k)[k])


def compute_box_distance(
    depth_map_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    *,
    frac: float = BOX_FRAC,
    mode: str = "center",      # "center" or "bottom"
    q: float = BOX_Q,
    subsample: int = BOX_SUBSAMP,
) -> float | None:
    """
    Matches your main pipeline logic:
      - "center": central frac x frac
      - "bottom": bottom frac of height + central 50% width band
      - distance = low percentile (p10 by default)
      - x2,y2 treated as slice end (exclusive)
    """
    H, W = depth_map_m.shape[:2]

    # IMPORTANT: allow x1 == W or y1 == H (will become empty region -> None)
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    if mode == "bottom":
        ch = int(h * frac)
        if ch <= 0:
            return None
        y_start = max(y1, y2 - ch)

        center_band_width = int(w * 0.5)
        if center_band_width <= 0:
            return None
        cx = (x1 + x2) // 2
        x_start = max(x1, cx - center_band_width // 2)
        x_end = min(x2, x_start + center_band_width)
        if x_end <= x_start:
            return None

        patch = depth_map_m[y_start:y2, x_start:x_end]
    else:
        cw = int(w * frac)
        ch = int(h * frac)
        if cw <= 0 or ch <= 0:
            return None

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cx1 = max(0, cx - cw // 2)
        cy1 = max(0, cy - ch // 2)
        cx2 = min(W, cx1 + cw)
        cy2 = min(H, cy1 + ch)
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        patch = depth_map_m[cy1:cy2, cx1:cx2]

    if patch.size == 0:
        return None
    if subsample > 1:
        patch = patch[::subsample, ::subsample]

    valid = patch[(patch > 0.0) & np.isfinite(patch) & (patch < MAX_DEPTH_M)].reshape(-1)
    return _fast_percentile_1d(valid, q=q)


def infer_mode(det: dict) -> str:
    m = (det.get("distance_mode") or "").lower()
    if m in ("bottom", "center"):
        return m

    cat = (det.get("eval_category") or "").lower()
    if any(k in cat for k in ("car", "truck", "largeveh", "cyclist")):
        return "bottom"
    return "center"


def get_bbox_int(det: dict) -> tuple[int, int, int, int] | None:
    if det.get("bbox_xyxy_int") is not None:
        x1, y1, x2, y2 = det["bbox_xyxy_int"]
        return int(x1), int(y1), int(x2), int(y2)

    bb = det.get("bbox_xyxy")
    if not bb or len(bb) != 4:
        return None
    x1f, y1f, x2f, y2f = bb
    return int(np.floor(x1f)), int(np.floor(y1f)), int(np.ceil(x2f)), int(np.ceil(y2f))


def build_pred_index(pred_dir: Path):
    """
    Scan pred_dir once for fast lookup.
    Keys are filenames only (not full paths).
    """
    png_map = {}
    npy_map = {}
    for p in pred_dir.rglob("*.png"):
        png_map[p.name] = p
    for p in pred_dir.rglob("*.npy"):
        npy_map[p.name] = p
    return png_map, npy_map


def derive_candidates(img_entry: dict) -> list[str]:
    """
    Return candidate GT-style filenames we expect preds to be named after.
    Primary = gt_depth_path filename (best)
    Fallback = derived from file_name pattern
    """
    candidates = []
    gt_path = img_entry.get("gt_depth_path")
    if gt_path:
        candidates.append(Path(gt_path).name)

    img_name = img_entry.get("file_name", "")
    parts = img_name.split("_image_")
    if len(parts) == 3:
        prefix, frame_str, cam_str = parts
        candidates.append(f"{prefix}_groundtruth_depth_{frame_str}_image_{cam_str}")

    # dedupe while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def load_pred_depth_for_entry(img_entry: dict, png_index: dict, npy_index: dict) -> tuple[np.ndarray | None, str | None]:
    """
    Try candidates:
      - <gt_stem>_pred_m.npy (meters float32) preferred
      - <gt_name>.png (uint16/256.0) fallback
    Returns (depth_map_m or None, chosen_path or None)
    """
    tried = []
    for gt_name in derive_candidates(img_entry):
        gt_stem = Path(gt_name).stem
        npy_name = f"{gt_stem}_pred_m.npy"

        tried.append(npy_name)
        p_npy = npy_index.get(npy_name)
        if p_npy is not None and p_npy.exists():
            depth_m = np.load(str(p_npy)).astype(np.float32)
            if depth_m.ndim == 3:
                depth_m = depth_m.squeeze()
            return depth_m, str(p_npy)

        tried.append(gt_name)
        p_png = png_index.get(gt_name)
        if p_png is not None and p_png.exists():
            im = cv2.imread(str(p_png), cv2.IMREAD_UNCHANGED)
            if im is None:
                return None, None
            if im.ndim == 3:
                im = im[..., 0]
            return (im.astype(np.float32) / 256.0), str(p_png)

    # nothing found
    if not QUIET_MISSING:
        print(f"[WARN] no prediction depth found for {img_entry.get('file_name')} (tried {len(tried)} names)")
    return None, None


# MAIN
def main():
    assert OBJ_GT_JSON.exists(), f"Missing GT JSON: {OBJ_GT_JSON}"
    assert PRED_DEPTH_DIR.exists(), f"Missing PRED_DEPTH_DIR: {PRED_DEPTH_DIR}"

    with open(OBJ_GT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data:
        raise RuntimeError("Unexpected JSON structure: expected a dict with key 'images'.")

    images = data["images"]
    print(f"Loaded {len(images)} images from {OBJ_GT_JSON}")

    png_index, npy_index = build_pred_index(PRED_DEPTH_DIR)
    print(f"Indexed preds: {len(png_index)} png, {len(npy_index)} npy")

    out = dict(data)
    out["pred_depth_dir"] = str(PRED_DEPTH_DIR)
    out["pred_eval_generated_at"] = datetime.now().isoformat(timespec="seconds")

    missing_pred = 0
    used_pred = 0

    new_images = []

    for idx, img_entry in enumerate(images, 1):
        dets = img_entry.get("detections", [])
        new_img = dict(img_entry)
        new_dets = []

        depth_pred, pred_path = load_pred_depth_for_entry(img_entry, png_index, npy_index)
        new_img["pred_depth_path"] = pred_path  # helpful for debugging

        if depth_pred is None:
            missing_pred += 1
        else:
            used_pred += 1
            # resize to recorded image size if needed
            W = int(new_img.get("width", depth_pred.shape[1]))
            H = int(new_img.get("height", depth_pred.shape[0]))
            if depth_pred.shape[:2] != (H, W):
                depth_pred = cv2.resize(depth_pred, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_pred = np.clip(depth_pred.astype(np.float32), 0.0, MAX_DEPTH_M)

        for det in dets:
            new_det = dict(det)

            gt_dist = det.get("gt_distance_m", None)
            bbox = get_bbox_int(det)
            mode = infer_mode(det)

            if depth_pred is None or gt_dist is None or bbox is None:
                new_det["ground_distance_predict"] = None
                new_det["wrongly_distance_m"] = None
            else:
                x1, y1, x2, y2 = bbox
                pred_dist = compute_box_distance(
                    depth_pred, x1, y1, x2, y2,
                    frac=BOX_FRAC, mode=mode, q=BOX_Q, subsample=BOX_SUBSAMP
                )
                if pred_dist is None:
                    new_det["ground_distance_predict"] = None
                    new_det["wrongly_distance_m"] = None
                else:
                    new_det["ground_distance_predict"] = float(pred_dist)
                    new_det["wrongly_distance_m"] = float(gt_dist - pred_dist)  # gt - pred

            new_dets.append(new_det)

        new_img["detections"] = new_dets
        new_images.append(new_img)

        if idx % 50 == 0 or idx == len(images):
            print(f"Processed {idx}/{len(images)}")

    out["images"] = new_images

    OUT_ERR_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ERR_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {OUT_ERR_JSON}")
    print(f"Pred available for {used_pred}/{len(images)} images; missing for {missing_pred}/{len(images)} images.")


if __name__ == "__main__":
    main()
