from pathlib import Path
import json
import cv2
import numpy as np

"""
Compute per-object distance error: in 935 images that have at least one object entry in obj_depth_gt.json
- There are 65 image that does not contain any object has been detected by yolo model

INPUT
- obj_depth_gt.json: each object has
    image, det_id, yolo_class_id, yolo_class_name,
    eval_category, confidence, bbox_xyxy, gt_distance_m
- Predicted depth maps from metric model in:
    pred_metric_kitti_vkitti_vits/
    - uint16 KITTI PNG: <gt_name>.png
    - optional float32 .npy: <gt_name>_pred_m.npy

OUTPUT
- obj_depth_with_pred.json:
    same fields as input +
    - ground_distance_predict
    - wrongly_distance_m = gt_distance_m - ground_distance_predict

TO DO
- check .npy vs .png img in input predict depth maps from metric model in pred_metric_kitti_vkitti_vits/

"""

# ========= CONFIG =========
ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth")   

OBJ_GT_JSON      = ROOT / "obj_depth_gt.json"
PRED_DEPTH_DIR   = ROOT / "pred_metric_kitti_vkitti_vits_onnx_azure"     #"pred_metric_kitti_vkitti_vits_torch" for original model, pred_metric_kitti_vkitti_vits_onnx_azure for onnx model
OUT_ERR_JSON     = ROOT / "obj_depth_with_pred.json"

MAX_DEPTH_M = 80.0


# ========= HELPERS =========

def load_pred_depth_for_image(image_name: str) -> np.ndarray | None:
    """
    Given an image filename from obj_depth_gt.json, derive the corresponding
    GT-style name and then locate the prediction files.

    Example image name:
      2011_09_26_drive_0002_sync_image_0000000005_image_02.png

    GT/pred name:
      2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png

    We then look for:
      PRED_DEPTH_DIR / <gt_name>                   (uint16 PNG)
      PRED_DEPTH_DIR / (<gt_stem> + "_pred_m.npy") (float32 meters)
    """
    parts = image_name.split("_image_")
    if len(parts) != 3:
        print(f"[WARN] unexpected image filename pattern: {image_name}")
        return None

    prefix, frame_str, cam_str = parts
    gt_name = f"{prefix}_groundtruth_depth_{frame_str}_image_{cam_str}"

    png_path = PRED_DEPTH_DIR / gt_name
    npy_path = PRED_DEPTH_DIR / (Path(gt_name).stem + "_pred_m.npy")

    # Prefer float32 npy if available
    if npy_path.exists():
        depth_m = np.load(str(npy_path)).astype(np.float32)
        if depth_m.ndim == 3:
            depth_m = depth_m.squeeze()
        return depth_m

    if png_path.exists():
        im = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[WARN] failed to read prediction PNG: {png_path}")
            return None
        if im.ndim == 3:
            im = im[..., 0]
        depth_m = im.astype(np.float32) / 256.0
        return depth_m

    print(f"[WARN] no prediction depth found for {image_name} (expected {png_path} or {npy_path})")
    return None


def box_distance(depth_map_m: np.ndarray,
                 x1: int, y1: int, x2: int, y2: int,
                 mode: str, frac: float = 0.3) -> float | None:
    """
    Same logic as GT box_distance_from_gt:
      - mode="bottom": use bottom frac of bbox height (vehicles, cyclists)
      - mode="center": use central frac x frac region (persons)

    Returns median depth in meters, or None if invalid.
    """
    H, W = depth_map_m.shape[:2]

    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None

    if mode == "bottom":
        band_h = int(h * frac)
        if band_h <= 0:
            return None
        y_start = max(y1, y2 - band_h)
        patch = depth_map_m[y_start:y2, x1:x2]
    else:  # "center"
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

    valid = patch[(patch > 0.0) & np.isfinite(patch) & (patch < MAX_DEPTH_M)]
    if valid.size == 0:
        return None

    return float(np.median(valid))


def mode_for_category(eval_category: str) -> str:
    """
    Use same rule as before:
      - bottom for vehicles/cyclists
      - center for person
    """
    cat = (eval_category or "").lower()
    if any(k in cat for k in ["car", "truck", "largeveh", "cyclist"]):
        return "bottom"
    return "center"


# ========= MAIN =========
def main():
    assert OBJ_GT_JSON.exists(), f"Missing GT JSON: {OBJ_GT_JSON}"

    with open(OBJ_GT_JSON, "r", encoding="utf-8") as f:
        objs = json.load(f)

    print(f"Loaded {len(objs)} GT objects from {OBJ_GT_JSON}")

    # Group objects by image to avoid reloading depth for each object
    objs_by_image: dict[str, list[dict]] = {}
    for o in objs:
        img_name = o["image"]
        objs_by_image.setdefault(img_name, []).append(o)

    out_entries: list[dict] = []

    for idx, (img_name, obj_list) in enumerate(objs_by_image.items(), 1):
        print(f"[{idx}/{len(objs_by_image)}] Image {img_name} with {len(obj_list)} objects")

        depth_pred = load_pred_depth_for_image(img_name)
        if depth_pred is None:
            # no prediction -> all new fields None
            for o in obj_list:
                new_o = dict(o)
                new_o["ground_distance_predict"] = None
                new_o["wrongly_distance_m"] = None
                out_entries.append(new_o)
            continue

        H, W = depth_pred.shape[:2]

        for o in obj_list:
            x1, y1, x2, y2 = o["bbox_xyxy"]
            gt_dist = o.get("gt_distance_m", None)
            eval_cat = o.get("eval_category", "")

            mode = mode_for_category(eval_cat)
            pred_dist = box_distance(depth_pred, x1, y1, x2, y2, mode=mode, frac=0.3)

            new_o = dict(o)  # copy original fields

            if pred_dist is None or gt_dist is None:
                new_o["ground_distance_predict"] = None
                new_o["wrongly_distance_m"] = None
            else:
                new_o["ground_distance_predict"] = float(pred_dist)
                # your requested sign: gt - pred
                new_o["wrongly_distance_m"] = float(gt_dist - pred_dist)

            out_entries.append(new_o)

    # Save extended JSON
    with open(OUT_ERR_JSON, "w", encoding="utf-8") as f:
        json.dump(out_entries, f, indent=2)

    print(f"\nSaved per-object GT vs prediction distances to: {OUT_ERR_JSON}")


if __name__ == "__main__":
    main()
