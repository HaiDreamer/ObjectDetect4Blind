from pathlib import Path
import argparse
import cv2
import numpy as np

r"""
PER PIXEL accuracy comparation

Run code
  # Cityscapes dataset (default)
  python eval_kitti_subset.py

  # metric model: Evaluate KITTI-style depth predictions (uint16 PNG, depth[m] = value/256).
  python eval_kitti_subset.py --gt-dir "C:\Python\ObjectDetectRequireFile\put-in-metric-depth\groundtruth_depth" --pred-dir "C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_onnx"

  # relative predictions model
  python eval_kitti_subset.py --gt-dir "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\groundtruth_depth" --pred-dir "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\pred_affine_kitti16_100"

  # Pruned model (torch)
  python eval_kitti_subset.py --gt-dir "C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth" --pred-dir "C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_pruned1layer_torch_cpu"

NOTE
    missing predictions are omitted (skipped)

OUTPUT (depth anything v2 small model(original metric depth version))
    GT_DIR   = D:\ObjectDetection4Blind-pt2\CitySpace\depth_gt\val
    PRED_DIR = D:\ObjectDetection4Blind-pt2\CitySpace\depth_pred\val

    # Images evaluated: 500
    d1, d2, d3, AbsRel, SqRel, RMSE, MAE, RMSElog, SILog, log10
    0.696, 0.922, 0.977, 0.218, 2.007, 8.566, 5.138, 0.264, 24.526, 0.091
    
"""

def load_u16_as_meters(p: Path) -> np.ndarray:
    """
    Read a KITTI-format depth PNG and convert to meters.

    KITTI convention:
        depth_meters = uint16_value / 256.0
        0 means invalid pixel.
    """
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(p)
    if x.ndim != 2:
        x = x[..., 0]
    return x.astype(np.float32) / 256.0

def metrics(pred: np.ndarray, gt: np.ndarray,
            dmin: float = 1e-3, dmax: float = 80.0):

    pred = np.clip(pred, dmin, dmax)
    gt = gt.copy()
    gt[np.isinf(gt)] = 0
    gt[np.isnan(gt)] = 0

    valid = (gt > dmin) & (gt < dmax)
    if valid.sum() == 0:
        return tuple([float("nan")] * 10)

    p, g = pred[valid], gt[valid]

    thresh = np.maximum(p / g, g / p)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    absrel = np.mean(np.abs(p - g) / g)
    sqrel = np.mean(((p - g) ** 2) / g)
    rmse = np.sqrt(np.mean((p - g) ** 2))
    mae = np.mean(np.abs(p - g))

    rmselog = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))
    e = np.log(p) - np.log(g)
    silog = np.sqrt(np.mean(e ** 2) - (np.mean(e) ** 2)) * 100.0
    log10 = np.mean(np.abs(np.log10(p) - np.log10(g)))

    return d1, d2, d3, absrel, sqrel, rmse, mae, rmselog, silog, log10

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KITTI-style depth predictions (metric or relative+affine)."
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=Path(r"D:\ObjectDetection4Blind-pt2\CitySpace\depth_gt\val"),
        help="Directory containing GT uint16 KITTI depth PNGs.",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=Path(r"D:\ObjectDetection4Blind-pt2\CitySpace\depth_pred\val"),
        help="Directory containing prediction uint16 KITTI depth PNGs.",
    )
    args = parser.parse_args()

    GT_DIR   = args.gt_dir
    PRED_DIR = args.pred_dir

    print(f"GT_DIR   = {GT_DIR}")
    print(f"PRED_DIR = {PRED_DIR}")

    gts = sorted(GT_DIR.glob("*.png"))
    assert gts, f"No GT PNGs found in {GT_DIR}"

    accs = []
    missing = 0

    for gt_path in gts:
        # ── Match GT filename to pred filename ────────────────────────────────
        base      = gt_path.name.replace("_depth_gt.png", "")
        pred_path = PRED_DIR / f"{base}_depth_pred.png"

        if not pred_path.exists():
            print(f"[WARN] Missing prediction for {gt_path.name} → {pred_path}")
            missing += 1
            continue

        gt_m   = load_u16_as_meters(gt_path)
        pred_m = load_u16_as_meters(pred_path)

        if pred_m.shape != gt_m.shape:
            pred_m = cv2.resize(
                pred_m,
                (gt_m.shape[1], gt_m.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        accs.append(metrics(pred_m, gt_m))

    if not accs:
        raise RuntimeError("No valid GT/prediction pairs found. Check directories and filenames.")

    accs   = np.array(accs, dtype=np.float64)
    labels = ["d1", "d2", "d3", "AbsRel", "SqRel", "RMSE", "MAE", "RMSElog", "SILog", "log10"]

    print("\n# Images evaluated:", accs.shape[0])
    if missing > 0:
        print("# Images missing predictions:", missing)

    print(", ".join(labels))
    print(", ".join(f"{accs[:, i].mean():.3f}" for i in range(accs.shape[1])))


if __name__ == "__main__":
    main()