from pathlib import Path
import argparse
import cv2
import numpy as np

r"""
Run code
  # metric model: Evaluate KITTI-style depth predictions (uint16 PNG, depth[m] = value/256).
  python eval_kitti_subset.py --gt-dir "C:\Python\ObjectDetectRequireFile\put-in-metric-depth\groundtruth_depth" --pred-dir "C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_onnx"

  # relative predictions model
  python eval_kitti_subset.py --gt-dir "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\groundtruth_depth" --pred-dir "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\pred_affine_kitti16_100"

TODO
    Chcek output of this

    Done → C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits
    Total time: 2968.76 s | Avg: 2.969 s/img | Throughput: 0.34 img/s

Output (depth anything v2 small model(original relative version))
    Original model
		 - Avg speed: 3.524 s/img
		 - Memory: 97 MB
    d1, d2, d3, AbsRel, SqRel, RMSE, RMSElog, SILog, log10
    0.943, 0.988, 0.996, 0.084, 0.452, 3.581, 0.124, 12.326, 0.036

Output (depth anything v2 small model(original metric depth version))
    Original model
		 - Avg speed: 3.2.969 s/img
		 - Memory: 94.6 MB 
    d1, d2, d3, AbsRel, SqRel, RMSE, RMSElog, SILog, log10
    0.854, 0.969, 0.991, 0.119, 0.679, 4.668, 0.176, 16.453, 0.053

Algorithm
    INPUT: ground truth depth and RGB image
    OUTPUT: accuracy...
"""

def load_u16_as_meters(p: Path) -> np.ndarray:
    """
    Read a KITTI-format depth PNG and convert to meters.

    KITTI convention:
        depth_meters = uint16_value / 256.0
        0 means invalid pixel.
    """
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # x: raw uint16 image from disk
    if x is None:
        raise FileNotFoundError(p)
    if x.ndim != 2:
        # Ensure single-channel depth
        x = x[..., 0]
    return x.astype(np.float32) / 256.0


# def metrics(pred: np.ndarray, gt: np.ndarray,
#             dmin: float = 1e-3, dmax: float = 80.0):
#     """
#     Compute common monocular depth metrics on valid pixels.

#     Parameters
#     ----------
#     pred : np.ndarray (HxW, float32)
#         Predicted depth in meters.
#     gt   : np.ndarray (HxW, float32)
#         Ground-truth depth in meters. Values <= 0 are invalid.
#     dmin : float
#         Minimum depth considered valid for evaluation (to avoid log(0), etc.).
#     dmax : float
#         Maximum depth considered valid for evaluation (typical KITTI cap is 80 m).

#     Returns
#     -------
#     tuple of 9 floats:
#         d1, d2, d3: accuracy rates δ<1.25, δ<1.25^2, δ<1.25^3 (higher is better)
#         AbsRel: mean absolute relative error
#         SqRel: mean squared relative error
#         RMSE: root mean squared error (meters)
#         RMSElog: RMSE in log space
#         SILog: scale-invariant log error x 100 (lower is better)
#         log10: mean absolute log10 error
#     """

#     # pred, gt: local working copies (both in meters)
#     pred = np.clip(pred, dmin, dmax)  # clamp predictions to [dmin, dmax]
#     gt = gt.copy()
#     gt[np.isinf(gt)] = 0
#     gt[np.isnan(gt)] = 0

#     # valid: boolean mask where GT is in-range (what we will score on)
#     valid = (gt > dmin) & (gt < dmax)
#     if valid.sum() == 0:
#         # If no valid pixels, return NaNs so caller can handle gracefully.
#         return tuple([float("nan")] * 9)

#     # p, g: 1D arrays of predicted/GT depths over valid pixels
#     p, g = pred[valid], gt[valid]

#     # thresh: elementwise max(p/g, g/p) used for δ accuracies
#     thresh = np.maximum(p / g, g / p)

#     # d1, d2, d3: accuracy under multiplicative thresholds (higher is better)
#     d1 = (thresh < 1.25).mean()          # δ < 1.25
#     d2 = (thresh < 1.25 ** 2).mean()     # δ < 1.25^2
#     d3 = (thresh < 1.25 ** 3).mean()     # δ < 1.25^3

#     # AbsRel: mean absolute relative error |p - g| / g
#     absrel = np.mean(np.abs(p - g) / g)

#     # SqRel: mean squared relative error (p - g)^2 / g
#     sqrel = np.mean(((p - g) ** 2) / g)

#     # RMSE: sqrt(mean((p - g)^2)) in meters
#     rmse = np.sqrt(np.mean((p - g) ** 2))

#     # RMSElog: sqrt(mean((log p - log g)^2))
#     rmselog = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))

#     # e: per-pixel log difference used for SILog
#     e = np.log(p) - np.log(g)

#     # SILog: scale-invariant log error = sqrt(E[e^2] - (E[e])^2) × 100
#     silog = np.sqrt(np.mean(e ** 2) - (np.mean(e) ** 2)) * 100.0

#     # log10: mean absolute log10 error
#     log10 = np.mean(np.abs(np.log10(p) - np.log10(g)))

#     return d1, d2, d3, absrel, sqrel, rmse, rmselog, silog, log10

def metrics(pred: np.ndarray, gt: np.ndarray,
            dmin: float = 1e-3, dmax: float = 80.0):

    pred = np.clip(pred, dmin, dmax)
    gt = gt.copy()
    gt[np.isinf(gt)] = 0
    gt[np.isnan(gt)] = 0

    valid = (gt > dmin) & (gt < dmax)
    if valid.sum() == 0:
        return tuple([float("nan")] * 10)  # now 10 metrics

    p, g = pred[valid], gt[valid]

    thresh = np.maximum(p / g, g / p)
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    absrel = np.mean(np.abs(p - g) / g)
    sqrel = np.mean(((p - g) ** 2) / g)

    rmse = np.sqrt(np.mean((p - g) ** 2))

    # NEW: mean absolute error in meters
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
        default=Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth"),
        help="Directory containing GT uint16 KITTI depth PNGs (default: mini_gt_100).",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        #pred_metric_kitti_vkitti_vits_torch for original model, pred_metric_kitti_vkitti_vits_onnx_azure for onnx model, pred_metric_kitti_vkitti_vits_onnx_int8_cpu for int8 onnx model
        default=Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_onnx_int8_cpu"), 
        help=("Directory containing prediction uint16 KITTI depth PNGs. "
              "Default: metric model outputs (pred_metric_kitti_vkitti_vits)."),
    )
    args = parser.parse_args()

    GT_DIR = args.gt_dir
    PRED_DIR = args.pred_dir

    print(f"GT_DIR   = {GT_DIR}")
    print(f"PRED_DIR = {PRED_DIR}")

    gts = sorted(GT_DIR.glob("*.png"))
    assert gts, f"No GT PNGs found in {GT_DIR}"

    accs = []
    missing = 0

    for gt_path in gts:
        pred_path = PRED_DIR / gt_path.name  # prediction must share the same basename
        if not pred_path.exists():
            print(f"[WARN] Missing prediction for {gt_path.name} → {pred_path}")
            missing += 1
            continue

        gt_m = load_u16_as_meters(gt_path)
        pred_m = load_u16_as_meters(pred_path)

        # If shapes differ (shouldn't happen if you resized properly), resize pred to GT
        if pred_m.shape != gt_m.shape:
            pred_m = cv2.resize(
                pred_m,
                (gt_m.shape[1], gt_m.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        accs.append(metrics(pred_m, gt_m))

    if not accs:
        raise RuntimeError("No valid GT/prediction pairs found. Check directories and filenames.")

    accs = np.array(accs, dtype=np.float64)
    labels = ["d1","d2","d3","AbsRel","SqRel","RMSE","MAE","RMSElog","SILog","log10"]

    print("\n# Images evaluated:", accs.shape[0])
    if missing > 0:
        print("# Images missing predictions:", missing)

    print(", ".join(labels))
    print(", ".join(f"{accs[:, i].mean():.3f}" for i in range(accs.shape[1])))


if __name__ == "__main__":
    main()
