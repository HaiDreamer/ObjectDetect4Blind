from pathlib import Path
import cv2, numpy as np

'''
Why "output become strange" happens ?

Wrong model type for metric evaluation.
depth_anything_v2_vits.pth is a foundation (relative) depth model — it does not predict meters. 
DA-V2 only predicts absolute depth after metric fine-tuning (e.g., Hypersim/VKITTI2 checkpoints under metric_depth/). If you want “real meters, no GT fitting,” use those metric weights. 

Using only “median scale” on a relative map.
Relative models are typically scale- & shift-ambiguous (often closer to inverse depth). A simple median scale in depth space won't 
fix the shift term (and may be applied in the wrong domain), which explodes AbsRel/δ metrics. A standard fix is affine calibration in
inverse depth (fit a, b such that a*pred + b ≈ 1/gt, then invert). This kind of scale+shift correction is widely used with monocular 
relative depth.

Output (depth anything v2 small model(original version))
d1, d2, d3, AbsRel, SqRel, RMSE, RMSElog, SILog, log10
0.943, 0.988, 0.996, 0.084, 0.452, 3.581, 0.124, 12.326, 0.036

'''

# config
GT_DIR   = Path(r"C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main\mini_gt_100")                # folder of GT uint16 PNGs (subset)
PRED_DIR = Path(r"C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main\pred_affine_kitti16_100")    # folder of prediction uint16 PNGs

def load_u16_as_meters(p: Path) -> np.ndarray:
    """
    Read a KITTI-format depth PNG and convert to meters.

    Parameters
    ----------
    p : Path
        Full path to a single-channel uint16 PNG where KITTI encodes depth as:
        meters = uint16_value / 256.0 (0 means invalid pixel).

    Returns
    -------
    np.ndarray (float32, HxW)
        Depth in meters for each pixel.
    """
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # x: raw uint16 image from disk
    if x is None:
        raise FileNotFoundError(p)
    return x.astype(np.float32) / 256.0           # convert KITTI units -> meters  (meters = value / 256.0)

def metrics(pred: np.ndarray, gt: np.ndarray, dmin: float = 1e-3, dmax: float = 80.0):
    """
    Compute common monocular depth metrics on valid pixels.

    Parameters
    ----------
    pred : np.ndarray (HxW, float32)
        Predicted depth in meters.
    gt   : np.ndarray (HxW, float32)
        Ground-truth depth in meters. Values <= 0 are invalid.
    dmin : float
        Minimum depth considered valid for evaluation (to avoid log(0), etc.).
    dmax : float
        Maximum depth considered valid for evaluation (typical KITTI cap is 80 m).

    Returns
    -------
    tuple of 9 floats:
        d1, d2, d3: accuracy rates δ<1.25, δ<1.25^2, δ<1.25^3 (higher is better)
        AbsRel: mean absolute relative error
        SqRel: mean squared relative error
        RMSE: root mean squared error (meters)
        RMSElog: RMSE in log space
        SILog: scale-invariant log error x 100 (lower is better; KITTI ranks by √SILog)
        log10: mean absolute log10 error
    """

    # pred, gt: local working copies (both in meters)
    pred = np.clip(pred, dmin, dmax)  # clamp predictions to [dmin, dmax]
    gt   = gt.copy()
    gt[np.isinf(gt)] = 0         
    gt[np.isnan(gt)] = 0              

    # valid : boolean mask where GT is in-range (what we will score on)
    valid = (gt > dmin) & (gt < dmax)
    if valid.sum() == 0:
        # If no valid pixels, return NaNs so caller can handle gracefully.
        return tuple([float("nan")] * 9)

    # p, g : 1D arrays of predicted/GT depths over valid pixels
    p, g = pred[valid], gt[valid]

    # thresh : elementwise max(p/g, g/p) used for δ accuracies (classic BTS/KITTI eval)
    thresh = np.maximum(p / g, g / p)

    # d1, d2, d3 : accuracy under multiplicative thresholds (higher is better)
    d1 = (thresh < 1.25).mean()          # δ < 1.25
    d2 = (thresh < 1.25 ** 2).mean()     # δ < 1.25^2
    d3 = (thresh < 1.25 ** 3).mean()     # δ < 1.25^3

    # AbsRel : mean absolute relative error |p - g| / g
    absrel = np.mean(np.abs(p - g) / g)

    # SqRel : mean squared relative error (p - g)^2 / g
    sqrel  = np.mean(((p - g) ** 2) / g)

    # RMSE : sqrt(mean((p - g)^2)) in meters
    rmse   = np.sqrt(np.mean((p - g) ** 2))

    # RMSElog : sqrt(mean((log p - log g)^2))
    rmselog = np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))

    # e : per-pixel log difference used for SILog
    e = np.log(p) - np.log(g)

    # SILog : scale-invariant log error = sqrt(E[e^2] - (E[e])^2) × 100
    silog  = np.sqrt(np.mean(e ** 2) - (np.mean(e) ** 2)) * 100.0

    # log10 : mean absolute log10 error
    log10  = np.mean(np.abs(np.log10(p) - np.log10(g)))

    return d1, d2, d3, absrel, sqrel, rmse, rmselog, silog, log10


# Driver code (unchanged; reads all pairs and prints averaged metrics)
gts = sorted(GT_DIR.glob("*.png"))
assert gts, f"No GT PNGs found in {GT_DIR}"

accs = []
for gt_path in gts:
    pred_path = PRED_DIR / gt_path.name  # prediction must share the same basename
    gt_m   = load_u16_as_meters(gt_path)
    pred_m = load_u16_as_meters(pred_path)
    accs.append(metrics(pred_m, gt_m))

accs = np.array(accs, dtype=np.float64)
labels = ["d1","d2","d3","AbsRel","SqRel","RMSE","RMSElog","SILog","log10"]
print(", ".join(labels))
print(", ".join(f"{accs[:, i].mean():.3f}" for i in range(accs.shape[1])))
