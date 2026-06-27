from pathlib import Path
import cv2
import numpy as np

"""
PER-PIXEL per-distance-range evaluation on Cityscapes
No object detection or segmentation needed — pure pixel-level GT vs pred.

Run:
    python eval_cityscapes_per_distance.py

Distance bins (meters):
    [0,10), [10,20), [20,40), [40,80]

Output
    ========== PER-DISTANCE ERROR STATS (PER-PIXEL) ==========
    GT bin (m)     N_pixels    mean|e|     med|e|  mean(gt-p)   med(gt-p)        MSE       RMSE   meanRel%    medRel%
    -----------------------------------------------------------------------------------------------------------------
    [0, 10)        16784746      1.536      1.289      -1.516      -1.285      5.912      2.431      21.78      19.27
    [10, 20)       11537699      3.265      2.316      -3.069      -2.266     22.610      4.755      23.04      17.38
    [20, 40)       10736410      4.501      3.406      -0.968      -1.207     38.365      6.194      15.94      12.08
    [40, 80]        8259497     15.725     14.105      14.909      13.945    355.134     18.845      26.75      26.24

"""

# ── Config ────────────────────────────────────────────────────────────────────
GT_DIR   = Path(r"D:\ObjectDetection4Blind-pt2\CitySpace\depth_gt\val")
PRED_DIR = Path(r"D:\ObjectDetection4Blind-pt2\CitySpace\depth_pred\val")

BINS   = [0.0, 10.0, 20.0, 40.0, 80.0]
DMIN   = 1e-3
DMAX   = 80.0

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_u16_as_meters(p: Path) -> np.ndarray:
    x = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if x is None:
        raise FileNotFoundError(p)
    if x.ndim != 2:
        x = x[..., 0]
    return x.astype(np.float32) / 256.0


def bin_label(lo, hi):
    return f"[{lo:.0f}, {hi:.0f}]" if hi == BINS[-1] else f"[{lo:.0f}, {hi:.0f})"


def main():
    gts = sorted(GT_DIR.glob("*.png"))
    assert gts, f"No GT PNGs found in {GT_DIR}"

    # Per-bin accumulators
    bin_pairs = [(BINS[i], BINS[i+1]) for i in range(len(BINS)-1)]
    stats = {bp: {"abs": [], "signed": [], "sq": [], "rel": []} for bp in bin_pairs}

    missing = 0

    for gt_path in gts:
        base      = gt_path.name.replace("_depth_gt.png", "")
        pred_path = PRED_DIR / f"{base}_depth_pred.png"

        if not pred_path.exists():
            print(f"[WARN] Missing pred for {gt_path.name}")
            missing += 1
            continue

        gt_m   = load_u16_as_meters(gt_path)
        pred_m = load_u16_as_meters(pred_path)

        if pred_m.shape != gt_m.shape:
            pred_m = cv2.resize(pred_m, (gt_m.shape[1], gt_m.shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        pred_m = np.clip(pred_m, DMIN, DMAX)

        # Valid pixels only (GT > 0)
        valid = (gt_m > DMIN) & (gt_m <= DMAX)

        gt_v   = gt_m[valid]
        pred_v = pred_m[valid]

        # Assign each pixel to a distance bin based on GT
        for (lo, hi) in bin_pairs:
            if hi == BINS[-1]:
                bin_mask = (gt_v >= lo) & (gt_v <= hi)
            else:
                bin_mask = (gt_v >= lo) & (gt_v <  hi)

            if bin_mask.sum() == 0:
                continue

            g = gt_v[bin_mask]
            p = pred_v[bin_mask]
            e = g - p

            stats[(lo, hi)]["abs"].append(np.abs(e))
            stats[(lo, hi)]["signed"].append(e)
            stats[(lo, hi)]["sq"].append(e ** 2)
            stats[(lo, hi)]["rel"].append(100.0 * np.abs(e) / g)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\nImages evaluated: {len(gts) - missing}  |  Missing: {missing}")

    header = (
        f"{'GT bin (m)':12s} "
        f"{'N_pixels':>10s} "
        f"{'mean|e|':>10s} "
        f"{'med|e|':>10s} "
        f"{'mean(gt-p)':>11s} "
        f"{'med(gt-p)':>11s} "
        f"{'MSE':>10s} "
        f"{'RMSE':>10s} "
        f"{'meanRel%':>10s} "
        f"{'medRel%':>10s}"
    )
    print("\n========== PER-DISTANCE ERROR STATS (PER-PIXEL) ==========")
    print(header)
    print("-" * len(header))

    for bp in bin_pairs:
        lo, hi = bp

        if not stats[bp]["abs"]:
            N = 0
            mean_abs = med_abs = mean_signed = med_signed = float("nan")
            mse = rmse = mean_rel = med_rel = float("nan")
        else:
            abs_arr    = np.concatenate(stats[bp]["abs"])
            signed_arr = np.concatenate(stats[bp]["signed"])
            sq_arr     = np.concatenate(stats[bp]["sq"])
            rel_arr    = np.concatenate(stats[bp]["rel"])

            N          = abs_arr.size
            mean_abs   = float(abs_arr.mean())
            med_abs    = float(np.median(abs_arr))
            mean_signed= float(signed_arr.mean())
            med_signed = float(np.median(signed_arr))
            mse        = float(sq_arr.mean())
            rmse       = float(np.sqrt(mse))
            mean_rel   = float(rel_arr.mean())
            med_rel    = float(np.median(rel_arr))

        print(
            f"{bin_label(lo, hi):12s} "
            f"{N:10d} "
            f"{mean_abs:10.3f} "
            f"{med_abs:10.3f} "
            f"{mean_signed:11.3f} "
            f"{med_signed:11.3f} "
            f"{mse:10.3f} "
            f"{rmse:10.3f} "
            f"{mean_rel:10.2f} "
            f"{med_rel:10.2f}"
        )


if __name__ == "__main__":
    main()