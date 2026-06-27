import os, glob, csv
import cv2, numpy as np
import torch, torch.nn as nn

'''
ALGORITHM for pixel-by-pixel error and how visually similar between teacher and student   
    Compute a teacher depth (full model) once.
    For each ViT block i, replace it with nn.Identity() → “student”.
    Compare student vs teacher with:
        meter errors (compute_metric_errors)
        image similarity (compute_visual_metrics_01)
    Save visuals using to_uint8_fixed_range and a colored diff map.
    Rank blocks by mean RMSE drift.

INPUT
    original metric depth model + some valid testing image    

OUTPUT
    Ranking (least important → most important) form block_change_summary_metric.csv
    Using primary “damage” RMSE_m (lower RMSE = less change = safer to prune first):
    Block 10 (RMSE 5.57)
    Block 11 (RMSE 6.49)
    Block 9 (RMSE 9.96)
    Block 6 (RMSE 10.72)
    Block 7 (RMSE 11.75)
    Block 8 (RMSE 12.06)
    Block 4 (RMSE 16.93)
    Block 3 (RMSE 16.99)
    Block 1 (RMSE 18.17)
    Block 5 (RMSE 18.27)
    Block 0 (RMSE 19.00)
    Block 2 (RMSE 19.72) most important (skipping it changes output most)    

'''

# ---------------- CONFIG ----------------
CKPT_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
IMG_DIR   = r"C:\Python\ObjectDetect4Blind\assets"
OUT_DIR   = r".\block_change_reports_metric"
MAX_IMGS  = 6
INPUT_SIZE = 518        # default image size, 518 is “nice” for ViT patch size 14

# For metric VKITTI (outdoor) models, max_depth should be 80 meters
MAX_DEPTH_METERS = 80.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# NOTE: visual metrics removed; we keep only metric error metrics: RMSE/MAE/MeanRel (AbsRel-style)

# ---------- helpers ----------
def to_uint8_fixed_range(arr, lo, hi):
    """Convert a float depth map (meters) into an 8-bit grayscale image for saving/visualization."""
    arr = np.asarray(arr, dtype=np.float32)
    x = (arr - lo) / (hi - lo + 1e-12)
    x = np.clip(x, 0, 1)
    return (x * 255.0 + 0.5).astype(np.uint8)

def compute_metric_errors(student_m, teacher_m, max_depth):
    """
    Compare metric depths (meters) without scale/shift alignment.

    Returns:
        rmse_m:   sqrt(mean((s - t)^2)) over valid pixels
        mae_m:    mean(|s - t|) over valid pixels
        meanrel:  mean(|s - t| / t) over valid pixels  (AbsRel definition)
    """
    s = np.asarray(student_m, dtype=np.float64)
    t = np.asarray(teacher_m, dtype=np.float64)

    mask = (
        np.isfinite(s) & np.isfinite(t) &
        (t > 1e-6) & (t <= max_depth) &
        (s >= 0.0) & (s <= max_depth)
    )
    if int(mask.sum()) < 100:
        return float("nan"), float("nan"), float("nan")

    diff = s[mask] - t[mask]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    mae  = float(np.mean(np.abs(diff)))
    meanrel = float(np.mean(np.abs(diff) / t[mask]))
    return rmse, mae, meanrel

def load_images():
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    paths = []
    if IMG_DIR and os.path.isdir(IMG_DIR):
        for e in exts:
            paths += glob.glob(os.path.join(IMG_DIR, e))

    imgs = []
    for p in paths[:MAX_IMGS]:
        im = cv2.imread(p)
        if im is not None:
            imgs.append((os.path.basename(p), im))

    # fallback 
    if not imgs:
        for i in range(MAX_IMGS):
            arr = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
            imgs.append((f"random_{i}.png", arr))
    return imgs

# for metric checkpoints, DepthAnythingV2 support max_depth and output meters.
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(
    encoder='vits',
    features=64,
    out_channels=[48,96,192,384],
    max_depth=MAX_DEPTH_METERS
).to(DEVICE).eval()

state = torch.load(CKPT_PATH, map_location="cpu")
# Some training scripts save {'model': state_dict, ...}
if isinstance(state, dict) and "model" in state:
    state = state["model"]
model.load_state_dict(state, strict=True)

vit = model.pretrained
nblocks = len(vit.blocks)
print(f"[Info] Encoder blocks: {nblocks}")

@torch.inference_mode()
def predict_depth_meters(bgr):
    # metric model returns HxW float depth in meters
    return model.infer_image(bgr, input_size=INPUT_SIZE)

# ---------- data ----------
imgs = load_images()

# teacher predictions once (full model)
teacher = {}
for name, im in imgs:
    teacher[name] = predict_depth_meters(im)

# CSV summary (metric errors)
csv_path = os.path.join(OUT_DIR, "block_change_summary_metric.csv")
with open(csv_path, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["block_idx", "RMSE_m", "MAE_m", "MeanRel"])

summary = []

# loop over blocks
for i in range(nblocks):
    print(f"[Eval] Skipping block {i} ...")
    original = vit.blocks[i]
    vit.blocks[i] = nn.Identity()  # ok on any device (no params)

    block_dir = os.path.join(OUT_DIR, f"skip_block_{i:02d}")
    os.makedirs(block_dir, exist_ok=True)

    rmse_list, mae_list, meanrel_list = [], [], []

    for name, im in imgs:
        t = teacher[name]
        s = predict_depth_meters(im)

        rmse, mae, meanRel = compute_metric_errors(s, t, MAX_DEPTH_METERS)
        rmse_list.append(rmse)
        mae_list.append(mae)
        meanrel_list.append(meanRel)

        # Visuals: [Teacher | Student | heatmap] with depth range 0 -> 80m
        t8 = to_uint8_fixed_range(t, 0.0, MAX_DEPTH_METERS)
        s8 = to_uint8_fixed_range(s, 0.0, MAX_DEPTH_METERS)
        ad = to_uint8_fixed_range(np.abs(t - s), 0.0, MAX_DEPTH_METERS)
        ad_c = cv2.applyColorMap(ad, cv2.COLORMAP_INFERNO)

        triptych = np.hstack([
            cv2.cvtColor(t8, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(s8, cv2.COLOR_GRAY2BGR),
            ad_c
        ])
        cv2.imwrite(os.path.join(block_dir, f"{os.path.splitext(name)[0]}metric.png"), triptych)

    # restore block
    vit.blocks[i] = original

    mean_rmse    = float(np.nanmean(rmse_list))
    mean_mae     = float(np.nanmean(mae_list))
    mean_meanrel = float(np.nanmean(meanrel_list))

    summary.append((i, mean_rmse, mean_mae, mean_meanrel))

    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            i,
            f"{mean_rmse:.6f}", f"{mean_mae:.6f}", f"{mean_meanrel:.6f}",
        ])

    print(f"[Block {i:02d}] RMSE={mean_rmse:.3f}m  MAE={mean_mae:.3f}m  MeanRel={mean_meanrel:.4f}")

# rank by RMSE (lower drift in meters => less important => prune-first)
summary.sort(key=lambda x: x[1])  # ascending RMSE

print("\n=== Block ranking (METRIC) ===")
for (i, rmse, mae, meanrel) in summary:
    print(f"Block {i:2d} : RMSE={rmse:.3f}m  MAE={mae:.3f}m  MeanRel={meanrel:.4f}")

print(f"\nVisuals per block in: {os.path.abspath(OUT_DIR)}\\skip_block_XX\\*metric.png")
print(f"CSV summary: {os.path.abspath(csv_path)}")
