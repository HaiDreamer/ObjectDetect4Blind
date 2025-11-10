import os, glob, csv, math
import cv2, numpy as np
import torch, torch.nn as nn

'''
How it is does ?
    running the full model once to get a “teacher” depth map for each image,
    skipping one encoder block at a time (replacing it with Identity()), re-inferencing to get a “student”,
    aligning the student to the teacher's scale, computing error/quality metrics, saving visuals, and logging a CSV 
        summary—then ranking blocks by how much they hurt quality when skipped. Depth Anything V2 is a DPT-style decoder with a 
        ViT backbone (DAv2 commonly uses DINOv2 features).
        
Result:
    block_change_report: which block affect most, visualize, compare -> choose block to pruning
        
Processing:
    Align the “student” depth to the “teacher” depth
    Solve a per-image least-squares fit on valid pixels to find alpha, beta such that:
    alpha * student + beta ≈ teacher
    In code: numpy.linalg.lstsq on the flattened arrays.
    Build the aligned student: s_aligned = alpha * student + beta
    Save alpha and beta for later averaging.
    Put both maps on the teacher's scale for fair comparison

    Compute the teacher's range:
    t_min = min(teacher)
    t_ptp = max(teacher) - min(teacher) + 1e-12

    Normalize both to [0,1] using the SAME (teacher) range:
    t01 = (teacher - t_min) / t_ptp
    s01 = (s_aligned - t_min) / t_ptp

    Per-image metrics (added to running lists)
    MSE01 = mean( (t01 - s01)^2 )
    MAE01 = mean( |t01 - s01| )
    PSNR = -10 * log10(MSE01) (range is [0,1]; the helper caps zero error at 99 dB)
    SSIM = structural_similarity(t01, s01, data_range=1.0) if skimage is available; otherwise NaN

    After processing all images for this block
    mean_mse = average of all per-image MSE01 values
    mean_mae = average of all per-image MAE01 values
    mean_psnr = average of all per-image PSNR values
    mean_ssim = average (nanmean) of all per-image SSIM values
    mean_a = average of all alpha values from the alignments
    mean_b = average of all beta values from the alignments      

Note:
    Scale & shift alignment: Because monocular depth is relative (affine-ambiguous), you align student → teacher by solving 
min(a, b) ||as + b - t||^2 per image before computing errors. This mirrors DPT/MiDaS evaluation practice for non-metric depth.
'''



# ---------------- CONFIG ----------------
CKPT_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits.pth"
IMG_DIR   = r"C:\Python\ObjectDetect4Blind\assets"  # your images (no labels needed)
OUT_DIR   = r".\block_change_reports"
MAX_IMGS  = 6
INPUT_SIZE = 518
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)

# optional SSIM
try:
    from skimage.metrics import structural_similarity as ssim_metric
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False

# ---------- helpers ----------
def align_student_to_teacher(student, teacher):
    """
    Solve alpha,beta in alpha * student + beta ~= teacher (least squares),
    and return aligned student.
    """
    s = student.astype(np.float64).ravel()
    t = teacher.astype(np.float64).ravel()
    mask = np.isfinite(s) & np.isfinite(t)
    if mask.sum() < 10:
        return student, 1.0, 0.0
    X = np.vstack([s[mask], np.ones_like(s[mask])]).T
    alpha, beta = np.linalg.lstsq(X, t[mask], rcond=None)[0]
    return alpha * student + beta, float(alpha), float(beta)

def to_uint8_shared_range(arr, ref_min, ref_ptp):
    """Map arr using a single shared (ref) range -> uint8 for visualization/metrics."""
    if not np.isfinite(ref_ptp) or ref_ptp <= 0:
        return np.zeros_like(arr, dtype=np.uint8)
    x = (arr - ref_min) / ref_ptp
    x = np.clip(x, 0, 1)
    return (x * 255.0 + 0.5).astype(np.uint8)

def psnr_from_mse01(mse01):
    """PSNR given MSE on [0,1] range."""
    if mse01 <= 0:
        return 99.0
    return 20.0 * math.log10(1.0) - 10.0 * math.log10(mse01)

def load_images():
    """Load up to MAX_IMGS from IMG_DIR or repo examples; else random."""
    exts = ("*.jpg","*.jpeg","*.png","*.bmp")
    paths = []
    if IMG_DIR and os.path.isdir(IMG_DIR):
        for e in exts: paths += glob.glob(os.path.join(IMG_DIR, e))
    if not paths:
        for e in exts: paths += glob.glob(os.path.join("assets","examples",e))
    imgs = []
    for p in paths[:MAX_IMGS]:
        im = cv2.imread(p)
        if im is not None:
            imgs.append((os.path.basename(p), im))
    if not imgs:
        for i in range(MAX_IMGS):
            arr = np.random.randint(0,255,(INPUT_SIZE,INPUT_SIZE,3),dtype=np.uint8)
            imgs.append((f"random_{i}.png", arr))
    return imgs

# ---------- model ----------
# Depth Anything V2 uses a DINOv2 ViT encoder + DPT-style decoder
# Ref: official repo run/infer code path.
from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384]).to(DEVICE).eval()
state = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state, strict=True)

vit = model.pretrained
try:
    nblocks = len(vit.blocks)
except Exception:
    nblocks = getattr(vit, "n_blocks", None)
print(f"[Info] Encoder blocks: {nblocks}")

@torch.inference_mode()
def predict_depth(bgr):
    # model handles internal preprocessing; returns HxW float32 (relative depth)
    d = model.infer_image(bgr, input_size=INPUT_SIZE)
    return d

# ---------- data ----------
imgs = load_images()

# teacher predictions once
teacher = {}
for name, im in imgs:
    teacher[name] = predict_depth(im)

# CSV summary
csv_path = os.path.join(OUT_DIR, "block_change_summary.csv")
with open(csv_path, "w", newline="") as fcsv:
    writer = csv.writer(fcsv)
    writer.writerow(["block_idx","MSE01","MAE01","PSNR","SSIM","alpha","beta"])

summary = []

# ---------- loop over blocks ----------
for i in range(nblocks):
    print(f"[Eval] Skipping block {i} ...")
    original = vit.blocks[i]
    vit.blocks[i] = nn.Identity()  # skip this block

    block_dir = os.path.join(OUT_DIR, f"skip_block_{i:02d}")
    os.makedirs(block_dir, exist_ok=True)

    mse_list, mae_list, psnr_list, ssim_list = [], [], [], []
    ab_list = []

    for name, im in imgs:
        t = teacher[name]
        s = predict_depth(im)

        # Align student -> teacher
        s_aligned, a, b = align_student_to_teacher(s, t)
        ab_list.append((a,b))

        # Use the TEACHER dynamic range for both maps (shared range)
        t_min = float(np.nanmin(t))
        t_ptp = float(np.nanmax(t) - np.nanmin(t) + 1e-12)

        t01 = (t - t_min) / t_ptp
        s01 = (s_aligned - t_min) / t_ptp

        # Metrics on [0,1]
        diff01 = t01 - s01
        mse01 = float(np.mean(diff01**2))
        mae01 = float(np.mean(np.abs(diff01)))
        psnr  = psnr_from_mse01(mse01)
        if HAVE_SSIM:
            ssim_val = float(ssim_metric(t01.astype(np.float32), s01.astype(np.float32), data_range=1.0))
        else:
            ssim_val = float('nan')

        mse_list.append(mse01); mae_list.append(mae01); psnr_list.append(psnr); ssim_list.append(ssim_val)

        # Visuals: show [Teacher | Student(aligned) | |T-S| heatmap] using shared range
        t8 = to_uint8_shared_range(t, t_min, t_ptp)
        s8 = to_uint8_shared_range(s_aligned, t_min, t_ptp)
        ad = to_uint8_shared_range(np.abs(t - s_aligned), 0.0, t_ptp)  # abs diff in teacher units
        ad_c = cv2.applyColorMap(ad, cv2.COLORMAP_INFERNO)
        triptych = np.hstack([cv2.cvtColor(t8, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(s8, cv2.COLOR_GRAY2BGR),
                              ad_c])
        cv2.imwrite(os.path.join(block_dir, f"{os.path.splitext(name)[0]}_T-S-D.png"), triptych)

    vit.blocks[i] = original  # restore

    mean_mse = float(np.mean(mse_list))
    mean_mae = float(np.mean(mae_list))
    mean_psnr = float(np.mean(psnr_list))
    mean_ssim = float(np.nanmean(ssim_list)) if HAVE_SSIM else float('nan')
    mean_a = float(np.mean([a for a,b in ab_list]))
    mean_b = float(np.mean([b for a,b in ab_list]))

    summary.append((i, mean_mse, mean_mae, mean_psnr, mean_ssim))

    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([i, f"{mean_mse:.8f}", f"{mean_mae:.8f}", f"{mean_psnr:.3f}", f"{mean_ssim:.5f}", f"{mean_a:.6f}", f"{mean_b:.6f}"])

    print(f"[Block {i:02d}] MSE01={mean_mse:.6f}  MAE01={mean_mae:.6f}  PSNR={mean_psnr:.2f}  SSIM={mean_ssim:.4f}  (alpha~{mean_a:.4f}, beta~{mean_b:.4f})")

# rank by average MSE01 (higher change => more important)
summary.sort(key=lambda x: x[1], reverse=True)
print("\n=== Block importance (higher change => more important) ===")
for i, mse, mae, psnr, ssim_v in summary:
    print(f"Block {i:2d} : MSE01={mse:.6f}  MAE01={mae:.6f}  PSNR={psnr:.2f}  SSIM={ssim_v:.4f}")

print(f"\nVisuals per block in: {os.path.abspath(OUT_DIR)}\\skip_block_XX\\*_T-S-D.png")
print(f"CSV summary: {os.path.abspath(csv_path)}")
