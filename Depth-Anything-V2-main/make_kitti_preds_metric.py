from pathlib import Path
import time
import cv2
import numpy as np
import sys

"""
INPUT
- RGB images: C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\image\*.png
- Ground truth depth map in KITTI format (uint16): C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth\*.png
- Model: outdoor VKITTI small depth_anything_v2_metric_vkitti_vits (PyTorch or ONNX)
OUTPUT
- Folder location: C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits*
- KITTI-style uint16 PNG (depth = value / 256) for compatibility with existing evaluation code and benchmarks, 
    and raw float32 arrays (.npy) in meters for precise, efficient analysis of per-object distances.

EXPLAINATION
Encode predictions as KITTI-style uint16 PNGs 
    mainly for compatibility, consistency, and practicality. 
    Using the standard KITTI encoding (value ∈ [0,65535], depth = value / 256, 0 = invalid) means all official and third-party 
        KITTI tools work directly on our predictions without any custom logic. 
    Reusing the same uint16x256 encoding as our previous relative-depth pipeline keeps the evaluation script identical and allows 
       fair, apples-to-apples comparison between metric and aligned-relative models. Storing depth as uint16 PNG instead of float32 
       greatly reduces file size and makes sharing and visualization easier. 
    Quantization step (≈4 mm) is tiny compared to typical model errors -> negligible impact on accuracy.
"""

# =========================================================
# CHOOSE BACKEND: "torch" (original .pth) or "onnx"
# =========================================================
MODE = "onnx"   # change to "onnx" to use ONNX / ORT model instead // or "torch" for original model

# ================== KITTI paths & output ==================
# Adjust these to your setup
KITTI_ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root")
IMG_DIR = KITTI_ROOT / "val_selection_cropped" / "image"
GT_DIR  = KITTI_ROOT / "val_selection_cropped" / "groundtruth_depth"

OUT_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of images to export (None = all)
N = 1000   

MAX_DEPTH = 80.0          # VKITTI outdoor metric model

# =========================================================
# TORCH BACKEND (original metric .pth)
# =========================================================
if MODE == "torch":
    import torch

    ROOT = Path(__file__).resolve().parent
    METRIC_DIR = ROOT / "metric_depth"
    sys.path.insert(0, str(METRIC_DIR))

    from depth_anything_v2.dpt import DepthAnythingV2

    # As in the official metric_depth README:
    # encoder='vits', dataset='vkitti', max_depth=80 for outdoor model 
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    ENCODER = "vits"          # using depth_anything_v2_metric_vkitti_vits.pth

    CKPT = Path(
        r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
    )

    assert CKPT.exists(), f"Missing checkpoint: {CKPT}"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def _strip_module(sd: dict):
        """Remove 'module.' prefix from DataParallel checkpoints, if present."""
        out = {}
        for k, v in sd.items():
            if isinstance(k, str) and k.startswith("module."):
                out[k[7:]] = v
            else:
                out[k] = v
        return out

    # Build model
    model = DepthAnythingV2(**{**model_configs[ENCODER], "max_depth": MAX_DEPTH})

    # Load state dict
    state = torch.load(str(CKPT), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _strip_module(state)
        model.load_state_dict(state, strict=True)
    else:
        # Rare case: checkpoint already a nn.Module
        model = state

    model.to(DEVICE).eval()

    @torch.inference_mode()
    def infer_metric_depth(bgr: np.ndarray) -> np.ndarray:
        """
        Run metric Depth Anything V2 on a BGR image (PyTorch).
        Returns HxW float32 depth map in meters.
        """
        # DepthAnythingV2.infer_image expects an OpenCV-style BGR image,
        # as in the official examples.
        depth = model.infer_image(bgr)
        return depth.astype(np.float32, copy=False)

# =========================================================
# ONNX BACKEND (FP16 .onnx / .ort)
# =========================================================
else:  # MODE == "onnx"
    import onnxruntime as ort

    # ONNX model path (FP16)
    ONNX_MODEL = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits_fp16.onnx"
    # If you want to use ORT-optimized model instead, change to:
    # ONNX_MODEL = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits_fp16.with_runtime_opt.ort"

    assert Path(ONNX_MODEL).exists(), f"Missing ONNX model: {ONNX_MODEL}"

    # Create ONNX Runtime inference session
    providers = ort.get_available_providers()
    print("ONNXRuntime providers:", providers)
    sess = ort.InferenceSession(ONNX_MODEL, providers=providers)

    # Get input/output names
    input_name  = sess.get_inputs()[0].name   # e.g. "input"
    output_name = sess.get_outputs()[0].name  # e.g. "depth"

    # IMPORTANT: the ONNX was exported at a fixed input size (e.g. 518x518).
    # If we feed arbitrary HxW, an internal Reshape will break with:
    #   input shape {1,384,25,86}, requested shape {1,384,7396}
    # So we must resize input to the export resolution (e.g. 518x518) before inference.
    EXPORT_SIZE = 518  # same H=W used during ONNX export

    def preprocess_bgr_for_depth_anything(bgr: np.ndarray) -> np.ndarray:
        """
        Preprocess BGR image for Depth Anything V2 ONNX model.
        This approximates the same pipeline as DepthAnythingV2.infer_image:
            - resize to EXPORT_SIZE x EXPORT_SIZE
            - BGR -> RGB
            - to float32 in [0,1]
            - normalize with ImageNet mean/std
            - HWC -> CHW
            - add batch dimension
        """
        # 0) resize to fixed resolution used in ONNX export
        bgr_resized = cv2.resize(bgr, (EXPORT_SIZE, EXPORT_SIZE), interpolation=cv2.INTER_LINEAR)

        # 1) BGR -> RGB
        rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)

        # 2) float32 in [0, 1]
        img = rgb.astype(np.float32) / 255.0

        # 3) normalize (ImageNet style)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # 4) HWC -> CHW
        img = img.transpose(2, 0, 1)  # (3, H, W)

        # 5) add batch dimension (N, C, H, W)
        img = np.expand_dims(img, axis=0)

        return img.astype(np.float32, copy=False)

    def infer_metric_depth(bgr: np.ndarray) -> np.ndarray:
        """
        Run metric Depth Anything V2 ONNX model on a BGR image.
        Returns HxW float32 depth map in meters (EXPORT_SIZE x EXPORT_SIZE before later resize).
        """
        inp = preprocess_bgr_for_depth_anything(bgr)
        out = sess.run([output_name], {input_name: inp})[0]
        depth = np.squeeze(out).astype(np.float32)  # (EXPORT_SIZE, EXPORT_SIZE)
        return depth

# =========================================================
# Common helpers
# =========================================================

def read_gt_shape(p: Path):
    """Just read the GT file to get (H, W) for resizing."""
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    return im.shape[:2]  # (H, W)

# ================== Export loop ==================
gts_all = sorted(GT_DIR.glob("*.png"))
gts = gts_all if N is None else gts_all[:N]
assert gts, f"No GT PNGs found in {GT_DIR}"

mode_str = "PyTorch" if MODE == "torch" else "ONNX/ORT"
print(f"Backend: {mode_str}")
print(f"Exporting {len(gts)} images → {OUT_DIR}")

t0 = time.perf_counter()

for i, gt_path in enumerate(gts, 1):
    # Map GT filename to corresponding RGB filename in val_selection_cropped
    img_name = gt_path.name.replace("_groundtruth_depth_", "_image_")
    img_path = IMG_DIR / img_name
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Missing RGB for {gt_path.name}\nExpected: {img_path}")

    # Metric depth prediction (HxW, meters) via selected backend
    pred_m = infer_metric_depth(img_bgr)

    # Resize to KITTI GT crop size if necessary
    gt_h, gt_w = read_gt_shape(gt_path)
    if pred_m.shape != (gt_h, gt_w):
        pred_m = cv2.resize(
            pred_m,
            (gt_w, gt_h),
            interpolation=cv2.INTER_LINEAR
        )

    # Clamp to valid metric range for this model (0..MAX_DEPTH)
    pred_m = np.clip(pred_m, 1e-3, MAX_DEPTH)

    # OPTIONAL: save raw float32 depth map (meters)
    npy_path = OUT_DIR / (gt_path.stem + "_pred_m.npy")
    np.save(str(npy_path), pred_m.astype(np.float32))

    # Save KITTI uint16 PNG: value = round(meters * 256.0), 0 = invalid
    pred_u16 = np.clip(
        np.rint(pred_m * 256.0),
        0,
        65535
    ).astype(np.uint16)

    ok = cv2.imwrite(str(OUT_DIR / gt_path.name), pred_u16)
    if not ok:
        raise RuntimeError(f"Failed to write: {OUT_DIR / gt_path.name}")

    if i % 25 == 0 or i == len(gts):
        print(f"{i}/{len(gts)} saved")

elapsed = time.perf_counter() - t0
imgs = len(gts)
sec_per_img = elapsed / max(imgs, 1)
ips = imgs / elapsed if elapsed > 0 else float("inf")

print("Done →", OUT_DIR)
print(f"Total time: {elapsed:.2f} s | Avg: {sec_per_img:.3f} s/img | Throughput: {ips:.2f} img/s")
