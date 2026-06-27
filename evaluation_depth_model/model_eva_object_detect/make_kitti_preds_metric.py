from pathlib import Path
import time
import cv2
import numpy as np
import sys

"""
TODO
    Adapt with another dataset, here that i use kitti dataset for validation

INPUT
- RGB images: C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\image\*.png
- Ground truth depth map in KITTI format (uint16): C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\groundtruth_depth\*.png
- Model: outdoor VKITTI small depth_anything_v2_metric_vkitti_vits (PyTorch or ONNX)

OUTPUT: 
- images with metric depth
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


# Mode: torch: original model
# output: mode onnx_int8 -> C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_onnx_int8_cpu
MODE = "torch"   #"torch", "onnx_fp16", "onnx_int8"
EXPORT_VARIANT = ""     # "pruned1layer" if use 
PRUNED_BLOCK_FALLBACK = 10

# KITTI paths & output 
KITTI_ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root")
IMG_DIR = KITTI_ROOT / "val_selection_cropped" / "image"
GT_DIR  = KITTI_ROOT / "val_selection_cropped" / "groundtruth_depth"
PRUNED1LAYER_CKPT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits_pruned_block10.pth")

# Base; final OUT_DIR is chosen per-backend
#"pred_metric_kitti_vkitti_vits_torch" for original model, pred_metric_kitti_vkitti_vits_onnx_azure for onnx model, pred_metric_kitti_vkitti_vits_pruned1layer for pruned 1 layer model
OUT_BASE = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_torch")     

# Number of images to export (None = all)
N = 1000
MAX_DEPTH = 80.0          # VKITTI outdoor metric model

# backend-specific setup
if MODE == "torch":
    import torch
    import torch.nn as nn

    # instead of ROOT = Path(__file__).resolve().parent
    DEPTH_ANYTHING_REPO = Path(r"C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main").resolve()
    METRIC_DIR = DEPTH_ANYTHING_REPO / "metric_depth"
    assert METRIC_DIR.exists(), f"metric_depth not found at: {METRIC_DIR}"

    # Make metric_depth importable 
    if str(METRIC_DIR) not in sys.path:
        sys.path.insert(0, str(METRIC_DIR)) 
    from depth_anything_v2.dpt import DepthAnythingV2

    
    ENCODER = "vits"
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    if EXPORT_VARIANT == "pruned1layer":
        CKPT = PRUNED1LAYER_CKPT
    else:
        CKPT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth")

    assert CKPT.exists(), f"Missing checkpoint: {CKPT}"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # per-mode output directory
    OUT_DIR = OUT_BASE.with_name(f"{OUT_BASE.name}_torch_{DEVICE.lower()}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build model
    model = DepthAnythingV2(**{**model_configs[ENCODER], "max_depth": MAX_DEPTH})

    def _strip_module(sd: dict):
        out = {}
        for k, v in sd.items():
            if k.startswith("module."):
                out[k[len("module."):]] = v
            else:
                out[k] = v
        return out

    state = torch.load(str(CKPT), map_location="cpu")

    # Extract state_dict + (optional) pruned_block metadata
    pruned_block = None
    if isinstance(state, dict):
        if "pruned_block" in state:
            pruned_block = int(state["pruned_block"])
        if "model" in state:
            sd = state["model"]
        elif "state_dict" in state:
            sd = state["state_dict"]
        else:
            # could already be a raw state_dict-like dict
            sd = state
    else:
        raise ValueError("Checkpoint format not supported; expected dict/state_dict.")

    sd = _strip_module(sd)

    # If pruned variant: ensure prune the same block BEFORE loading weights
    if EXPORT_VARIANT == "pruned1layer":
        if pruned_block is None:
            pruned_block = PRUNED_BLOCK_FALLBACK

        vit = model.pretrained
        nblocks = len(vit.blocks)
        assert 0 <= pruned_block < nblocks, f"pruned_block must be in [0, {nblocks-1}]"
        vit.blocks[pruned_block] = nn.Identity()

    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()

    @torch.inference_mode()     # disable auto_grad(model training), view tracking, version counter (cause it not need for model evaluation) => code run better 
    def infer_metric_depth(bgr: np.ndarray) -> np.ndarray:
        depth = model.infer_image(bgr, input_size=518)  # returns meters
        return depth.astype(np.float32, copy=False)

    mode_str = f"PyTorch ({DEVICE}) [{EXPORT_VARIANT}]"


else:  # MODE starts with "onnx"
    import onnxruntime as ort

    ONNX_FP16 = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits_fp16.onnx"
    ONNX_INT8 = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits_int8.onnx"

    if MODE == "onnx_fp16":
        ONNX_MODEL = ONNX_FP16
        model_tag = "fp16"
    elif MODE == "onnx_int8":
        ONNX_MODEL = ONNX_INT8
        model_tag = "int8"
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    assert Path(ONNX_MODEL).exists(), f"Missing ONNX model: {ONNX_MODEL}"

    # Pick providers explicitly (ORT recommends passing providers list), try GPU first and fall back to CPU
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    print("ONNXRuntime available:", avail)
    print("Using providers:", providers)

    ep_tag = providers[0].replace("ExecutionProvider", "").lower()

    OUT_DIR = OUT_BASE.with_name(f"{OUT_BASE.name}_onnx_{model_tag}_{ep_tag}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sess = ort.InferenceSession(ONNX_MODEL, providers=providers)

    # Get input/output names
    input_name  = sess.get_inputs()[0].name   # e.g. "input"
    output_name = sess.get_outputs()[0].name  # e.g. "depth"

    EXPORT_SIZE = 518

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
        # resize to fixed resolution used in ONNX export
        bgr_resized = cv2.resize(bgr, (EXPORT_SIZE, EXPORT_SIZE), interpolation=cv2.INTER_LINEAR)

        # BGR -> RGB
        rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)

        # scale pixel float32 in [0, 1]
        img = rgb.astype(np.float32) / 255.0

        # ImageNet normalize, match the backbone’s training distribution of depth anything v2
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[None, ...]     # change HWC to CHW, add batch dim => (1,C,H,W)
        return img.astype(np.float32, copy=False)   # array is float32, typical input dtype expected by ONNX models

    def infer_metric_depth(bgr: np.ndarray) -> np.ndarray:
        """
        Run metric Depth Anything V2 ONNX model on a BGR image.
        Returns HxW float32 depth map in meters (EXPORT_SIZE x EXPORT_SIZE before later resize).
        """
        inp = preprocess_bgr_for_depth_anything(bgr)
        out = sess.run([output_name], {input_name: inp})[0]
        depth = np.squeeze(out).astype(np.float32, copy=False)  # (EXPORT_SIZE, EXPORT_SIZE), meters
        return depth

    mode_str = f"ONNX/ORT {model_tag} ({ep_tag})"

# helpers 
def read_gt_shape(p: Path):
    """read the GT file to get (H, W) for resizing."""
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    return im.shape[:2]  # (H, W)

# export loop 
gts_all = sorted(GT_DIR.glob("*.png"))
gts = gts_all if N is None else gts_all[:N]
assert gts, f"No GT PNGs found in {GT_DIR}"

print(f"Backend: {mode_str}")
print(f"Exporting {len(gts)} images → {OUT_DIR}")

t0 = time.perf_counter()

for i, gt_path in enumerate(gts, 1):
    # Map GT filename to corresponding RGB filename
    img_name = gt_path.name.replace("_groundtruth_depth_", "_image_")
    img_path = IMG_DIR / img_name
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Missing RGB for {gt_path.name}\nExpected: {img_path}")

    # Predict metric depth (meters)
    pred_m = infer_metric_depth(img_bgr)

    # Resize to GT crop size if necessary
    gt_h, gt_w = read_gt_shape(gt_path)
    if pred_m.shape != (gt_h, gt_w):
        pred_m = cv2.resize(pred_m, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)

    # Clamp to valid range
    pred_m = np.clip(pred_m, 1e-3, MAX_DEPTH)

    # Save raw float32 meters
    npy_path = OUT_DIR / (gt_path.stem + "_pred_m.npy")
    np.save(str(npy_path), pred_m.astype(np.float32))

    # Save KITTI uint16 PNG: value = round(meters * 256.0), 0 = invalid
    pred_u16 = np.clip(np.rint(pred_m * 256.0), 0, 65535).astype(np.uint16)
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
