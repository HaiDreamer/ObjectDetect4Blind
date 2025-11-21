import os
import sys
import subprocess
from pathlib import Path
import warnings
import torch
import onnx
from onnxconverter_common import float16

'''
ALGORITHM
    Take a DepthAnythingV2 metric-depth PyTorch checkpoint → export it to ONNX FP32 → convert to ONNX FP16 → optionally convert to ORT format.

INPUT: original metric depth small model

OUTPUT
    For experiments / portability → use the FP16 ONNX model (.onnx)
    For production on ONNX Runtime (desktop/server) → use the .ort model
    For mobile app:
        If use ONNX Runtime Mobile, .onnx or .ort are both valid, but .ort is smaller & faster to load.
'''


# -------------------
# Paths & configs
# -------------------
ROOT = Path(__file__).resolve().parent

# Use metric_depth version of DepthAnythingV2
METRIC_ROOT = ROOT / "metric_depth"
sys.path.insert(0, str(METRIC_ROOT))
from depth_anything_v2.dpt import DepthAnythingV2  

CKPT_DIR  = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints"

# Metric checkpoint (PyTorch)
FP32_CKPT = os.path.join(CKPT_DIR, "depth_anything_v2_metric_vkitti_vits.pth")

# ONNX output paths (saved in the same directory)
ONNX_FP32 = os.path.join(CKPT_DIR, "depth_anything_v2_metric_vkitti_vits_fp32.onnx")
ONNX_FP16 = os.path.join(CKPT_DIR, "depth_anything_v2_metric_vkitti_vits_fp16.onnx")

OPSET = 18          # holding the ONNX opset version, widely supported by modern onnxruntime versions and works well with PyTorch’s exporter.

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}
ENCODER   = "vits"      # small metric depth model
MAX_DEPTH = 80.0        # outdoor VKITTI metric depth

def load_state_any_format(path):
    '''adapter for various checkpoint layouts'''
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    else:
        sd = obj
    # strip 'module.' if present
    sd = { (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
           for k, v in sd.items() }
    return sd

def build_metric_model():
    '''build metric depth model'''
    assert os.path.isfile(FP32_CKPT), f"Checkpoint not found: {FP32_CKPT}"
    cfg = {**MODEL_CONFIGS[ENCODER], "max_depth": MAX_DEPTH}
    model = DepthAnythingV2(**cfg)
    state = load_state_any_format(FP32_CKPT)
    model.load_state_dict(state, strict=True)
    model.eval().to("cpu")
    return model

def export_fp32(model, H=518, W=518):
    '''export to onnx float 32'''
    print("[1/3] Exporting ONNX FP32 ...")
    dummy = torch.randn(1, 3, H, W, dtype=torch.float32)    # shape (1, 3, 518, 518) in FP32.
    torch.onnx.export(                                      # export model to onnx format
        model,
        (dummy,),
        ONNX_FP32,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["depth"],
        dynamo=True,
        dynamic_shapes=[{0: "batch", 2: "h", 3: "w"}],
    )
    onnx.checker.check_model(onnx.load(ONNX_FP32))
    print(f"Exported FP32 ONNX: {ONNX_FP32}")

def convert_to_fp16():
    """Convert FP32 ONNX -> FP16 ONNX, silencing tiny truncation warnings."""
    print("[2/3] Converting ONNX to FP16 ...")

    # Load the freshly exported FP32 ONNX
    m = onnx.load(ONNX_FP32)

    # Silence the specific truncation warning during this conversion
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*will be truncated.*",
            category=UserWarning,
            module=r"onnxconverter_common\.float16",
        )
        m_fp16 = float16.convert_float_to_float16(
            m,
            keep_io_types=True,
            op_block_list=["Resize", "LayerNormalization", "Softmax"],
        )

    onnx.save(m_fp16, ONNX_FP16)
    onnx.checker.check_model(onnx.load(ONNX_FP16))
    print(f"Saved FP16 ONNX: {ONNX_FP16}")


def optional_to_ort():
    try:
        import onnxruntime  
    except Exception:
        print("[3/3] Skipping ORT conversion (onnxruntime not installed).")
        return
    module = "onnxruntime.tools.convert_onnx_models_to_ort"
    cmd = [sys.executable, "-m", module, ONNX_FP16, "--optimization_style", "Runtime"]
    print("[3/3] Converting ONNX FP16 -> ORT format ...")
    try:
        subprocess.run(cmd, check=True)
        print("Created .ort and required_operators*.config next to the ONNX.")
    except subprocess.CalledProcessError as e:
        print(f"ORT conversion failed with exit code {e.returncode}.")

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    model = build_metric_model()
    export_fp32(model)
    #convert_to_fp16()
    # optional_to_ort()

if __name__ == "__main__":
    main()
