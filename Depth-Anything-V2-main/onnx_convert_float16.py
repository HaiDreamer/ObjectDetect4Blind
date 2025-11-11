import os, io, torch, onnx
from depth_anything_v2.dpt import DepthAnythingV2
from onnxconverter_common import float16

CKPT_DIR   = r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints"
FP32_CKPT  = os.path.join(CKPT_DIR, "depth_anything_v2_vits.pth")
ONNX_FP32  = os.path.join(CKPT_DIR, "depth_anything_v2_vits_fp32.onnx")
ONNX_FP16  = os.path.join(CKPT_DIR, "depth_anything_v2_vits_fp16.onnx")
OPSET = 18

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

def load_state_any_format(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    else:
        sd = obj
    return { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }

def build_model():
    m = DepthAnythingV2(**MODEL_CONFIGS['vits'])
    m.load_state_dict(load_state_any_format(FP32_CKPT), strict=True)
    m.eval()
    return m

def export_fp32(model, H=518, W=518):
    print("[1/3] Exporting ONNX (dynamo) ...")
    dummy = torch.randn(1, 3, H, W, dtype=torch.float32)

    # ✅ FIX: use the list form of dynamic_shapes (no arg names needed)
    torch.onnx.export(
        model, (dummy,), ONNX_FP32,
        opset_version=OPSET, do_constant_folding=True,
        input_names=["input"], output_names=["depth"],
        dynamo=True,
        dynamic_shapes=[{0: "batch", 2: "h", 3: "w"}],  # <-- only INPUTS here
    )
    onnx.checker.check_model(onnx.load(ONNX_FP32))
    print(f"Exported FP32 ONNX: {ONNX_FP32}")

def convert_to_fp16():
    print("[2/3] Converting to FP16 ...")
    m = onnx.load(ONNX_FP32)
    m_fp16 = float16.convert_float_to_float16(
        m,
        keep_io_types=True,                       # I/O stays FP32 for app simplicity
        op_block_list=["Resize","LayerNormalization","Softmax"],  # conservative
    )
    onnx.save(m_fp16, ONNX_FP16)
    onnx.checker.check_model(onnx.load(ONNX_FP16))
    print(f"Saved FP16 ONNX: {ONNX_FP16}")

def optional_to_ort():
    import sys, subprocess, shutil
    try:
        import onnxruntime  # ensure it's installed and importable
    except Exception:
        print("[3/3] Skipping .ort packaging (onnxruntime not installed).")
        return

    # make sure the module is invokable as a CLI
    module = "onnxruntime.tools.convert_onnx_models_to_ort"
    cmd = [sys.executable, "-m", module, ONNX_FP16, "--optimization_style", "Runtime"]

    # optional: write outputs next to ONNX (default) or choose an output dir:
    # cmd += ["--output_dir", os.path.dirname(ONNX_FP16)]

    print("[3/3] Converting ONNX -> ORT format ...")
    try:
        subprocess.run(cmd, check=True)
        print("Created .ort and required_operators*.config next to the ONNX.")
    except subprocess.CalledProcessError as e:
        print(f"ORT conversion failed with exit code {e.returncode}.")


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    model = build_model()
    export_fp32(model)
    convert_to_fp16()
    optional_to_ort()

if __name__ == "__main__":
    main()
