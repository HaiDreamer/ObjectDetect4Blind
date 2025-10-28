import os
import io
import time
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic
from depth_anything_v2.dpt import DepthAnythingV2

# config
CKPT_DIR  = r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything"
FP32_CKPT = os.path.join(CKPT_DIR, "depth_anything_v2_vits.pth")        # FP32 checkpoint
INT8_CKPT = os.path.join(CKPT_DIR, "depth_anything_v2_vits_qv1.pth")    # output quantized model

# Official model configs (vits/vitb/vitl/vitg) Depth-Anything-V2  
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

def load_state_any_format(path):
    """Load a state_dict that might be nested or DataParallel-prefixed."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    else:
        sd = obj

    # strip "module." if present
    def strip_module(k):
        return k[7:] if k.startswith("module.") else k
    sd = {strip_module(k): v for k, v in sd.items()}
    return sd

def bytesize_of_model(model):
    """Rough on-disk size without writing: serialize to memory and count bytes."""
    buff = io.BytesIO()
    torch.save(model, buff)
    return buff.tell()

def benchmark_cpu(model, shape=(1, 3, 518, 518), runs=20, warmup=5):
    model.eval().to("cpu")
    x = torch.randn(*shape, dtype=torch.float32)
    with torch.inference_mode():
        # warmup
        for _ in range(warmup):
            _ = model(x)
        # measure
        t0 = time.perf_counter()
        for _ in range(runs):
            _ = model(x)
        t1 = time.perf_counter()
    return (t1 - t0) / runs

def main():
    # Build the FP32 model with the CORRECT encoder (vits) and load weights strictly
    print("[1/4] Loading FP32 model (encoder=vits) ...")
    encoder = 'vits'  
    model_fp32 = DepthAnythingV2(**model_configs[encoder])

    state = load_state_any_format(FP32_CKPT)
    # strict=True to surface shape/key problems immediately
    model_fp32.load_state_dict(state, strict=True)
    model_fp32.eval().to("cpu")

    # (Optional) quick CPU latency baseline
    print("[2/4] Benchmarking FP32 on CPU...")
    fp32_ms = benchmark_cpu(model_fp32) * 1000.0
    print(f"FP32 latency: {fp32_ms:.1f} ms per frame")

    # INT8 post-training dynamic quantization (Linear layers)
    print("[3/4] Quantizing (dynamic INT8) Linear layers...")
    qmodel = quantize_dynamic(
        model_fp32,
        {nn.Linear},            # ViT-heavy Linear layers → good CPU win
        dtype=torch.qint8       # weights int8; activations quantized dynamically at runtime
    )
    qmodel.eval().to("cpu")

    # quick INT8 latency check
    int8_ms = benchmark_cpu(qmodel) * 1000.0
    spd = fp32_ms / max(int8_ms, 1e-6)
    print(f"INT8 latency: {int8_ms:.1f} ms per frame  (speedup ~{spd:.2f}x)")

    # Save the quantized model as a .pth (CPU-only)
    print("[4/4] Saving quantized model...")
    torch.save(qmodel, INT8_CKPT)   # qmodel only: whole-model save; load later with torch.load(..., map_location='cpu')
    print(f"Saved: {INT8_CKPT}")

    # Compare serialized sizes
    fp32_size_mb = bytesize_of_model(model_fp32) / (1024**2)
    int8_size_mb = bytesize_of_model(qmodel) / (1024**2)
    shrink = fp32_size_mb / max(int8_size_mb, 1e-6)
    print(f"Estimated sizes -> FP32: {fp32_size_mb:.1f} MB | INT8: {int8_size_mb:.1f} MB (~{shrink:.2f}x smaller)")

if __name__ == "__main__":
    main()
