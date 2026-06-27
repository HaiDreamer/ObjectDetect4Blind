import os, io, time
from typing import List, Tuple
import torch
import torch.nn as nn
from depth_anything_v2.dpt import DepthAnythingV2
from torch.export import export as export_prog, save as export_save
import torch.compiler

'''
How to run ?(NO run in vs code)
    OPEN cmd
        check where cl.exe
            %comspec% /k "D:\VisualDevC++\VC\Auxiliary\Build\vcvars64.bat"
            where cl -> print directory for cl file -> ok
        cd C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main
        python quan_pt2e.py
'''

# PT2E (torchao) – x86 Inductor backend
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer

# -----------------------
# User config
# -----------------------
CKPT_DIR  = r"C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main\checkpoints"
FP32_CKPT = os.path.join(CKPT_DIR, "depth_anything_v2_vits.pth")
INT8_CKPT = os.path.join(CKPT_DIR, "depth_anything_v2_vits_pt2e.pth")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}
ENCODER = 'vits'
EXAMPLE_INPUT_SHAPE = (1, 3, 518, 518)

# Your sensitive names to keep FP32
SENSITIVE_NAME_HINTS = [
    "layernorm",      # LayerNorms (we'll also check isinstance(nn.LayerNorm))
    "patch_embed",    # patch embedding block
    "head", "pred"    # final prediction head (DPT head’s last layers)
]

# -----------------------
# Helpers
# -----------------------
def load_state_any_format(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj: sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict): sd = obj["model"]
    else: sd = obj
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    return sd

def bytesize_of_model(model):
    buff = io.BytesIO(); torch.save(model, buff); return buff.tell()

@torch.inference_mode()
def benchmark_cpu(model: nn.Module, runs=10, warmup=3, shape=EXAMPLE_INPUT_SHAPE) -> float:
    # Do NOT call .eval() on exported/prepared/converted PT2E graphs.
    x = torch.randn(*shape, dtype=torch.float32)
    for _ in range(warmup): _ = model(x)
    t0 = time.perf_counter()
    for _ in range(runs): _ = model(x)
    return (time.perf_counter() - t0) / max(1, runs)

def export_graph(model: nn.Module, example_inputs: Tuple[torch.Tensor, ...]):
    """Export with torch.export; fallback to capture_pre_autograd_graph on older torch."""
    try:
        return torch.export.export(model, example_inputs).module()  # torch >= 2.6
    except Exception:
        return torch._export.capture_pre_autograd_graph(model, example_inputs)  # older torch

def _node_under_sensitive_module(node, hints_lower):
    """
    Keep FP32 if the node belongs to:
      1) any LayerNorm
      2) patch embedding block
      3) final prediction head (DPT head)
    We use nn_module_stack FQNs + (when possible) module type or conv shape.
    """
    stack = node.meta.get("nn_module_stack", None) if hasattr(node, "meta") else None
    if not stack:
        return False

    # fast path: name hints for patch_embed / head / pred
    for fqn in stack.keys():
        low = fqn.lower()
        if "patch_embed" in low:
            return True
        if ("head" in low) or ("pred" in low):
            return True

    # type-aware path via recorded module types (when present)
    # stack: OrderedDict { "fqn": "ClassName" }
    # Treat any LayerNorm as sensitive
    for _, clsname in stack.items():
        if isinstance(clsname, str) and "layernorm" in clsname.lower():
            return True

    # shape-aware heuristic: final 1x1 Conv2d to 1 channel (depth head)
    # Some PT2E builds record node.target.module; otherwise skip gracefully.
    mod = getattr(node, "target", None)
    try:
        m = getattr(node, "owning_module", None) or getattr(mod, "_modules", None)
    except Exception:
        m = None
    # Fallback: inspect attached module if node has a bound module (GraphModule attr lookup)
    gm = getattr(node, "graph", None)
    root = getattr(gm, "owning_module", None) if gm is not None else None
    if root is not None and isinstance(getattr(node, "target", None), str):
        path = node.target
        try:
            sub = dict(root.named_modules()).get(path, None)
            if isinstance(sub, torch.nn.Conv2d):
                kH, kW = sub.kernel_size
                if sub.out_channels == 1 and kH == 1 and kW == 1:
                    return True
        except Exception:
            pass

    return False


def strip_quant_for_sensitive_modules(prepared_gm: torch.fx.GraphModule, hints: List[str]):
    """
    After prepare_pt2e(), remove quantization annotations from nodes whose
    module FQNs match your sensitive hints so they remain FP32 after convert_pt2e.
    """
    hints_lower = [h.lower() for h in hints]
    changed = False
    for n in prepared_gm.graph.nodes:
        qa = n.meta.get("quantization_annotation", None) if hasattr(n, "meta") else None
        if qa is None:
            continue
        if _node_under_sensitive_module(n, hints_lower):
            if hasattr(qa, "input_qspec_map"):
                qa.input_qspec_map.clear()
            if hasattr(qa, "output_qspec"):
                qa.output_qspec = None
            n.meta["quantization_annotation"] = qa
            changed = True
    if changed:
        prepared_gm.recompile()

# -----------------------
# Main
# -----------------------
def main():
    print("[1/6] Load FP32 model")
    model = DepthAnythingV2(**model_configs[ENCODER])
    rep = model.load_state_dict(load_state_any_format(FP32_CKPT), strict=True)
    if rep.missing_keys or rep.unexpected_keys:
        print("[load_state_dict] missing:", rep.missing_keys)
        print("[load_state_dict] unexpected:", rep.unexpected_keys)

    # Eager model can use eval()
    model.eval().to("cpu")

    print("[2/6] FP32 benchmark")
    fp32_ms = benchmark_cpu(model) * 1000.0
    print(f"FP32 latency: {fp32_ms:.1f} ms")

    print("[3/6] Export with torch.export (PT2E)")
    example_inputs = (torch.randn(*EXAMPLE_INPUT_SHAPE),)
    exported = export_graph(model, example_inputs)

    print("[4/6] Configure X86InductorQuantizer (A8W8)")
    quantizer = X86InductorQuantizer()
    quantizer.set_global(xiq.get_default_x86_inductor_quantization_config())

    print("[5/6] prepare_pt2e → strip sensitive → calibrate → convert_pt2e")
    prepared = prepare_pt2e(exported, quantizer)

    # Keep your sensitive modules in FP32
    strip_quant_for_sensitive_modules(prepared, SENSITIVE_NAME_HINTS)

    # Calibrate observers using a few samples from your data distribution
    with torch.inference_mode():
        for _ in range(16):
            _ = prepared(torch.randn(*EXAMPLE_INPUT_SHAPE))

    qmodel = convert_pt2e(prepared)

    example = torch.randn(*EXAMPLE_INPUT_SHAPE)
    ep = export_prog(qmodel, (example,))
    ep_path = os.path.join(CKPT_DIR, "depth_anything_v2_vits_pt2e.pth")
    export_save(ep, ep_path)
    print(f"Saved ExportedProgram: {ep_path}")

    qmodel = torch.compile(qmodel, backend="inductor", mode="max-autotune")
    int8_ms = benchmark_cpu(qmodel) * 1000.0
    print(f"INT8 latency: {int8_ms:.1f} ms (speedup ~{fp32_ms/max(1e-6,int8_ms):.2f}x)")

    print("[6/6] Save quantized model")
    example = torch.randn(*EXAMPLE_INPUT_SHAPE)
  
    @torch.compiler.disable
    def _export_uncompiled(mod, ex):
        return export_prog(mod, (ex,))

    # Export the UNCOMPILED model (qmodel), not qmodel_opt
    ep = _export_uncompiled(qmodel, example)

    ep_path = os.path.join(CKPT_DIR, "depth_anything_v2_vits_pt2e.ep")
    export_save(ep, ep_path)
    print(f"Saved ExportedProgram: {ep_path}")

    fp32_size_mb = bytesize_of_model(model)/(1024**2)
    int8_size_mb = bytesize_of_model(qmodel)/(1024**2)
    print(f"Sizes -> FP32: {fp32_size_mb:.1f} MB | INT8: {int8_size_mb:.1f} MB (~{fp32_size_mb/max(int8_size_mb,1e-6):.2f}x smaller)")

if __name__ == "__main__":
    main()
