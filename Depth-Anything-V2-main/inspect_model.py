import os, sys, csv
from collections import OrderedDict
import torch
import torch.nn as nn
from depth_anything_v2.dpt import DepthAnythingV2

CHECKPOINT_PATH = r"C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main\checkpoints\depth_anything_v2_vits.pth"

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

encoder = 'vits' 
model = DepthAnythingV2(**model_configs[encoder])

# 3) Load the checkpoint (CPU-safe)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
# Some checkpoints are saved as plain state_dict; others nested (e.g., {"state_dict": ...})
state_dict = ckpt.get("state_dict", ckpt)
# If keys are prefixed (e.g., "module."), strip them
clean_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_k = k
    if new_k.startswith("module."):
        new_k = new_k[len("module."):]
    clean_state_dict[new_k] = v

missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
print(f"[load_state_dict] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
if missing:
    print("  First few missing:", missing[:10])
if unexpected:
    print("  First few unexpected:", unexpected[:10])

model.eval()

# 4) Pretty print: total params and per-module listing
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("\n=== Model Size ===")
print(f"Total params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")

rows = []
print("\n=== Named Modules (name -> type, #params) ===")
for name, m in model.named_modules():
    # skip the root module line if you want only submodules; keep it here for completeness
    n_params = sum(p.numel() for p in m.parameters(recurse=False))
    mtype = m.__class__.__name__
    print(f"{name if name else '<root>':50s} -> {mtype:25s}  params={n_params:,}")
    rows.append((name if name else "<root>", mtype, n_params))

csv_path = os.path.join(os.path.dirname(CHECKPOINT_PATH), f"da_v2_{encoder}_layers.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "type", "num_params(recurse=False)"])
    writer.writerows(rows)
print(f"\nSaved layer inventory to: {csv_path}")

try:
    from torchinfo import summary
    # Set an input size you actually use; DA-V2 can infer on arbitrary sizes, e.g., 518x518
    H = W = 518
    print("\n=== torchinfo.summary ===")
    print(summary(model, input_size=(1, 3, H, W), verbose=1, col_names=("input_size","output_size","num_params")))
except Exception as e:
    print("\n[Info] Install 'torchinfo' for a compact table summary: pip install torchinfo")
    print("      Skipping torchinfo.summary. Error was:", repr(e))
