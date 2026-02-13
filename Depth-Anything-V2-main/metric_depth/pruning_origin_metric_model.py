import os, sys
import torch
import torch.nn as nn

'''
Use this after rank_vits_block_metric_depth
'''

# ---------------- CONFIG ----------------
REPO_ROOT = r"C:\Python\ObjectDetect4Blind\Depth-Anything-V2-main"  
CKPT_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints\depth_anything_v2_metric_vkitti_vits.pth"
SAVE_DIR  = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints"

BLOCK_TO_PRUNE = 10                 # choose 0..11
MAX_DEPTH_METERS = 80.0             # VKITTI outdoor metric models use 80m
INPUT_SIZE = 518

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

# import METRIC version (so max_depth is supported)
sys.path.insert(0, os.path.join(REPO_ROOT, "metric_depth"))
from depth_anything_v2.dpt import DepthAnythingV2

def strip_module_prefix(state_dict):
    """If keys are like 'module.xxx', strip 'module.' for single-GPU loading."""
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

# Build model (metric)
model = DepthAnythingV2(
    encoder="vits",
    features=64,
    out_channels=[48, 96, 192, 384],
    max_depth=MAX_DEPTH_METERS
).to(DEVICE).eval()

# Load checkpoint
ckpt = torch.load(CKPT_PATH, map_location="cpu")
if isinstance(ckpt, dict) and "model" in ckpt:
    sd = ckpt["model"]
else:
    sd = ckpt
sd = strip_module_prefix(sd)
model.load_state_dict(sd, strict=True)

# Prune (skip) ONE encoder block permanently
vit = model.pretrained
nblocks = len(vit.blocks)
assert 0 <= BLOCK_TO_PRUNE < nblocks, f"BLOCK_TO_PRUNE must be in [0, {nblocks-1}]"

vit.blocks[BLOCK_TO_PRUNE] = nn.Identity()

# Save new checkpoint (include metadata so you can reload correctly later)
os.makedirs(SAVE_DIR, exist_ok=True)
out_path = os.path.join(
    SAVE_DIR,
    f"depth_anything_v2_metric_vkitti_vits_pruned_block{BLOCK_TO_PRUNE:02d}.pth"
)

torch.save(
    {
        "model": model.state_dict(),
        "pruned_block": BLOCK_TO_PRUNE,
        "max_depth": MAX_DEPTH_METERS,
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "input_size": INPUT_SIZE,
    },
    out_path
)

print("Saved pruned checkpoint to:", out_path)
