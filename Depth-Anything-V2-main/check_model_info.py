import os, sys, torch

# ------------- CONFIG (edit only this path) -------------
CKPT_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits.pth"
# --------------------------------------------------------

# Import the official DA-V2 model class
from depth_anything_v2.dpt import DepthAnythingV2  # from the DA-V2 repo

def main():
    if not os.path.isfile(CKPT_PATH):
        print(f"[!] Checkpoint not found:\n    {CKPT_PATH}")
        sys.exit(1)

    # DA-V2 Small config (from the official README)
    model = DepthAnythingV2(
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384]
    )

    # Load your weights
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    vit = model.pretrained  # DINOv2 encoder used by DA-V2

    # Count blocks (works whether or not chunking is used; DA-V2 uses no chunking)
    try:
        num_blocks = len(vit.blocks)
    except Exception:
        num_blocks = getattr(vit, "n_blocks", "unknown")

    print("=== Encoder info ===")
    print(f"Encoder type      : vits (ViT-S/14)")
    print(f"Embedding dim     : {getattr(vit, 'embed_dim', 'unknown')}")
    print(f"Number of blocks  : {num_blocks}")
    print(f"Tap indices (DA-V2): {model.intermediate_layer_idx['vits']}")

if __name__ == "__main__":
    main()
