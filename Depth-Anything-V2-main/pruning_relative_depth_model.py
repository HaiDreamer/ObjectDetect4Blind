import os, json, cv2, numpy as np
import torch, torch.nn as nn

'''
TODO: check if it is possible ?! then check accuracy of model 
MAIN FUNCTION: Pruning 4 drop blocks(which are least important)

Pipeline: LOBO sensitivity scoring → pick lowest-impact blocks → rebuild encoder without them (structural prune) →
    retap decoder indices → (optional) Lₙ channel prune on decoder → save checkpoint → (recommended) short fine-tune.
    
'''

# ---------- EDIT THESE ----------
CKPT_IN   = r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits.pth"
CKPT_OUT  = r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned.pth"
META_OUT  = r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned.meta.json"
DROP_BLOCKS = [10, 1, 6, 11]   # <-- from your ranking (change as you wish)
DO_STRUCTURED_PRUNE_HEAD = False  # set True to prune decoder convs a bit
STRUCTURED_AMOUNT = 0.15          # 15% output channels pruned (if enabled)
TEST_IMAGE = r"C:\Python\ObjectDetect4Blind\assets"  # optional: path to any JPG/PNG for a quick sanity check
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------------

# --- import model (Depth Anything V2 Small) ---
from depth_anything_v2.dpt import DepthAnythingV2  # repo API

def load_model():
    m = DepthAnythingV2(encoder='vits', features=64, out_channels=[48,96,192,384])
    state = torch.load(CKPT_IN, map_location="cpu")
    m.load_state_dict(state, strict=True)
    return m

def compute_keep_order(nblocks, drop_list):
    drop_set = set(drop_list)
    keep_old = [i for i in range(nblocks) if i not in drop_set]
    # map old index -> new index after compaction
    old2new = {old:i for i, old in enumerate(keep_old)}
    return keep_old, old2new

def remap_taps(default_taps_old, keep_old, old2new, want_k=4):
    """
    Try to map the original tap indices to the nearest kept layers.
    If duplicates occur or we have < want_k distinct taps, fill by even spacing.
    """
    # nearest-kept mapping
    mapped = []
    for t in default_taps_old:
        # pick kept layer with min |k - t|
        nearest = min(keep_old, key=lambda k: abs(k - t))
        mapped.append(old2new[nearest])
    # dedup while preserving order
    uniq = []
    [uniq.append(x) for x in mapped if x not in uniq]
    # if not enough distinct, fill evenly over [0, len(keep)-1]
    L = len(keep_old)
    if len(uniq) < want_k:
        extra = []
        for j in range(want_k):
            idx = int(round((j+1) * (L / (want_k+1)) - 1))
            idx = max(0, min(L-1, idx))
            extra.append(idx)
        for e in extra:
            if e not in uniq: uniq.append(e)
            if len(uniq) == want_k: break
    # finally, clip to valid range and sort ascending (recommended)
    uniq = sorted(set([x for x in uniq if 0 <= x < L]))
    # ensure length == want_k by trunc/pad (pad from evenly spaced if needed)
    while len(uniq) > want_k: uniq.pop()  # drop largest
    while len(uniq) < want_k and L > 0:
        # add middle-ish indices not yet present
        for cand in [L//8, L//4, L//2, 3*L//4, 7*L//8]:
            cand = max(0, min(L-1, cand))
            if cand not in uniq:
                uniq.append(cand)
                if len(uniq) == want_k: break
        if len(uniq) < want_k and L == 1 and 0 not in uniq:
            uniq.append(0)
            break
    uniq = sorted(uniq)
    return uniq

def drop_vit_blocks_inplace(model, drop_blocks):
    vit = model.pretrained
    try:
        n = len(vit.blocks)
    except Exception:
        n = getattr(vit, "n_blocks", None)
    assert n is not None, "Cannot read vit depth"
    keep_old, old2new = compute_keep_order(n, drop_blocks)

    # Physically re-build ModuleList in kept order
    kept_modules = [vit.blocks[k] for k in keep_old]
    vit.blocks = nn.ModuleList(kept_modules)
    # keep a correct metadata if present
    if hasattr(vit, "n_blocks"):
        vit.n_blocks = len(kept_modules)

    # Retap decoder indices for 'vits'
    default_taps_old = [2, 5, 8, 11]  # DA-V2 default for vits
    new_taps = remap_taps(default_taps_old, keep_old, old2new, want_k=4)
    if hasattr(model, "intermediate_layer_idx"):
        model.intermediate_layer_idx['vits'] = new_taps
    return keep_old, new_taps

def optional_structured_prune_head(model, amount=0.15):
    """
    Light structured pruning on decoder (DPT head) convs: prune 15% of output channels per conv.
    """
    import torch.nn.utils.prune as prune
    pruned_layers = 0
    for m in model.depth_head.modules():
        if isinstance(m, nn.Conv2d):
            # prune output channels (dim=0) by L2 norm
            prune.ln_structured(m, name='weight', amount=amount, n=2, dim=0)
            prune.remove(m, 'weight')  # make permanent
            pruned_layers += 1
    return pruned_layers

@torch.inference_mode()
def quick_forward(model, img_path=None, size=518):
    # minimal smoke test
    H, W = 518, 518
    if img_path and os.path.isfile(img_path):
        bgr = cv2.imread(img_path)
        if bgr is None: bgr = np.random.randint(0,255,(H,W,3),dtype=np.uint8)
    else:
        bgr = np.random.randint(0,255,(H,W,3),dtype=np.uint8)
    d = model.infer_image(bgr, input_size=size)
    return d.shape, float(np.nanmin(d)), float(np.nanmax(d))

def main():
    print("[*] Loading model...")
    model = load_model().to(DEVICE).eval()

    vit = model.pretrained
    n_before = len(vit.blocks)
    print(f"    ViT blocks before: {n_before}")

    print(f"[*] Dropping blocks: {sorted(DROP_BLOCKS)}")
    keep_old, taps = drop_vit_blocks_inplace(model, DROP_BLOCKS)

    n_after = len(model.pretrained.blocks)
    print(f"    Kept old-layer IDs: {keep_old}")
    print(f"    New count: {n_after}")
    print(f"    Retapped indices (vits): {taps}")

    if DO_STRUCTURED_PRUNE_HEAD:
        print("[*] Structured-pruning DPT head ...")
        num = optional_structured_prune_head(model, amount=STRUCTURED_AMOUNT)
        print(f"    Pruned Conv2d layers in head: {num}")

    # quick forward
    shape, mn, mx = quick_forward(model, TEST_IMAGE, size=518)
    print(f"[*] Sanity forward: depth shape={shape}, min={mn:.4f}, max={mx:.4f}")

    # save checkpoint + metadata
    torch.save(model.state_dict(), CKPT_OUT)
    meta = {
        "base_ckpt": CKPT_IN,
        "drop_blocks": sorted(DROP_BLOCKS),
        "kept_blocks_old_ids": keep_old,
        "retapped_vits": taps,
        "structured_prune_head": bool(DO_STRUCTURED_PRUNE_HEAD),
        "structured_amount": STRUCTURED_AMOUNT if DO_STRUCTURED_PRUNE_HEAD else 0.0,
        "vit_blocks_after": n_after
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[*] Saved pruned weights -> {CKPT_OUT}")
    print(f"[*] Saved metadata -> {META_OUT}")

if __name__ == "__main__":
    main()