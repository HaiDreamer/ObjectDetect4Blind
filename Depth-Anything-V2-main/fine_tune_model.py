import os, json, time, random, argparse
from pathlib import Path
import numpy as np
import cv2
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

'''
Fine-tunes a pruned Depth Anything V2 model (ViT-S encoder) for relative monocular depth on a KITTI-style dataset (RGB + 16-bit PNG depth). 
Depth Anything V2 is a recent, high-quality foundation model for monocular depth; its official implementations commonly run at an input size of 518 by default. 
Rebuilds the encoder by dropping specified ViT blocks and retapping intermediate features per your pruning metadata, then loads a pruned checkpoint strictly.
Trains with scale-invariant objectives: SILog + scale/shift - aligned L1 + gradient loss (edges). SILog originates from Eigen et al. (scale-invariant depth). 
Evaluates with SILog on a held-out split and saves the best checkpoint when validation improves.
'''

# ---- DA-V2 ----
from depth_anything_v2.dpt import DepthAnythingV2

# ----------------- CLI -----------------
def get_args():
    p = argparse.ArgumentParser("Fine-tune pruned Depth Anything V2 (relative)")
    # pruned weights + meta from your pruning script
    p.add_argument("--pruned_ckpt", type=str,
                   default=r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned.pth")
    p.add_argument("--pruned_meta", type=str,
                   default=r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned.meta.json")

    # data (KITTI-style: RGBs + uint16 PNGs where meters = value/256, 0 invalid)
    p.add_argument("--rgb_dir", type=str, default=r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped\image")
    p.add_argument("--gt_dir",  type=str, default=r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped\groundtruth_depth")

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch",  type=int, default=1)
    p.add_argument("--img_size", type=int, default=518)      # DA-V2 default inference size
    p.add_argument("--max_depth", type=float, default=80.0)  # 20.0 for indoor
    p.add_argument("--lr_enc",  type=float, default=2e-5)
    p.add_argument("--lr_head", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=1500)
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--max_long", type=int, default=512)      # cap the long side to control N^2 attention

    # loss weights
    p.add_argument("--w_silog", type=float, default=0.5)  # SILog vs SSI-L1
    p.add_argument("--w_grad",  type=float, default=0.5)  # edge/gradient

    # output  (raw string to avoid Windows backslash escapes)
    p.add_argument("--outdir", type=str, default=r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints")
    # saving behavior
    p.add_argument("--save_last", action="store_true", help="also save last checkpoint every epoch")
    p.add_argument("--best_delta", type=float, default=1e-4, help="minimum improvement to update best")
    return p.parse_args()

# ----------------- Model rebuild (pruned) -----------------
def load_pruned_da_v2(pruned_ckpt: str, meta_json: str, device: str):
    enc_cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    m = DepthAnythingV2(**enc_cfg)

    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
    drop = set(meta.get("drop_blocks", []))
    taps = meta.get("retapped_vits", [2, 5, 8, 11])  # default taps

    vit = m.pretrained
    keep_ids = [i for i in range(len(vit.blocks)) if i not in drop]
    vit.blocks = nn.ModuleList([vit.blocks[i] for i in keep_ids])
    if hasattr(vit, "n_blocks"):
        vit.n_blocks = len(vit.blocks)
    if hasattr(m, "intermediate_layer_idx"):
        m.intermediate_layer_idx['vits'] = taps

    sd = torch.load(pruned_ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    sd = { (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v for k, v in sd.items() }
    m.load_state_dict(sd, strict=True)

    print(f"[PRUNED] kept {len(keep_ids)}/{len(keep_ids)+len(drop)} ViT blocks | taps={taps}")
    return m.to(device)

# ----------------- Helpers -----------------
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IM_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def _ceil_divisible(x, d=14):
    return int(((x + d - 1) // d) * d)

def _resize_keep_aspect(img, target_short, max_long=None, mult=14, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    # 1) scale so the short side hits target_short
    s = float(target_short) / float(min(h, w))
    nh, nw = int(round(h * s)), int(round(w * s))
    # 2) clamp long side
    if max_long is not None and max(nh, nw) > max_long:
        s2 = float(max_long) / float(max(nh, nw))
        nh, nw = int(round(nh * s2)), int(round(nw * s2))
    # 3) snap to ViT patch multiple (14 for DINOv2)
    nh, nw = _ceil_divisible(nh, mult), _ceil_divisible(nw, mult)
    return cv2.resize(img, (nw, nh), interpolation=interpolation)

# ----------------- Dataset -----------------
class KittiLikeRelDataset(Dataset):
    """
    Expects:
      rgb_dir/*.png|*.jpg
      gt_dir/*.png   (uint16: meters = val/256.0, 0 invalid)
    """
    def __init__(self, rgb_dir, gt_dir, size=518, max_depth=80.0, max_long=None):
        self.rgb_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.rgb_paths += sorted(Path(rgb_dir).glob(ext))
        assert self.rgb_paths, f"No RGB images in {rgb_dir}"
        self.gt_dir = Path(gt_dir)
        self.size = int(size)
        self.max_long = int(max_long) if max_long else None
        self.max_depth = float(max_depth)

        # --- sanity echo ---
        print("[SANITY] rgb_dir:", Path(rgb_dir))
        print("[SANITY] gt_dir :", self.gt_dir)
        print("[SANITY] #rgb:", len(self.rgb_paths), "| #gt png:", len(list(self.gt_dir.glob('*.png'))))
        try:
            sp = self.rgb_paths[0]
            gp = self._pair_name(sp)
            print("[SANITY] sample pair:")
            print("         rgb:", sp.name)
            print("         gt :", gp.name, "(exists:", gp.exists(), ")")
            if gp.exists():
                d0 = cv2.imread(str(gp), cv2.IMREAD_UNCHANGED)
                if d0 is not None:
                    valid = (d0 > 0).mean()*100.0
                    print(f"         gt dtype={d0.dtype}, min={d0.min()}, max={d0.max()}, valid%≈{valid:.2f}")
        except Exception as e:
            print("[SANITY] sample read failed:", e)

    def __len__(self): return len(self.rgb_paths)

    def _pair_name(self, rgb_path: Path):
        a_png = rgb_path.with_suffix(".png").name
        b = a_png.replace("_image_", "_groundtruth_depth_", 1)
        p = self.gt_dir / b
        if p.exists(): return p
        b2 = re.sub(r"(_sync)_image_", r"\1_groundtruth_depth_", a_png, count=1)
        p2 = self.gt_dir / b2
        if p2.exists(): return p2
        m = re.search(r"_(\d{10})_image_(0[23])\.png$", a_png)
        if m:
            frame, cam = m.groups()
            hits = sorted(self.gt_dir.glob(f"*{frame}_image_{cam}.png"))
            if hits: return hits[0]
        return self.gt_dir / a_png

    def __getitem__(self, i):
        rgb_p = self.rgb_paths[i]
        gt_p  = self._pair_name(rgb_p)

        # --- RGB ---
        bgr = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(rgb_p)
        bgr = _resize_keep_aspect(bgr, self.size, max_long=self.max_long, mult=14, interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]

        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1))
        rgb_t = (rgb_t - IM_MEAN) / IM_STD

        # --- Depth GT: uint16 → meters, NEAREST resize, mask ---
        gt_u16 = cv2.imread(str(gt_p), cv2.IMREAD_UNCHANGED)
        if gt_u16 is None:
            gt_m = np.zeros((H, W), np.float32); mask = np.zeros((H, W), np.uint8)
        else:
            gt_m_full = gt_u16.astype(np.float32) / 256.0  # KITTI: meters = val / 256
            mask_full = (gt_u16 > 0).astype(np.uint8)
            gt_m = cv2.resize(gt_m_full, (W, H), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask_full, (W, H), interpolation=cv2.INTER_NEAREST)

        gt_t   = torch.from_numpy(np.clip(gt_m, 1e-6, self.max_depth))
        mask_t = torch.from_numpy(mask.astype(np.uint8))

        if mask_t.sum() == 0:
            print(f"[WARN] zero valid depth in {gt_p.name} (pairing?)")

        return rgb_t, gt_t, mask_t

# ----------------- Losses -----------------
def silog_loss(pred, gt, mask, eps=1e-8):
    """
    Scale-Invariant Log loss (Eigen et al.). Uses mask over valid pixels.
    """
    m = mask.bool()
    p = torch.clamp(pred[m], min=eps)
    g = torch.clamp(gt[m],   min=eps)
    d = (torch.log(p) - torch.log(g))
    return torch.sqrt(torch.mean(d**2) - torch.mean(d)**2 + 1e-8)


def gradient_loss(pred, gt, mask):
    """
    L1 on depth gradients (edge/structure) with sparse masking.
    Mask AFTER finite differences to keep edges inside valid areas.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0); gt = gt.unsqueeze(0); mask = mask.unsqueeze(0)

    def grads(x):
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy

    dxp, dyp = grads(pred)
    dxg, dyg = grads(gt)

    mx = mask[..., :, 1:] & mask[..., :, :-1]
    my = mask[..., 1:, :] & mask[..., :-1, :]

    loss_x = (dxp - dxg).abs()[mx].mean() if mx.any() else pred.new_tensor(0.0)
    loss_y = (dyp - dyg).abs()[my].mean() if my.any() else pred.new_tensor(0.0)
    return loss_x + loss_y

def solve_scale_shift(y_rel, gt, mask, eps=1e-6):
    """
    Fit a and b (least squares) to minimize || a*y_rel + b - gt ||^2 on valid pixels.
    This is the standard per-image scale+shift alignment used for relative depth.
    """
    m = mask.bool()
    # If too few valid pixels, fall back to no alignment
    if m.sum().item() < 10:
        return 1.0, 0.0

    y = y_rel[m].view(-1)   # predicted relative depth (vectorized over valid pixels)
    g = gt[m].view(-1)      # ground-truth depth (vectorized over valid pixels)

    # Build design matrix A = [y, 1]
    A = torch.stack([y, torch.ones_like(y)], dim=1)  # shape [N, 2]

    # Solve least squares A [a b]^T ≈ g
    if hasattr(torch.linalg, "lstsq"):  # PyTorch ≥ 1.9
        x = torch.linalg.lstsq(A, g).solution  # shape [2]
    else:  # fallback (deprecated API kept for older versions)
        x, _ = torch.lstsq(g.unsqueeze(1), A)
        x = x[:2, 0]

    a, b = x[0].item(), x[1].item()
    return a, b

# ----------------- Forward helper -----------------
def forward_relative_depth(model, rgb_batch):
    return model(rgb_batch)  # (B,H',W')

# ----------------- Train -----------------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build pruned model
    model = load_pruned_da_v2(args.pruned_ckpt, args.pruned_meta, device=device)
    model.train()

    # Data
    ds = KittiLikeRelDataset(args.rgb_dir, args.gt_dir,
                             size=args.img_size, max_depth=args.max_depth, max_long=args.max_long)
    n = len(ds)
    idx = list(range(n)); random.shuffle(idx)
    cut = int(0.9 * n)
    train_ds = torch.utils.data.Subset(ds, idx[:cut])
    val_ds   = torch.utils.data.Subset(ds, idx[cut:])

    # Windows/CPU-aware DataLoader
    use_cuda = (device == "cuda")
    nworkers = 0 if (os.name == "nt" or not use_cuda) else 4
    dl  = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                     num_workers=nworkers, pin_memory=use_cuda)
    dlv = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                     num_workers=max(0, nworkers//2), pin_memory=use_cuda)

    # Optimizer with param split
    enc_params, head_params = [], []
    for n_, p in model.named_parameters():
        if not p.requires_grad: continue
        (enc_params if n_.startswith("pretrained") else head_params).append(p)

    if len(enc_params)==0 or len(head_params)==0:
        print("[WARN] empty param group; using all params with lr_head")
        opt = AdamW(model.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
    else:
        print("#enc params", sum(p.numel() for p in enc_params),
              "#head params", sum(p.numel() for p in head_params))
        opt = AdamW([
            {"params": enc_params,  "lr": args.lr_enc},
            {"params": head_params, "lr": args.lr_head},
        ], weight_decay=args.weight_decay)

    scaler = GradScaler(enabled=args.mixed_precision)

    # --- quick end-to-end sanity check ---
    with torch.no_grad():
        rgb_t, gt_m, mask_u8 = next(iter(dl))
        print("[SANITY] batch shapes:", tuple(rgb_t.shape), tuple(gt_m.shape), tuple(mask_u8.shape))
        B, C, H, W = rgb_t.shape
        tokens = (H//14)*(W//14)
        print(f"[SANITY] tokens/image={tokens}  attn_N^2={tokens*tokens:,}  HxW={H}x{W}")
        rgb_t = rgb_t.to(device, non_blocking=True)
        y0 = forward_relative_depth(model, rgb_t)
        print("[SANITY] model out shape:", tuple(y0.shape), "| min/max:", float(y0.min()), float(y0.max()))

    global_step = 0
    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0

        for rgb_t, gt_m, mask_u8 in dl:
            rgb_t = rgb_t.to(device, non_blocking=True)
            gt_m  = gt_m.to(device, non_blocking=True)
            mask  = mask_u8.to(device, non_blocking=True).bool()

            with autocast(enabled=args.mixed_precision):
                rel_raw = forward_relative_depth(model, rgb_t)       # (B,H',W') raw logits
                # Smooth positivity to avoid gradient death at the lower clamp
                rel_pos = F.softplus(rel_raw)                         # >0 with nonzero gradient

                rel_for_si = rel_pos + 1e-6                           # avoid log(0) only
                rel_for_l1 = torch.clamp(rel_pos, 1e-6, args.max_depth)

                # per-image scale+shift alignment
                ssi_l1 = rel_for_l1.new_tensor(0.0)
                grad_l = rel_for_l1.new_tensor(0.0)
                valid_imgs = 0
                for b in range(rel_for_l1.size(0)):
                    if mask[b].any():
                        a, b0 = solve_scale_shift(rel_for_l1[b], gt_m[b], mask[b])
                        y = a * rel_for_l1[b] + b0
                        y = torch.clamp(y, 1e-6, args.max_depth)
                        m = mask[b]
                        ssi_l1 = ssi_l1 + (y[m] - gt_m[b][m]).abs().mean()
                        grad_l = grad_l + gradient_loss(y, gt_m[b], m)
                        valid_imgs += 1

                if valid_imgs > 0:
                    ssi_l1 = ssi_l1 / valid_imgs
                    grad_l = grad_l / valid_imgs

                si = silog_loss(rel_for_si, gt_m, mask)

                loss = args.w_silog * si + (1.0 - args.w_silog) * ssi_l1 + args.w_grad * grad_l

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # one-time grad presence check
            if global_step == 0:
                g = sum((p.grad is not None) for p in model.parameters())
                print(f"[dbg] params with grad = {g}")

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            # debug signal: valid%, clamp hit rates
            with torch.no_grad():
                valid_pct = mask.float().mean().item() * 100.0
                clip_lo = (rel_for_l1 <= 1e-6).float().mean().item() * 100.0
                clip_hi = (rel_for_l1 >= args.max_depth - 1e-6).float().mean().item() * 100.0
                if global_step % 50 == 0:
                    print(f"[dbg] valid%={valid_pct:.2f}  clip_lo%={clip_lo:.2f}  clip_hi%={clip_hi:.2f}  loss={float(loss):.4f}")

            run_loss += float(loss)
            global_step += 1

        # ---- quick validation (SILog) ----
        model.eval()
        vals = []
        with torch.no_grad(), autocast(enabled=args.mixed_precision):
            for rgb_t, gt_m, mask_u8 in dlv:
                rgb_t = rgb_t.to(device, non_blocking=True)
                gt_m  = gt_m.to(device, non_blocking=True)
                mask  = mask_u8.to(device, non_blocking=True).bool()
                rel   = forward_relative_depth(model, rgb_t)
                rel   = F.softplus(rel) + 1e-6  # positivity + avoid log(0)
                vals.append(silog_loss(rel, gt_m, mask).item())
        val_silog = float(np.mean(vals)) if vals else 0.0

        print(f"Epoch {epoch}/{args.epochs} | train {run_loss/len(dl):.4f} | val SILog {val_silog:.3f} | {time.time()-t0:.1f}s")

        # ---- saving ----
        improved = (val_silog < best_val - args.best_delta)
        if improved:
            best_val = val_silog
            out_ckpt = os.path.join(args.outdir, "depth_anything_v2_vits_pruned_rel_best.pth")
            os.makedirs(args.outdir, exist_ok=True)
            torch.save(model.state_dict(), out_ckpt)
            with open(os.path.join(args.outdir, "train_cfg.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "img_size": args.img_size, "max_depth": args.max_depth,
                    "lr_enc": args.lr_enc, "lr_head": args.lr_head,
                    "w_silog": args.w_silog, "w_grad": args.w_grad,
                    "max_long": args.max_long
                }, f, indent=2)
            print(f"[saved BEST] {out_ckpt}")

        if args.save_last:
            last_ckpt = os.path.join(args.outdir, f"last_epoch{epoch:03d}.pth")
            torch.save(model.state_dict(), last_ckpt)
            print(f"[saved LAST] {last_ckpt}")

if __name__ == "__main__":
    main()
