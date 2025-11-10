from pathlib import Path
import time
import cv2, numpy as np, torch
from depth_anything_v2.dpt import DepthAnythingV2
from collections import OrderedDict
import json  

"""
Export DA-V2 (relative) predictions on KITTI val_selection_cropped with
per-image affine alignment in inverse depth, then save KITTI-format uint16 PNGs.

Relative monocular depth models (like depth_anything_v2_vits.pth) don't predict meters; their outputs are only accurate up to scale and shift.

- Model: depth_anything_v2_vits.pth
- Save: uint16 PNG, value = round(meters * 256.0), 0 = invalid

Input: model
Output: predicted images to compare with the labelled one (NEXT step: run eval_kitti_subset.py)
"""

# ------- config -------
N = 100         # number of images to export (set None to do all)
torch.serialization.add_safe_globals([DepthAnythingV2])

# ------- paths -------
KITTI_ROOT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root")
IMG_DIR = KITTI_ROOT / "val_selection_cropped" / "image"
GT_DIR  = KITTI_ROOT / "val_selection_cropped" / "groundtruth_depth"

OUT_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\pred_affine_kitti16_100")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------- model (relative DA-V2) -------
from collections import OrderedDict
from torch.ao.quantization import quantize_dynamic  # dynamic INT8 Linear  (weights-only)

CKPT = Path(r"C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned.pth")
BASE_FP32 = CKPT.with_name("depth_anything_v2_vits.pth")  
assert CKPT.exists(), f"Missing checkpoint: {CKPT}"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}

def _strip_module(sd: dict):
    return { (k[7:] if isinstance(k, str) and k.startswith('module.') else k): v for k, v in sd.items() }

def _is_statedict(x) -> bool:
    return isinstance(x, (dict, OrderedDict))

def _has_packed(sd: dict) -> bool:
    # heuristic: INT8 dynamic state_dict thường có '_packed_params' ở Linear
    return any('_packed_params' in k for k in sd.keys())

def _load_raw(path: Path):
    obj = torch.load(str(path), map_location='cpu')
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj, obj['state_dict']
    return obj, obj

def _load_fp32_model_from(path: Path):
    m = DepthAnythingV2(**cfg)
    sd = torch.load(str(path), map_location='cpu')
    if _is_statedict(sd):
        m.load_state_dict(_strip_module(sd), strict=True)
        return m
    # hiếm khi lưu nguyên model dưới tên FP32
    return sd if isinstance(sd, torch.nn.Module) else m

raw_obj, raw_sd = _load_raw(CKPT)
name = CKPT.name.lower()

try:
    # ---------- Case 1: FP32 ----------
    if ('depth_anything_v2_vits.pth' in name) and not ('_q' in name or 'qv1' in name or 'pruned' in name):
        model = DepthAnythingV2(**cfg)
        if _is_statedict(raw_sd):
            model.load_state_dict(_strip_module(raw_sd), strict=True)
        else:
            model = raw_obj  # full model pickle (ít gặp)
        print("Loaded FP32 model (vits).")

    # ---------- Case 2: _q ----------
    elif 'depth_anything_v2_vits_q.pth' in name:
        if isinstance(raw_obj, torch.nn.Module):
            model = raw_obj
            print("Loaded full quantized model object (_q).")
        elif _is_statedict(raw_sd):
            sd = _strip_module(raw_sd)
            if _has_packed(sd):
                # tạo kiến trúc quantized rồi load
                float_m = DepthAnythingV2(**cfg)
                qmodel  = quantize_dynamic(float_m, {torch.nn.Linear}, dtype=torch.qint8)
                try:
                    qmodel.load_state_dict(sd, strict=False)
                    model = qmodel
                    print("Loaded INT8-dynamic state_dict into quantized arch (_q).")
                except Exception as e:
                    # Fallback: tạo INT8 lại từ FP32 gốc
                    if BASE_FP32.exists():
                        print(f"[warn] Failed to load INT8 state_dict (_q): {e}\n→ Rebuild INT8 from FP32 base.")
                        base = _load_fp32_model_from(BASE_FP32).eval()
                        model = quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
                    else:
                        raise
            else:
                model = DepthAnythingV2(**cfg)
                model.load_state_dict(sd, strict=True)
                print("Loaded FP32 state_dict under _q name.")
        else:
            raise ValueError("Unsupported checkpoint format for _q file.")

    # ---------- Case 3 ----------
    elif 'depth_anything_v2_vits_qt2e.pth' in name:
        if isinstance(raw_obj, torch.nn.Module):
            model = raw_obj
            print("Loaded full quantized model object (qv1).")
        elif _is_statedict(raw_sd):
            sd = _strip_module(raw_sd)
            if _has_packed(sd):
                float_m = DepthAnythingV2(**cfg)
                qmodel  = quantize_dynamic(float_m, {torch.nn.Linear}, dtype=torch.qint8)
                try:
                    qmodel.load_state_dict(sd, strict=False)
                    model = qmodel
                    print("Loaded INT8-dynamic state_dict into quantized arch (qv1).")
                except Exception as e:
                    if BASE_FP32.exists():
                        print(f"[warn] Failed to load INT8 state_dict (qv1): {e}\n→ Rebuild INT8 from FP32 base.")
                        base = _load_fp32_model_from(BASE_FP32).eval()
                        model = quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
                    else:
                        raise
            else:
                model = DepthAnythingV2(**cfg)
                model.load_state_dict(sd, strict=True)
                print("Loaded FP32 state_dict (qv1).")
        else:
            raise ValueError("Unsupported checkpoint format for qv1 file.")

    # ---------- Case 4: PRUNED ----------
    elif 'depth_anything_v2_vits_pruned.pth' in name:
        # Guess meta path next to the checkpoint, fallback to a standard name.
        meta_guess = CKPT.with_suffix('.meta.json')
        meta_path = meta_guess if meta_guess.exists() else CKPT.with_name('depth_anything_v2_vits_pruned.meta.json')
        assert meta_path.exists(), f"Missing metadata JSON for pruned model: {meta_path}"

        # Load meta (drop blocks + retapped indices)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        drop = set(meta.get("drop_blocks", []))
        taps = meta.get("retapped_vits", [2, 5, 8, 11])  # DA-V2 default for vits if absent

        # Build fresh model, then physically compact ViT blocks and retap
        model = DepthAnythingV2(**cfg)
        vit = model.pretrained
        total = len(vit.blocks)
        kept_ids = [i for i in range(total) if i not in drop]
        vit.blocks = torch.nn.ModuleList([vit.blocks[k] for k in kept_ids])
        if hasattr(vit, "n_blocks"):
            vit.n_blocks = len(vit.blocks)
        if hasattr(model, "intermediate_layer_idx"):
            model.intermediate_layer_idx['vits'] = taps

        # Load pruned weights
        sd = torch.load(str(CKPT), map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        sd = _strip_module(sd)
        model.load_state_dict(sd, strict=True)
        print(f"Loaded PRUNED model (kept {len(kept_ids)}/{total} ViT blocks).")

    # ---------- Fallback generic ----------
    else:
        if isinstance(raw_obj, torch.nn.Module):
            model = raw_obj
            print("Loaded full model object (generic).")
        elif _is_statedict(raw_sd):
            model = DepthAnythingV2(**cfg)
            model.load_state_dict(_strip_module(raw_sd), strict=False)
            print("Loaded generic state_dict into DepthAnythingV2.")
        else:
            raise ValueError(f"Unknown checkpoint layout: {type(raw_obj)}")

except Exception as e:
    # luôn có model chạy được bằng cách build từ FP32 rồi dynamic quant
    if BASE_FP32.exists():
        print(f"[FATAL LOAD] {e}\n→ Final fallback: build INT8-dynamic from FP32 base.")
        base = _load_fp32_model_from(BASE_FP32).eval()
        model = quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
    else:
        raise

model = model.to(DEVICE).eval()

def read_gt_meters(p: Path):
    """KITTI depth format: uint16 PNG where meters = value / 256.0, 0 = invalid."""
    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(p)
    return im.astype(np.float32) / 256.0

# ======================= PRUNED MODEL EVAL (ADDED) =======================
# Evaluate a PRUNED DA-V2 model (structural layer-drop + retap) on KITTI
# - Rebuilds pruned arch from *.meta.json (drop_blocks + retapped indices)
# - Aligns predictions to GT (modes: 'inv_affine' matches your exporter; also 'median'/'scale'/'scale_shift')
# - Reports KITTI metrics: AbsRel, SqRel, RMSE, RMSE_log, δ<1.25/1.25^2/1.25^3, SIlog
import json, math

def _rebuild_pruned_model_from_meta(pruned_ckpt: Path, meta_json: Path, cfg, device='cpu'):
    """
    Rebuild the DA-V2 pruned architecture (drop blocks + retap) and load weights.
    Expects meta generated by your pruning script.
    """
    from depth_anything_v2.dpt import DepthAnythingV2
    m = DepthAnythingV2(**cfg)

    with open(meta_json, "r") as f:
        meta = json.load(f)
    drop = set(meta.get("drop_blocks", []))
    taps = meta.get("retapped_vits", [1,3,5,7])  # fallback

    # apply structural changes
    vit = m.pretrained
    kept_ids = [i for i in range(len(vit.blocks)) if i not in drop]
    vit.blocks = torch.nn.ModuleList([vit.blocks[k] for k in kept_ids])
    if hasattr(vit, "n_blocks"):
        vit.n_blocks = len(vit.blocks)
    if hasattr(m, "intermediate_layer_idx"):
        m.intermediate_layer_idx['vits'] = taps

    # load weights
    state = torch.load(str(pruned_ckpt), map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # strip "module."
    state = { (k[7:] if isinstance(k,str) and k.startswith('module.') else k): v for k,v in state.items() }
    m.load_state_dict(state, strict=True)
    return m.to(device).eval()

def _silog(pred, gt, mask, eps=1e-8):
    # scale-invariant log error (as % like common reports)
    d = (np.log(np.maximum(pred[mask], eps)) - np.log(np.maximum(gt[mask], eps)))
    return float(100.0 * math.sqrt(np.mean(d**2) - (np.mean(d))**2))

def _kitti_metrics(pred_m, gt_m, valid_mask, min_m=1e-3, max_m=80.0):
    p = np.clip(pred_m, min_m, max_m)
    g = np.clip(gt_m, min_m, max_m)
    m = valid_mask

    diff = p[m] - g[m]
    abs_rel = float(np.mean(np.abs(diff) / g[m]))
    sq_rel  = float(np.mean((diff**2) / g[m]))
    rmse    = float(np.sqrt(np.mean(diff**2)))
    rmse_l  = float(np.sqrt(np.mean((np.log(np.maximum(p[m],1e-8)) - np.log(np.maximum(g[m],1e-8)))**2)))

    ratio = np.maximum(p[m] / g[m], g[m] / p[m])
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25**2))
    d3 = float(np.mean(ratio < 1.25**3))

    silog = _silog(p, g, m)
    return dict(abs_rel=abs_rel, sq_rel=sq_rel, rmse=rmse, rmse_log=rmse_l, delta1=d1, delta2=d2, delta3=d3, silog=silog)

def _align_prediction(y_rel, gt_m, mode='inv_affine'):
    """
    Align relative prediction to GT for fair monocular evaluation.
    modes:
      - 'median'      : single-scale via median ratio (classic Eigen eval)
      - 'scale'       : least-squares single scale in depth
      - 'scale_shift' : least-squares scale+shift in depth
      - 'inv_affine'  : least-squares affine in inverse depth (matches your exporter)
    """
    eps = 1e-6
    valid = gt_m > 0
    if valid.sum() < 10:
        return np.zeros_like(gt_m), valid

    if mode == 'median':
        s = np.median(gt_m[valid]) / np.maximum(np.median(y_rel[valid]), eps)
        pred_m = s * y_rel
    elif mode == 'scale':
        # min || a*y - gt ||^2
        a = np.dot(y_rel[valid], gt_m[valid]) / (np.dot(y_rel[valid], y_rel[valid]) + eps)
        pred_m = a * y_rel
    elif mode == 'scale_shift':
        # min || a*y + b - gt ||^2
        A = np.stack([y_rel[valid], np.ones_like(y_rel[valid])], axis=1)
        a, b = np.linalg.lstsq(A, gt_m[valid], rcond=None)[0]
        pred_m = a * y_rel + b
    elif mode == 'inv_affine':
        # min || a*y + b - (1/gt) ||^2, then invert back to meters
        A = np.stack([y_rel[valid], np.ones_like(y_rel[valid])], axis=1)
        target = 1.0 / (gt_m[valid] + eps)
        a, b = np.linalg.lstsq(A, target, rcond=None)[0]
        pred_m = 1.0 / np.maximum(a * y_rel + b, eps)
    else:
        raise ValueError(f"Unknown align mode: {mode}")
    return pred_m.astype(np.float32), valid

# Choose files
gts_all = sorted(GT_DIR.glob("*.png"))
gts = gts_all if N is None else gts_all[:N]
assert gts, f"No GT PNGs found in {GT_DIR}"

print(f"Device: {DEVICE.upper()} | Exporting {len(gts)} images → {OUT_DIR}")

# -------- timing start --------
t0 = time.perf_counter()

for i, gt_path in enumerate(gts, 1):
    # map GT filename to the corresponding RGB filename
    img_name = gt_path.name.replace("_groundtruth_depth_", "_image_")
    img_path = IMG_DIR / img_name
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Missing RGB for {gt_path.name}\nExpected: {img_path}")

    # predict relative depth (HxW float32); DA-V2 API returns a NumPy map
    # (model.infer_image expects BGR image, like OpenCV)
    with torch.inference_mode():
        pred_rel = model.infer_image(img_bgr).astype(np.float32)

    # load GT in meters (uint16/256.0)
    gt_m = read_gt_meters(gt_path)

    # resize prediction to GT shape if needed
    if pred_rel.shape != gt_m.shape:
        pred_rel = cv2.resize(pred_rel, (gt_m.shape[1], gt_m.shape[0]), interpolation=cv2.INTER_LINEAR)

    # affine fit in inverse depth: a*y + b ≈ 1/gt on valid pixels
    eps = 1e-6
    valid = gt_m > 0
    if valid.sum() == 0:
        # If no valid GT (shouldn't happen on this split), write zeros
        pred_u16 = np.zeros_like(gt_m, dtype=np.uint16)
    else:
        y = pred_rel
        A = np.stack([y[valid], np.ones_like(y[valid])], axis=1)
        bvec = 1.0 / (gt_m[valid] + eps)
        a, b = np.linalg.lstsq(A, bvec, rcond=None)[0]

        # invert back to meters, clamp to KITTI range
        pred_m = 1.0 / np.maximum(a * y + b, eps)
        pred_m = np.clip(pred_m, 1e-3, 80.0)

        # save KITTI uint16 PNG: value = round(meters * 256.0), 0 = invalid
        pred_u16 = np.clip(np.rint(pred_m * 256.0), 0, 65535).astype(np.uint16)

    ok = cv2.imwrite(str(OUT_DIR / gt_path.name), pred_u16)
    if not ok:
        raise RuntimeError(f"Failed to write: {OUT_DIR/gt_path.name}")

    if i % 25 == 0 or i == len(gts):
        print(f"{i}/{len(gts)} saved")

# -------- timing end --------
if DEVICE == 'cuda':
    torch.cuda.synchronize()  # ensure all GPU work is finished before stopping the clock
elapsed = time.perf_counter() - t0
imgs = len(gts)
sec_per_img = elapsed / max(imgs, 1)
ips = imgs / elapsed if elapsed > 0 else float('inf')

print("Done →", OUT_DIR)
print(f"Total time: {elapsed:.2f} s | Avg: {sec_per_img:.3f} s/img | Throughput: {ips:.2f} img/s")
