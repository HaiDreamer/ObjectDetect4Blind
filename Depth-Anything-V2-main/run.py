import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2
from collections import OrderedDict
from torch.ao.quantization import quantize_dynamic  # dynamic INT8 (Linear)
import json
from pathlib import Path
import onnxruntime as ort  

'''
Run locally:
    python app.py
Run origin model:
    python run.py --encoder vits --precision int8 --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --outdir depth_vis --pred-only
    -> save only depth predictions
    OR
    python run.py --encoder vitl --precision int8 --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --outdir depth_vis
    -> save side-by-side comparison of input and depth prediction
Run prunned model
    python run.py --encoder vits --precision fp32 --use-pruned --pruned-ckpt "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned.pth" --pruned-meta "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned.meta.json" --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --outdir depth_vis
Run onnx model
    python run.py --onnx-ckpt "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_fp16.onnx" --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --input-size 518 --outdir depth_vis --pred-only
    
For video:
    python run_video.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis
'''

class ONNXDepthAnything:
    """
    Minimal adapter to mirror DepthAnythingV2.infer_image() using ONNX Runtime.

    This fixes static-shape ONNX exports that hard-code a ViT token sequence length
    (e.g., 127x127 patches + 1 class token = 16130). We detect/try the expected
    token grid and letterbox the input to (tokens_side*14, tokens_side*14) so all
    internal Reshape/Add ops see the baked sequence length.
    """

    def __init__(self, onnx_path: str, input_size: int = 518, providers=None):
        import onnxruntime as ort
        self.input_size = int(input_size)
        self.onnx_path = onnx_path
        self._tokens_side = None  # will cache detected/validated tokens per side
        self._tried_autodetect = False

        if providers is None:
            providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        avail = [p for p in providers if p in ort.get_available_providers()]
        if not avail:
            avail = ["CPUExecutionProvider"]

        self.sess = ort.InferenceSession(onnx_path, providers=avail)
        self.inp = self.sess.get_inputs()[0]
        self.out = self.sess.get_outputs()[0]
        self.input_name = self.inp.name
        self.output_name = self.out.name

    # no-op so you can call .to().eval() like a torch.nn.Module
    def to(self, device):
        self._device = device
        return self
    def eval(self):
        return self

    @staticmethod
    def _normalize_rgb01(img_rgb01: np.ndarray) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, None, :]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, None, :]
        return (img_rgb01 - mean) / std

    def _resize_letterbox_square_multiple14(self, img: np.ndarray, tokens_side: int):
        """
        Letterbox to a square target = tokens_side*14, keeping AR via padding.
        Returns (padded_img, (top,bottom,left,right), (inner_h,inner_w)).
        """
        target = 14 * int(tokens_side)
        h, w = img.shape[:2]
        scale = target / max(h, w)
        new_h = int(np.floor((h * scale) / 14.0) * 14)
        new_w = int(np.floor((w * scale) / 14.0) * 14)
        new_h = max(14, min(target, new_h))
        new_w = max(14, min(target, new_w))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        pad_h, pad_w = target - new_h, target - new_w
        top = pad_h // 2; bottom = pad_h - top
        left = pad_w // 2; right = pad_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)
        return padded, (top, bottom, left, right), (new_h, new_w)

    def _maybe_autodetect_tokens_side(self):
        """
        Try to read the expected sequence length from ONNX (positional embeddings),
        else leave None and we'll probe with a few common grids.
        """
        if self._tried_autodetect:
            return
        self._tried_autodetect = True
        try:
            import onnx, math
            m = onnx.load(self.onnx_path)
            seq_lens = []
            for init in m.graph.initializer:
                # look for [1, L, C] tensors (pos_embed etc.)
                if len(init.dims) == 3 and init.dims[0] == 1 and init.dims[1] > 1000:
                    seq_lens.append(int(init.dims[1]))
            if seq_lens:
                L = max(seq_lens)  # pick the largest candidate
                # Common ViT cases: L = (t*t) + 1 (cls), or (t*t) (+maybe +4 registers)
                for off in (0, 1, 4, 5):
                    t2 = L - off
                    t = int(round(math.sqrt(t2)))
                    if t * t == t2 and t % 1 == 0 and t >= 16:
                        self._tokens_side = t
                        break
        except Exception:
            pass  # best-effort; will fall back to probing

    def _choose_tokens_side_by_probe(self, bgr: np.ndarray):
        """
        If autodetect failed, try a few common token grids once and cache the one that runs.
        """
        for t in ([self._tokens_side] if self._tokens_side else []) + [127, 55, 37]:
            try:
                rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                padded, _, _ = self._resize_letterbox_square_multiple14(rgb01, t)
                x = np.transpose(self._normalize_rgb01(padded), (2, 0, 1))[None]
                # one dry run; if it doesn't throw, we lock this tokens_side
                _ = self.sess.run([self.output_name], {self.input_name: x})[0]
                self._tokens_side = t
                return
            except Exception:
                continue
        # If nothing worked, last resort: keep 37 (518px) to fail consistently with a clear error
        self._tokens_side = 37

    def infer_image(self, bgr: np.ndarray, input_size: int | None = None) -> np.ndarray:
        # ignore input_size; keep existing behavior
        h0, w0 = bgr.shape[:2]
        self._maybe_autodetect_tokens_side()
        if self._tokens_side is None:
            self._choose_tokens_side_by_probe(bgr)
        rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb01, (top, bottom, left, right), (inner_h, inner_w) = \
            self._resize_letterbox_square_multiple14(rgb01, tokens_side=self._tokens_side)
        chw = np.transpose(self._normalize_rgb01(rgb01), (2, 0, 1))[None]
        out = self.sess.run([self.output_name], {self.input_name: chw})[0]
        if out.ndim == 4: out = out[0]
        if out.ndim == 3: out = out[0] if out.shape[0] == 1 else out.mean(axis=0)
        elif out.ndim != 2: raise RuntimeError(f"Unexpected ONNX output shape: {out.shape}")
        out = out[top: top + inner_h, left: left + inner_w]
        if out.shape != (h0, w0):
            out = cv2.resize(out.astype(np.float32), (w0, h0), interpolation=cv2.INTER_LINEAR)
        return out.astype(np.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--precision', type=str, default='int8', choices=['fp32', 'int8'],
                        help='Choose checkpoint/flow: fp32 loads FP32; int8 loads *_q if present or quantizes from FP32.')
    parser.add_argument('--use-pruned', action='store_true',
                        help='load a structurally pruned checkpoint + meta')
    parser.add_argument('--pruned-ckpt', type=str, default='',
                        help='path to *_pruned.pth (weights saved after structural prune)')
    parser.add_argument('--pruned-meta', type=str, default='',
                        help='path to companion *.meta.json (contains drop_blocks & retapped indices)')
    # ONNX path 
    parser.add_argument('--onnx-ckpt', type=str, default='',
                        help='path to ONNX model (*.onnx); when set, runs with ONNX Runtime')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    # pick config for the chosen encoder
    enc_cfg = model_configs[args.encoder]

    # Load checkpoint (supports quantized or normal small depth model)
    # Paths for these cases model
    CKPT = f'C:/Python/ObjectDetectRequireFile/put-in-depth-anything/checkpoints/depth_anything_v2_{args.encoder}.pth'
    BASE_FP32 = f'C:/Python/ObjectDetectRequireFile/put-in-depth-anything/checkpoints/depth_anything_v2_{args.encoder}_fp16.onnx'
    cfg = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}

    def _is_state_dict(x) -> bool:
        return isinstance(x, (dict, OrderedDict))

    def _strip_module(sd: dict):
        # remove DataParallel 'module.' prefix
        return { (k[7:] if isinstance(k, str) and k.startswith('module.') else k): v for k, v in sd.items() }

    def _has_packed(sd: dict) -> bool:
        # dynamic-quant Linear thường có key '_packed_params'
        return any('_packed_params' in k for k in sd.keys())

    def _load_raw(path: str):
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, dict) and 'state_dict' in obj:
            return obj, obj['state_dict']
        return obj, obj

    def _load_fp32_model_from(path: str):
        m = DepthAnythingV2(**model_configs[args.encoder])
        sd = torch.load(path, map_location='cpu')
        if _is_state_dict(sd):
            m.load_state_dict(_strip_module(sd), strict=True)
            return m
        # rare: full model saved
        return sd if isinstance(sd, torch.nn.Module) else m

    depth_anything = None  # will hold the runnable model (PyTorch or ONNX)

    # === NEW: ONNX fast-path ===
    if args.onnx_ckpt:
        if not os.path.exists(args.onnx_ckpt):
            raise FileNotFoundError(f"ONNX checkpoint not found: {args.onnx_ckpt}")
        print(f"[ONNX] Loading model: {args.onnx_ckpt}")
        depth_anything = ONNXDepthAnything(args.onnx_ckpt, input_size=args.input_size)
        print(f"[ONNX] Providers: {depth_anything.sess.get_providers()}")

    # === PyTorch paths (only if not using ONNX) ===
    if depth_anything is None:
        # === NEW: choose checkpoint flow based on --precision
        if args.precision == 'int8':
            # Prefer *_q or *_qv1, else fall back to building INT8 from FP32
            if not os.path.exists(CKPT):
                qv1 = CKPT.replace("_q.pth", "_qv1.pth")
                if os.path.exists(qv1):
                    CKPT = qv1
                elif os.path.exists(BASE_FP32):
                    CKPT = BASE_FP32  # we'll quantize dynamically below
                else:
                    raise FileNotFoundError(f"INT8/FP32 checkpoints not found for encoder '{args.encoder}'")
        else:  # 'fp32'
            if not os.path.exists(BASE_FP32):
                # If only *_q exists, we can't "dequantize" it—fail clearly
                raise FileNotFoundError(
                    f"FP32 checkpoint not found for encoder '{args.encoder}'. Expected: {BASE_FP32}"
                )
            CKPT = BASE_FP32

        name = os.path.basename(CKPT).lower()
        raw_obj, raw_sd = _load_raw(CKPT)

        # ======================= PRUNED LOADER (ADDED) =======================
        def _rebuild_pruned_model_from_meta(pruned_ckpt: str,
                                            meta_json: str,
                                            cfg_local: dict,
                                            device: str = 'cpu'):
            """
            Rebuild Depth Anything V2 with structurally-pruned ViT blocks and retapped decoder,
            then load the pruned weights. Expects meta JSON produced by your prune script:
            {
              "drop_blocks": [ ... ],
              "retapped_vits": [ ... ]
            }
            """
            # fresh model skeleton for this encoder
            m = DepthAnythingV2(**cfg_local)

            # read pruning metadata
            with open(meta_json, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            drop = set(meta.get('drop_blocks', []))
            taps = meta.get('retapped_vits', [2, 5, 8, 11])  # safe fallback for vits

            # physically compact ViT encoder
            vit = m.pretrained
            total = len(vit.blocks)
            kept_ids = [i for i in range(total) if i not in drop]
            vit.blocks = torch.nn.ModuleList([vit.blocks[k] for k in kept_ids])
            if hasattr(vit, 'n_blocks'):
                vit.n_blocks = len(vit.blocks)

            # retap decoder indices (very important after compaction)
            if hasattr(m, 'intermediate_layer_idx'):
                # make sure 'vits' key exists for all ViT sizes in DA-V2
                m.intermediate_layer_idx['vits'] = taps

            # load pruned state dict
            sd_local = torch.load(pruned_ckpt, map_location='cpu')
            if isinstance(sd_local, dict) and 'state_dict' in sd_local:
                sd_local = sd_local['state_dict']
            sd_local = { (k[7:] if isinstance(k, str) and k.startswith('module.') else k): v for k, v in sd_local.items() }
            m.load_state_dict(sd_local, strict=True)

            print(f"[PRUNED] kept {len(kept_ids)}/{total} ViT blocks | taps={taps}")
            return m.to(device).eval()

        # ----- PRUNED fast-path (ADDED) -----
        loaded_via_pruned = False
        if args.use_pruned or name.endswith(f"depth_anything_v2_{args.encoder}_pruned.pth"):
            # resolve paths
            pruned_ckpt = args.pruned_ckpt if args.pruned_ckpt else CKPT
            if not os.path.exists(pruned_ckpt):
                raise FileNotFoundError(f"Pruned checkpoint not found: {pruned_ckpt}")

            # find companion meta.json (either provided or next to ckpt)
            if args.pruned_meta:
                meta_json = args.pruned_meta
            else:
                p = Path(pruned_ckpt)
                guess = p.with_suffix('.meta.json')
                alt   = p.with_name(f"depth_anything_v2_{args.encoder}_pruned.meta.json")
                meta_json = str(guess if guess.exists() else alt)
            if not os.path.exists(meta_json):
                raise FileNotFoundError(f"Metadata JSON for pruned model not found: {meta_json}")

            # rebuild pruned arch + load weights
            model = _rebuild_pruned_model_from_meta(pruned_ckpt, meta_json, enc_cfg, device=DEVICE)

            # optional: dynamic INT8 for Linear if user asked for int8
            if args.precision == 'int8':
                print("[PRUNED] Converting to INT8-dynamic (Linear only).")
                model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

            depth_anything = model.to(DEVICE).eval()
            loaded_via_pruned = True

        if not loaded_via_pruned:
            try:
                # -------- Case A: FP32 (depth_anything_v2_<enc>.pth) --------
                if name.endswith(f"depth_anything_v2_{args.encoder}.pth"):
                    if isinstance(raw_obj, torch.nn.Module):
                        model = raw_obj
                        print("Loaded full FP32 model object.")
                    elif _is_state_dict(raw_sd):
                        model = DepthAnythingV2(**model_configs[args.encoder])
                        model.load_state_dict(_strip_module(raw_sd), strict=True)
                        print("Loaded FP32 model from state_dict.")
                    else:
                        raise TypeError(f"Unexpected checkpoint type: {type(raw_obj)}")

                    # If user asked for int8 but we landed on FP32, quantize now
                    if args.precision == 'int8':
                        print("Converting FP32 → INT8-dynamic (Linear only).")
                        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

                # -------- Case B: INT8 quantized (_q or _qv1) --------
                elif name.endswith(f"depth_anything_v2_{args.encoder}_q.pth") or name.endswith(f"depth_anything_v2_{args.encoder}_qv1.pth"):
                    if args.precision == 'fp32':
                        # User explicitly requested FP32 but provided only INT8 file
                        raise RuntimeError("Requested --precision fp32, but selected checkpoint is INT8 (*.pth with _q/_qv1).")

                    if isinstance(raw_obj, torch.nn.Module):
                        model = raw_obj
                        print("Loaded full quantized model object.")
                    elif _is_state_dict(raw_sd):
                        sd = _strip_module(raw_sd)
                        if _has_packed(sd):
                            # Build quantized arch first, then load INT8 weights
                            float_m = DepthAnythingV2(**model_configs[args.encoder])
                            qmodel  = quantize_dynamic(float_m, {torch.nn.Linear}, dtype=torch.qint8)
                            try:
                                qmodel.load_state_dict(sd, strict=False)
                                model = qmodel
                                print("Loaded INT8-dynamic state_dict into quantized architecture.")
                            except Exception as e:
                                # Version-format mismatch: rebuild INT8 from FP32 base
                                if os.path.exists(BASE_FP32):
                                    print(f"[warn] Failed to load INT8 state_dict: {e}\n→ Rebuild INT8 from FP32 base.")
                                    base = _load_fp32_model_from(BASE_FP32).eval()
                                    model = quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
                                else:
                                    raise
                        else:
                            # It was actually FP32 weights under a *_q* name
                            model = DepthAnythingV2(**model_configs[args.encoder])
                            model.load_state_dict(sd, strict=True)
                            print("Loaded FP32 state_dict (named like *_q*).")
                            if args.precision == 'int8':
                                print("Converting FP32 → INT8-dynamic (Linear only).")
                                model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    else:
                        raise TypeError(f"Unexpected checkpoint type: {type(raw_obj)}")
                    
                # ---------- Case C: PRUNED ----------
                elif 'depth_anything_v2_vits_pruned.pth' in name:
                    # Guess meta path next to the checkpoint, fallback to a standard name.
                    meta_guess = Path(CKPT).with_suffix('.meta.json')
                    meta_path = meta_guess if meta_guess.exists() else Path(CKPT).with_name('depth_anything_v2_vits_pruned.meta.json')
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

                # -------- Fallback generic --------
                else:
                    if isinstance(raw_obj, torch.nn.Module):
                        model = raw_obj
                        print("Loaded full model object (generic).")
                    elif _is_state_dict(raw_sd):
                        model = DepthAnythingV2(**model_configs[args.encoder])
                        model.load_state_dict(_strip_module(raw_sd), strict=False)
                        print("Loaded generic state_dict into DepthAnythingV2.")
                    else:
                        raise TypeError(f"Unknown checkpoint layout: {type(raw_obj)}")

                    # Honor precision if we fell into a generic path
                    if args.precision == 'int8':
                        print("Converting FP32 → INT8-dynamic (Linear only).")
                        model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

            except Exception as e:
                # Final safety: always produce a runnable model
                if os.path.exists(BASE_FP32):
                    print(f"[FATAL LOAD] {e}\n→ Final fallback: build INT8-dynamic from FP32 base.")
                    base = _load_fp32_model_from(BASE_FP32).eval()
                    model = quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8)
                else:
                    raise

            depth_anything = model.to(DEVICE).eval()

    # === Load images ===
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # === Inference loop ===
    for k, filename in enumerate(filenames):
        print(f'Progress {k + 1}/{len(filenames)}: {filename}')
        raw_image = cv2.imread(filename)

        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')

        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            cv2.imwrite(output_path, combined_result)

    print(f"\n All results saved to: {args.outdir}")
