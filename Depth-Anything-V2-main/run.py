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

'''
Run locally:
    python app.py
Running:
    python run.py --encoder vits --precision int8 --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --outdir depth_vis --pred-only
    -> save only depth predictions
Running:
    python run.py --encoder vitl --precision int8 --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --outdir depth_vis
    -> save side-by-side comparison of input and depth prediction

For video:
    python run_video.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis
'''

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
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }

    # Load checkpoint (supports quantized or normal)

    # Paths for three cases
    CKPT = f'C:/Python/ObjectDetectRequireFile/put-in-depth-anything/checkpoints/depth_anything_v2_{args.encoder}.pth'
    BASE_FP32 = f'C:/Python/ObjectDetectRequireFile/put-in-depth-anything/checkpoints/depth_anything_v2_{args.encoder}_q.pth'

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
        elif name.endswith(f"depth_anything_v2_{args.encoder}_q.pth") or \
             name.endswith(f"depth_anything_v2_{args.encoder}_qv1.pth"):
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
