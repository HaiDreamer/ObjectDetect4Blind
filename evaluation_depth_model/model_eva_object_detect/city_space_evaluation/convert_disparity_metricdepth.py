import numpy as np
import json
from PIL import Image
import os

EVAL_H, EVAL_W = 256, 512  # as per paper

def load_depth_gt(city, filename_base, split="val"):
    base = r"D:\ObjectDetection4Blind-pt2\CitySpace"
    disparity_path = fr"{base}\disparity\{split}\{city}\{filename_base}_disparity.png"
    camera_path    = fr"{base}\camera\{split}\{city}\{filename_base}_camera.json"

    disparity = np.array(Image.open(disparity_path)).astype(np.float32)

    with open(camera_path) as f:
        camera = json.load(f)

    baseline = camera["extrinsic"]["baseline"]
    focal    = camera["intrinsic"]["fx"]

    # Convert disparity to metric depth (meters)
    disparity[disparity > 0] = (disparity[disparity > 0] - 1) / 256.0
    depth_gt = np.zeros_like(disparity)
    mask = disparity > 0
    depth_gt[mask] = baseline * focal / disparity[mask]

    # Resize to 256x512 — use NEAREST to avoid interpolating invalid pixels
    depth_gt_resized = np.array(
        Image.fromarray(depth_gt).resize((EVAL_W, EVAL_H), Image.NEAREST)
    )

    return depth_gt_resized  # float32, meters, shape (256, 512)


def load_image(city, filename_base, split="val"):
    base = r"D:\ObjectDetection4Blind-pt2\CitySpace"
    img_path = fr"{base}\leftImg8bit\{split}\{city}\{filename_base}_leftImg8bit.png"

    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((EVAL_W, EVAL_H), Image.BILINEAR)  # bilinear for RGB

    return img_resized  # PIL Image, 256x512


def save_depth_as_kitti(depth_meters, city, filename_base, split="val"):
    out_dir = fr"D:\ObjectDetection4Blind-pt2\CitySpace\depth_pred\{split}\{city}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = fr"{out_dir}\{filename_base}_depth_pred.png"

    depth_encoded = (depth_meters * 256.0).clip(0, 65535).astype(np.uint16)
    Image.fromarray(depth_encoded).save(out_path)

    return out_path


def load_depth_pred(city, filename_base, split="val"):
    base = r"D:\ObjectDetection4Blind-pt2\CitySpace"
    path = fr"{base}\depth_pred\{split}\{city}\{filename_base}_depth_pred.png"
    raw = np.array(Image.open(path)).astype(np.float32)
    return raw / 256.0  # back to meters