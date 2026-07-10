import cv2
import subprocess

def _watch(name: str, proc: subprocess.Popen):
    rc = proc.wait()
    print(f"[{name}] finished with exit code {rc}")

def _ensure_depth_size(depth_bgr, H, W):
    if depth_bgr is None:
        return None
    if (depth_bgr.shape[0], depth_bgr.shape[1]) != (H, W):
        depth_bgr = cv2.resize(depth_bgr, (W, H), interpolation=cv2.INTER_NEAREST)
    return depth_bgr
