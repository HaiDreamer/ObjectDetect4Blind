from pathlib import Path

PRED_DEPTH_DIR = Path(r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_pruned1layer_torch_cpu")

print("drive_0047 files:", len(list(PRED_DEPTH_DIR.rglob("*2011_10_03_drive_0047_sync*"))))
print("frame 0785 cam 02 files:", list(PRED_DEPTH_DIR.rglob("*0000000785*image_02*"))[:10])
print("frame 0791 cam 03 files:", list(PRED_DEPTH_DIR.rglob("*0000000791*image_03*"))[:10])
