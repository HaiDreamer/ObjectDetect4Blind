# RULE
    ROI% is hyperparameter
    ablation (1 pixel(middle in bb/region), 10%, 20%, 30%, 100%) in KITTI GT depth then report metric
    Take value 10% min of these as standard
        Với cảnh báo va chạm, bạn thường muốn gần với khoảng cách nhỏ nhất hợp lý (gần nhất trên vật thể), nhưng tránh outlier.
        Dùng percentile thấp p10 trong ROI thay vì min

# STANDARD
    Ground-truth KITTI depth (C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root)
        [text](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
    Sanity check: range 5-80m
    Pick 100% ROI làm “reference”
    Use model for object detection and segmentation for bb, NO use model depth


# CITATION
@inproceedings{Uhrig2017THREEDV,
  author = {Jonas Uhrig and Nick Schneider and Lukas Schneider and Uwe Franke and Thomas Brox and Andreas Geiger},
  title = {Sparsity Invariant CNNs},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2017}
}
