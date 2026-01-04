# RULE
    ROI% is hyperparameter
    ablation (1 pixel, 10%, 20%, 30%, 100%) in KITTI GT depth then report metric


# STANDARD
    Ground-truth KITTI depth (C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root)
        [text](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
    Sanity check: range 5-80m


# NOTE
GT depth KITTI không đầy đủ 100% pixel (pixel không có GT sẽ = 0)
Depth PNG scale: nhiều loader KITTI depth convert uint16 / 256.0 để ra mét (và 0 là invalid)


# CITATION
@inproceedings{Uhrig2017THREEDV,
  author = {Jonas Uhrig and Nick Schneider and Lukas Schneider and Uwe Franke and Thomas Brox and Andreas Geiger},
  title = {Sparsity Invariant CNNs},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2017}
}
