# Main step
- Input: C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root\val_selection_cropped\image (1000 image in total)
- Object detection: use YOLO model (for more accuracy) yolov8m.pt model
    Detect what ? (form KITTI) "person","bicycle","car","cyclis" ~ "motorcycle + human","tram" ~ "bus","truck"
                  (from YOLO model) person, bicycle, car, motorcycle, bus, truck
    Final evaluation categories:
        Person
        Cyclist/Bicycle (optional but nice)
        Car (Car + Van)
        Truck
        Large vehicle (Tram ↔ Bus) (optional if you care about them)
        Everything else (tree, traffic light, crosswalk, pole…) is not labeled in KITTI object detection, so you can't do proper GT-based distance error for those with KITTI.
    Save coordinate of bounding box (1 image = 1 txt/json file, name = image name)
    
- Image segmentation: use ? model 


# Need to do
- Quantize metric depth model to mobile app 
- Improvement possibilities
    accuracy of distance (from camera to sidewalk != from feet to sidewalk)
    use the median instead of mean (less sensitive to background/occlusion), or average only a central region of the box (to avoid including background at the edges).
- Danger message ?
    Evaluate speed/ predict movement (of movable object ?)
- Accuracy?
    The only reliable way: calibrate on your own data (OR like what we calculate eval_kitti_subset?)
    Take 10-50 picture with true distance of object -> calculate wrongly 
    -> Check +- distance wrongly approximation

# Important note

- "Depth Anything V2" standard models → relative depth
    output a dense depth map where:
        larger values = farther, smaller = closer
        but the absolute scale is arbitrary (not calibrated to meters)
        The authors explicitly describe these as relative depth models.
        Can still get per-pixel depth ordering and do things like occlusion reasoning, but cannot directly say "this point is 3.2 m away" without some additional scaling or calibration.

"Depth Anything V2 – Metric VKITTI" models → metric depth in meters
    A fine-tuned versions of the same architecture on Virtual KITTI 2 with metric depth labels with:
        Input: 1 RGB image
        Output: per-pixel depth map
        Now the depth values are trained to match real metric distances (meters) for outdoor scenes.

So, which one ?
    If only need shape / relative geometry (which point is closer/farther), use the relative Depth Anything V2 model – it's simpler and very general.
    If need actual distance from the camera/user in meters (e.g., "how far is this car?"), use a Metric VKITTI model (for outdoor)

# Optimization
- Problems: Metric depth models are full-image networks, not per pixel estimate distance -> cannot depth only around boxes/borders
                and Metric depth models rely on global context to recover scale correctly 
- Segmentation
    Focus on the lower image band. Why?
        For sidewalk / ground, the relevant "nearest" danger is usually in the bottom part of the image (closer to the camera). Restrict the search to, say, the bottom 1/3 or 1/2 of the frame
    
- Object detection (improve accuracy?)
    Human/Traffic light/Tree/Electric Pole: middle of bb
    Car/Bicycle/Truck/Motorbike: lower bounding box 


# For android mobile app -> How this connects to your mobile app idea

Mobile app (Android / Java or Kotlin)
    Capture photo (or camera frame).
    Send JPEG/PNG and (ideally) the camera intrinsics to your server via HTTP.
Server (Python + DA-V2)
    Run Depth Anything V2 (preferably metric model).
    Get depth map.
    Use object detection / tapping coordinates from app to choose the object region.
    Compute median depth for that region → Z_metric.
    Optionally compute full 3D distance using intrinsics.
    Return distance in meters to the app as JSON.
App UI
    Display: per object distance 

# Algorithm for evaluate distance

OBJECT DETECTION -> bounding box of each object
- Labelled include: "person","bicycle","car","motorcycle","bus","truck", "traffic light", "tree", "perdestrian_crossing_sign", "electric_pole"
    traffic light for user knows when able to go cross over (but not have distinguish color yet)
- Metric depth model: depth_map_m[y, x] ≈ distance (in meters) from camera to the visible surface at pixel (x, y)
    -> each pixel has its own distance estimation
    object_distance ≈ 1/N*[sum of [​(x,y)∈box] for each ​depth_map_m[y,x]]

- YOLO detection have as lines in text file contains "class cx cy w h conf"
    For each line:
        transform normalized box center + size → integer pixel coordinates (x1, y1, x2, y2) in image space.
        clamp box to image bounds
        extract the patch of the depth map that corresponds to the bounding box
        remove invalid / zero depths
        average remaining depths
        -> in each bounding box, average per-pixel distance estimation for overall distance of an object

- Why it is effective ?
    The bounding box usually tightly covers one object.
    The depth inside that region is roughly similar (all pixels are on that object's surface), especially near the center.
    Averaging reduces the effect of noise; depth maps are often noisy per pixel.

- Limitation    
    Occlusions: If other objects are in front of or behind the target inside the same box, the average mixes different surfaces.
    Loose bounding boxes: If the box is too large, it includes background, so average depth might be too far
    Perspective: For large objects extending in depth (e.g., a long car at an angle), one depth number can't capture the entire shape

SEGMENTATION
- Labelled include: crosswalk, tree line, sidewalk, stairs
- Per-object distance algorithm = sample depth map in the object region (box or mask) → clean invalid pixels → aggregate (mean/median)
- Evaluate nearest sidewalk distance 

---

# Understanding dataset (validation & test)
- link: https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction
    (download link 3 and 4)

- dataset: each sample is a RGB image + a depth map
    that depth is a metric (real-world distance) in front of camera, NOT relative or normalized
    pair: (RGB image, ground-truth depth map) where each valid pixel stores distance in meters from LiDAR

- "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped\image"
    Cropped camera images from the KITTI raw driving sequences.
    Shortly understand: original image

- "C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped\velodyne_raw"  
    Type: Sparse depth maps (projected raw LiDAR returns) in KITTI-style PNGs or binary files, depending on the release.
    What they are:
        Direct projection of LiDAR points into the camera frame.
        Much sparser than groundtruth_depth (no accumulation / filtering).
    Use:
        For depth completion tasks (input = sparse depth + RGB → dense depth).
        Some methods reconstruct GT or evaluate directly using these sparse maps.
    Note: If only do monocular (RGB-only) depth prediction,  usually don't need this folder.

- C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped\intrinsics
    Type: text files or small matrices per image (e.g. *.txt), usually camera calibration / intrinsics.
    Content:
        Camera intrinsic parameters K
        ​Sometimes additional metadata (e.g. rectification, projection matrices).
    Use:
        Needed if you want to convert depth to 3D point clouds, back-project pixels to world coordinates, or do geometry-aware losses.
        Many monocular-depth evaluation scripts that only care about per-pixel meters won't use this directly.
    Note: If only do monocular (RGB-only) depth prediction,  usually don't need this folder.

- C:\Python\ObjectDetectRequireFile\put-in-depth-anything\kitti_root\val_selection_cropped\groundtruth_depth
    stored in uint16, 1 channel(encoded depth value)
    depth_meters = uint16_value / 256.0, 0 = invalid
    -> dark pixel: near or 0 depth, light pixel: farther depth

    **Compare ground truth pixel vs real life** 
    KITTI evaluation caps depth at 80 m (dmax=80) when computing metrics

    have noise and small error
        Errors along object boundaries (foreground/background mixing)
        larger errors at long distance
        problems in dynamic objects

    SAFE assume 
        Physically consistent "best effort"
            It is derived from real LiDAR sensor data and careful processing.
            For most pixels with GT, it's quite accurate (cm-level to few-cm errors typically, much smaller than deep model errors).
        Good enough as reference
            The entire community uses it as the reference to evaluate depth models (MAE, RMSE, SILog, etc.) on KITTI's benchmark.

    Conclusion
        KITTI depth GT from LiDAR + stereo, with careful processing
        Semi-dense, only around 16% of pixels in the original KITTI GT have valid depth, other = 0 -> masked out
        In the experimental range, we consider KITTI ground truth depth as an approximation of the true (metric) distance, with sensor errors small compared to model errors, although noise and outliers still exist in some regions.
        -> consider ground truth image as "true"

# Note
- ignore very tiny ground truth distance (< 0.5m) or very large distance (>80 m)
- Person and Person in bicycle/moto has slightly different distance (different way evaluating -> different result!)
WITH FRAC = 0.3
- Those 209 are almost certainly objects where earlier in your GT-distance script we couldn't compute a valid GT distance because:
    the depth patch inside the bbox had no valid KITTI GT pixels (all 0 or >80 m)
    "C:\Python\ObjectDetectRequireFile\put-in-metric-depth\obj_depth_vis\2011_09_26_drive_0036_sync_image_0000000299_image_02.png"
    the bbox was degenerate (zero area)
    box_distance_from_gt returned None (the bounding box was too small)
    Parts of vehicles near image border
- To reduce number "209"
    Use a larger frac (e.g. 0.5 instead of 0.3), so the sampled patch is bigger and more likely to contain some GT pixels.
    Fallback strategy: if valid.size == 0 in bottom mode, try center mode, or vice versa.
    Log a few examples: print a few objects where gt_distance_m is None and visualize their bboxes + GT depth to see the pattern.


# Report
**WITH FRAC = 0.3, ORIGINAL model vs ground truth distance error estimate**
- In ≈2.6% of detected objects, the KITTI LiDAR GT inside that bounding box had no valid depth points (or the bbox was too tiny),we excluded them from distance-error statistics.
GT distance (m):
  mean:   26.247
  median: 21.834
  min:    2.711
  max:    79.938
Pred distance (m):
  mean:   23.955
  median: 20.306
  min:    3.137
  max:    75.554
Signed error (gt - pred) (m):
  mean:   2.292
  median: 1.078
  min:    -46.277
  max:    52.157
Absolute error |gt - pred| (m):
  mean:   4.120
  median: 1.941
  min:    0.001
  max:    52.157
Min signed error: C:\Python\ObjectDetectRequireFile\put-in-metric-depth\obj_depth_vis\2011_09_26_drive_0095_sync_image_0000000227_image_03.png
Wrost: C:\Python\ObjectDetectRequireFile\put-in-metric-depth\obj_depth_vis\2011_09_26_drive_0036_sync_image_0000000242_image_03.png

**Per-class error**
Class                     N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
---------------------------------------------------------------------------------------------
Car                    6328      4.270      2.015      2.845      1.372      14.65       9.99
Cyclist/Bicycle         384      1.205      0.632     -0.292     -0.047      12.30       5.66
LargeVehicle            149      4.169      2.221      0.693      0.224      15.10      12.01
Person                  345      4.845      3.108     -2.053     -2.281      30.13      24.70
Truck                   504      3.948      2.167      0.772      0.698      15.01       9.56

**Per-distance range error**
========== PER-DISTANCE ERROR STATS ==========
GT bin (m)        N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-------------------------------------------------------------------------------------
[0, 10)        1347      1.160      0.669     -0.221      0.342      16.21       9.26
[10, 20)       2201      2.156      1.228      0.184      0.735      14.73       8.76
[20, 40)       2612      4.156      2.865      1.784      1.930      14.41      10.25
[40, 80)       1550      9.420      7.190      8.326      6.826      16.59      13.53

**Per-distance range/class error**
========== PER-CLASS + PER-DISTANCE ERROR STATS ==========

Class: Car
GT bin (m)        N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-------------------------------------------------------------------------------------
[0, 10)         980      1.037      0.655      0.036      0.471      14.19       8.95
[10, 20)       1699      2.147      1.255      0.556      0.963      14.62       8.85
[20, 40)       2238      3.951      2.739      2.191      2.070      13.58       9.94
[40, 80]       1411      9.578      7.321      8.588      7.077      16.69      13.89

Class: Cyclist/Bicycle
GT bin (m)        N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-------------------------------------------------------------------------------------
[0, 10)         155      1.197      0.484     -0.756     -0.188      18.02       6.86
[10, 20)        208      1.089      0.654     -0.076     -0.027       8.21       4.68
[20, 40)         21      2.403      1.818      0.991      1.452      10.69       7.56
[40, 80]          0        nan        nan        nan        nan        nan        nan

Class: LargeVehicle
GT bin (m)        N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-------------------------------------------------------------------------------------
[0, 10)          24      1.052      1.015      0.712      0.772      14.95      13.59
[10, 20)         32      2.076      1.493     -0.111      0.202      13.12      10.67
[20, 40)         62      4.754      3.719     -1.618     -1.866      17.12      13.42
[40, 80]         31      7.570      4.405      6.130      3.595      13.25      10.35

Class: Person
GT bin (m)        N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-------------------------------------------------------------------------------------
[0, 10)         116      2.387      2.001     -2.162     -1.962      33.50      26.00
[10, 20)        136      3.813      3.072     -3.212     -2.743      27.18      23.76
[20, 40)         76      8.669      7.152     -2.229     -3.488      30.32      24.79
[40, 80]         17     12.788     11.253      8.743     10.169      29.84      27.03

Class: Truck
GT bin (m)        N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-------------------------------------------------------------------------------------
[0, 10)          72      0.824      0.634      0.238      0.249      12.36      10.63
[10, 20)        126      2.276      0.890     -0.652      0.249      14.05       6.40
[20, 40)        215      4.688      3.228      0.026      1.204      17.02      11.34
[40, 80]         91      6.985      5.740      4.927      3.884      13.67      11.43

**Per area-range error**
Value A1 and A2
    about 1/3 of that class's objects have area < A1 → "small" for that class
    about 1/3 have area between A1 and A2 → "medium"
    about 1/3 have area ≥ A2 → "large"

Per-class area statistics (pixel^2):
  Cyclist/Bicycle     : min=629.0, max=61218.0, A1(33%)=3040.890380859375, A2(66%)=7094.60107421875
  Person              : min=232.0, max=41160.0, A1(33%)=1328.7200927734375, A2(66%)=4605.201171875
  Car                 : min=168.0, max=99902.0, A1(33%)=1260.0, A2(66%)=4724.1015625
  LargeVehicle        : min=672.0, max=169936.0, A1(33%)=3820.320068359375, A2(66%)=14118.48046875
  Truck               : min=700.0, max=170724.0, A1(33%)=2861.900146484375, A2(66%)=9018.78125

Skipped due to missing gt/pred: 209
Skipped due to invalid bbox (w<=0 or h<=0): 0

========== PER-CLASS + PER-SIZE ERROR STATS ==========

Class: Car
  Area stats: min=168.0, max=99902.0, A1(33%)=1260.0, A2(66%)=4724.1
SizeBin         N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-----------------------------------------------------------------------------------
small        2087      8.061      5.631      5.218      4.197      19.30      12.91
medium       2089      3.514      2.281      2.306      1.701      13.77       9.81
large        2152      1.327      0.854      1.065      0.789      10.99       8.63

Class: Cyclist/Bicycle
  Area stats: min=629.0, max=61218.0, A1(33%)=3040.9, A2(66%)=7094.6
SizeBin         N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-----------------------------------------------------------------------------------
small         127      1.044      0.533      0.293      0.074       6.23       3.92
medium        126      1.117      0.758     -0.121     -0.131       9.62       5.76
large         131      1.445      0.628     -1.024     -0.310      20.77       8.36

Class: LargeVehicle
  Area stats: min=672.0, max=169936.0, A1(33%)=3820.3, A2(66%)=14118.5
SizeBin         N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-----------------------------------------------------------------------------------
small          49      7.723      5.039      2.793      1.819      19.51      11.93
medium         49      3.102      2.532     -0.905     -1.434      12.40      12.53
large          51      1.778      1.211      0.211      0.534      13.47      11.59

Class: Person
  Area stats: min=232.0, max=41160.0, A1(33%)=1328.7, A2(66%)=4605.2
SizeBin         N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-----------------------------------------------------------------------------------
small         114      8.432      6.725     -1.656     -2.529      30.90      24.77
medium        114      3.847      3.209     -2.506     -2.657      29.11      23.76
large         117      2.323      2.014     -2.000     -1.913      30.37      25.56

Class: Truck
  Area stats: min=700.0, max=170724.0, A1(33%)=2861.9, A2(66%)=9018.8
SizeBin         N    mean|e|     med|e| mean(gt-p)  med(gt-p)   meanRel%    medRel%
-----------------------------------------------------------------------------------
small         166      6.552      4.881     -0.427      0.685      20.14      12.89
medium        166      4.269      3.003      2.131      1.634      15.82      10.11
large         172      1.124      0.796      0.616      0.500       9.28       7.01

Total objects used across all (class, size_bin): 7710

# To do
- Check with frac = 0.3 and frac = 0.5
- Reduce number "209"
- Table of wrongly accuracy
    Sanity check for json file
    Per-class error ("Person", "Cyclist/Bicycle", "Car", "Truck", "LargeVeh")
        Count object
        Mean abs error (MAE) with MAE = 1/N * (per object: |ground truth - predict|)
        Mean signed error (bias) with Bias = 1/N * (per object: (ground truth - predict))
            < 0: model tends to predict farther (overestimates)
            > 0: predict closer than reality (underestimates)
        Mean relative error: RelError = 1/N * (per object: |ground truth - predict| / ground truth)
    Per distance error 0-10, 10-20, 20-40, 40-80(m)
        Count objects
        Mean / median absolute error
        Mean relative error
    Error vith object size: (pixel area)
        Define: small area, medium area, large area ... ?

# References
    [1] KITTI base dataset + sensors (Velodyne HDL-64E)
    A. Geiger, P. Lenz, and R. Urtasun,
    "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite,"
    in Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2012, pp. 3354–3361. DOI: 10.1109/CVPR.2012.6248074.

    [2] KITTI depth completion / prediction dataset (LiDAR-based GT, 93k images)
    J. Uhrig, N. Schneider, L. Schneider, U. Franke, T. Brox, and A. Geiger,
    "Sparsity Invariant CNNs,"
    in Proc. Int. Conf. on 3D Vision (3DV), IEEE, 2017, pp. 11–20. DOI: 10.1109/3DV.2017.00012.

    [3] Official KITTI depth benchmark (GT description + metrics like SILog)
    A. Geiger et al.,
    "The KITTI Vision Benchmark Suite – Depth Completion / Depth Prediction Evaluation,"
    KITTI official website, accessed 2025.

    [4] LiDAR range accuracy (cm-level) for Velodyne HDL-64E S2
    C. Glennie and D. Lichti,
    "Static Calibration and Analysis of the Velodyne HDL-64E S2 for High Accuracy Mobile Scanning,"
    Remote Sensing, vol. 2, no. 6, pp. 1610–1624, 2010. DOI: 10.3390/rs2061610.

    [5] Standard depth metrics (RMSE, AbsRel, SILog) on KITTI
    D. Eigen, C. Puhrsch, and R. Fergus,
    "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network,"
    in Advances in Neural Information Processing Systems 27 (NeurIPS 2014), pp. 2366–2374. MIT Press, 2014.

    @inproceedings{Geiger2012KITTI,
    author    = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
    title     = {Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2012},
    pages     = {3354--3361},
    doi       = {10.1109/CVPR.2012.6248074}
    }

    @inproceedings{Uhrig2017SparsityInvariantCNNs,
    author    = {Jonas Uhrig and Nick Schneider and Lukas Schneider and Uwe Franke and Thomas Brox and Andreas Geiger},
    title     = {Sparsity Invariant {CNN}s},
    booktitle = {Proceedings of the International Conference on 3D Vision (3DV)},
    year      = {2017},
    pages     = {11--20},
    doi       = {10.1109/3DV.2017.00012}
    }

    @article{Glennie2010VelodyneCalibration,
    author    = {Craig Glennie and Derek Lichti},
    title     = {Static Calibration and Analysis of the Velodyne {HDL-64E} {S}2 for High Accuracy Mobile Scanning},
    journal   = {Remote Sensing},
    volume    = {2},
    number    = {6},
    pages     = {1610--1624},
    year      = {2010},
    doi       = {10.3390/rs2061610}
    }

    @inproceedings{Eigen2014Depth,
    author    = {David Eigen and Christian Puhrsch and Rob Fergus},
    title     = {Depth Map Prediction from a Single Image using a Multi-Scale Deep Network},
    booktitle = {Advances in Neural Information Processing Systems 27 (NeurIPS 2014)},
    pages     = {2366--2374},
    year      = {2014}
    }

    @misc{KITTIdepthEval,
    author       = {Andreas Geiger and Philipp Lenz and Christoph Stiller and Raquel Urtasun},
    title        = {{The KITTI Vision Benchmark Suite -- Depth Completion and Depth Prediction Evaluation}},
    howpublished = {\url{https://www.cvlibs.net/datasets/kitti/eval_depth_all.php}},
    note         = {Accessed: 2025-11-19},
    year         = {2017}
    }