r'''
USAGE
    check accuracy ~ ROI% is hyperparameter
    ablation (1 pixel(middle in bb/region), 10%, 20%, 30%, 100%) in KITTI GT depth then report metric
    
HOW TO DO
    Evaluate distance: same as main_distance.py
    NEED to eliminate object with low confidence score 

WARNING
    Ground truth kitti dataset not include all of our label in segmentation/object detection
    nhiều loader KITTI depth convert uint16 / 256.0 để ra mét (và 0 là invalid)    -> need to guard this, clean invalid pixel before evaluate distance

INPUT
    metric depth kitti ground truth dataset
    labelled bb from object detect and poligon from segmentation

OUTPUT
    Pick 100% ROI (for “reference”)
        label: "person","bicycle","car","motorcycle","bus","truck", "traffic light", "tree", "perdestrian_crossing_sign", "electric_pole"
    OBJECT DETECTION
        1 middle pixel based on mod ("center"/"bottom") 
            Timing: {'eval_seconds_total_after_depth_ready': 0.030954604735597968, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.03095460473559797, 'dist_seconds_total_bbox_to_roi_to_distance_eval': 0.021674592746421695, 'dist_objects_counted': 3282, 'avg_ms_per_bbox_distance_eval': 0.00660408066618577, 'timer': 'time.perf_counter'}
        1% aread of bb
            Timing: {'eval_seconds_total_after_depth_ready': 0.09054800122976303, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.09054800122976303, 'dist_seconds_total_bbox_to_roi_to_distance_eval': 0.08119609579443932, 'dist_objects_counted': 3282, 'avg_ms_per_bbox_distance_eval': 0.024739821997086936, 'timer': 'time.perf_counter'}
               N     MAE    MedAE  Bias_mean(gt-p)  Bias_median(gt-p)      MSE     RMSE  MeanRel%  MedRel% setting
            2864 0.79236 0.128906        -0.692195          -0.078125 5.036238 2.244156  5.325081 0.946382    roi1
        10% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.1469209019560367, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.1469209019560367, 'dist_seconds_total_bbox_to_roi_to_distance_eval': 0.13671449571847916, 'dist_objects_counted': 3282, 'avg_ms_per_bbox_distance_eval': 0.04165584878686141, 'timer': 'time.perf_counter'}
        20% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.15072330157272518, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.15072330157272518, 'dist_seconds_total_bbox_to_roi_to_distance_eval': 0.1399661006871611, 'dist_objects_counted': 3282, 'avg_ms_per_bbox_distance_eval': 0.04264658765605152, 'timer': 'time.perf_counter'}
        30% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.15316309547051787, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.15316309547051787, 'dist_seconds_total_bbox_to_roi_to_distance_eval': 0.14404860162176192, 'dist_objects_counted': 3282, 'avg_ms_per_bbox_distance_eval': 0.04389049409560083, 'timer': 'time.perf_counter'}
        100% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.2554758987389505, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.2554758987389505, 'dist_seconds_total_bbox_to_roi_to_distance_eval': 0.2465037046931684, 'dist_objects_counted': 3282, 'avg_ms_per_bbox_distance_eval': 0.07510777108262291, 'timer': 'time.perf_counter'}
    SEGMENTATION
        label: crosswalk, tree line, sidewalk, stairs

        1 middle pixel based on mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.12516650347970426, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.12516650347970426, 'mask_seconds_total_polygon_to_mask': 0.03055690019391477, 'mask_regions_counted': 202, 'avg_ms_per_region_polygon_to_mask': 0.15127178313819192, 'pick_seconds_total_band_to_percentile_to_pick': 0.09147660247981548, 'pick_regions_attempted': 202, 'avg_ms_per_region_band_to_percentile_to_pick': 0.45285446772185883, 'timer': 'time.perf_counter'}
        1% aread of polygon
            Timing: {'eval_seconds_total_after_depth_ready': 0.15514869755133986, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.15514869755133986, 'mask_seconds_total_polygon_to_mask': 0.03038700087927282, 'mask_regions_counted': 202, 'avg_ms_per_region_polygon_to_mask': 0.15043069742214266, 'pick_seconds_total_band_to_percentile_to_pick': 0.1213568989187479, 'pick_regions_attempted': 202, 'avg_ms_per_region_band_to_percentile_to_pick': 0.6007767273205341, 'timer': 'time.perf_counter'}
        10% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.13046710402704775, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.13046710402704775, 'mask_seconds_total_polygon_to_mask': 0.03360820189118385, 'mask_regions_counted': 202, 'avg_ms_per_region_polygon_to_mask': 0.16637723708506857, 'pick_seconds_total_band_to_percentile_to_pick': 0.09319890104234219, 'pick_regions_attempted': 202, 'avg_ms_per_region_band_to_percentile_to_pick': 0.4613806982294168, 'timer': 'time.perf_counter'}
        20% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.12618000246584415, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.12618000246584415, 'mask_seconds_total_polygon_to_mask': 0.02799810189753771, 'mask_regions_counted': 202, 'avg_ms_per_region_polygon_to_mask': 0.1386044648392956, 'pick_seconds_total_band_to_percentile_to_pick': 0.09538270183838904, 'pick_regions_attempted': 202, 'avg_ms_per_region_band_to_percentile_to_pick': 0.47219159325935167, 'timer': 'time.perf_counter'}
        30% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.1685799052938819, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.1685799052938819, 'mask_seconds_total_polygon_to_mask': 0.029900600435212255, 'mask_regions_counted': 202, 'avg_ms_per_region_polygon_to_mask': 0.14802277443174383, 'pick_seconds_total_band_to_percentile_to_pick': 0.13532109884545207, 'pick_regions_attempted': 202, 'avg_ms_per_region_band_to_percentile_to_pick': 0.6699064299279806, 'timer': 'time.perf_counter'}
        100% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.16105039743706584, 'eval_images_counted': 1000, 'avg_eval_ms_per_image_after_depth_ready': 0.16105039743706584, 'mask_seconds_total_polygon_to_mask': 0.030047500738874078, 'mask_regions_counted': 202, 'avg_ms_per_region_polygon_to_mask': 0.14875000365779248, 'pick_seconds_total_band_to_percentile_to_pick': 0.12767930352129042, 'pick_regions_attempted': 202, 'avg_ms_per_region_band_to_percentile_to_pick': 0.6320757600063882, 'timer': 'time.perf_counter'}

'''

    