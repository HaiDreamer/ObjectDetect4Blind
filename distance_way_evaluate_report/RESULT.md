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
            Timing: {'eval_seconds_total_after_depth_ready': 0.03638590034097433, 'eval_images_counted': 1000, 'eval_objects_attempted': 3282, 'avg_eval_ms_per_image_after_depth_ready': 0.03638590034097433, 'avg_eval_ms_per_object_attempted': 0.011086502236738065, 'timer': 'time.perf_counter'}
        10% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.12437780061736703, 'eval_images_counted': 1000, 'eval_objects_attempted': 3282, 'avg_eval_ms_per_image_after_depth_ready': 0.12437780061736703, 'avg_eval_ms_per_object_attempted': 0.03789695326549879, 'timer': 'time.perf_counter'}
        20% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.12600520066916943, 'eval_images_counted': 1000, 'eval_objects_attempted': 3282, 'avg_eval_ms_per_image_after_depth_ready': 0.12600520066916943, 'avg_eval_ms_per_object_attempted': 0.03839280946653548, 'timer': 'time.perf_counter'}
        30% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.12977859168313444, 'eval_images_counted': 1000, 'eval_objects_attempted': 3282, 'avg_eval_ms_per_image_after_depth_ready': 0.12977859168313444, 'avg_eval_ms_per_object_attempted': 0.03954253250552542, 'timer': 'time.perf_counter'}
        100% area of bb mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.12977859168313444, 'eval_images_counted': 1000, 'eval_objects_attempted': 3282, 'avg_eval_ms_per_image_after_depth_ready': 0.12977859168313444, 'avg_eval_ms_per_object_attempted': 0.03954253250552542, 'timer': 'time.perf_counter'}
    SEGMENTATION
        label: crosswalk, tree line, sidewalk, stairs

        1 middle pixel based on mod ("center"/"bottom")

        10% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.6681950998026878, 'eval_images_counted': 1000, 'eval_regions_attempted': 145, 'avg_eval_ms_per_image_after_depth_ready': 0.6681950998026878, 'avg_eval_ms_per_region_attempted': 4.608242067604743, 'timer': 'time.perf_counter'}
        20% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 1.3298907007556409, 'eval_images_counted': 1000, 'eval_regions_attempted': 145, 'avg_eval_ms_per_image_after_depth_ready': 1.3298907007556409, 'avg_eval_ms_per_region_attempted': 9.171660005211315, 'timer': 'time.perf_counter'}
        30% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 0.6380813051946461, 'eval_images_counted': 1000, 'eval_regions_attempted': 145, 'avg_eval_ms_per_image_after_depth_ready': 0.6380813051946461, 'avg_eval_ms_per_region_attempted': 4.400560725480318, 'timer': 'time.perf_counter'}
        100% area of polygon mod ("center"/"bottom")
            Timing: {'eval_seconds_total_after_depth_ready': 1.361293802736327, 'eval_images_counted': 1000, 'eval_regions_attempted': 145, 'avg_eval_ms_per_image_after_depth_ready': 1.361293802736327, 'avg_eval_ms_per_region_attempted': 9.388233122319498, 'timer': 'time.perf_counter'}

'''

    