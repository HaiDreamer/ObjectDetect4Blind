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

        10% area of bb mod ("center"/"bottom")

        20% area of bb mod ("center"/"bottom")

        30% area of bb mod ("center"/"bottom")

        100% area of bb mod ("center"/"bottom")

    SEGMENTATION
        label: crosswalk, tree line, sidewalk, stairs

        1 middle pixel based on mod ("center"/"bottom")

        10% area of polygon mod ("center"/"bottom")

        20% area of polygon mod ("center"/"bottom")

        30% area of polygon mod ("center"/"bottom")

        100% area of polygon mod ("center"/"bottom")

'''