# Labels (What we have for detection)
OBJECT DETECTION: person, bicycle, car, motorcycle, bus, truck, traffic light, tree, electric_pole, pedestrian_crossing_sign
SEGMENTATION: cross walk, tree line, sidewalk, stairs

# Normally
Walking speed: 1.2-1.4m/s for normal people, blind people tend to move slower and stand for a longer time  
    Source: Bohannon RW, Andrews AW. “Normal walking speed: a descriptive meta-analysis.” Physiotherapy, 2011;97(3):182–189
            Bohannon RW. “Comfortable and maximum walking speed of adults aged 20–79 years: reference values and determinants.” Age and Ageing, 1997;26(1):15–19.
            Middleton A, Fritz SL, Lusardi MM. “Walking speed: the functional vital sign.” Journal of Geriatric Physical Therapy, 2015;38(1):3–10.

Reaction time: 
    Average human reaction time to a visual stimulus is around 0.25 s, and to an auditory stimulus around 0.17 s
        -> If a user has about 1–2 seconds to respond to an obstacle, this is physiologically sufficient time to perceive the information, process it, and begin adjusting their path
    Source:
        Backyard Brains. (n.d.). The science of your reaction time.
        Biology Insights. (n.d.). How fast is a human’s reaction time?
        ScienceOxygen. (n.d.). Do we respond faster to visual or auditory stimuli?

Cane system:
    White cane(normal cane)
        detect range ~ 1-1.5m
        Source: WHO, 2024; Leader Dogs, 2024; White cane article
            “White cane” – Wikipedia.
            World Health Organization. “Assistive product specification for procurement: White canes.”
    Smart cane
        Detect range: 1.5-2m
        Source: [Smart Cane for Visually Impaired…, 2022; Low-Cost Smart Cane…, 2024]
            Smart Cane for Visually Impaired with Obstacle, Water Detection and GPS
            Low-Cost Smart Cane for Visually Impaired People with Pathway Surface Transition Points.

Model depth anything v2 metric kitti outdoor (1e-3m - 80m) -> for monocular camera
    Source:
        “Metric Depth Estimation – Depth Anything V2.”
        Depth-Anything-V2-Metric-VKITTI-Large README, Hugging Face.

 
# “Best” warning / decision algorithm based on your code MAIN_distance.py
TODO: need teammate/teacher consider this!

Problem
    Given these objects + distances, which ones should I warn about, how strongly, and how often?

Define danger zone in the image
    spatial zones (e.g., left / center / right, near / mid / far)
        Distance range: (on my own design)
            VERY_NEAR:  0.4  – 1.5 m   (emergency / strong warning)             
                but user can sense it by cane ? hwo can i avoid it duplicate cane or apply this as the "last save" ?! but it can make user feel annoying because we have "NEAR"
            NEAR:       1.5  – 3.5 m   (high priority; main danger zone)
            MID:        3.5  – 7.0 m   (early warning)
            FAR:        7.0  – 15.0 m  (context only, low-priority info)
            >15 m:      ignore or map to "FAR_INFO"

        In camera
            left: left 1/3 image
            right: right 1/3 image
            center: the left image

Risk scoring (need to know why we choose these number!)
    type_weight: Per detection
        (no matter moving/non-moving)
            car, truck, bus: 1.0x
            motorcycle: 0.95x
            bicycle/human: 0.9x
        (non-movable object)
            electric pole, traffic light, tree, tree line, sidewalk, stairs, pedestrian_crossing_sign: 0.4x
    direction_weight: Center object are dangerous
        If center: 1.0x 
        Others: 0.7x            
    dist_weight: Closer -> more dangerous (need distance band)
        Near: 1.0x      # TTC ~ 1–3.5 s
        Mid: 0.5x       # TTC ~ 3.5–7 s
        Far: 0.2x       # TTC > 7 s
    size_weight: Larger object(in img) => more likely to closer to user, need to know how much pixel is big, small..
        area_ratio = bbox_area / (W * H)
            small  : area_ratio < 0.02      # < 2% diện tích ảnh
            medium : 0.02 ≤ area_ratio < 0.10  # 2–10%
            large  : area_ratio ≥ 0.10      # > 10%
        Danger-coefficient:
            small  = 0.7x
            medium = 1.0x
            large  = 1.3x

FORMULA: risk_score = type_weight * direction_weight * dist_weight * size_weight (higher value is more dangerous)

# Notification logic
    Per frame, pick only the top 1–2 risks above a threshold 
    Message: short, fast, effective(can be scale with danger value)
    Real systems must avoid repeating “Car ahead… Car ahead… Car ahead…” every frame
        combine detection with a simple tracker to maintain object IDs and only notify on events    

