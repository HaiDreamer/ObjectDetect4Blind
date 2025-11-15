# Important note

- “Depth Anything V2” standard models → relative depth
    output a dense depth map where:
        larger values = farther, smaller = closer
        but the absolute scale is arbitrary (not calibrated to meters)
        The authors explicitly describe these as relative depth models.
        Can still get per-pixel depth ordering and do things like occlusion reasoning, but cannot directly say “this point is 3.2 m away” without some additional scaling or calibration.

“Depth Anything V2 – Metric VKITTI” models → metric depth in meters
    A fine-tuned versions of the same architecture on Virtual KITTI 2 with metric depth labels with:
        Input: 1 RGB image
        Output: per-pixel depth map
        Now the depth values are trained to match real metric distances (meters) for outdoor scenes.

So, which one ?
    If only need shape / relative geometry (which point is closer/farther), use the relative Depth Anything V2 model – it’s simpler and very general.
    If need actual distance from the camera/user in meters (e.g., “how far is this car?”), use a Metric VKITTI model (for outdoor)

# Need to do
- Quantize metric depth model to mobile app -> compare accuracy?
- How can i check +- distance wrongly approximate ? -> need to know how to calculate error of model (like relative model has done!)
- Improvement possibilities
    accuracy of distance (from camera to sidewalk != from feet to sidewalk)
    use the median instead of mean (less sensitive to background/occlusion), or average only a central region of the box (to avoid including background at the edges).
- Danger message ?
    Evaluate speed/ predict movement (of movable object ?)

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
    Display: “Distance to object: 2.35 m”.

# Algorithm for evaluate distance

OBJECT DETECTION -> bounding box of each object
- Labelled include: "person","bicycle","car","motorcycle","bus","truck","traffic light" "tree" "perdestrian_crossing" "electric_pole"
    traffic light for user knows when able to go cross over (but nto have distinguish color yet)
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
    The depth inside that region is roughly similar (all pixels are on that object’s surface), especially near the center.
    Averaging reduces the effect of noise; depth maps are often noisy per pixel.

- Limitation    
    Occlusions: If other objects are in front of or behind the target inside the same box, the average mixes different surfaces.
    Loose bounding boxes: If the box is too large, it includes background, so average depth might be too far
    Perspective: For large objects extending in depth (e.g., a long car at an angle), one depth number can’t capture the entire shape

SEGMENTATION
- Labelled include: crossword, tree line, sidewalk, stairs
- Per-object distance algorithm = sample depth map in the object region (box or mask) → clean invalid pixels → aggregate (mean/median)
- Evaluate nearest sidewalk distance 