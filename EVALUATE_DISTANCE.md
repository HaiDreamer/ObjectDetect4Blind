# Important note
use metric depth anything v2 model (VKITTI outdoor) -> output is already meter

# Need to do
Quantize to mobile app -> compare accuracy

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