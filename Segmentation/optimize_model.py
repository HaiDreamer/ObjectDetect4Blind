from ultralytics import YOLO

m = YOLO(r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\segment_best.pt")

# quick speed/throughput baseline on your GPU
m.predict(
    source = r"C:\Python\ObjectDetect4Blind\assets\demo01.jpg",
    imgsz = 640,
    device = 'cpu',
    half = False,
    conf = 0.25,
    iou = 0.7,
    retina_masks = False
)
   
