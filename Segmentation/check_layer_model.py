from ultralytics import YOLO

'''
FOR: checking layer of model (for quantizing)

OUTPUT
    Ultralytics YOLOv8m-seg (the dump literally says YOLOv8m-seg summary).
    Size/compute: 191 layers, 27,241,964 parameters, ~104.7 GFLOPs (typically reported at the default input size used by the tool).

    How to read model.X...
        This is a Sequential model where:
            model.0 ... model.9 = backbone feature extractor (Conv + C2f blocks + SPPF)
            model.10 ... model.21 = neck (FPN/PAN) (Upsample + Concat + C2f + downsample Conv)
            model.22 = segmentation head (Segment)
        The most important part: model.22 (Segment head)


'''

MODEL_PATH = r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg.pt"

model = YOLO(MODEL_PATH)          # load .pt in Python :contentReference[oaicite:1]{index=1}

# 1) Structured summary (layers/params); detailed=True gives a more layer-level view :contentReference[oaicite:2]{index=2}
model.info(detailed=True, verbose=True)

# 2) Print the actual PyTorch module (full architecture repr) :contentReference[oaicite:3]{index=3}
print(model.model)

# 3) List modules (nice “layer-by-layer” listing)
for name, m in model.model.named_modules():
    if name:  # skip root
        print(f"{name:45s} {m.__class__.__name__}")
