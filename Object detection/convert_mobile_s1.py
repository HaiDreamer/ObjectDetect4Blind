'''
Use python 3.11 version

Input: pretrained model (<filename>.pt)
Output: export model to .pte

Why?
    ExecuTorch on Android 
    my model from YOLO, which is implemented by Pytorch
    executorch need Python 3.10 - 3.12
'''

# from pathlib import Path
# from ultralytics import YOLO

# weights = Path(r"C:\Python\DeepLearningProject\Object detection\models\yolov8n.pt").resolve()

# m = YOLO(str(weights))
# exported = m.export(
#     format="onnx",
#     imgsz=640,
#     dynamic=False,
#     opset=12,
#     simplify=True,
# )

# print(f"ONNX saved to: {exported}")  # saved in ...\models\yolo

from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

models_dir = Path(r"C:\Python\DeepLearningProject\Object detection\models")
fp32 = models_dir / "yolov8n.onnx"
int8 = models_dir / "yolov8n_int8_dynamic.onnx"

quantize_dynamic(
    model_input=str(fp32),
    model_output=str(int8),
    weight_type=QuantType.QInt8,   # or QuantType.QUInt8 depending on the model
    per_channel=False,             # keep default; set True only if your ops support it
    reduce_range=False             # leave False unless you have a reason to enable
)

print("Saved:", int8)
