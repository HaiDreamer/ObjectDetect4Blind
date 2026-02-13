import onnx
import pathlib as Path

def info(path):
    m = onnx.load(path)
    print("\n==", path)
    print("ir_version:", m.ir_version)
    print("opset_import:", [(o.domain, o.version) for o in m.opset_import])

info(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best.onnx")
info(r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2.onnx")
