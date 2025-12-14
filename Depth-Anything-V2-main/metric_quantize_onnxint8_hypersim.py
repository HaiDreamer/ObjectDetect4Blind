import os
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

"""
Convert DepthAnythingV2 metric-depth ONNX (FP32) -> INT8 (dynamic quantization).

INPUT:
    depth_anything_v2_metric_hypersim_vits_fp32.onnx  (float32)

OUTPUT:
    depth_anything_v2_metric_hypersim_vits_int8.onnx  (int8 weights)

WARNING: 
    WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
"""

# -------------------
# Paths
# -------------------
ROOT = Path(__file__).resolve().parent

CKPT_DIR = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints"

MODEL_FP32 = os.path.join(
    CKPT_DIR, "depth_anything_v2_metric_hypersim_vits_fp32.onnx"
)
MODEL_INT8 = os.path.join(
    CKPT_DIR, "depth_anything_v2_metric_hypersim_vits_int8.onnx"
)


def quantize_to_int8():
    assert os.path.isfile(MODEL_FP32), f"FP32 ONNX not found: {MODEL_FP32}"
    print("[INT8] Quantizing FP32 ONNX -> INT8 (dynamic quantization)...")
    print("       input :", MODEL_FP32)
    print("       output:", MODEL_INT8)

    # Dynamic quantization: quantizes weights to INT8; activations stay FP32.
    # This is the simplest way and works for many CNN models.
    # Docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
    quantize_dynamic(
        model_input=MODEL_FP32,
        model_output=MODEL_INT8,
        weight_type=QuantType.QInt8,  # or QuantType.QUInt8
        # You can also add:
        # op_types_to_quantize=["Conv", "MatMul"],
        # per_channel=True,
    )

    # Optional: sanity check the resulting model
    m = onnx.load(MODEL_INT8)
    onnx.checker.check_model(m)

    print("[INT8] Saved quantized ONNX model:", MODEL_INT8)


if __name__ == "__main__":
    quantize_to_int8()
