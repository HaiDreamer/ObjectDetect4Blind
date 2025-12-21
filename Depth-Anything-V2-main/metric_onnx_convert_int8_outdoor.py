import os
from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

"""
Convert DepthAnythingV2 metric-depth ONNX (FP32) -> INT8 (dynamic quantization).

INPUT:
    depth_anything_v2_metric_vkitti_vits_fp32.onnx  (float32)

OUTPUT:
    depth_anything_v2_metric_vkitti_vits_int8.onnx  (int8 weights)

Bug (has solved)
    hit ORT_NOT_IMPLEMENTED and explicitly state onnxruntime does not support ConvInteger, so dynamic quantization fails for CNNs unless you exclude conv nodes.
    
"""

# -------------------
# Paths
# -------------------
ROOT = Path(__file__).resolve().parent

CKPT_DIR = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints"

MODEL_FP32 = os.path.join(
    CKPT_DIR, "depth_anything_v2_metric_vkitti_vits_fp32.onnx"
)
MODEL_INT8 = os.path.join(
    CKPT_DIR, "depth_anything_v2_metric_vkitti_vits_int8.onnx"
)


def quantize_to_int8():
    assert os.path.isfile(MODEL_FP32), f"FP32 ONNX not found: {MODEL_FP32}"
    print("[INT8] Quantizing FP32 ONNX -> INT8 (dynamic quantization)...")
    print("       input :", MODEL_FP32)
    print("       output:", MODEL_INT8)

    m = onnx.load(MODEL_FP32)
    nodes = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
    print("Quantizing nodes:", len(nodes))
    
    quantize_dynamic(
        model_input=MODEL_FP32,
        model_output=MODEL_INT8,
        weight_type=QuantType.QInt8,  # or QuantType.QUInt8
        nodes_to_quantize=nodes,
    )

    # Optional: sanity check the resulting model
    m = onnx.load(MODEL_INT8)
    onnx.checker.check_model(m)

    print("[INT8] Saved quantized ONNX model:", MODEL_INT8)


if __name__ == "__main__":
    quantize_to_int8()
