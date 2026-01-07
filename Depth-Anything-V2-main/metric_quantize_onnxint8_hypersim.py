import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

CKPT_DIR = r"C:\Python\ObjectDetectRequireFile\put-in-metric-depth\checkpoints"
MODEL_FP32 = os.path.join(CKPT_DIR, "depth_anything_v2_metric_hypersim_vits_fp32.onnx")
MODEL_INT8 = os.path.join(CKPT_DIR, "depth_anything_v2_metric_hypersim_vits_int8.onnx")

def quantize_to_int8_matmul_only():
    assert os.path.isfile(MODEL_FP32), f"FP32 ONNX not found: {MODEL_FP32}"

    m = onnx.load(MODEL_FP32)
    nodes = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
    print("MatMul/Gemm nodes:", len(nodes))

    quantize_dynamic(
        model_input=MODEL_FP32,
        model_output=MODEL_INT8,
        weight_type=QuantType.QInt8,
        nodes_to_quantize=nodes,  # allow-list: ONLY these nodes are quantized
    )

    # sanity: ensure ConvInteger doesn't exist
    q = onnx.load(MODEL_INT8)
    ops = {n.op_type for n in q.graph.node}
    print("Has ConvInteger?", "ConvInteger" in ops)

if __name__ == "__main__":
    quantize_to_int8_matmul_only()
