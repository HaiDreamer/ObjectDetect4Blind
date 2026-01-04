# Preparation

**Note**
    Fix input shapes and avoid Python-only layers or dynamic control flow before export.
    Plan preprocessing & postprocessing on-device (resize/normalize, NMS)

1. Pick backend: ONNX Runtime Mobile (Android)
2. A trained FP32 model + export path
    PyTorch→ONNX/TensorRT
3. A small, representative calibration dataset (for PTQ / INT8)
    Use raw images preprocessed exactly like inference (same resize/normalize/letterbox, same input dtype/shape). 
    Size: a few hundred samples is typically enough; OpenVINO often uses ~300 samples.
    Have a small “representative” calibration set (200–1000 images typical)
4. Tools
    ONNX Runtime: PTQ with calibrators (per-channel/symmetric options).
    PyTorch: torchao (static/PTQ, QAT tutorials)
5. Accuracy & speed evaluation setup (need dataset)
6. Right quantization settings (“knobs”)
    Defaults that usually help detectors: per-channel (weights) + per-tensor (activations), often symmetric for weights. These settings commonly preserve accuracy better.
7. (Optional) Training loop for QAT
    If PTQ hurts mAP too much, switch to Quantization-Aware Training (adds fake-quant nodes during training to recover accuracy)    
    
    
    