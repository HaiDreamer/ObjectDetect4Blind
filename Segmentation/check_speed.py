import time, statistics
from ultralytics import YOLO
import torch

"""
TODO: estimate for some img => avg speed
      NEED to check this code again

FOR      
    Checking speed with onnx model (onnx model is fine for mobile app)

MODELS input:
    C:\\Python\\ObjectDetectRequireFile\\put-in-segment\\models\\best_seg.pt
    C:\\Python\\ObjectDetectRequireFile\\put-in-segment\\models\\best_seg_fp32.onnx (can skip)
    C:\\Python\\ObjectDetectRequireFile\\put-in-segment\\models\\best_seg_fp16.onnx
    C:\\Python\\ObjectDetectRequireFile\\put-in-segment\\models\\best_seg_int8_static_qdq.onnx  

OUTPUT:
    avg: ... ms  p50: ... ms  p95: ... ms
    FPS (avg): ...
    Ultralytics speed breakdown (ms): {'preprocess': ..., 'inference': ..., 'postprocess': ...}

    Loading C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp32.onnx for ONNX Runtime inference...
    Using ONNX Runtime CPUExecutionProvider
    avg: 545.61 ms  p50: 547.12 ms  p95: 561.28 ms
    FPS (avg): 1.83
    Ultralytics speed breakdown (ms): {'preprocess': 1.437499886378646, 'inference': 174.897100077942, 'postprocess': 353.1311999540776}

    Loading C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp16.onnx for ONNX Runtime inference...
    Using ONNX Runtime CPUExecutionProvider
    avg: 811.96 ms  p50: 811.53 ms  p95: 895.79 ms
    FPS (avg): 1.79
    Ultralytics speed breakdown (ms): {'preprocess': 1.727099996060133, 'inference': 215.48339980654418, 'postprocess': 562.1941999997944}
    
    Loading C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_int8_static_qdq.onnx for ONNX Runtime inference...
    Using ONNX Runtime CPUExecutionProvider
    avg: 581.36 ms  p50: 517.97 ms  p95: 741.89 ms
    FPS (avg): 1.72
    Ultralytics speed breakdown (ms): {'preprocess': 1.8054000101983547, 'inference': 112.89449990727007, 'postprocess': 585.1584000047296}

"""

MODEL = r"C:\Python\ObjectDetectRequireFile\put-in-segment\models\best_seg_fp16.onnx"
IMAGE = r"C:\Python\ObjectDetectRequireFile\put-in-segment\GroupProject_Seg\images\val\ds1__ds1__5_jpg.rf.a5c2692d1064450c606737b0b475dce7.jpg"

IMGSZ = 640
WARMUP = 5
RUNS = 50

# Match your style: GPU if available else CPU.
# For ONNX on CPU, force DEVICE="cpu"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print(f"\nLoading {MODEL} ...")
print(f"DEVICE={DEVICE} | torch.cuda.is_available()={torch.cuda.is_available()}")

model = YOLO(MODEL)

# warmup (for fair timing)
for _ in range(WARMUP):
    model.predict(IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)

times_ms = []
last_res = None

for _ in range(RUNS):
    t0 = time.perf_counter()
    last_res = model.predict(IMAGE, imgsz=IMGSZ, device=DEVICE, verbose=False)
    times_ms.append((time.perf_counter() - t0) * 1000)
    
#avg: mean latency over all runs
#p50: median latency (typical case)
#p95: “slow tail” latency (worst spikes)
times_ms_sorted = sorted(times_ms)
p50 = times_ms_sorted[len(times_ms_sorted) // 2]
p95 = times_ms_sorted[int(len(times_ms_sorted) * 0.95) - 1]

avg_ms = statistics.mean(times_ms)
print(f"avg: {avg_ms:.2f} ms  p50: {p50:.2f} ms  p95: {p95:.2f} ms")
print(f"FPS (avg): {1000.0 / avg_ms:.2f}")

# Ultralytics internal breakdown (if available)
try:
    print("Ultralytics speed breakdown (ms):", last_res[0].speed)
except Exception:
    print("No results[0].speed in this version (but console output still shows Speed: ...).")
