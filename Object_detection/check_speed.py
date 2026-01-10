import time, statistics
from ultralytics import YOLO

'''
TODO: check model speed of some img => speed
    need to check this code again

Path of model to check
    C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp32.onnx
    C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp16.onnx
    C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8dyn_mm.onnx        (NO NEED CHECK, it is useless)
    C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8_static_qdq.onnx

OUTPUT
    Loading C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp32.onnx for ONNX Runtime inference...
    Using ONNX Runtime CPUExecutionProvider
    Memory: 259 MB
    avg: 369.87 ms  p50: 369.30 ms  p95: 398.12 ms
    FPS (avg): 2.70
    Ultralytics speed breakdown (ms): {'preprocess': 1.9152999157086015, 'inference': 364.5574999973178, 'postprocess': 1.1073000496253371}

    Loading C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_fp16.onnx for ONNX Runtime inference...
    Using ONNX Runtime CPUExecutionProvider
    Memory: 130 MB
    avg: 365.63 ms  p50: 365.13 ms  p95: 385.69 ms
    FPS (avg): 2.73
    Ultralytics speed breakdown (ms): {'preprocess': 1.9799999427050352, 'inference': 370.6091999774799, 'postprocess': 1.2391000054776669}

    Loading C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8_static_qdq.onnx for ONNX Runtime inference...
    Using ONNX Runtime CPUExecutionProvider
    avg: 177.42 ms  p50: 177.11 ms  p95: 182.60 ms
    FPS (avg): 5.64
    Ultralytics speed breakdown (ms): {'preprocess': 1.8453000811859965, 'inference': 166.38760000932962, 'postprocess': 1.1771999998018146}
'''

MODEL = r"C:\Python\ObjectDetectRequireFile\put-in-obj-detect\models\best-lan2_int8_static_qdq.onnx"
IMAGE = r"C:\Python\ObjectDetect4Blind\assets\demo01.jpg"

model = YOLO(MODEL)

# warmup (for fair timing)
for _ in range(5):
    model.predict(IMAGE, imgsz=640, verbose=False)

times_ms = []
last_res = None
for _ in range(50):
    t0 = time.perf_counter()
    last_res = model.predict(IMAGE, imgsz=640, verbose=False)
    times_ms.append((time.perf_counter() - t0) * 1000)

times_ms_sorted = sorted(times_ms)
p50 = times_ms_sorted[len(times_ms_sorted)//2]
p95 = times_ms_sorted[int(len(times_ms_sorted)*0.95) - 1]

print(f"avg: {statistics.mean(times_ms):.2f} ms  p50: {p50:.2f} ms  p95: {p95:.2f} ms")
print(f"FPS (avg): {1000.0/statistics.mean(times_ms):.2f}")

# Ultralytics internal breakdown (if available in your version)
try:
    print("Ultralytics speed breakdown (ms):", last_res[0].speed)
except Exception:
    print("No results[0].speed in this version (but console output still shows Speed: ...).")
