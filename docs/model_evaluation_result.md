Table 1: Global performance and depth accuracy per-pixel of different quantized models
of the metric Depth Anything V2 ViT-S outdoor model

+----------------------------+-------------+----------+------+------+------+--------+---------+---------+---------+
| Model                      | Speed (s/img)| Memory   |  d1  |  d2  |  d3  | AbsRel |  SqRel  |  RMSE  | SILog  |
+----------------------------+-------------+----------+------+------+------+--------+---------+---------+---------+
| FP32 (original)            |   3.313     | 94.6 MB  | 0.854|0.969 |0.991 | 0.119  | 0.679   | 4.668  | 16.453 |
| FP16 ONNX                  |   0.442     | 48.5 MB  | 0.806|0.963 |0.991 | 0.134  | 0.758   | 4.801  | 16.877 |
| INT8 ONNX                  |   0.290     | 34.7 MB  | 0.797|0.943 |0.978 | 0.145  | 1.000   | 5.667  | 19.614 |
| Pruned 1-layer 	          |   3.096     | 87.8 MB  | 0.744|0.925 |0.983 | 0.174  | 1.125   | 5.881  | 22.299 |
+----------------------------+-------------+----------+------+------+------+--------+---------+---------+---------+

Done → C:\Python\ObjectDetectRequireFile\put-in-metric-depth\pred_metric_kitti_vkitti_vits_pruned1layer_torch_cpu
Total time: 3678.20 s | Avg: 3.678 s/img | Throughput: 0.27 img/s

original model
Avg speed: 2.969 s/img
- Memory: 94.6 MB 

Quantized .onnx float16 model
- 1.95x reduced memory Float32 -> float16 onnx(94.6MB -> 48.5MB)
- Avg: 0.394 s/img(7.5355x faster)

int8 model
Speed avg: 0.290 s/img
Memory: 34.7 MB

pruned 1 layer model
Avg: 3.678 s/img
Memory: 87.8 MB


| GT bin (m) | Model           | mean|e| (m) ↓ | med|e| (m) ↓ | meanRel% ↓ | medRel% ↓ | RMSE (m) ↓ |
| ---------- | --------------- | ------------: | -----------: | ---------: | --------: | ---------: |
| [0,10)     | FP32 (original) |         0.719 |        0.433 |      11.62 |      6.76 |      1.344 |
|            | FP16 ONNX       |         0.748 |        0.466 |      11.98 |      7.08 |      1.252 |
|            | INT8 ONNX       |         0.811 |        0.536 |      12.87 |      8.32 |      1.307 |
|            | Pruned 1-layer  |         1.992 |        1.571 |      32.93 |     25.10 |      2.613 |
| [10,20)    | FP32 (original) |         1.589 |        0.972 |      11.09 |      6.50 |      2.558 |
|            | FP16 ONNX       |         1.950 |        1.349 |      13.53 |      9.53 |      2.815 |
|            | INT8 ONNX       |         2.031 |        1.460 |      14.10 |     10.59 |      2.818 |
|            | Pruned 1-layer  |         3.047 |        2.650 |      21.82 |     18.78 |      3.711 |
| [20,40)    | FP32 (original) |         3.029 |        2.162 |      11.10 |      7.86 |      4.287 |
|            | FP16 ONNX       |         4.156 |        3.435 |      15.11 |     13.05 |      5.428 |
|            | INT8 ONNX       |         4.410 |        3.747 |      16.03 |     14.18 |      5.663 |
|            | Pruned 1-layer  |         2.957 |        2.249 |      11.06 |      8.23 |      3.992 |
| [40,80]    | FP32 (original) |         7.296 |        4.836 |      14.03 |      9.67 |     10.527 |
|            | FP16 ONNX       |         8.893 |        7.072 |      17.46 |     15.27 |     12.273 |
|            | INT8 ONNX       |         9.536 |        7.994 |      18.91 |     16.52 |     12.781 |
|            | Pruned 1-layer  |         9.790 |        7.461 |      18.32 |     15.90 |     13.073 |

error pruned1layer
| Class                        | Model           | mean|e| (m) ↓ | med|e| (m) ↓ | meanRel% ↓ | medRel% ↓ | RMSE (m) ↓ |
| ---------------------------- | --------------- | ------------: | -----------: | ---------: | --------: | ---------: |
| **Car**                      | FP32 (original) |         1.837 |        0.860 |       9.19 |      6.33 |      3.456 |
|                              | FP16 ONNX       |         2.427 |        1.145 |      11.44 |      8.66 |      4.280 |
|                              | INT8 ONNX       |         2.596 |        1.284 |      12.31 |      9.76 |      4.454 |
|                              | Pruned 1-layer  |         1.837 |        0.860 |       9.19 |      6.33 |      3.456 |
| **Cyclist**                  | FP32 (original) |         0.690 |        0.497 |       7.92 |      5.69 |      0.937 |
|                              | FP16 ONNX       |         1.196 |        1.001 |      13.61 |     11.36 |      1.603 |
|                              | INT8 ONNX       |         1.210 |        1.003 |      13.88 |     12.24 |      1.573 |
|                              | Pruned 1-layer  |         0.690 |        0.497 |       7.92 |      5.69 |      0.937 |
| **LargeVeh**                 | FP32 (original) |         0.799 |        0.414 |       8.38 |      6.22 |      1.230 |
|                              | FP16 ONNX       |         2.155 |        1.434 |      16.81 |     17.20 |      3.020 |
|                              | INT8 ONNX       |         2.296 |        1.478 |      17.76 |     18.01 |      3.245 |
|                              | Pruned 1-layer  |         0.799 |        0.414 |       8.38 |      6.22 |      1.230 |
| **Person**                   | FP32 (original) |         2.154 |        1.624 |      21.10 |     18.95 |      2.975 |
|                              | FP16 ONNX       |         2.973 |        2.518 |      28.85 |     26.85 |      3.625 |
|                              | INT8 ONNX       |         3.014 |        2.629 |      29.17 |     28.10 |      3.644 |
|                              | Pruned 1-layer  |         2.154 |        1.624 |      21.10 |     18.95 |      2.975 |
| **Truck**                    | FP32 (original) |         1.571 |        1.017 |      11.85 |      9.20 |      2.563 |
|                              | FP16 ONNX       |         1.581 |        0.857 |      13.11 |     11.30 |      2.333 |
|                              | INT8 ONNX       |         1.513 |        0.860 |      12.79 |     12.30 |      2.262 |
|                              | Pruned 1-layer  |         1.571 |        1.017 |      11.85 |      9.20 |      2.563 |
| **electric pole**            | FP32 (original) |         2.751 |        1.863 |      24.87 |     18.89 |      3.757 |
|                              | FP16 ONNX       |         2.516 |        2.019 |      22.86 |     22.51 |      3.016 |
|                              | INT8 ONNX       |         2.945 |        2.231 |      26.41 |     24.51 |      3.512 |
|                              | Pruned 1-layer  |         2.751 |        1.863 |      24.87 |     18.89 |      3.757 |
| **motocycle**                | FP32 (original) |         1.011 |        0.567 |      15.28 |      7.09 |      1.373 |
|                              | FP16 ONNX       |         0.788 |        0.374 |      10.69 |      4.26 |      1.198 |
|                              | INT8 ONNX       |         0.752 |        0.406 |      10.35 |      4.96 |      1.139 |
|                              | Pruned 1-layer  |         1.011 |        0.567 |      15.28 |      7.09 |      1.373 |
| **pedestrian crossing sign** | FP32 (original) |         1.929 |        1.064 |      17.83 |     11.28 |      2.792 |
|                              | FP16 ONNX       |         3.252 |        2.879 |      26.02 |     17.95 |      3.805 |
|                              | INT8 ONNX       |         3.086 |        2.379 |      26.13 |     18.14 |      3.620 |
|                              | Pruned 1-layer  |         1.929 |        1.064 |      17.83 |     11.28 |      2.792 |
| **tree**                     | FP32 (original) |         4.950 |        4.160 |      34.91 |     31.47 |      6.430 |
|                              | FP16 ONNX       |         4.351 |        3.010 |      30.06 |     23.66 |      5.972 |
|                              | INT8 ONNX       |         4.454 |        3.217 |      30.85 |     27.34 |      6.060 |
|                              | Pruned 1-layer  |         4.950 |        4.160 |      34.91 |     31.47 |      6.430 |
| **crosswalk**                | FP32 (original) |         1.594 |        1.594 |      12.58 |     12.58 |      1.727 |
|                              | FP16 ONNX       |         0.401 |        0.401 |       3.16 |      3.16 |      0.430 |
|                              | INT8 ONNX       |         0.548 |        0.548 |       4.33 |      4.33 |      0.597 |
|                              | Pruned 1-layer  |         0.037 |        0.037 |       0.27 |      0.27 |      0.045 |
| **sidewalk**                 | FP32 (original) |         0.575 |        0.574 |       8.85 |      8.89 |      0.628 |
|                              | FP16 ONNX       |         0.225 |        0.216 |       3.57 |      3.53 |      0.276 |
|                              | INT8 ONNX       |         0.233 |        0.202 |       3.73 |      3.07 |      0.287 |
|                              | Pruned 1-layer  |         0.333 |        0.336 |       5.20 |      5.34 |      0.388 |
| **stairs**                   | FP32 (original) |         2.367 |        1.117 |      12.94 |      9.98 |      3.651 |
|                              | FP16 ONNX       |         2.657 |        1.164 |      14.57 |      7.54 |      4.130 |
|                              | INT8 ONNX       |         2.528 |        0.942 |      13.88 |      6.74 |      3.881 |
|                              | Pruned 1-layer  |         3.185 |        2.828 |      18.33 |     16.24 |      3.909 |
| **tree line**                | FP32 (original) |         0.677 |        0.574 |      14.77 |      6.75 |      0.931 |
|                              | FP16 ONNX       |         0.696 |        0.466 |      14.87 |      7.21 |      0.880 |
|                              | INT8 ONNX       |         0.800 |        0.655 |      16.61 |      9.70 |      0.979 |
|                              | Pruned 1-layer  |         0.974 |        0.615 |      21.04 |      9.12 |      1.256 |


Gotcha — here it is in simple plain text, no LaTeX.

Let’s define for each detection i:
g_i = ground-truth distance (meters)
p_i = predicted distance (meters)
e_i = error = g_i - p_i

You have N detections in a bin.

mean|e| (Mean absolute error, MAE)
Formula:
For each sample: absolute error_i = |e_i| = |g_i - p_i|
Then:
mean|e| = (1 / N) * sum over i of |g_i - p_i|
Meaning:
Average absolute error in meters.
“On average, the prediction is off by this many meters, ignoring direction.”
meanRel% (Mean relative absolute error, in percent)
Formula:
For each sample: rel_error_i = 100 * |g_i - p_i| / g_i
Then:
meanRel% = (1 / N) * sum over i of (100 * |g_i - p_i| / g_i)
Meaning:
Average percentage error relative to the true distance.
“On average, the prediction is off by this many percent of the true distance.”
This is essentially the same idea as the commonly used absolute relative error (“Abs Rel”) in depth estimation, just expressed as a percentage rather than a fraction.
RMSE (Root mean squared error)
Formula:
For each sample: squared_error_i = (g_i - p_i)^2
First compute MSE (mean squared error):
MSE = (1 / N) * sum over i of (g_i - p_i)^2
Then:
RMSE = sqrt(MSE)
Meaning:
Error in meters, like MAE, but large errors are penalized more because of the square.
“Typical error in meters, with extra weight on big mistakes (outliers).”