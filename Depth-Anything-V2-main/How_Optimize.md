# Requirements
Ensure both accuracy not drop too much and reduce size of model
After compress: compare model(accuracy and size) before compression and after compression

# KITTI Depth Evaluation — Quick Reference (Depth Anything V2)

Dataset / format (for the KITTI depth_prediction benchmark)
• Ground truth (GT) PNGs are uint16 where meters = value / 256.0 and 0 = invalid.
• Your predictions must use the same format and filenames as GT.
• KITTI ranks methods by √SILog. (The devkit also prints AbsRel, SqRel, RMSE, RMSElog.)

# Accuracy metrics 
(p = prediction in meters, g = ground-truth in meters; averages are over valid GT pixels)
↑ means higher is better; ↓ means lower is better.

1) δ accuracies (δ1, δ2, δ3)   ↑
   Definition: threshold = max(p/g, g/p). Then
     δ1 = mean(threshold < 1.25)
     δ2 = mean(threshold < 1.25^2)
     δ3 = mean(threshold < 1.25^3)
   Intuition: % of pixels whose prediction is within a multiplicative factor of GT.
2) AbsRel (Absolute Relative Error)   ↓
   AbsRel = mean(|p − g| / g)
   Intuition: proportional error relative to ground truth.
3) SqRel (Squared Relative Error)   ↓
   SqRel = mean((p − g)^2 / g)
   Intuition: like AbsRel but penalizes large errors more.
4) RMSE (Root Mean Squared Error, meters)   ↓
   RMSE = sqrt(mean((p − g)^2))
5) RMSElog (Log RMSE)   ↓
   RMSElog = sqrt(mean((log p − log g)^2))
6) SILog (Scale-Invariant Log Error)   ↓
   Let e = log p − log g.
   SILog = sqrt( mean(e^2) − (mean(e))^2 ) × 100
   Notes: scale-invariant; uniform rescaling of predictions affects it less.
          KITTI’s leaderboard ranks by √SILog (lower is better).
7) log10 (Mean Absolute Log10 Error)   ↓
   log10 = mean(|log10 p − log10 g|)

Good sanity ranges on KITTI (relative monocular models with per-image alignment):
• δ1 ≳ 0.90, AbsRel ≈ 0.06–0.12, RMSE ≲ 4–6 m, SILog in low tens (e.g., ~8–15).
  (Exact values depend on model and split; use the same protocol to compare.)

Practical tips
• Evaluate only on pixels where GT > 0 (valid mask).
• Clamp depths to the benchmark range (typically 0–80 m for KITTI).
• Relative (affine-invariant) models (e.g., Depth Anything V2 foundation) should be aligned
  per-image before scoring: fit a, b so that a*y + b ≈ 1/g in inverse-depth, then invert.
• Metric DA-V2 checkpoints (VKITTI/Hypersim fine-tuned) output meters directly — do NOT align
  to GT; just save meters × 256 and evaluate.

# How to check accuracy of depth estimation model ? 

Inside file kitti_root (WHY kitti? we need to detect distance objects at outside, also evaluate distance)
Run make_kitti_preds

# Quantization
   Def: Quantization hiểu đơn giản hơn là kỹ thuật chuyển đổi định dạng của dữ liệu từ những kiểu dữ liệu có độ chính xác cao sang những kiểu dữ liệu có độ chính xác thấp hơn, qua đó làm giảm memory khi lưu trữ. Do đó làm tăng tốc độ inference là giảm latency khi load model. Tất nhiên là khi bạn làm giảm độ chính xác xuống thì thường accuracy của mô hình sẽ giảm một chút (không quá nhiều) so với mô hình gốc.

   Formula: quantize q = round(x/scale) + zero_point
   Technique:
      Full int quantization(quick win on cpu):  (done)
         4x smaller and 2-3x faster on CPU 
      Float 16 quantization 
         Have GPU 
         Minimal accuracy change

   Upgrade 1: keep sensitive layer with same accuracy type (FP32) instead of INT8 (not completed)
      What is sensitive layer ?
         LayerNorms
         patchEmbed
         final prediction head/last layer of encoder (DPT head)
      Sensitive scan ?

# Post training quantization
   Checking accuracy, size model
   -> Compare to original model

# Training-aware quantization (QAT)
   

# How depth anything model is trained

# Pruning
Reduce size of model while keep accuracy
   General Technique:
      Eliminate neurons(layers) that contribute less to the output
      Fine-tune (train for few epochs with small learning rate, for "surviving weight”)
      Re-do pruning (loops for several times until optimize both condition: battery use and model size)
