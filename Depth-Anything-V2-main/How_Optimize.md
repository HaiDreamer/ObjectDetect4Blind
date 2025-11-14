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
Original model
   Encoder type      : vits (ViT-S/14)
   Embedding dim     : 384
   Number of blocks  : 12
   Tap indices (DA-V2): [2, 5, 8, 11]

What is tap indices?
   tell vits encoder which transformer blocks to "tap" (extract) features form so the DPT-style decoder can fuse multi-scale information into a final depth map
   NEED to retap after pruning model

Which blocks are more important?
   Higher MSE / lower PSNR / lower SSIM ⇒ bigger change ⇒ that block is more important.

   Block remove each result
      block_idx,MSE01,MAE01,PSNR,SSIM,alpha,beta
      0,0.03052955,0.12560470,16.149,0.69851,1.135689,-0.677993
      1,0.00392269,0.03979474,25.063,0.89014,1.182750,0.046374
      2,0.00938331,0.06256544,21.388,0.84673,0.975481,-0.002830
      3,0.01577090,0.09039485,18.992,0.71892,1.664848,-0.786184
      4,0.01642881,0.08851652,19.033,0.75355,1.359850,-0.672919
      5,0.03093622,0.12595510,16.065,0.71414,1.799772,-0.742489
      6,0.00395356,0.04013912,25.583,0.85361,1.112795,-0.003922
      7,0.01580263,0.08242129,19.748,0.74281,1.217867,-0.587689
      8,0.01719892,0.09268131,18.030,0.73148,1.819362,-1.725465
      9,0.01279062,0.08651063,19.224,0.71973,1.761943,-1.303174
      10,0.00339999,0.04226129,25.057,0.82104,0.917169,-0.216517
      11,0.00884184,0.06561439,21.011,0.82367,1.007369,0.406152

   -> Based on MSE(Mean Square Error), we have
      Most → least important (by MSE01)
      Block 5 (most) -> 0 -> 8 -> 4 -> 7 -> 3 -> 9 -> 2 -> 11 -> 6 -> 1 -> 10 (least)



# Pruning (DONT DO THIS, BECAUSE QUAITIZATION FOR MOBILE APP HURT ACCURACY A LOT!)
Reduce size of model while keep accuracy
   General Technique:
      Eliminate neurons(layers) that contribute less to the output
      Fine-tune (train for few epochs with small learning rate, for "surviving weight”)
      Re-do pruning (loops for several times until optimize both condition: battery use and model size)

After pruning, consider a short fine-tune to recover a bit of quality (unfreeze top few encoder blocks + DPT head, small LR).

Requirement (for depth anything model)

Pruning keep 8/12 most important blocks

Fine-tune model
   Epoch 1/10 | train 4.5526 | val SILog 0.568 | 301.9s
   [saved BEST] C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned_rel_best.pthEpoch 1/10 | train 4.5601 | val SILog 0.567 | 255.4s
   [saved] C:\Python\ObjectDetectRequireFile\put-in-depth-anything\checkpoints\depth_anything_v2_vits_pruned_rel_best.pth
   Epoch 2/10 | train 4.5645 | val SILog 0.568 | 301.2s
   Epoch 3/10 | train 4.5645 | val SILog 0.568 | 286.9s
   Epoch 4/10 | train 4.5645 | val SILog 0.568 | 266.4s
   Epoch 5/10 | train 4.5645 | val SILog 0.568 | 237.0s
   Epoch 9/10 | train 4.5645 | val SILog 0.568 | 259.8s

"""
Export DA-V2 (relative) predictions on KITTI val_selection_cropped with
per-image affine alignment in inverse depth, then save KITTI-format uint16 PNGs.

Relative monocular depth models (like depth_anything_v2_vits.pth) don't predict meters; their outputs are only accurate up to scale and shift.

- Model: depth_anything_v2_vits.pth
- Save: uint16 PNG, value = round(meters * 256.0), 0 = invalid

Input: model
Output: predicted images to compare with the labelled one (NEXT step: run eval_kitti_subset.py)
"""