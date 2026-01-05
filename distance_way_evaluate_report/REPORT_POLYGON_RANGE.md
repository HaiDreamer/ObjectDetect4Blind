# RULE
    ROI% is hyperparameter
    ablation (1 pixel(middle in bb/region), 10%, 20%, 30%, 100%) in KITTI GT depth then report metric
    Take value 5%, 10% min of these ????
        Với cảnh báo va chạm, bạn thường muốn gần với khoảng cách nhỏ nhất hợp lý (gần nhất trên vật thể), nhưng tránh outlier.
        Dùng percentile thấp (p5 hoặc p10) trong ROI thay vì min

# PROBLEM
What model we should use for bounding box of object detection and region from segmentation
    Use our model or their own model to improve accuracy ???

# HINT
Ước lượng “độ không chắc chắn” đơn giản từ ROI
    spread = p50 - p10 hoặc std của ROI
    spread lớn → depth đang nhiễu/occlusion → tăng margin hoặc giảm mức tin cậy.
    (Trong safety-critical, việc xét uncertainty là hướng phổ biến).


GPTTT
Với report ablation ROI% trên KITTI GT depth (mục tiêu là độ ổn định của phép lấy khoảng cách theo ROI, không phải “model tốt/xấu”), mình khuyên:
Chọn “model ngoài” (hoặc GT label) cho report chính
Lý do: ROI (bbox/mask) là “đầu vào” của phép đo khoảng cách. Nếu dùng custom model accuracy thấp, nhiễu bbox/mask sẽ trộn lẫn vào kết quả → bạn sẽ khó kết luận “ROI 10% ổn định hơn 1 pixel” hay chỉ vì bbox/mask lệch.
Cấu hình hợp lý cho report (ít biến nhiễu nhất):
Ưu tiên dùng GT label nếu dataset có (ví dụ KITTI object labels cho một số class như car/pedestrian/cyclist). GT luôn sạch nhất cho mục đích ablation.
Với class/region KITTI không có GT, dùng model ngoài có label-set phù hợp street-scene.
Sau đó, bạn có thể thêm 1 phần phụ lục: “same ablation nhưng dùng custom model của mình” để chứng minh kết luận ROI có còn đúng khi label noise tăng.
Model đề xuất để cover đúng label bạn cần
A) Segmentation (crosswalk, sidewalk, tree line) + nhiều object street-scene
Mask2Former panoptic trained on Mapillary Vistas là lựa chọn “gọn” nhất vì:
Mapillary Vistas có các class street-scene mà bạn cần như sidewalk, crosswalk (plain / zebra), traffic light, traffic sign front/back, utility-pole, vegetation (trees…), person, bicycle, motorcycle, bus, car, truck… 
CVF Open Access
Mask2Former là framework unified cho panoptic/instance/semantic segmentation (tức là bạn lấy được mask và suy ra bbox từ mask luôn). 
Hugging Face
+1
=> Với report, bạn có thể dùng 1 model này làm nguồn:

Segmentation labels: sidewalk, crosswalk, vegetation→tree line
Object instances/bboxes: person, bicycle, car, motorcycle, bus, truck, traffic light, (utility pole ≈ electric_pole), traffic sign (gần nhất với pedestrian_crossing_sign nếu bạn chấp nhận “traffic-sign front”). 
CVF Open Access
+1
Note: “tree line” thường không phải class riêng; bạn map bằng vegetation trong Mapillary (nó mô tả trees/vertical vegetation). 
CVF Open Access

B) “stairs”
Mapillary Vistas không có stairs (bạn thấy find không ra). Với stairs, dùng semantic segmentation trained on ADE20K vì ADE20K label list có stairs / stairway / step. 
Hugging Face
=> Practical: chạy thêm 1 model ADE20K chỉ để lấy class stairs.

C) “pedestrian_crossing_sign” (label rất specific)
Mapillary có traffic-sign front nhưng không phân loại chi tiết loại biển (pedestrian crossing vs stop vs …). 
CVF Open Access
Nếu bạn bắt buộc phải đúng “pedestrian crossing sign”, lựa chọn nhanh để bootstrap là open-vocabulary detector (Grounding DINO / OWL-ViT) với prompt “pedestrian crossing sign”. Grounding DINO được thiết kế cho open-set detection dựa trên text queries. 
ECVA
+1

(Nhưng open-vocab thường chậm hơn và bbox đôi khi kém “tight” hơn detector closed-set.)
Vậy dùng custom hay model ngoài?
Report ablation ROI% (mục tiêu khoa học/định lượng): dùng GT labels + model ngoài mạnh (ít nhiễu) làm main result.
Nếu report nhằm chứng minh pipeline deploy: thêm phần 2 chạy lại với custom model của bạn (và ghi rõ: đây là “end-to-end including label noise”).
Nếu bạn muốn mình đề xuất “combo tối giản” đúng với code hiện tại (onnxruntime + tốc độ), mình sẽ gợi ý 2 đường:
1 model Mapillary panoptic làm cả bbox+seg (đơn giản, ít mismatch), + 1 model ADE20K cho stairs
hoặc
YOLO (COCO) cho các object phổ biến + Mapillary semantic seg cho sidewalk/crosswalk/pole/sign/vegetation + ADE20K cho stairs (nhanh hơn, nhưng nhiều model hơn).
Bạn đang nghiêng về hướng nào: (1) một model panoptic làm tất cả hay (2) tách detection/seg để tối ưu tốc độ?

# STANDARD
    Ground-truth KITTI depth (C:\Python\ObjectDetectRequireFile\put-in-metric-depth\kitti_root)
        [text](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
    Sanity check: range 5-80m
    Pick 100% ROI làm “reference”
    Need model for object detection and segmentation for bb, NO use model depth



# NOTE
GT depth KITTI không đầy đủ 100% pixel (pixel không có GT sẽ = 0)
Depth PNG scale: nhiều loader KITTI depth convert uint16 / 256.0 để ra mét (và 0 là invalid)    -> need to guard this


# CITATION
@inproceedings{Uhrig2017THREEDV,
  author = {Jonas Uhrig and Nick Schneider and Lukas Schneider and Uwe Franke and Thomas Brox and Andreas Geiger},
  title = {Sparsity Invariant CNNs},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2017}
}
