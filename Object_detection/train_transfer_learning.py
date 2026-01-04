"""
Transfer Learning - YOLOv8x
Giữ 4 COCO classes + Fine-tune 5 classes mới
Compact version cho server training

Training stage
    Stage 1 (20 epochs): “warm up” the head while freezing many layers
    Stage 2 (30 epochs): fine-tune more layers with stronger augmentation & tuned loss weights
    Stage 3 (50 epochs): full fine-tune with low LR + mosaic reduction near the end

Accuracy number (evaluation)
    metrics.box.map50   # mAP@0.50 IoU
    metrics.box.map     # mAP@0.50:0.95 IoU (COCO-style)
    metrics.box.mp      # mean precision
    metrics.box.mr      # mean recall
"""

from ultralytics import YOLO
import torch
import os

# ============================================
# CẤU HÌNH ĐƯỜNG DẪN
# ============================================
DATA_YAML = "/storage/student5/blind/GroupProject_OD/data.yaml"     # train/val paths + class names + number of classes
PROJECT_DIR = "/home/student5/blind/yolo_training/results/detect"   # where to save
RUN_NAME = "ketquatraintransfer"
EPOCHS = 100
BATCH = 8
IMG_SIZE = 640
DEVICE = 0  # GPU 0, nếu không có GPU thì để 'cpu'

print("=" * 70)
print("🚀 TRANSFER LEARNING: YOLOv8x")
print("=" * 70)
print(f"📁 Data YAML: {DATA_YAML}")
print(f"📁 Project Dir: {PROJECT_DIR}")
print(f"📝 Run Name: {RUN_NAME}")
print(f"📊 Total Epochs: {EPOCHS}")
print(f"📦 Batch Size: {BATCH}")
print(f"🖼️  Image Size: {IMG_SIZE}")
print(f"💻 Device: {'GPU ' + str(DEVICE) if DEVICE != 'cpu' else 'CPU'}")
print("=" * 70)
print("📦 COCO classes: person, bicycle, motorcycle, car")
print("🆕 New classes: bus, electric pole, pedestrian crossing sign, tree, truck")
print("=" * 70)

# ============================================
# STAGE 1: Warm-up Detection Head (20 epochs)
# ============================================
print("\n🔥 STAGE 1: WARM-UP DETECTION HEAD (20 epochs)")

model = YOLO('yolov8x.pt')

model.train(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name=f"{RUN_NAME}_stage1_warmup",
    
    epochs=20,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    
    freeze=10,
    lr0=0.01,
    lrf=0.1,
    momentum=0.937,
    weight_decay=0.0005,
    
    augment=True,
    mosaic=1.0,
    mixup=0.1,
    
    patience=10,
    optimizer='AdamW',
    save_period=-1,
    verbose=True,
    plots=False
)

print("✅ Stage 1 completed\n")


# ============================================
# STAGE 2: Fine-tune với Class Weights (30 epochs)
# ============================================
print("🎯 STAGE 2: FINE-TUNE WITH CLASS WEIGHTS (30 epochs)")

stage1_best = os.path.join(PROJECT_DIR, f"{RUN_NAME}_stage1_warmup", "weights", "best.pt")
model = YOLO(stage1_best)

model.train(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name=f"{RUN_NAME}_stage2_finetune",
    
    epochs=30,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    
    freeze=5,
    lr0=0.005,
    lrf=0.05,
    momentum=0.937,
    weight_decay=0.0005,
    
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    augment=True,
    degrees=5.0,
    translate=0.2,
    scale=0.9,
    shear=2.0,
    perspective=0.001,
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    
    patience=15,
    optimizer='AdamW',
    save_period=-1,
    verbose=True,
    plots=False
)

print("✅ Stage 2 completed\n")


# ============================================
# STAGE 3: Full Fine-tune (50 epochs)
# ============================================
print("🚀 STAGE 3: FULL FINE-TUNE (50 epochs)")

stage2_best = os.path.join(PROJECT_DIR, f"{RUN_NAME}_stage2_finetune", "weights", "best.pt")
model = YOLO(stage2_best)

model.train(
    data=DATA_YAML,
    project=PROJECT_DIR,
    name=f"{RUN_NAME}_stage3_full",
    
    epochs=50,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    
    freeze=0,
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    augment=True,
    degrees=3.0,
    translate=0.15,
    scale=0.7,
    shear=1.0,
    mosaic=0.8,
    mixup=0.1,
    copy_paste=0.05,
    
    patience=25,
    optimizer='AdamW',
    close_mosaic=10,
    save_period=-1,
    verbose=True,
    plots=False
)

print("✅ Stage 3 completed\n")


# ============================================
# FINAL VALIDATION
# ============================================
print("=" * 70)
print("📊 FINAL VALIDATION")
print("=" * 70)

# model evaluation
stage3_best = os.path.join(PROJECT_DIR, f"{RUN_NAME}_stage3_full", "weights", "best.pt")
best_model = YOLO(stage3_best)
metrics = best_model.val(data=DATA_YAML)

print(f"\n🎯 Overall Performance:")
print(f"   mAP50    : {metrics.box.map50:.4f}")
print(f"   mAP50-95 : {metrics.box.map:.4f}")
print(f"   Precision: {metrics.box.mp:.4f}")
print(f"   Recall   : {metrics.box.mr:.4f}")

# Per-class results
print(f"\n📋 Per-Class mAP50:")
classes = ['bicycle', 'bus', 'car', 'electric pole', 'motorcycle', 
           'pedestrian crossing sign', 'person', 'tree', 'truck']
           
coco_classes = ['bicycle', 'car', 'motorcycle', 'person']
new_classes = ['bus', 'electric pole', 'pedestrian crossing sign', 'tree', 'truck']

for i, cls in enumerate(classes):
    marker = "📦" if cls in coco_classes else "🆕"
    print(f"{marker} {cls:30s}: {metrics.box.map50[i]:.4f}")

# Summary
coco_idx = [0, 2, 4, 6]
new_idx = [1, 3, 5, 7, 8]
coco_avg = metrics.box.map50[coco_idx].mean()
new_avg = metrics.box.map50[new_idx].mean()

print(f"\n📦 COCO Avg (4 classes): {coco_avg:.4f}")
print(f"🆕 New Avg (5 classes) : {new_avg:.4f}")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETED!")
print(f"📁 Best model: {stage3_best}")
print("=" * 70)

# Lưu summary
summary_file = os.path.join(PROJECT_DIR, f"{RUN_NAME}_summary.txt")
with open(summary_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("TRAINING SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Overall Performance:\n")

    # metric model
    f.write(f"  mAP50    : {metrics.box.map50:.4f}\n")
    f.write(f"  mAP50-95 : {metrics.box.map:.4f}\n")
    f.write(f"  Precision: {metrics.box.mp:.4f}\n")
    f.write(f"  Recall   : {metrics.box.mr:.4f}\n\n")
    # end metric model

    f.write(f"COCO Classes Avg: {coco_avg:.4f}\n")
    f.write(f"New Classes Avg : {new_avg:.4f}\n\n")
    f.write(f"Best Model: {stage3_best}\n")
    f.write("=" * 70 + "\n")

print(f"\n✅ Summary saved: {summary_file}")
