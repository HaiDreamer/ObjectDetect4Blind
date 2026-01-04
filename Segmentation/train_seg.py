import sys
import subprocess
from pathlib import Path
import shutil

'''
1) Phần chuyển label: bbox → polygon hình chữ nhật      abs need this 
'''

# setup bb 
DATASET_PATH = "/storage/student5/blind/GroupProject_Seg"
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def bbox_to_rect_poly_line(parts):
    # parts: [cls, cx, cy, w, h] normalized
    cls = parts[0]
    cx, cy, w, h = map(float, parts[1:5])
    x1, y1 = clamp01(cx - w/2), clamp01(cy - h/2)
    x2, y2 = clamp01(cx + w/2), clamp01(cy + h/2)
    poly = [x1, y1,  x2, y1,  x2, y2,  x1, y2]  # 4 points
    return cls + " " + " ".join(f"{p:.6f}" for p in poly)


def convert_labels_dir(lbl_dir: Path):
    changed_files = 0
    changed_lines = 0

    for txt in lbl_dir.rglob("*.txt"):
        lines = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        if not lines:
            continue

        new_lines = []
        changed = False

        for line in lines:
            parts = line.split()
            if len(parts) == 5:  # bbox-only -> convert
                new_lines.append(bbox_to_rect_poly_line(parts))
                changed = True
                changed_lines += 1
            else:
                new_lines.append(line)

        if changed:
            shutil.copy2(txt, str(txt) + ".bak")  # backup
            txt.write_text("\n".join(new_lines) + "\n")
            changed_files += 1

    return changed_files, changed_lines

for split in ["train", "val"]:
    d = Path(DATASET_PATH) / "labels" / split
    if d.exists():
        f, l = convert_labels_dir(d)
        print(f"[{split}] converted files: {f}, converted bbox-lines: {l}")
    else:
        print(f"[{split}] not found:", d)

        
# ================== CẤU HÌNH CỦA BẠN ==================
DATA_YAML = "/storage/student5/blind/GroupProject_Seg/data.yaml"

# Nơi lưu kết quả (runs)
PROJECT_DIR = "/home/student5/blind/yolo_training/results_seg/segment"
RUN_NAME = "train_results"

EPOCHS = 10
BATCH = 1
IMG_SIZE = 1024
DEVICE = 1          # GPU index (0,1,...) hoặc "cpu"
WORKERS = 8
SAVE_PERIOD = 1     # lưu checkpoint mỗi epoch
# ======================================================


def run(cmd, **kwargs):
    """Chạy lệnh shell và in ra để dễ debug."""
    print("\n[RUN]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, **kwargs)


def ensure_ultralytics():
    """
    Đảm bảo có ultralytics. Nếu môi trường đã setup sẵn thì sẽ bỏ qua.
    Nếu bạn đang dùng venv/conda: có thể bỏ '--user' trong pip install.
    """
    try:
        import ultralytics  # noqa
        print(f"[INFO] ultralytics đã có sẵn: {ultralytics.__version__}")
        return
    except Exception:
        print("[INFO] Chưa có ultralytics -> đang cài ...")
        run([sys.executable, "-m", "pip", "install", "--user", "-U", "ultralytics"])


def main():
    data_yaml_path = Path(DATA_YAML)
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Không tìm thấy data.yaml: {data_yaml_path}")

    ensure_ultralytics()

    # Import sau khi đảm bảo đã cài
    from ultralytics import YOLO

    # ✅ Model segmentation
    model = YOLO("yolov8m-seg.pt")  # tự tải về nếu chưa có

    # Train, default: use validation dataset (information val in data.yaml)
    results = model.train(
        data=str(data_yaml_path),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT_DIR,
        name=RUN_NAME,
        task="segment",    
        save=True,
        save_period=SAVE_PERIOD,
        pretrained=True,
        exist_ok=True,      # chạy lại không bị lỗi trùng thư mục
        verbose=True,
    )

    print("\n[INFO] Train xong!")
    print(f"  Output: {PROJECT_DIR}/{RUN_NAME}")
    print(f"  Weights: {PROJECT_DIR}/{RUN_NAME}/weights")
    print("  - best.pt  : model tốt nhất")
    print("  - last.pt  : model epoch cuối")
    print(f"  - epoch*.pt: checkpoint từng epoch (save_period={SAVE_PERIOD})")

    return results


if __name__ == "__main__":
    main()
