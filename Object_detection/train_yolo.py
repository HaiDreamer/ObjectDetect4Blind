import os
import sys
import subprocess
from pathlib import Path

# ================== Cấu hình của BẠN ==================
# sửa lại cho đúng đường dẫn dataset & nơi lưu runs
DATA_YAML = "/storage/student5/blind/GroupProject_OD/data.yaml"
PROJECT_DIR = "/home/student5/blind/yolo_training/results/detect"
RUN_NAME = "ketquatrainlan2"
EPOCHS = 100
BATCH = 8
IMG_SIZE = 640
DEVICE = 0      # nếu không có GPU thì để 'cpu'
# ======================================================


def run(cmd, **kwargs):
    """Chạy lệnh shell và in ra cho dễ debug."""
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_pip():
    """Đảm bảo có pip (cài bằng get-pip.py nếu chưa có)."""
    try:
        import pip  # noqa
        print(f"[INFO] pip đã có sẵn: {pip.__version__}")
        return
    except Exception:
        print("[INFO] pip chưa có, đang cài bằng get-pip.py ...")

    import urllib.request

    url = "https://bootstrap.pypa.io/get-pip.py"
    script = "get-pip.py"

    print(f"[INFO] Tải {url} ...")
    urllib.request.urlretrieve(url, script)

    run([sys.executable, script, "--user"])
    print("[INFO] Cài pip xong.")


def pip_install(packages):
    """Cài các package cần thiết bằng pip --user."""
    cmd = [sys.executable, "-m", "pip", "install", "--user"] + list(packages)
    run(cmd)


def main():
    # 1. Đảm bảo có pip
    ensure_pip()

    # 2. Cài torch + ultralytics + opencv nếu chưa có
    #    (chạy nhiều lần cũng không sao, pip sẽ bỏ qua bản đã cài)
    pip_install(["torch", "torchvision", "torchaudio",
                 "ultralytics", "opencv-python-headless"])

    # 3. Import YOLO sau khi đã cài xong
    from ultralytics import YOLO

    data_yaml_path = Path(DATA_YAML)
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy data.yaml tại: {data_yaml_path}")

    print(f"[INFO] Dùng data.yaml: {data_yaml_path}")

    # 4. Khởi tạo model (YOLO sẽ tự tải yolov8n.pt nếu chưa có)
    model = YOLO("yolov8x.pt")

    # 5. Train
    model.train(
        data=str(data_yaml_path),
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        project=PROJECT_DIR,
        name=RUN_NAME,
        device=DEVICE,
        save=True,
        save_period=1,   # 👉 lưu weights sau MỖI epoch: epoch1.pt, epoch2.pt,...
    )

    print("\n[INFO] Train xong!")
    print(f"  Weights nằm trong: {PROJECT_DIR}/{RUN_NAME}/weights")
    print("  - best.pt : model tốt nhất")
    print("  - last.pt : model ở epoch cuối")
    print("  - epoch*.pt : checkpoint từng epoch (do save_period=1)")


if __name__ == "__main__":
    main()

