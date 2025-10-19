import cv2
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

# === Thư mục chứa video ===
video_dir = r"D:\Ảnh"        # 👉 thay bằng thư mục chứa 16 video
output_root = "frames_30_unique"   # 👉 thư mục để lưu kết quả

# === Tạo thư mục đích nếu chưa có ===
os.makedirs(output_root, exist_ok=True)

# === Lấy danh sách tất cả video trong thư mục ===
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

def extract_frames(video_name):
    video_path = os.path.join(video_dir, video_name)
    video_stem = os.path.splitext(video_name)[0]  # tên video không có đuôi
    output_dir = os.path.join(output_root, video_stem)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    save_every = 30  # 👉 Lưu mỗi 30 frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % save_every == 0:
            # Tạo tên ảnh duy nhất (gồm tên video + số frame + mã ngẫu nhiên)
            unique_id = uuid.uuid4().hex[:6]  # 6 ký tự ngẫu nhiên
            frame_name = f"{video_stem}_frame_{frame_count:05d}_{unique_id}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ {video_name}: {saved_count} frames saved in '{output_dir}'")

# === Dùng đa luồng để xử lý nhanh nhiều video cùng lúc ===
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(extract_frames, video_files)

print("🎬 Hoàn tất trích xuất tất cả video!")
