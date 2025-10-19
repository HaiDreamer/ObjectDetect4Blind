from roboflow import Roboflow
import os
from tqdm import tqdm

rf = Roboflow(api_key="g3NROnPvu1ialUbxBEeW")

workspace_id = "object-detection-7datn"
project_id = "od-ywhgw"
project = rf.workspace(workspace_id).project(project_id)

root_dir = r"D:\Photo\Night"

# 👇 Thêm .heic và .heif
image_extensions = (".jpg", ".jpeg", ".png", ".heic", ".heif")

uploaded = 0

for folder, _, files in os.walk(root_dir):
    for file in tqdm(files, desc=f"Uploading from {folder}"):
        if file.lower().endswith(image_extensions):
            image_path = os.path.join(folder, file)
            try:
                project.upload(
                    image_path=image_path,
                    split="train",
                    batch_name="video_frames",
                    num_retry_uploads=3
                )
                uploaded += 1
            except Exception as e:
                print(f"❌ Error uploading {file}: {e}")

print(f"✅ Done! Uploaded {uploaded} images to {workspace_id}/{project_id}")
