from roboflow import Roboflow
import os
from tqdm import tqdm

rf = Roboflow(api_key="g3NROnPvu1ialUbxBEeW")

workspace_id = "object-detection-7datn"
project_id = "testyolo-ezuzf"
project = rf.workspace(workspace_id).project(project_id)

root_dir = r"C:\Users\OS\Downloads\GroupProject_OD"
image_extensions = (".jpg", ".jpeg", ".png", ".heic", ".heif")

uploaded = 0

for split in ["train", "valid",]:
    image_root = os.path.join(root_dir, split, "images")
    label_root = os.path.join(root_dir, split, "labels")

    if not os.path.exists(image_root):
        continue

    print(f"\n=== Split: {split} ===")
    print("image_root:", image_root)
    print("label_root:", label_root)

    for folder, _, files in os.walk(image_root):
        for file in tqdm(files, desc=f"Uploading from {folder}"):
            if file.lower().endswith(image_extensions):
                image_path = os.path.join(folder, file)

                rel_path = os.path.relpath(image_path, image_root)   
                base, _ = os.path.splitext(rel_path)                 
                label_path = os.path.join(label_root, base + ".txt") 

                kwargs = {
                    "image_path": image_path,
                    "split": split,          
                    "batch_name": "yolo_import",
                    "num_retry_uploads": 3
                }
                if os.path.exists(label_path):
                    kwargs["annotation_path"] = label_path

                try:
                    project.upload(**kwargs)
                    uploaded += 1
                except Exception as e:
                    print(f"Error uploading {file}: {e}")

print(f"\nDone! Uploaded {uploaded} images to {workspace_id}/{project_id}")
