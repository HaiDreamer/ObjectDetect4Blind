import cv2
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

video_dir = r"D:\Photo\Night"        
output_root = "frames_50_unique"     

# Create output directory 
os.makedirs(output_root, exist_ok=True)

# Get list of all video files in the directory
video_files = [
    f for f in os.listdir(video_dir)
    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
]

def extract_frames(video_name):
    video_path = os.path.join(video_dir, video_name)
    video_stem = os.path.splitext(video_name)[0]  
    output_dir = os.path.join(output_root, video_stem)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    save_every = 50  # save one frame every 50 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % save_every == 0:
            # Create a unique frame name (video name + frame index + random ID)
            unique_id = uuid.uuid4().hex[:6]  # 6 random characters
            frame_name = f"{video_stem}_frame_{frame_count:05d}_{unique_id}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"{video_name}: {saved_count} frames saved in '{output_dir}'")

# Use multithreading to process multiple videos faster
with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(extract_frames, video_files)

print("Finished extracting frames from all videos!")
