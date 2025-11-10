# ObjectDetect4Blind

A multi-threaded pipeline that runs **three vision models in parallel** — **Depth Estimation** (Depth-Anything-V2), **Object Detection** (YOLO), and **Segmentation** — on images or video. Outputs are saved to `./output`.

---

## Features
- Orchestrates **depth**, **detection**, and **segmentation** together via `MAIN.py`
- Supports **image** and **video** inputs
- Simple, folder-based **model checkpoint placement**

---

## Project Layout

RECOMMEND: put file respitory the same as me, or u need to fix code for file/model location
```text

C:/Python/
├────ObjectDetect4Blind/
│    ├── .vscode
│    ├── MAIN.py                         # Launches multithreading 3 models
│    ├── assets/                         # Example input images
│    ├── output/                         # Outputs produced by MAIN.py
│    ├── Depth-Anything-V2-main/         # Depth estimation module
│    │   ├── app.py
│    │   ├── run.py
│    │   └──run_video.py                 # Temp not developed
│    ├── Object detection/               # Object detection module
│    │   └── main.py                     # Usage entrypoint for detection
│    └── Segmentation/                   # Segmentation module
│        └── test_model.py               # Usage entrypoint for segmentation
└────ObjectDetectRequireFile/
     ├── put-in-depth-anything
     ├── put-in-obj-detect                   
     ├── put-in-segment                   
     └── output/ 

```

## Setup

### Model file
https://drive.google.com/file/d/1DwhseV8bqV_qw7CIuWS7pnMRJ9au9lpE/view

### Python versions (important)
This repo currently expects **two Python interpreters** when running `MAIN.py`:
- **YOLO / Object detection:** Python **3.11**
- **Depth estimation:** Python **3.13**

`MAIN.py` starts each model using the interpreter paths you configure.  
**Edit `MAIN.py`** and replace the hard-coded interpreter/virtual-env paths with the ones on your machine (see comments in the file).

> ✅ Tip: If you manage multiple Python versions, set up two virtual environments (e.g., with Conda or pyenv) and point `MAIN.py` to their `python` executables.

---

## Quickstart

### A) Run the full pipeline (multithreading)
1. Open `MAIN.py` and update the Python paths/env activation commands for:
   - Detection, Segmentation (Python 3.11)
   - Depth (Python 3.13)
2. From the project root:
   python MAIN.py

### B) Run Depth-Anything-V2 only
1. Local demo server
    - python app.py
2. Single img(depth estimation only)
    - python run.py --encoder vits --precision int8 --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --outdir depth_vis --pred-only
3. Single image (side-by-side input + depth)
    - python run.py --encoder vits --precision int8 --img-path "C:\Python\ObjectDetect4Blind\assets\demo01.jpg" --outdir depth_vis
4. Video
    - python run_video.py --encoder vitl \ --video-path assets/examples_video \ --outdir video_depth_vis
