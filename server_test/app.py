import cv2
import numpy as np
import os, sys, time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware

"""
HOW TO RUN (WEB APP)

  cd C:\Python\ObjectDetect4Blind\server_test
  uvicorn app:app --reload --host 0.0.0.0 --port 8000

Then open in browser: http://127.0.0.1:8000/    

FOR MOBILE: http://192.168.1.68:8000  (my own ivp4 internet -> ipconfig in terminal)

NEXT: 
  assume that depth anything model is inside server student5@ict14:~$, where models in 
    /storage/student5/models/depth_anything_v2_vits.pth how can i still run like this

  Performance: If many images are processed, reusing model servers or processes instead of 
    spawning Python interpreters per image would reduce overhead.
"""

# =========================
# PATH / IMPORT SETUP
# =========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)  # so we can import pipeline.py from same folder

from pipeline import run_full_pipeline_for_image  # YOLO+depth+seg pipeline

# =========================
# FASTAPI APP
# =========================

app = FastAPI()

# allow any device origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# for user upload img from their device
UPLOAD_DIR = Path(CURRENT_DIR) / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

'''Return html page (simple version)
- A file input and submit button.
- A small client-side script that:
- Reads the selected file, posts it to /predict using fetch (FormData).
- Measures client-side elapsed time.
- If the response is OK, converts the returned blob to an object URL and sets it as the src of an <img> to show the result.
- Shows server error text in a <pre> if the response fails.
'''
@app.get("/", response_class=HTMLResponse)
def index():        
    return """
    <html>
    <head><title>Full pipeline demo</title></head>
    <body>
      <h1>Upload image to run YOLO + Depth + Segmentation overlay</h1>
      <form id="form">
        <input type="file" id="file" name="file" accept="image/*" />
        <button type="submit">Submit</button>
      </form>
      <h2>Result:</h2>
      <img id="result" style="max-width: 512px;"/>
      <pre id="log"></pre>
      <script>
        const form = document.getElementById('form');
        const fileInput = document.getElementById('file');
        const resultImg = document.getElementById('result');
        const logEl = document.getElementById('log');

        form.addEventListener('submit', async (e) => {
          e.preventDefault();
          const file = fileInput.files[0];
          if (!file) {
            alert("Please choose an image first.");
            return;
          }
          const fd = new FormData();
          fd.append('file', file);

          const t0 = performance.now();
          const res = await fetch('/predict', {
            method: 'POST',
            body: fd
          });
          const t1 = performance.now();

          if (!res.ok) {
            const text = await res.text();
            logEl.textContent = `Server error (${res.status}): ${text}`;
            resultImg.src = "";
            return;
          }

          const blob = await res.blob();
          const url = URL.createObjectURL(blob);
          resultImg.src = url;
          logEl.textContent = `Client total time: ${((t1 - t0)/1000).toFixed(3)} s`;
        });
      </script>
    </body>
    </html>
    """

'''POST
- Read uploaded bytes: await file.read().
- Decode image into OpenCV array: np.frombuffer -> cv2.imdecode. If decoding fails, return 400.
- Save the uploaded image to UPLOAD_DIR as <stem>.jpg with cv2.imwrite. If write fails, return 500.
- Call your pipeline:
- final_path = run_full_pipeline_for_image(upload_path, class_names=None, seg_args=None)
- This function launches YOLO, depth, and segmentation tools, waits for them and writes the final overlay PNG. Any exception raised inside is caught and returned as 500 with the error message.
- Read the final PNG with cv2.imread, encode it as PNG with cv2.imencode, and return its bytes with media_type="image/png".
- There is an outer try/except that catches unexpected crashes and returns a 500.
'''
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1) decode upload
        data = await file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return Response(content=b"Could not decode image", status_code=400)

        H, W = img.shape[:2]
        print(f"[PREDICT] received image: {file.filename} ({W}x{H})")

        # 2) save upload for pipeline
        stem = Path(file.filename).stem or "upload"
        upload_path = UPLOAD_DIR / f"{stem}.jpg"
        if not cv2.imwrite(str(upload_path), img):
            return Response(content=b"Failed to save upload", status_code=500)
        print(f"[PREDICT] saved upload to: {upload_path}")

        # 3) run full pipeline
        t0 = time.perf_counter()
        try:
            final_path = run_full_pipeline_for_image(upload_path, class_names=None, seg_args=None)
        except Exception as e:
            # this is where YOLO / depth / seg errors will appear
            msg = f"Pipeline error: {repr(e)}"
            print("[PREDICT]", msg)
            return Response(content=msg.encode("utf-8"), status_code=500)
        elapsed = time.perf_counter() - t0
        print(f"[PREDICT] full pipeline took {elapsed:.3f} s (~{elapsed/60:.2f} min)")

        # 4) read final image and return
        final_img = cv2.imread(str(final_path))
        if final_img is None:
            return Response(content=b"Failed to read final overlay image", status_code=500)

        success, encoded_image = cv2.imencode(".png", final_img)
        if not success:
            return Response(status_code=500)

        return Response(content=encoded_image.tobytes(), media_type="image/png")

    except Exception as e:
        # fallback
        msg = f"Server crash: {repr(e)}"
        print("[PREDICT]", msg)
        return Response(content=msg.encode("utf-8"), status_code=500)
