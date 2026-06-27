import json
from pathlib import Path
import cv2
import numpy as np

JSON_PATH = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\segment_json_KITTI_val.json"
OUT_DIR   = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\debug_seg_vis"

N_IMAGES = 50                 # how many images to export
ALPHA_FILL = 0.35             # polygon fill transparency (0..1)

def flat_to_pts(seg_flat):
    """
    [x1,y1,x2,y2,...] -> (N,1,2) int32 for OpenCV
    Ex: [10.2, 5.7, 20.9, 6.1, 15.0, 30.0] becomes
        [[10.2,  5.7],
        [20.9,  6.1],
        [15.0, 30.0]]
    """
    if not seg_flat or len(seg_flat) < 6 or (len(seg_flat) % 2) != 0:
        return None         # not valid case
    pts = np.array(seg_flat, dtype=np.float32).reshape(-1, 2)
    pts = np.round(pts).astype(np.int32)
    return pts.reshape(-1, 1, 2)

def color_for_id(class_id: int):
    '''deterministic random color per class_id (BGR)'''
    rng = np.random.default_rng(int(class_id) + 12345)
    bgr = rng.integers(40, 255, size=3, dtype=np.int32)
    return int(bgr[0]), int(bgr[1]), int(bgr[2])

def draw_instance(img_bgr, inst):
    '''
    :param img_bgr: image array with bgr format
    :param inst: dictionary with segmentation_xy, class_id, class_name, confidence as key
    '''
    pts = flat_to_pts(inst.get("segmentation_xy", []))      
    if pts is None:
        return img_bgr

    cls_id = int(inst.get("class_id", 0))
    cls_name = str(inst.get("class_name", cls_id))
    conf = inst.get("confidence", None)

    color = color_for_id(cls_id)

    # fill polygon with transparency
    overlay = img_bgr.copy()
    cv2.fillPoly(overlay, [pts], color)                 # filled polygon 
    img_bgr = cv2.addWeighted(overlay, ALPHA_FILL, img_bgr, 1 - ALPHA_FILL, 0)  # transparency

    # outline polygon in img_bgr img with point in pts, auto connect last point to first point, with 2 pixel line
    cv2.polylines(img_bgr, [pts], True, color, 2)

    # label near first vertex of the polygon
    x, y = int(pts[0, 0, 0]), int(pts[0, 0, 1])
    label = f"{cls_name}" + (f" {float(conf):.2f}" if conf is not None else "")
    # x, max(15, y) avoid text from being too close to the top border   
    cv2.putText(img_bgr, label, (x, max(15, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)  

    return img_bgr

def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(JSON_PATH).read_text(encoding="utf-8"))
    images = data.get("images", [])

    saved = 0
    for im in images:
        if saved >= N_IMAGES:
            break

        file_path = im.get("file_path")
        file_name = im.get("file_name", f"img_{saved}.png")
        instances = im.get("instances", [])

        if not file_path:
            continue

        img = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if img is None:
            print("WARN: cannot read", file_path)
            continue

        for inst in instances:
            img = draw_instance(img, inst)

        out_path = out_dir / f"{Path(file_name).stem}__regions.png"
        cv2.imwrite(str(out_path), img)
        print("Saved:", out_path)
        saved += 1

if __name__ == "__main__":
    main()
