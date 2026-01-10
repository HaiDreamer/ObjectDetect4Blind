import json
from pathlib import Path
import numpy as np
import pandas as pd

MAX_DEPTH = 80.0
MIN_DEPTH = 5.0  # your sanity range

def load_distance_json(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)

    rows = []
    for im in j.get("images", []):
        fn = im.get("file_name")
        for obj in im.get("objects", []):
            det = obj.get("distance_detail", {}) or {}
            rows.append({
                "file_name": fn,
                "obj_id": obj.get("id"),
                "class_name": obj.get("class_name"),
                "confidence": obj.get("confidence"),
                "excluded_low_conf": bool(obj.get("excluded_low_conf", False)),
                "distance_m": obj.get("distance_m"),
                "valid_px": det.get("valid_px"),
                "mode": det.get("mode"),
                "roi_method": det.get("roi_method"),
                "q": det.get("q"),
            })

    df = pd.DataFrame(rows)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    return df

df = load_distance_json(r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\100%_bb_distance_json_KITTI_val_GT.json")

# keep only objects you actually evaluated
eval_df = df[(~df["excluded_low_conf"]) & df["distance_m"].notna()].copy()

print("Total objects:", len(df))
print("Evaluated objects:", len(eval_df))
print("Coverage (%):", 100.0 * len(eval_df) / max(1, len(df)))

# range sanity check (you already clamp < MAX_DEPTH, but this catches weirdness)
out_of_range = eval_df[(eval_df["distance_m"] < MIN_DEPTH) | (eval_df["distance_m"] > MAX_DEPTH)]
print("Out of sanity range [5,80]m:", len(out_of_range))

# per-class counts + basic distance stats
summary = eval_df.groupby("class_name")["distance_m"].agg(
    N="count",
    mean="mean",
    median="median",
    p10=lambda x: np.nanpercentile(x, 10),
    p90=lambda x: np.nanpercentile(x, 90),
)
print(summary.sort_values("N", ascending=False).to_string())
