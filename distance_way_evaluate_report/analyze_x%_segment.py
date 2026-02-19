import json
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-6

# paths
# base line for compare with another ablations
REF_JSON = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\10%_seg_distance_json_KITTI_val_GT.json"

ABLATIONS = {
    #"single_pixel": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\1pixel_seg_distance_json_KITTI_val_GT.json",
    "quantile_band_1%": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\1%_seg_distance_json_KITTI_val_GT.json",
    #"quantile_band_10%": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\10%_seg_distance_json_KITTI_val_GT.json",
    #"quantile_band_20%": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\20%_seg_distance_json_KITTI_val_GT.json",
    #"quantile_band_30%": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\30%_seg_distance_json_KITTI_val_GT.json",
}

JOIN_STRATEGY = "inner"  # intersection only
OUT_CSV = "seg_ablation_overall_short.csv"

def load_seg_distance_json(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)

    rows = []
    for im in j.get("images", []):
        fn = im.get("file_name")
        regions = im.get("regions", []) or []
        for r in regions:
            rows.append({
                "file_name": fn,
                "region_id": r.get("id"),
                "class_name": r.get("class_name"),
                "excluded_low_conf": bool(r.get("excluded_low_conf", False)),
                "distance_m": r.get("distance_m", None),
            })

    df = pd.DataFrame(rows)
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["key"] = df["file_name"].astype(str) + "|" + df["region_id"].astype(str)
    return df


def merge_ref_ablation(ref_df: pd.DataFrame, abl_df: pd.DataFrame, join: str = "inner") -> pd.DataFrame:
    '''keep only evaluated objects (not excluded + has distance)'''
    ref = ref_df[(~ref_df["excluded_low_conf"]) & ref_df["distance_m"].notna()].copy()
    abl = abl_df[(~abl_df["excluded_low_conf"]) & abl_df["distance_m"].notna()].copy()

    if join != "inner":
        raise ValueError("This script supports inner only.")

    m = ref[["key", "distance_m"]].rename(columns={"distance_m": "gt"}).merge(
        abl[["key", "distance_m"]].rename(columns={"distance_m": "p"}), on="key", how="inner",
    )
    return m


def overall_metrics(m: pd.DataFrame) -> dict:
    """
    mean|e|  = mean absolute error (MAE)
    meanRel% = mean(|e| / |gt|) * 100
    RMSE (m) = sqrt(mean(e^2))
    """
    if len(m) == 0:
        return {"N": 0, "mean|e|": np.nan, "meanRel%": np.nan, "RMSE (m)": np.nan}

    e = m["gt"] - m["p"]
    abs_e = e.abs()
    mse = float((e ** 2).mean())
    rmse = float(np.sqrt(mse))  # RMSE = sqrt(MSE) :contentReference[oaicite:0]{index=0}
    meanrel = float((abs_e / np.maximum(m["gt"].abs(), EPS) * 100.0).mean())

    return {"N": int(len(m)), "mean|e|": float(abs_e.mean()), "meanRel%": meanrel, "RMSE (m)": rmse}


def main():
    ref_df = load_seg_distance_json(REF_JSON)

    rows = []
    for setting, path in ABLATIONS.items():
        abl_df = load_seg_distance_json(path)
        m = merge_ref_ablation(ref_df, abl_df, join=JOIN_STRATEGY)

        r = overall_metrics(m)
        r["setting"] = setting
        rows.append(r)

    out = pd.DataFrame(rows)[["setting", "N", "mean|e|", "meanRel%", "RMSE (m)"]].sort_values("setting")
    out.to_csv(OUT_CSV, index=False)

    # table for console
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved:", OUT_CSV)


if __name__ == "__main__":
    main()
