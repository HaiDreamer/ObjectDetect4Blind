import json
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-6  # for meanRel% denominator safety

def load_distance_json(path: str | Path) -> pd.DataFrame:
    path = Path(path)       # normalize path
    with path.open("r", encoding="utf-8") as f:
        j = json.load(f)

    # take img name with their obj with its distance
    rows = []
    for im in j.get("images", []):
        fn = im.get("file_name")
        for obj in im.get("objects", []):
            det = obj.get("distance_detail", {}) or {}
            rows.append({
                "file_name": fn,
                "obj_id": obj.get("id"),
                "class_name": obj.get("class_name"),
                "excluded_low_conf": bool(obj.get("excluded_low_conf", False)),
                "distance_m": obj.get("distance_m"),
                "valid_px": det.get("valid_px", 0),
            })

    df = pd.DataFrame(rows)     # 2d data with file_name, obj_id, class_name, excluded_low_conf, distance_m, valid_px, key as KEY
    df["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")     # convert val to number, anything not a number becomes NaN
    df["key"] = df["file_name"].astype(str) + "|" + df["obj_id"].astype(str)     
    return df

def merge_ref_ablation(ref_df: pd.DataFrame, abl_df: pd.DataFrame) -> pd.DataFrame:
    '''keep only evaluated objects (not being excluded + has distance)'''
    ref = ref_df[(~ref_df["excluded_low_conf"]) & ref_df["distance_m"].notna()].copy()
    abl = abl_df[(~abl_df["excluded_low_conf"]) & abl_df["distance_m"].notna()].copy()
    # inner join = only objects that have distance in BOTH ref and ablation
    m = ref[["key", "class_name", "distance_m"]].rename(columns={"distance_m": "gt"}).merge(
        abl[["key", "distance_m"]].rename(columns={"distance_m": "p"}), on="key", how="inner",
    )
    return m

def overall_error_row(m: pd.DataFrame) -> dict:
    # m has columns: gt, p, class_name
    if len(m) == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "MedAE": np.nan,
            "Bias_mean(gt-p)": np.nan,
            "Bias_median(gt-p)": np.nan,
            "MSE": np.nan,
            "RMSE": np.nan,
            "MeanRel%": np.nan,
            "MedRel%": np.nan,
        }

    e = m["gt"] - m["p"]
    abs_e = e.abs()
    sq_e = e ** 2
    rel_pct = abs_e / np.maximum(m["gt"].abs(), EPS) * 100.0

    mse = float(sq_e.mean())
    return {
        "N": int(len(m)),
        "MAE": float(abs_e.mean()),
        "MedAE": float(abs_e.median()),
        "Bias_mean(gt-p)": float(e.mean()),
        "Bias_median(gt-p)": float(e.median()),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),  # RMSE = sqrt(MSE)
        "MeanRel%": float(rel_pct.mean()),
        "MedRel%": float(rel_pct.median()),
    }

# per-class error
def per_class_error_table(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()
    m["e"] = m["gt"] - m["p"]
    m["abs_e"] = m["e"].abs()
    m["sq_e"] = m["e"] ** 2
    m["rel_pct"] = m["abs_e"] / np.maximum(m["gt"].abs(), EPS) * 100.0

    def agg(g):
        mse = float(g["sq_e"].mean())
        return pd.Series({
            "N": int(len(g)),
            "mean|e|": float(g["abs_e"].mean()),
            "med|e|": float(g["abs_e"].median()),
            "mean(gt-p)": float(g["e"].mean()),
            "med(gt-p)": float(g["e"].median()),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "meanRel%": float(g["rel_pct"].mean()),
            "medRel%": float(g["rel_pct"].median()),
        })

    out = m.groupby("class_name", dropna=False).apply(agg, include_groups=False).reset_index()
    return out.sort_values(["N", "class_name"], ascending=[False, True])

# Main RUNNNN
REF_100 = r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\100%_bb_distance_json_KITTI_val_GT.json"

ABLATIONS = {
    #"pixel_center": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\1pixel_bb_distance_json_KITTI_val_GT.json",
    "roi1": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\1%_bb_distance_json_KITTI_val_GT.json",
    #"roi10": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\10%_bb_distance_json_KITTI_val_GT.json",
    #"roi20": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\20%_bb_distance_json_KITTI_val_GT.json",
    #"roi30": r"C:\Python\ObjectDetect4Blind\distance_way_evaluate_report\30%_bb_distance_json_KITTI_val_GT.json",
}

ref_df = load_distance_json(REF_100)

overall_rows = []
# save per-class each setting
perclass_tables = []

for setting, path in ABLATIONS.items():
    abl_df = load_distance_json(path)
    m = merge_ref_ablation(ref_df, abl_df)

    row = overall_error_row(m)
    row["setting"] = setting
    overall_rows.append(row)

    # per-class
    # t = per_class_error_table(m)
    # t.insert(0, "setting", setting)
    # perclass_tables.append(t)

overall_df = pd.DataFrame(overall_rows).sort_values("setting")
overall_df.to_csv("ablation_overall_error_stats.csv", index=False)
print(overall_df.to_string(index=False))

# save per-class
# if perclass_tables:
#     all_perclass = pd.concat(perclass_tables, ignore_index=True)
#     all_perclass.to_csv("ablation_per_class_error_stats.csv", index=False)
