import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from pathlib import Path

df = pd.DataFrame({
    "Model": ["Original", "ONNX FP16", "ONNX INT8", "Pruned 1-layer"],
    "Avg time (s/img)": [2.969, 0.394, 0.290, 3.678],
    "Speed vs FP32": [1.00, 7.54, 10.24, 0.81],
    "Memory (MB)": [94.6, 48.5, 34.7, 87.8],
})

out_dir = Path(r"C:\Python\ObjectDetect4Blind\visualize\output"); out_dir.mkdir(exist_ok=True)

t0 = df.loc[df["Model"] == "Original", "Avg time (s/img)"].iloc[0]
m0 = df.loc[df["Model"] == "Original", "Memory (MB)"].iloc[0]

df_plot = pd.DataFrame({
    "Model": df["Model"],
    "Latency gain (x)": t0 / df["Avg time (s/img)"],
    "Memory gain (x)": m0 / df["Memory (MB)"],
    "Speedup vs FP32 (x)": df["Speed vs FP32"],
})

fig, ax = plt.subplots(figsize=(8.5, 4.8))
parallel_coordinates(df_plot, "Model", ax=ax)  # không set màu -> dùng mặc định
ax.set_title("Normalized comparison (higher is better)")
ax.set_ylabel("Gain (×)")
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

fig.savefig(out_dir / "quant_parallel_coordinates.png", dpi=300, bbox_inches="tight")
fig.savefig(out_dir / "quant_parallel_coordinates.pdf", bbox_inches="tight")
plt.close(fig)
