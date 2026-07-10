import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your CSV file
csv_path = r"C:\Python\ObjectDetectRequireFile\ketquatrainlan2\results.csv"

# Read CSV
df = pd.read_csv(csv_path)

# 4 metrics for object detection report
metrics = [
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
    "metrics/precision(B)",
    "metrics/recall(B)"
]

# Output image path
output_path = Path(csv_path).parent / "detection_metrics.png"

# Create 2x2 subplot figure
plt.figure(figsize=(12, 8))

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.plot(df["epoch"], df[metric], marker="o", markersize=3, linewidth=2)
    plt.title(metric.replace("metrics/", "").replace("(B)", "").strip(), fontsize=11)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Mark best value
    best_idx = df[metric].idxmax()
    best_epoch = df.loc[best_idx, "epoch"]
    best_value = df.loc[best_idx, metric]

    plt.scatter(best_epoch, best_value, s=50)
    # plt.annotate(
    #     f"Best: {best_value:.4f}\nEpoch: {int(best_epoch)}",
    #     (best_epoch, best_value),
    #     textcoords="offset points",
    #     xytext=(8, -18),
    #     fontsize=9
    # )

plt.suptitle("Object Detection Metrics", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save image
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved image to: {output_path}")