import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# OUTPUT
# =========================
OUT_DIR = "plots_table5"
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def add_labels(ax, bars, fmt="{:.3f}", fontsize=11, ypad=0.0):
    """Add value labels on top of bars (same style as your code)."""
    for b in bars:
        h = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h + ypad,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="bold",
        )

# =========================
# TABLE 5.7: Speed & Memory
# =========================
models_57 = ["Original\nFP32", "ONNX\nFP16", "ONNX\nINT8", "Pruned\n1-layer"]
x57 = np.arange(len(models_57))

avg_time_s = [2.969, 0.394, 0.290, 3.678]          # Avg time (s/img)
speed_vs_fp32 = [1.00, 7.54, 10.24, 0.81]          # Speed vs FP32 (x)
memory_mb = [94.6, 48.5, 34.7, 87.8]               # Memory (MB)

fig, axs = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Speed & Memory Comparison vs Original (FP32)", fontsize=16)

b1 = axs[0].bar(x57, avg_time_s)
axs[0].set_title("Avg Time (s/img)")
axs[0].set_xticks(x57)
axs[0].set_xticklabels(models_57)
axs[0].set_ylim(0, max(avg_time_s) * 1.20)
axs[0].grid(axis="y", linestyle="--", alpha=0.5)
add_labels(axs[0], b1, fmt="{:.3f}", fontsize=11)

b2 = axs[1].bar(x57, speed_vs_fp32)
axs[1].set_title("Speed vs FP32 (×)")
axs[1].set_xticks(x57)
axs[1].set_xticklabels(models_57)
axs[1].set_ylim(0, max(speed_vs_fp32) * 1.20)
axs[1].grid(axis="y", linestyle="--", alpha=0.5)
add_labels(axs[1], b2, fmt="{:.2f}×", fontsize=11)

b3 = axs[2].bar(x57, memory_mb)
axs[2].set_title("Memory (MB)")
axs[2].set_xticks(x57)
axs[2].set_xticklabels(models_57)
axs[2].set_ylim(0, max(memory_mb) * 1.20)
axs[2].grid(axis="y", linestyle="--", alpha=0.5)
add_labels(axs[2], b3, fmt="{:.1f}", fontsize=11)

save_fig(fig, "table_5_7_speed_memory.png")

# =========================
# TABLE 5.8: Per-distance error metrics (split into 3 figures)
# =========================
bins = ["[0, 10)", "[10, 20)", "[20, 40)", "[40, 80)"]
x58 = np.arange(len(bins))
w = 0.20  # 4 models -> 0.20 works well

models_58 = ["Original", "FP16 ONNX", "INT8 ONNX", "Pruned 1-layer"]

mean_abs_e = np.array([
    [0.719, 1.589, 3.029, 7.296],  # Original
    [0.748, 1.950, 4.156, 8.893],  # FP16 ONNX
    [0.811, 2.031, 4.410, 9.536],  # INT8 ONNX
    [1.992, 3.047, 2.957, 9.790],  # Pruned 1-layer
])

mean_rel = np.array([
    [11.62, 11.09, 11.10, 14.03],
    [11.98, 13.53, 15.11, 17.46],
    [12.87, 14.10, 16.03, 18.91],
    [32.93, 21.82, 11.06, 18.32],
])

rmse = np.array([
    [1.344, 2.558, 4.287, 10.527],
    [1.252, 2.815, 5.428, 12.273],
    [1.307, 2.818, 5.663, 12.781],
    [2.613, 3.711, 3.992, 13.073],
])

def plot_grouped_metric_figure(
    data, title, ylabel, fmt, out_name,
    ylim_pad=1.20, label_fontsize=6, ypad_frac=0.02
):
    n_models = data.shape[0]
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * w

    fig, ax = plt.subplots(figsize=(7.5, 5))
    fig.suptitle("Per-distance Range Error Metrics", fontsize=16)

    # push labels higher by a fixed fraction of chart max (data units)
    ypad = ypad_frac * data.max()

    for i in range(n_models):
        bars = ax.bar(x58 + offsets[i], data[i], w, label=models_58[i])
        add_labels(ax, bars, fmt=fmt, fontsize=label_fontsize, ypad=ypad)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x58)
    ax.set_xticklabels(bins)
    ax.set_ylim(0, data.max() * ylim_pad)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)

    save_fig(fig, out_name)

plot_grouped_metric_figure(
    mean_abs_e,
    title="mean|e| (m)",
    ylabel="Meters",
    fmt="{:.3f}",
    out_name="table_5_8_mean_abs_e.png",
)

plot_grouped_metric_figure(
    mean_rel,
    title="meanRel (%)",
    ylabel="Percent",
    fmt="{:.2f}",
    out_name="table_5_8_mean_rel.png",
)

plot_grouped_metric_figure(
    rmse,
    title="RMSE (m)",
    ylabel="Meters",
    fmt="{:.3f}",
    out_name="table_5_8_rmse.png",
)


# =========================
# TABLE 5.9: Per-class mean absolute error
# (split into 2 panels for readability)
# =========================
classes = [
    "Car", "Bicycle", "Bus", "Person", "Truck", "Electric pole", "Motorcycle",
    "Ped. sign", "Tree", "Crosswalk", "Sidewalk", "Stairs", "Tree-lined"
]

orig =  [1.837, 0.690, 0.799, 2.154, 1.571, 2.751, 1.011, 1.929, 4.950, 1.594, 0.575, 2.367, 0.677]
fp16 =  [2.427, 1.196, 2.155, 2.973, 1.581, 2.516, 0.788, 3.252, 4.351, 0.401, 0.225, 2.657, 0.696]
int8 =  [2.596, 1.210, 2.296, 3.014, 1.513, 2.945, 0.752, 3.086, 4.454, 0.548, 0.233, 2.528, 0.800]
prun =  [2.772, 1.720, 2.526, 4.703, 2.031, 4.797, 2.521, 3.194, 5.017, 0.037, 0.333, 3.185, 0.974]

series_59 = {
    "Original": np.array(orig),
    "FP16 ONNX": np.array(fp16),
    "INT8 ONNX": np.array(int8),
    "Pruned 1-layer": np.array(prun),
}

# split for readability
split_idx = 7
classes_a, classes_b = classes[:split_idx], classes[split_idx:]

fig, axs = plt.subplots(2, 1, figsize=(18, 10), sharey=True)
fig.suptitle("Per-class mean|e| (m)", fontsize=16)

def plot_class_panel(ax, cls_list, start_idx):
    x = np.arange(len(cls_list))
    w2 = 0.20
    keys = list(series_59.keys())
    offsets = (np.arange(len(keys)) - (len(keys) - 1) / 2) * w2

    for i, k in enumerate(keys):
        vals = series_59[k][start_idx:start_idx + len(cls_list)]
        bars = ax.bar(x + offsets[i], vals, w2, label=k)
        add_labels(ax, bars, fmt="{:.3f}", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cls_list, rotation=25, ha="right")
    ax.set_ylabel("Meters")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plot_class_panel(axs[0], classes_a, start_idx=0)
axs[0].set_title("Classes (1/2)")
axs[0].legend(loc="upper left")

plot_class_panel(axs[1], classes_b, start_idx=split_idx)
axs[1].set_title("Classes (2/2)")
axs[1].legend(loc="upper left")

# a bit of headroom for labels
ymax_59 = max(max(orig), max(fp16), max(int8), max(prun))
axs[0].set_ylim(0, ymax_59 * 1.20)

save_fig(fig, "table_5_9_per_class_mae.png")

print("✅ Done. All figures saved in:", OUT_DIR)
