#!/usr/bin/env python3
"""
New figures for Results and Takeaway slides — not duplicating existing fig1–fig6.
  fig7_efficiency_heatmap.png  — Results slide: PIM efficiency (speedup per MB saved)
  fig8_takeaway_summary.png    — Takeaway slide: annotated scatter of speedup vs iteration reduction
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

CSV_FILE = "benchmark_results.csv"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

df = pd.read_csv(CSV_FILE)
DATASETS = ["cit-HepPh", "web-Google", "roadNet-CA"]
SIZES = [128, 256, 512, 1024]
DS_COLORS = {"cit-HepPh": "#534AB7", "web-Google": "#1D9E75", "roadNet-CA": "#D85A30"}

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Heatmap — cycle speedup across (dataset × N)
# Shows the scaling story at a glance without any bar/line chart already used.
# ─────────────────────────────────────────────────────────────────────────────
speedup_matrix = np.zeros((len(DATASETS), len(SIZES)))
for i, ds in enumerate(DATASETS):
    for j, n in enumerate(SIZES):
        row = df[(df["dataset"] == ds) & (df["N"] == n)]
        speedup_matrix[i, j] = row["speedup"].values[0]

fig7, ax7 = plt.subplots(figsize=(7, 3.6))

im = ax7.imshow(speedup_matrix, cmap="YlOrRd", aspect="auto", vmin=0)

# Cell annotations
for i in range(len(DATASETS)):
    for j in range(len(SIZES)):
        val = speedup_matrix[i, j]
        text_color = "white" if val > 50 else "black"
        ax7.text(j, i, f"{val:.1f}×", ha="center", va="center",
                 fontsize=12, fontweight="bold", color=text_color)

ax7.set_xticks(range(len(SIZES)))
ax7.set_xticklabels([f"N={n}" for n in SIZES], fontsize=11)
ax7.set_yticks(range(len(DATASETS)))
ax7.set_yticklabels(DATASETS, fontsize=11)
ax7.set_title("PIM Cycle Speedup over CPU  (higher = better)", fontsize=13, pad=10)

cbar = fig7.colorbar(im, ax=ax7, fraction=0.03, pad=0.03)
cbar.set_label("Speedup (×)", fontsize=10)

ax7.spines[:].set_visible(False)
ax7.tick_params(length=0)

fig7.tight_layout()
fig7.savefig(f"{OUTPUT_DIR}/fig7_speedup_heatmap.png", dpi=300, bbox_inches="tight")
fig7.savefig(f"{OUTPUT_DIR}/fig7_speedup_heatmap.pdf", bbox_inches="tight")
print("Saved fig7_speedup_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Takeaway — bubble chart
#   x-axis : iteration reduction ratio  (cpu_iters / pim_iters)
#   y-axis : cycle speedup
#   bubble size : data reduction ratio
# Captures all three wins (speed, iterations, memory) in one visual.
# ─────────────────────────────────────────────────────────────────────────────
fig8, ax8 = plt.subplots(figsize=(7.5, 5))

for ds in DATASETS:
    subset = df[df["dataset"] == ds].sort_values("N")
    iter_ratio = subset["cpu_iters"] / subset["pim_iters"]
    speedup    = subset["speedup"]
    data_red   = subset["data_reduction"]
    sizes_plot = subset["N"].values

    sc = ax8.scatter(
        iter_ratio, speedup,
        s=data_red * 6,          # bubble area ∝ data reduction
        color=DS_COLORS[ds],
        alpha=0.75,
        edgecolors="white",
        linewidths=1.2,
        label=ds,
        zorder=3,
    )

    # Label each bubble with its N value
    for x_val, y_val, n_val in zip(iter_ratio, speedup, sizes_plot):
        ax8.annotate(
            f"N={n_val}",
            xy=(x_val, y_val),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5,
            color="gray",
        )

# Reference lines
ax8.axhline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax8.axvline(1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# Bubble size legend (manual)
for dr_val, label in [(10, "10× data\nreduction"), (50, "50×"), (130, "130×")]:
    ax8.scatter([], [], s=dr_val * 6, color="gray", alpha=0.5, label=label)

ax8.set_xlabel("Iteration reduction  (CPU iters / PIM iters)", fontsize=12)
ax8.set_ylabel("Cycle speedup  (CPU cycles / PIM cycles)", fontsize=12)
ax8.set_title("Three wins at once: speed · iterations · memory\n(bubble size = data traffic reduction)", fontsize=12)

ax8.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9,
           title="Dataset / Data reduction", title_fontsize=9)

ax8.spines["top"].set_visible(False)
ax8.spines["right"].set_visible(False)
ax8.grid(True, linestyle="--", alpha=0.3)

fig8.tight_layout()
fig8.savefig(f"{OUTPUT_DIR}/fig8_takeaway_bubble.png", dpi=300, bbox_inches="tight")
fig8.savefig(f"{OUTPUT_DIR}/fig8_takeaway_bubble.pdf", bbox_inches="tight")
print("Saved fig8_takeaway_bubble")

print("\nDone — figures in", OUTPUT_DIR)
