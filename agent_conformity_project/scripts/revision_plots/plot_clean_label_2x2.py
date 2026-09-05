import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("results/figures/revision_final")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Same restrained palette as the revised peer-count figure
blue = "#2B4C7E"
red = "#A23B48"
purple = "#6A4C93"
text_color = "#222222"
grid_color = "#D0D0D0"

groups = [
    "Qwen2.5-3B\nCSQA",
    "Gemma2-2B\nCSQA",
    "Qwen2.5-3B\nMMLU",
    "Gemma2-2B\nMMLU",
]

# Primary direct CR
source = np.array([87.79, 70.51, 74.83, 41.88])
model  = np.array([13.70, 11.11, 34.18,  6.82])

x = np.arange(len(groups))
width = 0.32

fig, ax = plt.subplots(figsize=(6.8, 4.3))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars_source = ax.bar(
    x - width/2,
    source,
    width,
    color=red,
    label="Source"
)

bars_model = ax.bar(
    x + width/2,
    model,
    width,
    color=blue,
    label="Model"
)

ax.set_ylabel("Conformity rate (%)", color=text_color, labelpad=8)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylim(0, 100)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(text_color)
ax.spines["bottom"].set_color(text_color)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

ax.tick_params(axis="both", colors=text_color, width=1.0, length=4)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.8,
    color=grid_color
)
ax.set_axisbelow(True)

leg = ax.legend(
    frameon=False,
    ncol=2,
    loc="upper right"
)
for t in leg.get_texts():
    t.set_color(text_color)

# Numeric labels
for bars in [bars_source, bars_model]:
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width()/2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=purple
        )

fig.tight_layout(pad=0.5)

fig.savefig(
    OUT / "clean_label_2x2_cr.pdf",
    bbox_inches="tight",
    facecolor="white"
)
fig.savefig(
    OUT / "clean_label_2x2_cr.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print("SAVED:")
print(OUT / "clean_label_2x2_cr.pdf")
print(OUT / "clean_label_2x2_cr.png")
