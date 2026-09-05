import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("results/figures/revision_final")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Audited Qwen2.5-3B / CSQA direct CR (%)
labels = [
    "Sequential-Final",
    "No-History",
    "Answer-History",
    "Answer+Confidence",
    "Length-Matched",
    "Neutral-V2",
    "Short-Ack",
]

values = [
    57.60,
    57.60,
    20.56,
    5.35,
    8.35,
    14.35,
    54.82,
]

# Restrained paper palette
blue = "#2B4C7E"
red = "#A23B48"
purple = "#6A4C93"
text_color = "#222222"
grid_color = "#D0D0D0"

# First four = retained-history decomposition
# Last three = controls
colors = [
    blue,
    blue,
    purple,
    purple,
    red,
    red,
    red,
]

y = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(7.0, 4.6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.barh(
    y,
    values,
    height=0.62,
    color=colors
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlabel("Conformity rate (%)")
ax.set_xlim(0, 65)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(text_color)
ax.spines["bottom"].set_color(text_color)

ax.tick_params(axis="both", colors=text_color)

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7,
    color=grid_color
)
ax.set_axisbelow(True)

for bar, value in zip(bars, values):
    ax.text(
        value + 1.0,
        bar.get_y() + bar.get_height()/2,
        f"{value:.1f}",
        va="center",
        ha="left",
        fontsize=10,
        color=text_color
    )

# Separate primary decomposition from additional controls
ax.axhline(
    3.5,
    linestyle=":",
    linewidth=1.0,
    color="#999999"
)

fig.tight_layout(pad=0.5)

fig.savefig(
    OUT / "qwen3b_csqa_7condition_cr.pdf",
    bbox_inches="tight",
    facecolor="white"
)
fig.savefig(
    OUT / "qwen3b_csqa_7condition_cr.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print("SAVED:")
print(OUT / "qwen3b_csqa_7condition_cr.pdf")
print(OUT / "qwen3b_csqa_7condition_cr.png")
