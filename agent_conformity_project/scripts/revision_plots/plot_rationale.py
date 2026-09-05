import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/figures/revision_final")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

blue = "#2B4C7E"
red = "#A23B48"
purple = "#6A4C93"
text_color = "#222222"
grid_color = "#D0D0D0"

labels = ["Bare", "Rationale"]
values = [57.60, 85.44]

fig, ax = plt.subplots(figsize=(4.3, 4.1))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bars = ax.bar(
    labels,
    values,
    width=0.55,
    color=[blue, red]
)

ax.set_ylabel("Conformity rate (%)", color=text_color, labelpad=8)
ax.set_ylim(0, 100)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(text_color)
ax.spines["bottom"].set_color(text_color)

ax.tick_params(axis="both", colors=text_color)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.8,
    color=grid_color
)
ax.set_axisbelow(True)

for bar, value in zip(bars, values):
    ax.annotate(
        f"{value:.1f}",
        (bar.get_x() + bar.get_width()/2, value),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        color=purple
    )

# Effect annotation, not a title
ax.text(
    0.5,
    94,
    r"$\Delta$CR = +27.84 pp",
    ha="center",
    va="center",
    fontsize=11,
    color=purple
)

fig.tight_layout(pad=0.5)

fig.savefig(
    OUT / "rationale_qwen3b_csqa.pdf",
    bbox_inches="tight",
    facecolor="white"
)
fig.savefig(
    OUT / "rationale_qwen3b_csqa.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print("SAVED:")
print(OUT / "rationale_qwen3b_csqa.pdf")
print(OUT / "rationale_qwen3b_csqa.png")
