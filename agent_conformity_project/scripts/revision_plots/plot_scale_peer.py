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
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

peers = [1, 3, 5]
qwen3b = [1.07, 66.38, 57.60]
qwen7b = [85.89, 100.00, 100.00]

# 更接近论文风格：深蓝、酒红、暗紫
blue = "#2B4C7E"
red = "#A23B48"
purple = "#6A4C93"
text_color = "#222222"
grid_color = "#D0D0D0"

fig, ax = plt.subplots(figsize=(5.8, 4.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    peers, qwen3b,
    marker="o",
    markersize=7.5,
    linewidth=2.6,
    color=blue,
    label="Qwen2.5-3B"
)

ax.plot(
    peers, qwen7b,
    marker="s",
    markersize=7.5,
    linewidth=2.6,
    color=red,
    label="Qwen2.5-7B"
)

ax.set_xlabel("Number of wrong peers", color=text_color, labelpad=8)
ax.set_ylabel("Conformity rate (%)", color=text_color, labelpad=8)

ax.set_xticks([1, 3, 5])
ax.set_xlim(0.85, 5.15)
ax.set_ylim(0, 105)
ax.margins(x=0.02, y=0.04)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)
ax.spines["left"].set_color(text_color)
ax.spines["bottom"].set_color(text_color)
ax.tick_params(axis="both", colors=text_color, width=1.0, length=4)

ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.8, color=grid_color)

leg = ax.legend(
    frameon=False,
    loc="lower right",
    handlelength=2.2
)
for t in leg.get_texts():
    t.set_color(text_color)

for x, y in zip(peers, qwen3b):
    ax.annotate(
        f"{y:.1f}",
        (x, y),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=11,
        color=purple
    )

for x, y in zip(peers, qwen7b):
    offset = -18 if y >= 99 else 8
    va = "top" if y >= 99 else "bottom"
    ax.annotate(
        f"{y:.1f}",
        (x, y),
        xytext=(0, offset),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=11,
        color=purple
    )

fig.tight_layout(pad=0.5)

fig.savefig(
    OUT / "scale_peer_qwen_csqa.pdf",
    bbox_inches="tight",
    facecolor="white"
)
fig.savefig(
    OUT / "scale_peer_qwen_csqa.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print("SAVED:")
print(OUT / "scale_peer_qwen_csqa.pdf")
print(OUT / "scale_peer_qwen_csqa.png")
