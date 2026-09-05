import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("results/figures/revision_final")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

steps = [1, 2, 3, 4, 5]

no_history = [1.07, 52.25, 66.38, 60.81, 57.60]
answer_history = [1.07, 30.84, 38.12, 23.13, 20.56]
answer_conf = [1.07, 12.63, 14.78, 7.49, 5.35]

blue = "#2B4C7E"
red = "#A23B48"
purple = "#6A4C93"
text_color = "#222222"
grid_color = "#D0D0D0"

fig, ax = plt.subplots(figsize=(6.4, 4.3))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.plot(
    steps,
    no_history,
    marker="o",
    markersize=7,
    linewidth=2.4,
    color=blue,
    label="No-History"
)

ax.plot(
    steps,
    answer_history,
    marker="s",
    markersize=7,
    linewidth=2.4,
    color=red,
    label="Answer-History"
)

ax.plot(
    steps,
    answer_conf,
    marker="^",
    markersize=7,
    linewidth=2.4,
    color=purple,
    label="Answer+Confidence"
)

ax.set_xlabel("Peer-evidence step")
ax.set_ylabel("Conformity rate (%)")

ax.set_xticks(steps)
ax.set_xlim(0.8, 5.2)
ax.set_ylim(0, 75)
ax.set_yticks([0, 20, 40, 60])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(text_color)
ax.spines["bottom"].set_color(text_color)

ax.tick_params(axis="both", colors=text_color)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.7,
    color=grid_color
)
ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    loc="upper left"
)

# Final-step values only; avoid cluttering every point
for value, color in [
    (no_history[-1], blue),
    (answer_history[-1], red),
    (answer_conf[-1], purple),
]:
    ax.text(
        5.08,
        value,
        f"{value:.1f}",
        va="center",
        ha="left",
        fontsize=10,
        color=color
    )

fig.tight_layout(pad=0.5)

fig.savefig(
    OUT / "stepwise_trajectory_qwen3b_csqa.pdf",
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT / "stepwise_trajectory_qwen3b_csqa.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print("SAVED:")
print(OUT / "stepwise_trajectory_qwen3b_csqa.pdf")
print(OUT / "stepwise_trajectory_qwen3b_csqa.png")
