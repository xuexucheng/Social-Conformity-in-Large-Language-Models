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
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Unified paper palette
blue = "#2B4C7E"
red = "#A23B48"
purple = "#6A4C93"
text_color = "#222222"
grid_color = "#D0D0D0"

protocols = ["Exp1", "Exp2", "Exp3", "Exp4"]
x = np.arange(4)

# Four-protocol shared-valid-subset CR (%)
csqa = {
    "Qwen2.5-3B": [24.4, 58.8, 3.0, 17.5],
    "Phi-3.5-mini": [67.3, 81.7, 15.3, 50.7],
    "Gemma2-2B": [2.4, 4.5, 11.1, 5.6],
}

mmlu = {
    "Qwen2.5-3B": [42.3, 55.2, 4.9, 34.6],
    "Phi-3.5-mini": [36.6, 67.2, 28.5, 33.3],
    "Gemma2-2B": [1.6, 2.8, 12.2, 4.2],
}

styles = {
    "Qwen2.5-3B": dict(color=blue, marker="o"),
    "Phi-3.5-mini": dict(color=red, marker="s"),
    "Gemma2-2B": dict(color=purple, marker="^"),
}

fig, axes = plt.subplots(
    1, 2,
    figsize=(8.2, 3.9),
    sharey=True
)

fig.patch.set_facecolor("white")

for ax, data, dataset in zip(
    axes,
    [csqa, mmlu],
    ["CSQA", "MMLU"]
):
    ax.set_facecolor("white")

    for model, values in data.items():
        ax.plot(
            x,
            values,
            linewidth=2.3,
            markersize=6.5,
            label=model,
            **styles[model]
        )

    ax.set_xticks(x)
    ax.set_xticklabels(protocols)

    # Dataset name is an axis label, not a panel title
    ax.set_xlabel(dataset, labelpad=7, color=text_color)

    ax.set_ylim(0, 90)
    ax.set_yticks([0, 20, 40, 60, 80])

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.8,
        color=grid_color
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(text_color)
    ax.spines["bottom"].set_color(text_color)

    ax.tick_params(
        axis="both",
        colors=text_color,
        width=1.0,
        length=4
    )

axes[0].set_ylabel(
    "Conformity rate (%)",
    color=text_color,
    labelpad=8
)

# One shared legend
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    frameon=False,
    ncol=3,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.03),
    handlelength=2.2,
    columnspacing=1.7
)

fig.subplots_adjust(
    left=0.10,
    right=0.99,
    top=0.97,
    bottom=0.25,
    wspace=0.15
)

fig.savefig(
    OUT / "main_protocol_cr.pdf",
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT / "main_protocol_cr.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

print("SAVED:")
print(OUT / "main_protocol_cr.pdf")
print(OUT / "main_protocol_cr.png")
