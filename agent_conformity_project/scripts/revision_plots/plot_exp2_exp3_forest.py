import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

CSV = Path(
    "analysis/recovered_main500/main6_paired_analysis/"
    "main_paired_comparisons.csv"
)

OUT = Path("results/figures/revision_final")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Same model palette as the other manuscript figures
BLUE = "#2B4C7E"
RED = "#A23B48"
PURPLE = "#6A4C93"
TEXT = "#222222"
GRID = "#D0D0D0"

df = pd.read_csv(CSV)

d = df[
    (df["condition1"].str.lower() == "exp2") &
    (df["condition2"].str.lower() == "exp3") &
    (df["metric"] == "conformity_rate")
].copy()

assert len(d) == 6, f"Expected 6 rows, found {len(d)}"

def short_model(x):
    if "Qwen" in x:
        return "Qwen2.5-3B"
    if "Phi" in x:
        return "Phi-3.5-mini"
    if "gemma" in x.lower():
        return "Gemma2-2B"
    return x

def short_dataset(x):
    if x == "CommonsenseQA":
        return "CSQA"
    return x

d["model_short"] = d["model"].map(short_model)
d["dataset_short"] = d["dataset"].map(short_dataset)
d["label"] = d["model_short"] + " / " + d["dataset_short"]

order = [
    "Qwen2.5-3B / CSQA",
    "Qwen2.5-3B / MMLU",
    "Phi-3.5-mini / CSQA",
    "Phi-3.5-mini / MMLU",
    "Gemma2-2B / CSQA",
    "Gemma2-2B / MMLU",
]

d["label"] = pd.Categorical(
    d["label"],
    categories=order,
    ordered=True
)
d = d.sort_values("label")

# Convert proportions to percentage points
d["delta_pp"] = d["delta"] * 100
d["low_pp"] = d["ci_low"] * 100
d["high_pp"] = d["ci_high"] * 100

color_map = {
    "Qwen2.5-3B": BLUE,
    "Phi-3.5-mini": RED,
    "Gemma2-2B": PURPLE,
}

y = list(range(len(d)))[::-1]

fig, ax = plt.subplots(figsize=(8.4, 5.0))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.axvline(
    0,
    color="#555555",
    linestyle="--",
    linewidth=1.2,
    zorder=1
)

for yi, (_, row) in zip(y, d.iterrows()):
    delta = row["delta_pp"]
    low = row["low_pp"]
    high = row["high_pp"]
    color = color_map[row["model_short"]]

    ax.errorbar(
        delta,
        yi,
        xerr=[[delta - low], [high - delta]],
        fmt="o",
        markersize=7.5,
        color=color,
        ecolor=color,
        elinewidth=2.0,
        capsize=4,
        capthick=1.6,
        zorder=3
    )

    # Put label outside CI so text does not overlap error bar
    if delta < 0:
        ax.text(
            low - 1.2,
            yi,
            f"{delta:+.2f}",
            ha="right",
            va="center",
            fontsize=11,
            color=color
        )
    else:
        ax.text(
            high + 1.2,
            yi,
            f"{delta:+.2f}",
            ha="left",
            va="center",
            fontsize=11,
            color=color
        )

ax.set_yticks(y)
ax.set_yticklabels(d["label"].astype(str))

ax.set_xlabel("Change in conformity rate (percentage points)")
ax.set_xlim(-82, 20)

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.7,
    alpha=0.65,
    color=GRID
)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(TEXT)
ax.spines["bottom"].set_color(TEXT)
ax.tick_params(axis="both", colors=TEXT)

fig.tight_layout(pad=0.5)

fig.savefig(
    OUT / "exp2_exp3_delta_cr_forest.pdf",
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    OUT / "exp2_exp3_delta_cr_forest.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

# Save exactly what was plotted for auditability
d[
    [
        "dataset",
        "model",
        "paired_n",
        "rate1",
        "rate2",
        "delta",
        "ci_low",
        "ci_high",
        "mcnemar_b",
        "mcnemar_c",
        "p_raw",
        "p_holm",
    ]
].to_csv(
    OUT / "exp2_exp3_delta_cr_forest_source.csv",
    index=False
)

print("===== FORMAL EXP2 -> EXP3 CR RESULTS =====")
for _, r in d.iterrows():
    print(
        f"{str(r['label']):28s} "
        f"N={int(r['paired_n']):3d}  "
        f"ΔCR={r['delta_pp']:+6.2f} pp  "
        f"95% CI=[{r['low_pp']:+6.2f}, {r['high_pp']:+6.2f}]  "
        f"Holm p={r['p_holm']:.3e}"
    )

print("\nSAVED:")
print(OUT / "exp2_exp3_delta_cr_forest.pdf")
print(OUT / "exp2_exp3_delta_cr_forest.png")
print(OUT / "exp2_exp3_delta_cr_forest_source.csv")
