import matplotlib
matplotlib.use("Agg")

import json
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("analysis/recovered_main500")
OUT = Path("results/figures/revision_final")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "Exp1": "exp1_all_at_once_final.jsonl",
    "Exp2": "exp2_sequential_context_final_only.jsonl",
    "Exp3": "exp3_sequential_answer_each_step.jsonl",
    "Exp4": "exp4_all_at_once_self_iter5.jsonl",
}

CELLS = [
    ("Qwen2.5-3B", "CSQA", ROOT / "qwen_csqa"),
    ("Qwen2.5-3B", "MMLU", ROOT / "qwen_mmlu"),
    ("Gemma2-2B", "CSQA", ROOT / "gemma_csqa"),
    ("Gemma2-2B", "MMLU", ROOT / "gemma_mmlu"),
]

BLUE = "#2B4C7E"
RED = "#A23B48"
TEXT = "#222222"
GRID = "#D0D0D0"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize(rows):
    dc = []
    dd = []

    valid = 0

    for r in rows:
        c = r.get("correct_answer")
        d = r.get("distractor")

        before = r.get("initial_option_logprobs") or {}
        after = r.get("attack_option_logprobs") or {}

        if (
            c not in before or
            c not in after or
            d not in before or
            d not in after
        ):
            continue

        # Some recovered JSONL rows store logprobs as strings or null.
        # Coerce safely and exclude only rows without finite numeric values.
        try:
            before_c = float(before[c])
            after_c = float(after[c])
            before_d = float(before[d])
            after_d = float(after[d])
        except (TypeError, ValueError):
            continue

        vals = [before_c, after_c, before_d, after_d]

        if not all(np.isfinite(v) for v in vals):
            continue

        dc.append(after_c - before_c)
        dd.append(after_d - before_d)
        valid += 1

    if valid == 0:
        raise RuntimeError("No valid option-level logprob rows.")

    return {
        "n": valid,
        "delta_correct": float(np.mean(dc)),
        "delta_target": float(np.mean(dd)),
        "se_correct": float(np.std(dc, ddof=1) / np.sqrt(valid)),
        "se_target": float(np.std(dd, ddof=1) / np.sqrt(valid)),
    }


summary_rows = []

for model, dataset, directory in CELLS:
    for exp, fn in FILES.items():
        path = directory / fn

        if not path.exists():
            raise FileNotFoundError(path)

        rows = load_rows(path)
        s = summarize(rows)

        summary_rows.append({
            "model": model,
            "dataset": dataset,
            "experiment": exp,
            **s,
        })


# Audit table
csv_path = OUT / "logprob_shift_main500_source.csv"

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "dataset",
            "experiment",
            "n",
            "delta_correct",
            "delta_target",
            "se_correct",
            "se_target",
        ],
    )
    writer.writeheader()
    writer.writerows(summary_rows)


print("===== MAIN500 OPTION-LEVEL LOGPROB SHIFT =====")

for r in summary_rows:
    print(
        f"{r['model']:12s} {r['dataset']:4s} {r['experiment']} "
        f"N={r['n']:3d}  "
        f"Δlogp(C)={r['delta_correct']:+7.3f}  "
        f"Δlogp(D)={r['delta_target']:+7.3f}"
    )


fig, axes = plt.subplots(
    2, 2,
    figsize=(8.2, 6.2),
    sharex=True,
    sharey=True,
)

fig.patch.set_facecolor("white")

exp_order = ["Exp1", "Exp2", "Exp3", "Exp4"]
x = np.arange(4)

for ax, (model, dataset, _) in zip(axes.flat, CELLS):

    subset = [
        r for r in summary_rows
        if r["model"] == model and r["dataset"] == dataset
    ]

    lookup = {r["experiment"]: r for r in subset}

    correct = [lookup[e]["delta_correct"] for e in exp_order]
    target = [lookup[e]["delta_target"] for e in exp_order]

    correct_ci = [1.96 * lookup[e]["se_correct"] for e in exp_order]
    target_ci = [1.96 * lookup[e]["se_target"] for e in exp_order]

    ax.axhline(
        0,
        color="#666666",
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )

    ax.errorbar(
        x,
        correct,
        yerr=correct_ci,
        marker="o",
        markersize=6.0,
        linewidth=2.0,
        capsize=3,
        color=BLUE,
        label="Correct option",
    )

    ax.errorbar(
        x,
        target,
        yerr=target_ci,
        marker="s",
        markersize=6.0,
        linewidth=2.0,
        capsize=3,
        color=RED,
        label="Target distractor",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(exp_order)
    ax.set_ylim(-8, 13)
    ax.set_yticks([-5, 0, 5, 10])

    # Panel identifier inside the axes, not a title above it
    ax.text(
        0.03,
        0.95,
        f"{model} / {dataset}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color=TEXT,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.65,
        color=GRID,
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)

    ax.tick_params(axis="both", colors=TEXT)


axes[0, 0].set_ylabel("Mean log-probability shift")
axes[1, 0].set_ylabel("Mean log-probability shift")

handles, labels = axes[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    frameon=False,
    ncol=2,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
)

fig.subplots_adjust(
    left=0.10,
    right=0.99,
    top=0.98,
    bottom=0.12,
    hspace=0.20,
    wspace=0.15,
)

fig.savefig(
    OUT / "logprob_shift_main500.pdf",
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    OUT / "logprob_shift_main500.png",
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

print("\nSAVED:")
print(OUT / "logprob_shift_main500.pdf")
print(OUT / "logprob_shift_main500.png")
print(csv_path)
