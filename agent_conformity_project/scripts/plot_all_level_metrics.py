import json
import os

import matplotlib.pyplot as plt

from src.config import RESULT_DIR


METRICS_TO_PLOT = [
    ("attack_accuracy", "Attack Accuracy"),
    ("conformity_rate", "Conformity Rate"),
    ("wrong_conformity_rate", "Wrong Conformity Rate"),
    ("change_rate", "Change Rate"),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_single_run(levels, values, ylabel, title, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(levels, values, marker="o")
    plt.xlabel("Attack Level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_multi_run(levels, run_to_values, ylabel, title, save_path):
    plt.figure(figsize=(8, 5))
    for run_name, values in run_to_values.items():
        plt.plot(levels, values, marker="o", label=run_name)
    plt.xlabel("Attack Level")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    level_analysis_dir = os.path.join(RESULT_DIR, "level_analysis")
    combined_json_path = os.path.join(level_analysis_dir, "all_runs_attack_level_metrics.json")
    combined = load_json(combined_json_path)

    # 每个 run 单独画图
    for run_name, data in combined.items():
        levels = [x["attack_level"] for x in data]
        plot_dir = os.path.join(level_analysis_dir, run_name, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        for metric_key, metric_label in METRICS_TO_PLOT:
            values = [x[metric_key] for x in data]
            save_path = os.path.join(plot_dir, f"{metric_key}_by_level.png")
            plot_single_run(
                levels,
                values,
                metric_label,
                f"{metric_label} by Attack Level ({run_name})",
                save_path,
            )

    # 所有 run 统一画对比图
    combined_plot_dir = os.path.join(level_analysis_dir, "plots_combined")
    os.makedirs(combined_plot_dir, exist_ok=True)

    run_names = sorted(
        combined.keys(),
        key=lambda x: int(x.split("_n")[-1]) if "_n" in x else 10**9
    )

    if run_names:
        levels = [x["attack_level"] for x in combined[run_names[0]]]

        for metric_key, metric_label in METRICS_TO_PLOT:
            run_to_values = {
                run_name: [x[metric_key] for x in combined[run_name]]
                for run_name in run_names
            }
            save_path = os.path.join(combined_plot_dir, f"{metric_key}_comparison.png")
            plot_multi_run(
                levels,
                run_to_values,
                metric_label,
                f"{metric_label} by Attack Level Across Dataset Sizes",
                save_path,
            )

    print(f"[OK] saved per-run plots and combined plots under {level_analysis_dir}")


if __name__ == "__main__":
    main()