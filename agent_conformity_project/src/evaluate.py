import json
import os
import shutil

from src.config import RESULT_DIR, RUN_DIR, RUN_NAME
from src.metrics import (
    build_attack_comparison,
    build_defense_comparison,
    compute_clean_metrics,
    compute_attack_metrics,
    compute_defense_metrics,
)
from src.social_experiment import INPUT_MODES


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    clean_path = os.path.join(RESULT_DIR, "results_clean.jsonl")
    clean_rows = load_jsonl(clean_path)
    attack_metrics = {}
    defense_metrics = {}
    for input_mode in INPUT_MODES:
        attack_path = os.path.join(RESULT_DIR, f"results_attack_{input_mode}.jsonl")
        defense_path = os.path.join(RESULT_DIR, f"results_defense_{input_mode}.jsonl")
        attack_metrics[input_mode] = compute_attack_metrics(load_jsonl(attack_path))
        defense_metrics[input_mode] = compute_defense_metrics(load_jsonl(defense_path))

    metrics = {
        "run_name": RUN_NAME,
        "clean": compute_clean_metrics(clean_rows),
        "attack": attack_metrics,
        "attack_comparison": build_attack_comparison(attack_metrics),
        "defense": defense_metrics,
        "defense_comparison": build_defense_comparison(defense_metrics),
    }

    latest_metrics_path = os.path.join(RESULT_DIR, "metrics.json")
    run_metrics_path = os.path.join(RUN_DIR, "metrics.json")

    with open(latest_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    shutil.copy2(latest_metrics_path, run_metrics_path)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[OK] metrics saved to {latest_metrics_path}")
    print(f"[OK] metrics archived to {run_metrics_path}")


if __name__ == "__main__":
    main()
