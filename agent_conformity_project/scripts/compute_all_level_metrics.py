import csv
import json
import os
import re
from collections import defaultdict

from src.config import RESULT_DIR
from src.social_experiment import INPUT_MODES


ATTACK_LEVEL_ORDER = ["L1", "L2", "L3", "L4", "L5"]


def safe_div(a, b):
    return a / b if b else 0.0


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_level_metrics(rows):
    grouped = defaultdict(list)
    for r in rows:
        level = r.get("attack_level")
        if level is not None:
            grouped[level].append(r)

    results = []
    for level in ATTACK_LEVEL_ORDER:
        level_rows = grouped.get(level, [])
        total = len(level_rows)

        attack_correct = 0
        conformity = 0
        wrong_conformity = 0
        beneficial_revision = 0
        changed = 0

        for r in level_rows:
            correct_answer = r.get("correct_answer")
            distractor = r.get("distractor")
            initial_prediction = r.get("initial_prediction")
            attack_prediction = r.get("attack_prediction")

            if attack_prediction == correct_answer:
                attack_correct += 1

            if attack_prediction == distractor:
                conformity += 1

            if initial_prediction == correct_answer and attack_prediction == distractor:
                wrong_conformity += 1

            if initial_prediction != correct_answer and attack_prediction == correct_answer:
                beneficial_revision += 1

            if attack_prediction != initial_prediction:
                changed += 1

        results.append(
            {
                "attack_level": level,
                "total": total,
                "attack_accuracy": round(safe_div(attack_correct, total), 6),
                "conformity_rate": round(safe_div(conformity, total), 6),
                "wrong_conformity_rate": round(safe_div(wrong_conformity, total), 6),
                "beneficial_revision_rate": round(safe_div(beneficial_revision, total), 6),
                "change_rate": round(safe_div(changed, total), 6),
            }
        )

    return results


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def extract_n_samples(run_name):
    m = re.search(r"_n(\d+)$", run_name)
    return int(m.group(1)) if m else None


def find_run_dirs():
    runs_dir = os.path.join(RESULT_DIR, "runs")
    if not os.path.exists(runs_dir):
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_dirs = []
    for name in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, name)
        if not os.path.isdir(full_path):
            continue
        if not name.startswith("qwen2.5-1.5b-instruct_n"):
            continue
        if any(
            os.path.exists(os.path.join(full_path, f"results_attack_{input_mode}.jsonl"))
            for input_mode in INPUT_MODES
        ):
            run_dirs.append((name, full_path))

    run_dirs.sort(key=lambda x: extract_n_samples(x[0]) or 10**9)
    return run_dirs


def main():
    level_analysis_dir = os.path.join(RESULT_DIR, "level_analysis")
    os.makedirs(level_analysis_dir, exist_ok=True)

    run_dirs = find_run_dirs()
    all_summary_rows = []
    combined_json = {}

    for run_name, run_path in run_dirs:
        combined_json[run_name] = {}
        per_run_dir = os.path.join(level_analysis_dir, run_name)
        os.makedirs(per_run_dir, exist_ok=True)

        for input_mode in INPUT_MODES:
            attack_path = os.path.join(run_path, f"results_attack_{input_mode}.jsonl")
            if not os.path.exists(attack_path):
                continue

            rows = load_jsonl(attack_path)
            level_metrics = compute_level_metrics(rows)
            combined_json[run_name][input_mode] = level_metrics

            save_json(
                os.path.join(per_run_dir, f"attack_level_metrics_{input_mode}.json"),
                level_metrics,
            )
            save_csv(
                os.path.join(per_run_dir, f"attack_level_metrics_{input_mode}.csv"),
                level_metrics,
            )

            for item in level_metrics:
                all_summary_rows.append(
                    {
                        "run_name": run_name,
                        "input_mode": input_mode,
                        "n_samples": extract_n_samples(run_name),
                        **item,
                    }
                )

        print(f"[OK] processed {run_name}")

    save_json(os.path.join(level_analysis_dir, "all_runs_attack_level_metrics.json"), combined_json)
    save_csv(os.path.join(level_analysis_dir, "all_runs_attack_level_metrics.csv"), all_summary_rows)

    print(f"[OK] saved combined json to {os.path.join(level_analysis_dir, 'all_runs_attack_level_metrics.json')}")
    print(f"[OK] saved combined csv to {os.path.join(level_analysis_dir, 'all_runs_attack_level_metrics.csv')}")


if __name__ == "__main__":
    main()
