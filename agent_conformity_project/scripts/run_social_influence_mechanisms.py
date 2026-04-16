import os
import shutil
from collections import defaultdict

from src.config import RESULT_DIR, RUN_DIR
from src.dataset_loader import load_dataset
from src.experiment_io import write_csv, write_json, write_jsonl
from src.social_experiment import INPUT_MODES, run_attack_pass, run_initial_pass
from src.social_mechanism import build_mechanism_conditions


def parse_input_modes():
    raw = os.getenv("MECHANISM_INPUT_MODES")
    if not raw:
        return INPUT_MODES
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    return [mode for mode in INPUT_MODES if mode in selected] or INPUT_MODES


def extract_answer(opinion):
    if isinstance(opinion, dict):
        return opinion.get("answer")
    return opinion


def build_row(item, input_mode, condition, initial_result, attack_result):
    answers = [extract_answer(opinion) for opinion in condition["opinions"]]
    target_vote_count = sum(1 for answer in answers if answer == item["distractor"])
    correct_vote_count = sum(1 for answer in answers if answer == item["correct_answer"])

    return {
        "input_mode": input_mode,
        "dimension": condition["dimension"],
        "condition_name": condition["condition_name"],
        "id": item["id"],
        "question": item["question"],
        "correct_answer": item["correct_answer"],
        "distractor": item["distractor"],
        "opinion_count": len(condition["opinions"]),
        "target_vote_count": target_vote_count,
        "correct_vote_count": correct_vote_count,
        "opinions": condition["opinions"],
        "initial_prediction": initial_result["prediction"],
        "initial_confidence": initial_result["confidence"],
        "attack_prediction": attack_result["prediction"],
        "attack_confidence": attack_result["confidence"],
        "changed": attack_result["prediction"] != initial_result["prediction"],
        "conformed_to_target": attack_result["prediction"] == item["distractor"],
        "wrong_conformity": (
            initial_result["prediction"] == item["correct_answer"]
            and attack_result["prediction"] == item["distractor"]
        ),
        "beneficial_revision": (
            initial_result["prediction"] != item["correct_answer"]
            and attack_result["prediction"] == item["correct_answer"]
        ),
        "raw_initial_output": initial_result["text"],
        "raw_attack_output": attack_result["text"],
        "attack_step_outputs": attack_result["step_outputs"],
    }


def summarize_condition(rows):
    total = len(rows)
    return {
        "total": total,
        "conformity_rate": sum(1 for row in rows if row["conformed_to_target"]) / total if total else 0.0,
        "wrong_conformity_rate": sum(1 for row in rows if row["wrong_conformity"]) / total if total else 0.0,
        "change_rate": sum(1 for row in rows if row["changed"]) / total if total else 0.0,
        "attack_accuracy": (
            sum(1 for row in rows if row["attack_prediction"] == row["correct_answer"]) / total if total else 0.0
        ),
        "avg_target_vote_count": sum(row["target_vote_count"] for row in rows) / total if total else 0.0,
        "avg_correct_vote_count": sum(row["correct_vote_count"] for row in rows) / total if total else 0.0,
    }


def build_dimension_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["input_mode"], row["dimension"], row["condition_name"])].append(row)

    summaries = []
    for (input_mode, dimension, condition_name), condition_rows in sorted(grouped.items()):
        summaries.append(
            {
                "input_mode": input_mode,
                "dimension": dimension,
                "condition_name": condition_name,
                **summarize_condition(condition_rows),
            }
        )
    return summaries


def main():
    latest_dir = os.path.join(RESULT_DIR, "experiment_3_social_influence_mechanisms")
    run_dir = os.path.join(RUN_DIR, "experiment_3_social_influence_mechanisms")
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    dataset = load_dataset()
    input_modes = parse_input_modes()
    all_rows = []

    for item in dataset:
        initial_result = run_initial_pass(item)
        conditions = build_mechanism_conditions(item)
        for input_mode in input_modes:
            for condition in conditions:
                attack_result = run_attack_pass(
                    item,
                    condition["opinions"],
                    condition["attack_level"],
                    initial_result,
                    input_mode=input_mode,
                )
                all_rows.append(build_row(item, input_mode, condition, initial_result, attack_result))

    summary_rows = build_dimension_summaries(all_rows)
    write_jsonl(os.path.join(latest_dir, "rows.jsonl"), all_rows)
    write_csv(
        os.path.join(latest_dir, "social_influence_mechanisms_rows.csv"),
        [{k: v for k, v in row.items() if not k.startswith("raw_") and k != "attack_step_outputs"} for row in all_rows],
    )
    write_csv(os.path.join(latest_dir, "condition_summary.csv"), summary_rows)
    write_json(os.path.join(latest_dir, "condition_summary.json"), summary_rows)

    for dimension in ["scaling_effect", "consensus_structure", "focus_distribution", "tone_weighting"]:
        dimension_rows = [row for row in summary_rows if row["dimension"] == dimension]
        write_json(os.path.join(latest_dir, f"{dimension}_summary.json"), dimension_rows)

    shutil.copytree(latest_dir, run_dir, dirs_exist_ok=True)
    print(f"[OK] social influence mechanisms saved to {latest_dir}")


if __name__ == "__main__":
    main()
