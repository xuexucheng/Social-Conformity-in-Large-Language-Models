import os
import shutil

from src.config import N_ATTACK_AGENTS, RESULT_DIR, RUN_DIR
from src.dataset_loader import load_dataset
from src.experiment_io import write_csv, write_json, write_jsonl
from src.semantic_exclusion import get_semantic_analyzer, select_semantic_groups
from src.social_experiment import INPUT_MODES, run_attack_pass, run_initial_pass


GROUP_SIZE = int(os.getenv("SEMANTIC_GROUP_SIZE", "50"))
ATTACK_LEVEL = os.getenv("SEMANTIC_ATTACK_LEVEL", "L3")
SEMANTIC_BACKEND = os.getenv("SEMANTIC_BACKEND", "auto")
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def build_candidate_row(item, analyzer):
    correct_text = item["options"][item["correct_answer"]]
    distractor_text = item["options"][item["distractor"]]
    profile = analyzer.profile(correct_text, distractor_text)
    return {
        "id": item["id"],
        "question": item["question"],
        "correct_answer": item["correct_answer"],
        "correct_text": correct_text,
        "target_option": item["distractor"],
        "target_text": distractor_text,
        **profile,
    }


def evaluate_group(items, group_name, analyzer, input_mode):
    rows = []
    for item in items:
        initial_result = run_initial_pass(item)
        attack_result = run_attack_pass(
            item,
            [item["distractor"]] * N_ATTACK_AGENTS,
            ATTACK_LEVEL,
            initial_result,
            input_mode=input_mode,
        )

        profile = analyzer.profile(
            item["options"][item["correct_answer"]],
            item["options"][item["distractor"]],
        )

        rows.append(
            {
                "input_mode": input_mode,
                "group": group_name,
                "id": item["id"],
                "question": item["question"],
                "correct_answer": item["correct_answer"],
                "target_option": item["distractor"],
                "correct_text": item["options"][item["correct_answer"]],
                "target_text": item["options"][item["distractor"]],
                "similarity_score": profile["similarity_score"],
                "contradiction_score": profile["contradiction_score"],
                "antonym_signal": profile["antonym_signal"],
                "exclusion_score": profile["exclusion_score"],
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
        )
    return rows


def summarize(rows):
    groups = {}
    input_modes = sorted({row["input_mode"] for row in rows})
    for input_mode in input_modes:
        groups[input_mode] = {}
        for group_name in sorted({row["group"] for row in rows if row["input_mode"] == input_mode}):
            group_rows = [
                row for row in rows if row["input_mode"] == input_mode and row["group"] == group_name
            ]
            total = len(group_rows)
            groups[input_mode][group_name] = {
                "total": total,
                "conformity_rate": (
                    sum(1 for row in group_rows if row["conformed_to_target"]) / total if total else 0.0
                ),
                "wrong_conformity_rate": (
                    sum(1 for row in group_rows if row["wrong_conformity"]) / total if total else 0.0
                ),
                "change_rate": (
                    sum(1 for row in group_rows if row["changed"]) / total if total else 0.0
                ),
                "clean_accuracy": (
                    sum(1 for row in group_rows if row["initial_prediction"] == row["correct_answer"]) / total
                    if total
                    else 0.0
                ),
                "attack_accuracy": (
                    sum(1 for row in group_rows if row["attack_prediction"] == row["correct_answer"]) / total
                    if total
                    else 0.0
                ),
                "avg_exclusion_score": (
                    sum(row["exclusion_score"] for row in group_rows) / total if total else 0.0
                ),
            }

    comparisons = {}
    for input_mode, mode_groups in groups.items():
        if "near_similar" in mode_groups and "far_exclusive" in mode_groups:
            comparisons[input_mode] = {
                "conformity_rate_gap_far_minus_near": (
                    mode_groups["far_exclusive"]["conformity_rate"]
                    - mode_groups["near_similar"]["conformity_rate"]
                ),
                "wrong_conformity_rate_gap_far_minus_near": (
                    mode_groups["far_exclusive"]["wrong_conformity_rate"]
                    - mode_groups["near_similar"]["wrong_conformity_rate"]
                ),
            }

    return {
        "experiment_name": "Semantic Exclusion Test",
        "attack_level": ATTACK_LEVEL,
        "group_size": GROUP_SIZE,
        "groups": groups,
        "comparisons": comparisons,
    }


def main():
    latest_dir = os.path.join(RESULT_DIR, "experiment_2_semantic_exclusion_test")
    run_dir = os.path.join(RUN_DIR, "experiment_2_semantic_exclusion_test")
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    dataset = load_dataset()
    analyzer = get_semantic_analyzer(SEMANTIC_BACKEND, SEMANTIC_MODEL_NAME)
    candidate_rows = [build_candidate_row(item, analyzer) for item in dataset]
    candidate_map = {item["id"]: item for item in dataset}

    near_candidates, far_candidates = select_semantic_groups(candidate_rows, GROUP_SIZE)
    near_items = [candidate_map[row["id"]] for row in near_candidates]
    far_items = [candidate_map[row["id"]] for row in far_candidates]

    evaluated_rows = []
    for input_mode in INPUT_MODES:
        evaluated_rows.extend(evaluate_group(near_items, "near_similar", analyzer, input_mode))
        evaluated_rows.extend(evaluate_group(far_items, "far_exclusive", analyzer, input_mode))

    write_jsonl(os.path.join(latest_dir, "rows.jsonl"), evaluated_rows)
    write_csv(os.path.join(latest_dir, "semantic_candidates.csv"), candidate_rows)
    write_csv(
        os.path.join(latest_dir, "semantic_exclusion_results.csv"),
        [{k: v for k, v in row.items() if not k.startswith("raw_") and k != "attack_step_outputs"} for row in evaluated_rows],
    )
    summary = summarize(evaluated_rows)
    summary["semantic_backend"] = getattr(analyzer, "backend_name", SEMANTIC_BACKEND)
    summary["semantic_model_name"] = SEMANTIC_MODEL_NAME
    write_json(os.path.join(latest_dir, "summary.json"), summary)

    shutil.copytree(latest_dir, run_dir, dirs_exist_ok=True)
    print(f"[OK] semantic exclusion test saved to {latest_dir}")


if __name__ == "__main__":
    main()
