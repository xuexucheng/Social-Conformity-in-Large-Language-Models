import json
import os
import shutil
import traceback
from collections import defaultdict

from src.config import RESULT_DIR, RUN_DIR
from src.dataset_loader import load_dataset
from src.experiment_io import write_csv, write_json, write_jsonl
from src.social_experiment import INPUT_MODES, run_attack_pass, run_initial_pass
from src.social_mechanism import build_mechanism_conditions


SUPPORTED_DIMENSIONS = {
    "input_mode_comparison",
}


def parse_input_modes():
    raw = os.getenv("MECHANISM_INPUT_MODES")
    if not raw:
        return INPUT_MODES
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    return [mode for mode in INPUT_MODES if mode in selected] or INPUT_MODES


def parse_int_env(name, default, minimum=None):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        value = max(minimum, value)
    return value


def parse_save_every():
    return parse_int_env("SAVE_EVERY", 20, minimum=1)


def parse_resume_enabled():
    return os.getenv("RESUME_RUN", "1") == "1"


def parse_sample_limit():
    return parse_int_env("MAX_ITEMS", 0, minimum=0)


def parse_start_index():
    return parse_int_env("START_INDEX", 0, minimum=0)


def parse_dimensions():
    raw = os.getenv("MECHANISM_DIMENSIONS")
    if not raw:
        return []
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    return [dimension for dimension in selected if dimension in SUPPORTED_DIMENSIONS]


def parse_copy_each_save():
    return os.getenv("COPY_TO_RUN_DIR_EACH_SAVE", "0") == "1"


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


def build_row_key(row):
    return (
        row["id"],
        row["input_mode"],
        row["dimension"],
        row["condition_name"],
    )


def load_existing_jsonl(path):
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def prepare_dataset(dataset, start_index, max_items):
    if start_index:
        dataset = dataset[start_index:]
    if max_items:
        dataset = dataset[:max_items]
    return dataset


def filter_conditions(conditions, selected_dimensions):
    if not selected_dimensions:
        return conditions
    return [condition for condition in conditions if condition["dimension"] in selected_dimensions]


def save_outputs(latest_dir, run_dir, all_rows, errors, copy_to_run_dir):
    summary_rows = build_dimension_summaries(all_rows)

    write_jsonl(os.path.join(latest_dir, "rows.jsonl"), all_rows)
    write_csv(
        os.path.join(latest_dir, "social_influence_mechanisms_rows.csv"),
        [{k: v for k, v in row.items() if not k.startswith("raw_") and k != "attack_step_outputs"} for row in all_rows],
    )
    write_csv(os.path.join(latest_dir, "condition_summary.csv"), summary_rows)
    write_json(os.path.join(latest_dir, "condition_summary.json"), summary_rows)
    write_jsonl(os.path.join(latest_dir, "errors.jsonl"), errors)

    for dimension in sorted({row["dimension"] for row in all_rows}):
        dimension_rows = [row for row in summary_rows if row["dimension"] == dimension]
        write_json(os.path.join(latest_dir, f"{dimension}_summary.json"), dimension_rows)

    if copy_to_run_dir:
        shutil.copytree(latest_dir, run_dir, dirs_exist_ok=True)


def main():
    latest_dir = os.path.join(RESULT_DIR, "experiment_3_social_influence_mechanisms")
    run_dir = os.path.join(RUN_DIR, "experiment_3_social_influence_mechanisms")
    os.makedirs(latest_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    dataset = load_dataset()
    input_modes = parse_input_modes()
    save_every = parse_save_every()
    resume_enabled = parse_resume_enabled()
    start_index = parse_start_index()
    max_items = parse_sample_limit()
    selected_dimensions = parse_dimensions()
    copy_each_save = parse_copy_each_save()

    dataset = prepare_dataset(dataset, start_index, max_items)
    total_items = len(dataset)

    rows_path = os.path.join(latest_dir, "rows.jsonl")
    errors_path = os.path.join(latest_dir, "errors.jsonl")
    existing_rows = load_existing_jsonl(rows_path) if resume_enabled else []
    existing_errors = load_existing_jsonl(errors_path) if resume_enabled else []

    all_rows = list(existing_rows)
    errors = list(existing_errors)
    completed_row_keys = {build_row_key(row) for row in all_rows}
    print(f"[INFO] loaded dataset size = {total_items}")
    print(f"[INFO] input_modes = {input_modes}")
    print(f"[INFO] save_every = {save_every}")
    print(f"[INFO] resume_enabled = {resume_enabled}")
    print(f"[INFO] start_index = {start_index}")
    print(f"[INFO] max_items = {max_items if max_items else 'all'}")
    print(f"[INFO] selected_dimensions = {selected_dimensions or 'all'}")
    print(f"[INFO] copy_to_run_dir_each_save = {copy_each_save}")
    print(f"[INFO] resumed_rows = {len(existing_rows)}")
    print(f"[INFO] resumed_errors = {len(existing_errors)}")

    processed_items = 0

    for item_index, item in enumerate(dataset, start=1):
        item_id = item.get("id", f"item_{item_index}")
        conditions = filter_conditions(build_mechanism_conditions(item), selected_dimensions)
        target_row_keys = {
            (item_id, input_mode, condition["dimension"], condition["condition_name"])
            for input_mode in input_modes
            for condition in conditions
        }
        remaining_row_keys = target_row_keys - completed_row_keys

        if resume_enabled and not remaining_row_keys:
            print(
                f"[INFO] item {item_index}/{total_items} | id={item_id} | skipped (already completed)",
                flush=True,
            )
            processed_items += 1
            continue

        print(f"[INFO] item {item_index}/{total_items} | id={item_id} | start", flush=True)

        try:
            initial_result = run_initial_pass(item)

            print(
                f"[INFO] item {item_index}/{total_items} | id={item_id} | "
                f"conditions={len(conditions)} | input_modes={len(input_modes)} | remaining={len(remaining_row_keys)}",
                flush=True,
            )

            for input_mode in input_modes:
                for condition_index, condition in enumerate(conditions, start=1):
                    row_key = (
                        item_id,
                        input_mode,
                        condition["dimension"],
                        condition["condition_name"],
                    )
                    if resume_enabled and row_key in completed_row_keys:
                        continue

                    try:
                        print(
                            f"[INFO] item {item_index}/{total_items} | id={item_id} | "
                            f"mode={input_mode} | condition {condition_index}/{len(conditions)} | "
                            f"{condition['dimension']}::{condition['condition_name']}",
                            flush=True,
                        )

                        attack_result = run_attack_pass(
                            item,
                            condition["opinions"],
                            condition["attack_level"],
                            initial_result,
                            input_mode=input_mode,
                        )

                        row = build_row(item, input_mode, condition, initial_result, attack_result)
                        all_rows.append(row)
                        completed_row_keys.add(build_row_key(row))

                    except Exception as inner_exc:
                        errors.append(
                            {
                                "stage": "attack_pass",
                                "item_index": item_index,
                                "item_id": item_id,
                                "input_mode": input_mode,
                                "dimension": condition.get("dimension"),
                                "condition_name": condition.get("condition_name"),
                                "error_type": type(inner_exc).__name__,
                                "error_message": str(inner_exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        print(
                            f"[WARN] item {item_index}/{total_items} | id={item_id} | "
                            f"mode={input_mode} | condition={condition.get('condition_name')} failed: {inner_exc}",
                            flush=True,
                        )

        except Exception as outer_exc:
            errors.append(
                {
                    "stage": "initial_pass_or_condition_build",
                    "item_index": item_index,
                    "item_id": item_id,
                    "error_type": type(outer_exc).__name__,
                    "error_message": str(outer_exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(
                f"[WARN] item {item_index}/{total_items} | id={item_id} failed before attack loop: {outer_exc}",
                flush=True,
            )

        processed_items += 1

        if processed_items % save_every == 0:
            save_outputs(latest_dir, run_dir, all_rows, errors, copy_to_run_dir=copy_each_save)
            print(
                f"[INFO] autosaved at item {processed_items}/{total_items} | "
                f"rows={len(all_rows)} | errors={len(errors)}",
                flush=True,
            )

        print(
            f"[INFO] item {item_index}/{total_items} | id={item_id} | done | "
            f"current_rows={len(all_rows)} | current_errors={len(errors)}",
            flush=True,
        )

    save_outputs(latest_dir, run_dir, all_rows, errors, copy_to_run_dir=True)
    print(
        f"[OK] social influence mechanisms saved to {latest_dir} | "
        f"rows={len(all_rows)} | errors={len(errors)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
