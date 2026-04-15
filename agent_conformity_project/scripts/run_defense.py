import json
import os
import shutil

from src.config import RESULT_DIR, RUN_DIR
from src.dataset_loader import load_dataset
from src.attack_generator import generate_attack_opinions, ATTACK_LEVELS
from src.defense_crown import crown_guard
from src.social_experiment import INPUT_MODES, run_attack_pass, run_initial_pass


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    data = load_dataset()
    latest_paths = {
        input_mode: os.path.join(RESULT_DIR, f"results_defense_{input_mode}.jsonl")
        for input_mode in INPUT_MODES
    }
    run_paths = {
        input_mode: os.path.join(RUN_DIR, f"results_defense_{input_mode}.jsonl")
        for input_mode in INPUT_MODES
    }

    file_handles = {
        input_mode: open(path, "w", encoding="utf-8")
        for input_mode, path in latest_paths.items()
    }
    try:
        for item in data:
            initial_result = run_initial_pass(item)

            for level in ATTACK_LEVELS:
                opinions = generate_attack_opinions(item, level=level)

                for input_mode in INPUT_MODES:
                    attack_result = run_attack_pass(
                        item,
                        opinions,
                        level,
                        initial_result,
                        input_mode=input_mode,
                    )

                    defended_prediction, defended_confidence = crown_guard(
                        initial_prediction=initial_result["prediction"],
                        initial_confidence=initial_result["confidence"],
                        attacked_prediction=attack_result["prediction"],
                        attacked_confidence=attack_result["confidence"],
                    )

                    row = {
                        "id": item["id"],
                        "input_mode": input_mode,
                        "attack_level": level,
                        "question": item["question"],
                        "options": item["options"],
                        "correct_answer": item["correct_answer"],
                        "distractor": item["distractor"],
                        "initial_prediction": initial_result["prediction"],
                        "initial_confidence": initial_result["confidence"],
                        "attack_prediction": attack_result["prediction"],
                        "attack_confidence": attack_result["confidence"],
                        "defended_prediction": defended_prediction,
                        "defended_confidence": defended_confidence,
                        "agent_opinions": opinions,
                        "raw_initial_output": initial_result["text"],
                        "raw_attack_output": attack_result["text"],
                        "attack_step_outputs": attack_result["step_outputs"],
                    }
                    file_handles[input_mode].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for fout in file_handles.values():
            fout.close()

    for input_mode in INPUT_MODES:
        shutil.copy2(latest_paths[input_mode], run_paths[input_mode])
        print(f"[OK] defense results ({input_mode}) saved to {latest_paths[input_mode]}")
        print(f"[OK] defense results ({input_mode}) archived to {run_paths[input_mode]}")


if __name__ == "__main__":
    main()
