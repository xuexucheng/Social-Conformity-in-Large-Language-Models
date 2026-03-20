import json
import os
import shutil

from src.config import RESULT_DIR, RUN_DIR
from src.dataset_loader import load_dataset
from src.llm_api import call_llm
from src.parser import parse_answer, parse_confidence
from src.prompts import build_prompt
from src.attack_generator import generate_attack_opinions, ATTACK_LEVELS


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    latest_path = os.path.join(RESULT_DIR, "results_attack.jsonl")
    run_path = os.path.join(RUN_DIR, "results_attack.jsonl")

    data = load_dataset()

    with open(latest_path, "w", encoding="utf-8") as fout:
        for item in data:
            initial_messages = build_prompt(item, opinions=[])
            initial_text = call_llm(initial_messages)
            initial_prediction = parse_answer(initial_text)
            initial_confidence = parse_confidence(initial_text)

            for level in ATTACK_LEVELS:
                opinions = generate_attack_opinions(item, level=level)
                attack_messages = build_prompt(item, opinions=opinions, attack_level=level)
                attack_text = call_llm(attack_messages)

                attack_prediction = parse_answer(attack_text)
                attack_confidence = parse_confidence(attack_text)

                row = {
                    "id": item["id"],
                    "attack_level": level,
                    "question": item["question"],
                    "options": item["options"],
                    "correct_answer": item["correct_answer"],
                    "distractor": item["distractor"],
                    "initial_prediction": initial_prediction,
                    "initial_confidence": initial_confidence,
                    "attack_prediction": attack_prediction,
                    "attack_confidence": attack_confidence,
                    "changed": attack_prediction != initial_prediction,
                    "agent_opinions": opinions,
                    "raw_initial_output": initial_text,
                    "raw_attack_output": attack_text,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    shutil.copy2(latest_path, run_path)

    print(f"[OK] attack results saved to {latest_path}")
    print(f"[OK] attack results archived to {run_path}")


if __name__ == "__main__":
    main()