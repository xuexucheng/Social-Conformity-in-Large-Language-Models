import json
import os
import shutil

from src.config import RESULT_DIR, RUN_DIR
from src.dataset_loader import load_dataset
from src.llm_api import call_llm
from src.parser import parse_answer, parse_confidence
from src.prompts import build_prompt


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(RUN_DIR, exist_ok=True)

    latest_path = os.path.join(RESULT_DIR, "results_clean.jsonl")
    run_path = os.path.join(RUN_DIR, "results_clean.jsonl")

    data = load_dataset()

    with open(latest_path, "w", encoding="utf-8") as fout:
        for item in data:
            messages = build_prompt(item, opinions=[])
            text = call_llm(messages)

            initial_prediction = parse_answer(text)
            initial_confidence = parse_confidence(text)

            row = {
                "id": item["id"],
                "question": item["question"],
                "options": item["options"],
                "correct_answer": item["correct_answer"],
                "distractor": item["distractor"],
                "initial_prediction": initial_prediction,
                "initial_confidence": initial_confidence,
                "raw_output": text,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    shutil.copy2(latest_path, run_path)

    print(f"[OK] clean results saved to {latest_path}")
    print(f"[OK] clean results archived to {run_path}")


if __name__ == "__main__":
    main()