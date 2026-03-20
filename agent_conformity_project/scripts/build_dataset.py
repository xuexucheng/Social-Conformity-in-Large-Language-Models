import json
import os
import random

from src.config import RAW_DATA_DIR, DATA_SPLIT, DATA_PATH, N_SAMPLES, SEED


def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        return [data]
    except json.JSONDecodeError:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def convert_choices_to_options(choices):
    if isinstance(choices, dict) and "label" in choices and "text" in choices:
        labels = choices["label"]
        texts = choices["text"]
        return {label: text for label, text in zip(labels, texts)}

    if isinstance(choices, dict):
        return choices

    raise ValueError("Unsupported choices/options format.")


def main():
    random.seed(SEED)

    in_path = os.path.join(RAW_DATA_DIR, f"{DATA_SPLIT}.json")
    raw_data = load_json_or_jsonl(in_path)

    sampled = raw_data[:N_SAMPLES]
    dataset = []

    for i, item in enumerate(sampled):
        question = item["question"]

        if "options" in item:
            options = convert_choices_to_options(item["options"])
        elif "choices" in item:
            options = convert_choices_to_options(item["choices"])
        else:
            raise ValueError("Each raw item must contain 'options' or 'choices'.")

        correct_answer = item.get("correct_answer")
        if correct_answer is None:
            correct_answer = item.get("answerKey")

        if not correct_answer:
            continue

        wrong_options = [k for k in options.keys() if k != correct_answer]
        distractor = random.choice(wrong_options)

        dataset.append(
            {
                "id": item.get("id", i),
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "distractor": distractor,
            }
        )

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"[OK] dataset built: {DATA_PATH}, size={len(dataset)}")


if __name__ == "__main__":
    main()