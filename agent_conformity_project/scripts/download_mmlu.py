import argparse
import json
import os
import random
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import DATA_PATH, RAW_DATA_DIR, SEED  # noqa: E402


LABELS = ["A", "B", "C", "D"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download MMLU from Hugging Face and convert it to this project's dataset format."
    )
    parser.add_argument("--dataset", default="cais/mmlu")
    parser.add_argument("--subject", default="all")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=0, help="0 means keep all rows.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--raw-data-dir", default=RAW_DATA_DIR)
    parser.add_argument(
        "--output",
        default=None,
        help="Converted project-format output. Defaults to data/datasets/mmlu_<subject>_<split>.json.",
    )
    parser.add_argument(
        "--activate",
        action="store_true",
        help=f"Also write the converted dataset to the active DATA_PATH ({DATA_PATH}).",
    )
    return parser.parse_args()


def normalize_answer(answer):
    if isinstance(answer, int):
        return LABELS[answer]

    text = str(answer).strip().upper()
    if text in LABELS:
        return text
    if text.isdigit():
        return LABELS[int(text)]

    raise ValueError(f"Unsupported MMLU answer value: {answer!r}")


def convert_row(row, index, rng):
    choices = list(row["choices"])
    options = {label: choice for label, choice in zip(LABELS, choices)}
    correct_answer = normalize_answer(row["answer"])
    wrong_options = [label for label in options if label != correct_answer]

    return {
        "id": f"mmlu_{row.get('subject', 'unknown')}_{index}",
        "source": "mmlu",
        "subject": row.get("subject"),
        "question": row["question"],
        "options": options,
        "correct_answer": correct_answer,
        "distractor": rng.choice(wrong_options),
    }


def main():
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with `pip install datasets`."
        ) from exc

    rng = random.Random(args.seed)
    dataset = load_dataset(args.dataset, args.subject, split=args.split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    raw_rows = [dict(row) for row in dataset]
    converted = [convert_row(row, index, rng) for index, row in enumerate(raw_rows)]
    output_path = args.output or os.path.join(
        "data", "datasets", f"mmlu_{args.subject}_{args.split}.json"
    )

    os.makedirs(args.raw_data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    raw_name = f"mmlu_{args.subject}_{args.split}.json"
    raw_path = os.path.join(args.raw_data_dir, raw_name)
    with open(raw_path, "w", encoding="utf-8") as fout:
        json.dump(raw_rows, fout, ensure_ascii=False, indent=2)

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(converted, fout, ensure_ascii=False, indent=2)

    if args.activate:
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        with open(DATA_PATH, "w", encoding="utf-8") as fout:
            json.dump(converted, fout, ensure_ascii=False, indent=2)

    print(f"[OK] raw MMLU saved: {raw_path}, size={len(raw_rows)}")
    print(f"[OK] project dataset saved: {output_path}, size={len(converted)}")
    if args.activate:
        print(f"[OK] active dataset updated: {DATA_PATH}, size={len(converted)}")


if __name__ == "__main__":
    main()
