#!/usr/bin/env python3
"""Select a generic label token-length matched to ``Model`` on every item."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import conditions as C


CANDIDATE_LABELS = (
    "Source",
    "Speaker",
    "Responder",
    "Participant",
    "Entity",
    "Origin",
    "Channel",
)


def load_dataset(path, limit=None):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data[:limit] if limit else data


def prompt_count(tokenizer, messages, model_name):
    from src.llm_api import normalize_messages_for_model

    normalized = normalize_messages_for_model(messages, model_name)
    return len(
        tokenizer.apply_chat_template(
            normalized, tokenize=True, add_generation_prompt=True
        )
    )


def select_label(tokenizer, dataset, model_name, candidates=CANDIDATE_LABELS):
    results = []
    for candidate in candidates:
        differences = []
        for item in dataset:
            initial = {
                "messages": C.build_initial_messages(item),
                "text": "ANSWER: A\nCONFIDENCE: 100",
            }
            neutral = C.build_final_messages(
                item, initial, "neutral_source_label", neutral_label=candidate
            )
            model = C.build_final_messages(item, initial, "model_source_label")
            differences.append(
                prompt_count(tokenizer, neutral, model_name)
                - prompt_count(tokenizer, model, model_name)
            )
        results.append(
            {
                "label": candidate,
                "exact_match_n": sum(difference == 0 for difference in differences),
                "item_n": len(differences),
                "difference_values": sorted(set(differences)),
            }
        )
    exact = [row for row in results if row["exact_match_n"] == row["item_n"]]
    if not exact:
        raise ValueError(
            "no curated generic label is prompt-token matched to 'Model' on every item; "
            f"audit={results}"
        )
    return {"neutral_label": exact[0]["label"], "candidate_audit": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True
    )
    dataset = load_dataset(args.data_path, args.limit)
    candidates = tuple(args.candidate) + CANDIDATE_LABELS
    payload = {
        "model_name": args.model_name,
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "dataset_path": str(Path(args.data_path).resolve()),
        "n_items": len(dataset),
        "model_label": C.MODEL_LABEL,
        **select_label(tokenizer, dataset, args.model_name, candidates),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"NEUTRAL_LABEL={payload['neutral_label']}")
    print(f"OUTPUT_JSON={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
