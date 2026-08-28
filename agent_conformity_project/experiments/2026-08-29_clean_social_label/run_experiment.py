#!/usr/bin/env python3
"""Run the clean label-only social-framing control.

Formal runs should use ``--reference-jsonl`` so every condition replays the
same archived private baseline instead of making a fresh baseline call.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import conditions as C  # noqa: E402
from src.config import DATA_PATH, MAX_TOKENS, MODEL_NAME, RESULT_DIR, TEMPERATURE  # noqa: E402
from src.llm_api import call_llm_with_logprobs, normalize_messages_for_model  # noqa: E402
from src.parser import parse_answer, parse_confidence  # noqa: E402


OPTION_LABELS = frozenset("ABCDE")
OUTPUT_SUBDIR = "2026-08-29_clean_social_label"


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def standardize(result, options):
    text = (result.get("text") or "").strip()
    prediction = parse_answer(text, options=options)
    return {
        "text": text,
        "prediction": prediction,
        "confidence": parse_confidence(text),
        "valid": prediction in OPTION_LABELS,
    }


def model_call(messages, options):
    result = call_llm_with_logprobs(
        messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_logprobs=20,
    )
    return standardize(result, options)


def load_dataset(path, limit=None):
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return data[:limit] if limit else data


def load_references(path):
    references = {}
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("id", row.get("item_id"))
            if item_id is None:
                raise ValueError(f"reference row lacks id at {path}:{line_number}")
            item_id = str(item_id)
            if item_id in references:
                raise ValueError(f"duplicate reference id {item_id!r}")
            raw = row.get("raw_initial_output", row.get("raw_output"))
            initial = row.get("initial_prediction", row.get("initial_answer"))
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"reference row {item_id!r} lacks raw initial output")
            references[item_id] = {**row, "raw_initial_output": raw, "initial_prediction": initial}
    return references


def validate_references(references, dataset):
    fields = ("question", "options", "correct_answer", "distractor")
    for item in dataset:
        item_id = str(item.get("id"))
        reference = references.get(item_id)
        if reference is None:
            raise ValueError(f"missing initial-answer reference for item {item_id!r}")
        drift = [field for field in fields if reference.get(field) != item.get(field)]
        if drift:
            raise ValueError(f"reference metadata drift for {item_id!r}: {drift}")


def prompt_tokens(tokenizer, messages):
    normalized = normalize_messages_for_model(messages, MODEL_NAME)
    return len(
        tokenizer.apply_chat_template(
            normalized, tokenize=True, add_generation_prompt=True
        )
    )


def run_item(item, conditions, neutral_label, initial_reference, tokenizer=None):
    initial_messages = C.build_initial_messages(item)
    if initial_reference is None:
        initial = model_call(initial_messages, item["options"])
        source = "model_call"
    else:
        initial = standardize(
            {"text": initial_reference["raw_initial_output"]}, item["options"]
        )
        if initial_reference.get("initial_prediction") is not None and (
            initial["prediction"] != initial_reference.get("initial_prediction")
        ):
            raise ValueError(f"reference initial-answer parse mismatch for {item['id']!r}")
        source = "reference_jsonl"
    initial_result = {"messages": initial_messages, "text": initial["text"]}
    rows = {}
    for condition in conditions:
        messages = C.build_final_messages(
            item, initial_result, condition, neutral_label=neutral_label
        )
        final = model_call(messages, item["options"])
        rows[condition] = {
            "schema_version": "1.0",
            "experiment": "clean_social_label",
            "condition": condition,
            "model_name": MODEL_NAME,
            "id": item["id"],
            "source": item.get("source"),
            "subject": item.get("subject"),
            "question": item["question"],
            "options": item["options"],
            "correct_answer": item["correct_answer"],
            "distractor": item["distractor"],
            "n_sources": C.N_SOURCES,
            "neutral_label": neutral_label,
            "model_label": C.MODEL_LABEL,
            "initial_answer_source": source,
            "initial_prediction": initial["prediction"],
            "raw_initial_output": initial["text"],
            "final_prediction": final["prediction"],
            "final_confidence": final["confidence"],
            "raw_final_output": final["text"],
            "final_valid": final["valid"],
            "changed_from_initial": final["prediction"] != initial["prediction"],
            "chose_distractor": final["prediction"] == item["distractor"],
            "is_correct": final["prediction"] == item["correct_answer"],
            "final_messages": messages,
            "final_prompt_token_count": (
                prompt_tokens(tokenizer, messages) if tokenizer is not None else None
            ),
        }
    return rows


def read_completed_ids(path):
    ids = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                ids.append(str(row.get("id")))
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in resume file {path}")
    return set(ids)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", default=",".join(C.CONDITIONS))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reference-jsonl", default="")
    parser.add_argument("--label-selection-json", required=True)
    parser.add_argument("--compute-tokens", action="store_true")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    unknown = sorted(set(conditions) - set(C.CONDITIONS))
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    selection_path = Path(args.label_selection_json).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    neutral_label = selection.get("neutral_label")
    if not neutral_label:
        raise SystemExit("label selection JSON lacks neutral_label")

    dataset = load_dataset(DATA_PATH, args.limit)
    references = None
    reference_path = Path(args.reference_jsonl).resolve() if args.reference_jsonl else None
    if reference_path:
        references = load_references(reference_path)
        validate_references(references, dataset)

    tokenizer = None
    tokenizer_path = args.tokenizer_path or MODEL_NAME
    if args.compute_tokens:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=True
        )

    out_dir = Path(RESULT_DIR) / "runs" / args.run_name / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {condition: out_dir / f"{condition}.jsonl" for condition in conditions}
    completed = set()
    if args.resume:
        if not all(path.is_file() for path in paths.values()):
            raise SystemExit("resume requires every selected condition file")
        id_sets = {condition: read_completed_ids(path) for condition, path in paths.items()}
        completed = id_sets[conditions[0]]
        if any(ids != completed for ids in id_sets.values()):
            raise SystemExit("resume checkpoint is not aligned across conditions")
    elif any(path.exists() for path in paths.values()):
        raise SystemExit("output exists; choose a new run name or use --resume")

    manifest = {
        "experiment": "clean_social_label",
        "run_name": args.run_name,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": MODEL_NAME,
        "dataset_path": str(Path(DATA_PATH).resolve()),
        "dataset_md5": md5(DATA_PATH),
        "n_samples": len(dataset),
        "conditions": conditions,
        "primary_contrast": ["neutral_source_label", "model_source_label"],
        "neutral_label": neutral_label,
        "model_label": C.MODEL_LABEL,
        "label_selection_json": str(selection_path),
        "label_selection_json_md5": md5(selection_path),
        "reference_jsonl": str(reference_path) if reference_path else None,
        "reference_jsonl_md5": md5(reference_path) if reference_path else None,
        "decoding": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "top_logprobs": 20},
        "tokenizer_path": tokenizer_path if tokenizer is not None else None,
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "git_commit": git_commit(),
    }
    manifest_path = out_dir / "manifest.json"
    if not args.resume:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    mode = "a" if args.resume else "w"
    handles = {condition: path.open(mode, encoding="utf-8") for condition, path in paths.items()}
    try:
        for index, item in enumerate(dataset, start=1):
            if str(item["id"]) in completed:
                continue
            rows = run_item(
                item,
                conditions,
                neutral_label,
                (references or {}).get(str(item["id"])),
                tokenizer=tokenizer,
            )
            for condition, row in rows.items():
                handles[condition].write(json.dumps(row, ensure_ascii=False) + "\n")
                handles[condition].flush()
            if index % 25 == 0 or index == len(dataset):
                print(f"[OK] {index}/{len(dataset)} items", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    print(f"[DONE] outputs={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
