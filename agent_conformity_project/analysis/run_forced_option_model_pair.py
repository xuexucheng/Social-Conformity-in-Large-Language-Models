#!/usr/bin/env python3
"""Run P3 forced-option scoring for CSQA + MMLU with one model load."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.forced_option_scoring import (  # noqa: E402
    audit_against_dataset,
    build_private_prompt,
    candidate_labels,
    encode_candidates,
    file_md5,
    load_dataset_map,
    load_jsonl,
    score_candidates,
    sha256_text,
    stable_softmax,
    validate_existing_output,
)


MODEL_SPECS = {
    "qwen": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "model_path": REPO / "models/qwen25_3b_instruct",
    },
    "phi": {
        "model_name": "microsoft/Phi-3.5-mini-instruct",
        "revision": "2fe192450127e6a83f7441aef6e3ca586c338b77",
        "model_path": REPO / "models/phi35_mini_instruct",
    },
    "gemma": {
        "model_name": "google/gemma-2-2b-it",
        "revision": "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
        "model_path": REPO / "models/gemma2_2b_it",
    },
}


def dataset_specs(model_key: str):
    return [
        {
            "key": "csqa",
            "dataset_name": "CommonsenseQA",
            "input": PROJECT
            / "analysis/recovered_main500"
            / f"{model_key}_csqa"
            / "exp1_all_at_once_final.jsonl",
            "dataset": PROJECT
            / "data/datasets/commonsenseqa_five_wrong_guidance_refined_500.json",
        },
        {
            "key": "mmlu",
            "dataset_name": "MMLU",
            "input": PROJECT
            / "analysis/recovered_main500"
            / f"{model_key}_mmlu"
            / "exp1_all_at_once_final.jsonl",
            "dataset": PROJECT
            / "data/datasets/mmlu_all_validation.json",
        },
    ]


def prepare_dataset(spec):
    rows = load_jsonl(spec["input"])

    if len(rows) != 500:
        raise ValueError(
            f"{spec['key']}: expected 500 rows, got {len(rows)}"
        )

    dataset_map = load_dataset_map(spec["dataset"])
    audit_against_dataset(rows, dataset_map)

    return {
        **spec,
        "rows": rows,
        "dataset_md5": file_md5(spec["dataset"]),
    }


def score_one_dataset(
    *,
    prepared,
    model,
    tokenizer,
    model_key,
    model_name,
    revision,
    template_hash,
    output_dir,
    resume,
):
    import torch

    rows = prepared["rows"]
    dataset_name = prepared["dataset_name"]
    dataset_md5 = prepared["dataset_md5"]

    output_path = output_dir / f"{model_key}_{prepared['key']}.jsonl"

    completed = set()

    if output_path.exists():
        if not resume:
            raise FileExistsError(
                f"{output_path} already exists; use --resume"
            )

        completed = validate_existing_output(
            output_path,
            model_name,
            revision,
            dataset_md5,
        )

    print()
    print("=" * 72, flush=True)
    print(
        f"START {model_key}/{prepared['key']} "
        f"rows={len(rows)} already_done={len(completed)}",
        flush=True,
    )
    print("output:", output_path, flush=True)
    print("=" * 72, flush=True)

    mode = "a" if output_path.exists() else "w"

    with output_path.open(mode, encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            item_id = str(row.get("id", row.get("item_id")))

            if item_id in completed:
                continue

            labels = candidate_labels(row["options"])

            _, rendered, prompt_ids = build_private_prompt(
                tokenizer,
                row,
                model_name,
            )

            candidates = encode_candidates(tokenizer, labels)

            scores = score_candidates(
                model,
                tokenizer,
                prompt_ids,
                candidates,
                "cuda",
            )

            probabilities = stable_softmax(
                [scores[label] for label in labels]
            )
            probs = dict(zip(labels, probabilities))

            correct = str(row["correct_answer"]).strip().upper()
            target = str(row["distractor"]).strip().upper()

            if correct not in labels:
                raise ValueError(
                    f"{item_id}: correct={correct} absent from {labels}"
                )
            if target not in labels:
                raise ValueError(
                    f"{item_id}: target={target} absent from {labels}"
                )
            if correct == target:
                raise ValueError(
                    f"{item_id}: correct answer equals target distractor"
                )

            result = {
                "id": item_id,
                "model": model_name,
                "dataset": dataset_name,
                "correct_answer": correct,
                "target_distractor": target,
                "candidate_labels": list(labels),

                "score_A": scores.get("A"),
                "score_B": scores.get("B"),
                "score_C": scores.get("C"),
                "score_D": scores.get("D"),
                "score_E": scores.get("E"),

                "prob_A": probs.get("A"),
                "prob_B": probs.get("B"),
                "prob_C": probs.get("C"),
                "prob_D": probs.get("D"),
                "prob_E": probs.get("E"),

                "target_score": scores[target],
                "target_probability": probs[target],
                "correct_score": scores[correct],
                "correct_probability": probs[correct],

                "model_revision": revision,
                "tokenizer_revision": revision,
                "prompt_hash": sha256_text(rendered),
                "dataset_md5": dataset_md5,
                "chat_template_sha256": template_hash,
                "prompt_token_count": len(prompt_ids),

                "candidate_token_counts": {
                    label: len(candidates[label]["ids"])
                    for label in labels
                },
                "canonical_completions": {
                    label: candidates[label]["text"]
                    for label in labels
                },
            }

            out.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            out.flush()

            if index % 25 == 0 or index == len(rows):
                print(
                    f"[{prepared['key']} {index}/500] "
                    f"id={item_id} "
                    f"target={target} "
                    f"p_target={probs[target]:.6f}",
                    flush=True,
                )

    # Strict postcondition.
    final_rows = load_jsonl(output_path)

    if len(final_rows) != 500:
        raise RuntimeError(
            f"{output_path}: expected 500 final rows, "
            f"got {len(final_rows)}"
        )

    ids = [str(row["id"]) for row in final_rows]
    if len(set(ids)) != 500:
        raise RuntimeError(
            f"{output_path}: duplicate ids after scoring"
        )

    probability_failures = 0

    for row in final_rows:
        labels = row["candidate_labels"]

        total = sum(
            float(row[f"prob_{label}"])
            for label in labels
        )

        if abs(total - 1.0) > 1e-6:
            probability_failures += 1

        if prepared["key"] == "mmlu":
            if row["score_E"] is not None:
                raise RuntimeError("MMLU score_E must be null")
            if row["prob_E"] is not None:
                raise RuntimeError("MMLU prob_E must be null")

    if probability_failures:
        raise RuntimeError(
            f"{output_path}: "
            f"{probability_failures} probability sums != 1"
        )

    print(
        f"DATASET_COMPLETE {model_key}/{prepared['key']} "
        f"N={len(final_rows)}",
        flush=True,
    )

    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-key",
        required=True,
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/forced_option_plausibility",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
    )

    args = parser.parse_args()

    model_key = args.model_key
    spec = MODEL_SPECS[model_key]

    model_name = spec["model_name"]
    revision = spec["revision"]
    model_path = spec["model_path"]

    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = [
        prepare_dataset(x)
        for x in dataset_specs(model_key)
    ]

    print("===== RUN CONTRACT =====")
    print("model_key:", model_key)
    print("model_name:", model_name)
    print("model_revision:", revision)
    print("model_path:", model_path)

    for ds in prepared:
        print(
            ds["key"],
            "rows=",
            len(ds["rows"]),
            "dataset_md5=",
            ds["dataset_md5"],
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"

    template_hash = hashlib.sha256(
        (tokenizer.chat_template or "").encode("utf-8")
    ).hexdigest()

    print("chat_template_sha256:", template_hash)

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    print("cuda_device:", torch.cuda.get_device_name(0))
    print("===== LOADING MODEL ONCE =====", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model.eval()
    model.to("cuda")

    print("MODEL_LOADED", flush=True)

    for ds in prepared:
        score_one_dataset(
            prepared=ds,
            model=model,
            tokenizer=tokenizer,
            model_key=model_key,
            model_name=model_name,
            revision=revision,
            template_hash=template_hash,
            output_dir=output_dir,
            resume=args.resume,
        )

    print()
    print(
        f"MODEL_PAIR_SCORING_COMPLETE: {model_key}",
        flush=True,
    )


if __name__ == "__main__":
    main()
