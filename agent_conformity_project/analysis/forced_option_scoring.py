#!/usr/bin/env python3
"""Pre-social forced-option plausibility scoring.

For every item, reconstruct the exact private-baseline prompt used by the paper
artifact, apply the same model-specific message normalization, and score the
full canonical completion sequence "ANSWER: X" for every REAL option X.

CSQA therefore uses A-E; MMLU uses A-D. Probabilities are normalized only over
the options actually present in the item.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.llm_api import normalize_messages_for_model  # noqa: E402
from src.prompts import build_prompt  # noqa: E402


ALL_LABELS = tuple("ABCDE")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_labels(options: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(options, dict) or not options:
        raise ValueError("options must be a non-empty mapping")

    labels = tuple(str(k).strip().upper() for k in options)

    if labels not in (tuple("ABCD"), tuple("ABCDE")):
        raise ValueError(f"unsupported option labels/order: {labels}")

    return labels


def stable_softmax(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("scores cannot be empty")
    if not all(math.isfinite(x) for x in scores):
        raise ValueError(f"non-finite score: {scores}")

    m = max(scores)
    exps = [math.exp(x - m) for x in scores]
    z = sum(exps)
    return [x / z for x in exps]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON {path}:{line_no}: {exc}") from exc

            item_id = row.get("id", row.get("item_id"))
            if item_id is None:
                raise ValueError(f"missing id {path}:{line_no}")

            item_id = str(item_id)
            if item_id in seen:
                raise ValueError(f"duplicate id {item_id!r}")
            seen.add(item_id)

            rows.append(row)

    if not rows:
        raise ValueError(f"no rows in {path}")

    return rows


def load_dataset_map(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a JSON list: {path}")

    result = {}
    for i, row in enumerate(data):
        item_id = row.get("id", row.get("item_id"))
        if item_id is None:
            raise ValueError(f"dataset row {i} missing id")
        item_id = str(item_id)
        if item_id in result:
            raise ValueError(f"duplicate dataset id {item_id!r}")
        result[item_id] = row
    return result


def audit_against_dataset(
    rows: list[dict[str, Any]],
    dataset_map: dict[str, dict[str, Any]],
) -> None:
    fields = ("question", "options", "correct_answer", "distractor")

    for row in rows:
        item_id = str(row.get("id", row.get("item_id")))
        if item_id not in dataset_map:
            raise ValueError(f"id {item_id!r} absent from tracked dataset")

        ref = dataset_map[item_id]
        changed = [name for name in fields if row.get(name) != ref.get(name)]
        if changed:
            raise ValueError(
                f"tracked-dataset mismatch id={item_id!r}, fields={changed}"
            )


def build_private_prompt(tokenizer, row: dict[str, Any], model_name: str):
    item = {
        "question": row["question"],
        "options": row["options"],
    }

    messages = build_prompt(item, opinions=[])
    normalized = normalize_messages_for_model(messages, model_name)

    rendered = tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.apply_chat_template(
        normalized,
        tokenize=True,
        add_generation_prompt=True,
    )

    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("chat template produced invalid prompt token ids")

    return normalized, rendered, prompt_ids


def encode_candidates(tokenizer, labels: tuple[str, ...]):
    result = {}
    for label in labels:
        text = f"ANSWER: {label}"
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"candidate {text!r} tokenized to empty sequence")
        result[label] = {"text": text, "ids": ids}
    return result


def sequence_logprob_from_logits(
    logits,
    prompt_len: int,
    candidate_ids: list[int],
) -> float:
    """Sum log p(candidate_j | prompt, previous candidate tokens).

    logits[t] predicts token t+1, hence candidate token at sequence position
    prompt_len+j is scored from logits[prompt_len+j-1].
    """
    import torch

    if prompt_len < 1:
        raise ValueError("prompt_len must be >= 1")
    if not candidate_ids:
        raise ValueError("candidate_ids cannot be empty")

    positions = torch.arange(
        prompt_len - 1,
        prompt_len - 1 + len(candidate_ids),
        device=logits.device,
    )
    selected = logits.index_select(0, positions).float()
    log_probs = torch.log_softmax(selected, dim=-1)

    targets = torch.tensor(
        candidate_ids,
        device=logits.device,
        dtype=torch.long,
    )

    return float(
        log_probs.gather(1, targets.unsqueeze(1)).sum().item()
    )


def score_candidates(
    model,
    tokenizer,
    prompt_ids: list[int],
    candidates: dict[str, dict[str, Any]],
    device: str,
) -> dict[str, float]:
    import torch

    labels = list(candidates)
    sequences = [
        prompt_ids + candidates[label]["ids"]
        for label in labels
    ]

    max_len = max(map(len, sequences))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")

    input_ids = torch.full(
        (len(sequences), max_len),
        int(pad_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_len),
        dtype=torch.long,
        device=device,
    )

    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = torch.tensor(
            seq,
            dtype=torch.long,
            device=device,
        )
        attention_mask[i, : len(seq)] = 1

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    scores = {}
    prompt_len = len(prompt_ids)

    for i, label in enumerate(labels):
        cand_ids = candidates[label]["ids"]
        scores[label] = sequence_logprob_from_logits(
            output.logits[i],
            prompt_len,
            cand_ids,
        )

    del output, input_ids, attention_mask
    return scores


def validate_existing_output(
    path: Path,
    model_name: str,
    model_revision: str,
    dataset_md5: str,
) -> set[str]:
    if not path.exists():
        return set()

    done = set()
    for row in load_jsonl(path):
        if row.get("model") != model_name:
            raise ValueError("resume output model mismatch")
        if row.get("model_revision") != model_revision:
            raise ValueError("resume output model revision mismatch")
        if row.get("tokenizer_revision") != model_revision:
            raise ValueError("resume output tokenizer revision mismatch")
        if row.get("dataset_md5") != dataset_md5:
            raise ValueError("resume output dataset MD5 mismatch")
        done.add(str(row["id"]))

    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl).resolve()
    dataset_path = Path(args.dataset_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output_jsonl).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    rows = load_jsonl(input_path)
    if len(rows) != 500:
        raise ValueError(f"expected 500 main-run rows, got {len(rows)}")

    dataset_map = load_dataset_map(dataset_path)
    audit_against_dataset(rows, dataset_map)

    dataset_md5 = file_md5(dataset_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"

    template = tokenizer.chat_template or ""
    template_hash = sha256_text(template)

    # Prompt/tokenizer-only audit before touching a GPU.
    option_counts = {}
    prompt_lengths = []
    candidate_tokenizations = {}

    for row in rows:
        labels = candidate_labels(row["options"])
        option_counts[len(labels)] = option_counts.get(len(labels), 0) + 1

        _, rendered, prompt_ids = build_private_prompt(
            tokenizer,
            row,
            args.model_name,
        )
        prompt_lengths.append(len(prompt_ids))

        encoded = encode_candidates(tokenizer, labels)
        for label, info in encoded.items():
            candidate_tokenizations.setdefault(
                label,
                tuple(info["ids"]),
            )
            if candidate_tokenizations[label] != tuple(info["ids"]):
                raise ValueError(
                    f"candidate tokenization unexpectedly changed for {label}"
                )

        if not rendered:
            raise ValueError("empty rendered prompt")

    print("===== PREFLIGHT =====")
    print("dataset:", args.dataset_name)
    print("model:", args.model_name)
    print("model_revision:", args.model_revision)
    print("dataset_md5:", dataset_md5)
    print("rows:", len(rows))
    print("option_counts:", option_counts)
    print("prompt_tokens_min:", min(prompt_lengths))
    print("prompt_tokens_max:", max(prompt_lengths))
    print("chat_template_sha256:", template_hash)
    print(
        "candidate_token_ids:",
        {k: list(v) for k, v in candidate_tokenizations.items()},
    )

    if args.preflight_only:
        print("FORCED_OPTION_PREFLIGHT_OK")
        return

    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("GPU scoring requested but CUDA is unavailable")

    device = "cuda"

    print("===== LOAD MODEL =====", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.to(device)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed = (
        validate_existing_output(
            output_path,
            args.model_name,
            args.model_revision,
            dataset_md5,
        )
        if args.resume
        else set()
    )

    if output_path.exists() and not args.resume:
        raise FileExistsError(
            f"{output_path} exists; use --resume or remove explicitly"
        )

    mode = "a" if args.resume else "w"

    with output_path.open(mode, encoding="utf-8") as out:
        for index, row in enumerate(rows, 1):
            item_id = str(row.get("id", row.get("item_id")))
            if item_id in completed:
                continue

            labels = candidate_labels(row["options"])
            _, rendered, prompt_ids = build_private_prompt(
                tokenizer,
                row,
                args.model_name,
            )
            candidates = encode_candidates(tokenizer, labels)

            scores = score_candidates(
                model,
                tokenizer,
                prompt_ids,
                candidates,
                device,
            )

            probs_list = stable_softmax([scores[x] for x in labels])
            probs = dict(zip(labels, probs_list))

            correct = str(row["correct_answer"]).upper()
            target = str(row["distractor"]).upper()

            if correct not in labels:
                raise ValueError(f"{item_id}: correct {correct} absent from options")
            if target not in labels:
                raise ValueError(f"{item_id}: distractor {target} absent from options")
            if correct == target:
                raise ValueError(f"{item_id}: correct equals distractor")

            result = {
                "id": item_id,
                "model": args.model_name,
                "dataset": args.dataset_name,
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
                "model_revision": args.model_revision,
                "tokenizer_revision": args.model_revision,
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

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()

            if index % 25 == 0 or index == len(rows):
                print(
                    f"[{index}/{len(rows)}] "
                    f"id={item_id} target={target} "
                    f"p_target={probs[target]:.6f}",
                    flush=True,
                )

    print("FORCED_OPTION_SCORING_OK")
    print("output:", output_path)


if __name__ == "__main__":
    main()
