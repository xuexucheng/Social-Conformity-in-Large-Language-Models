#!/usr/bin/env python3
"""Select two independent neutral controls matched to observed self-history.

This is a tokenizer-only preflight.  It reads a completed
stepwise_answer_confidence_history JSONL, measures the token-length distribution
of the four intermediate turns that actually enter each final prompt, and
selects two distinct curated neutral phrasings at the modal length.  It never
runs a model and fails closed when fewer than two exact matches are available.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Iterable


PREFIXES = (
    "Acknowledged.",
    "Message received.",
    "Received.",
    "Noted.",
    "The message has been received.",
    "The message has been noted.",
    "Receipt of the message is noted.",
)

SUFFIXES = (
    "Proceeding to the next message.",
    "Continuing to the next message.",
    "Moving to the following message.",
    "Proceeding to the following message.",
    "Continuing with the next message.",
    "Moving forward to the next message.",
    "The next message can now follow.",
    "Continuing.",
)

EXPLICIT_CANDIDATES = (
    "Acknowledged. Message received. Proceeding to the next message.",
    "Message received. Acknowledged. Moving to the following message.",
    "Acknowledged. The message has been received. Proceeding to the next message.",
    "Received. The message has been noted. Continuing to the next message.",
    "Message received. It has been noted. Continuing to the following message.",
    "The message has been received. Acknowledged. Proceeding to the next message.",
    "The message has been noted. Received. Moving to the following message.",
    "Acknowledged. Proceeding directly to the next message now.",
    "Message received. Proceeding directly to the next message now.",
)

FORBIDDEN_STANCE_TERMS = (
    "answer",
    "confidence",
    "correct",
    "incorrect",
    "wrong",
    "agree",
    "disagree",
    "choice",
    "option",
    "decision",
    "opinion",
    "persuasive",
)


def curated_candidates(extra: Iterable[str] = ()) -> list[str]:
    candidates = list(EXPLICIT_CANDIDATES)
    candidates.extend(f"{prefix} {suffix}" for prefix in PREFIXES for suffix in SUFFIXES)
    candidates.extend(extra)
    output = []
    seen = set()
    for candidate in candidates:
        normalized = " ".join(str(candidate).split())
        if not normalized or normalized in seen:
            continue
        lowered = normalized.lower()
        if any(term in lowered for term in FORBIDDEN_STANCE_TERMS):
            raise ValueError(f"candidate contains a stance-bearing term: {normalized!r}")
        seen.add(normalized)
        output.append(normalized)
    return output


def load_observed_history(path: str | Path) -> list[str]:
    path = Path(path)
    turns = []
    seen_ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = row.get("id")
            if item_id is None:
                raise ValueError(f"missing id at {path}:{line_number}")
            if item_id in seen_ids:
                raise ValueError(f"duplicate id {item_id!r} at {path}:{line_number}")
            seen_ids.add(item_id)
            steps = row.get("step_outputs") or row.get("attack_step_outputs") or []
            if len(steps) < 5:
                raise ValueError(
                    f"item {item_id!r} has {len(steps)} step outputs; expected at least 5"
                )
            for step in steps[:4]:
                text = step.get("raw_text", step.get("text", step.get("output")))
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"item {item_id!r} contains an empty self-history turn")
                turns.append(text.strip())
    if not turns:
        raise ValueError(f"no observed self-history turns in {path}")
    return turns


def select_candidates(
    tokenizer: Any,
    observed_turns: list[str],
    candidates: list[str],
) -> dict[str, Any]:
    observed_counts = [
        len(tokenizer.encode(text, add_special_tokens=False)) for text in observed_turns
    ]
    distribution = collections.Counter(observed_counts)
    # Deterministic tie-break: most frequent, then shorter token length.
    target_tokens = min(
        distribution,
        key=lambda count: (-distribution[count], count),
    )
    candidate_rows = [
        {
            "text": text,
            "tokens": len(tokenizer.encode(text, add_special_tokens=False)),
        }
        for text in candidates
    ]
    matches = [row for row in candidate_rows if row["tokens"] == target_tokens]
    if len(matches) < 2:
        counts = collections.Counter(row["tokens"] for row in candidate_rows)
        raise ValueError(
            "fewer than two independent neutral candidates match the observed modal "
            f"self-history length ({target_tokens} tokens); candidate token counts={dict(sorted(counts.items()))}"
        )
    return {
        "target_tokens": target_tokens,
        "observed_turn_n": len(observed_counts),
        "observed_token_distribution": {
            str(key): value for key, value in sorted(distribution.items())
        },
        "neutral_v1": matches[0]["text"],
        "neutral_v2": matches[1]["text"],
        "neutral_v1_tokens": matches[0]["tokens"],
        "neutral_v2_tokens": matches[1]["tokens"],
        "matching_candidate_n": len(matches),
        "matching_candidates": matches,
        "all_candidate_counts": candidate_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--self-history-jsonl", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="additional human-reviewed neutral candidate; repeatable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True
    )
    turns = load_observed_history(args.self_history_jsonl)
    payload = {
        "model_name": args.model_name,
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "self_history_jsonl": str(Path(args.self_history_jsonl).resolve()),
        **select_candidates(tokenizer, turns, curated_candidates(args.candidate)),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"TARGET_TOKENS={payload['target_tokens']}")
    print(f"NEUTRAL_V1={payload['neutral_v1']!r}")
    print(f"NEUTRAL_V2={payload['neutral_v2']!r}")
    print(f"OUTPUT_JSON={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
