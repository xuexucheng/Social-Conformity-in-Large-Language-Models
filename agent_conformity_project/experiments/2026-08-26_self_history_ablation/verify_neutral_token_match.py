"""Fail-closed tokenizer preflight for the two neutral-control turns.

The control strings are resolved from conditions.py, their single source of
truth.  This script performs no model inference and is intended to run before
the vLLM server starts.
"""

import argparse
import os

import conditions as C


def count_control_tokens(tokenizer, model_name):
    neutral_v1 = C.resolve_neutral_turn(model_name)
    neutral_v2 = C.resolve_neutral_v2_turn(model_name)
    return {
        "neutral_v1": neutral_v1,
        "neutral_v2": neutral_v2,
        "neutral_v1_tokens": len(
            tokenizer.encode(neutral_v1, add_special_tokens=False)
        ),
        "neutral_v2_tokens": len(
            tokenizer.encode(neutral_v2, add_special_tokens=False)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify Neutral-V1/V2 token equality")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True
    )
    audit = count_control_tokens(tokenizer, args.model_name)
    print(f"Neutral-V1 text  = {audit['neutral_v1']!r}")
    print(f"Neutral-V2 text  = {audit['neutral_v2']!r}")
    print(f"Neutral-V1 tokens= {audit['neutral_v1_tokens']}")
    print(f"Neutral-V2 tokens= {audit['neutral_v2_tokens']}")
    if audit["neutral_v1_tokens"] != audit["neutral_v2_tokens"]:
        raise SystemExit(
            "[FATAL] Neutral-V2 is not token-length matched to Neutral-V1: "
            f"{audit['neutral_v2_tokens']} vs {audit['neutral_v1_tokens']}"
        )
    print("NEUTRAL_TOKEN_MATCH_OK")


if __name__ == "__main__":
    main()
