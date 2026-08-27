"""Exact per-tokenizer length check for the Length-Matched neutral turn.

Run on AutoDL (where each model + tokenizer is already present) BEFORE running a
model's batch, so the neutral turn is length-matched on that model's OWN
tokenizer against that model's OWN self-history turn. Report to the human and
DO NOT change the design silently.

Usage:
    python3 compute_neutral_token_match.py <MODEL_REPO> <ARCHIVED_EXP3_JSONL>

Example:
    python3 compute_neutral_token_match.py google/gemma-2-2b-it \
        results/runs/five_wrong_guidance_commonsenseqa500_gemma2_2b_20260504_215000/2026-04-29_five_wrong_guidance/exp3_sequential_answer_each_step.jsonl
"""

import collections
import json
import statistics
import sys

from transformers import AutoTokenizer

# Candidate family (all pass the neutrality rules). The first is the Qwen-confirmed default.
CANDIDATES = [
    "Acknowledged. Message received. Proceeding to the next message.",
    "Acknowledged. Message received. Continuing to the next message.",
    "Acknowledged. Message received. Continuing.",
    "Received. Acknowledged. Message received. Continuing.",
    "Acknowledged. Message received. Proceeding to the next message now.",
]


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    repo, exp3_path = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(repo)

    counts = collections.Counter()
    with open(exp3_path, encoding="utf-8") as f:
        for line in f:
            for s in (json.loads(line).get("attack_step_outputs") or []):
                counts[s.get("text") or ""] += 1
    lens = [len(tok.encode(t, add_special_tokens=False)) for t, c in counts.items() for _ in range(c)]
    target = statistics.mode(lens)
    print(f"model              : {repo}")
    print(f"tokenizer          : {type(tok).__name__}")
    print(f"self-history turns  : n={len(lens)} mode={target} median={int(statistics.median(lens))} "
          f"min={min(lens)} max={max(lens)}")
    print(f"target token length = {target}\n")
    print(f"{'tok':>4} {'Δ':>4}  candidate")
    print("-" * 80)
    best = None
    for c in CANDIDATES:
        n = len(tok.encode(c, add_special_tokens=False))
        d = n - target
        print(f"{n:>4} {d:>+4}  {c!r}")
        if best is None or abs(d) < abs(best[1]):
            best = (c, d, n)
    print(f"\nCLOSEST: {best[0]!r}  ({best[2]} tok, Δ{best[1]:+d})")
    print("If Δ != 0, report to the human before adopting; do not change the design silently.")


if __name__ == "__main__":
    main()
