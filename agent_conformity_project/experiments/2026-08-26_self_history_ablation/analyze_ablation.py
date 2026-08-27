"""Paired analysis for the Self-History Ablation.

Dependency-free (stdlib only). Reuses the repo's exact McNemar helper. Works on
both the new ablation JSONL schema and the archived Exp2/Exp3 schema.

For each comparison it reports, on the COMMON-VALID paired subset:
  * per-condition N, valid N, accuracy, harmful-conformity, target-distractor
    adoption, flip(change) rate
  * pairwise percentage-point differences
  * exact two-sided McNemar p (on correctness and on distractor adoption)
  * question-level paired bootstrap 95% CI (10,000 resamples; both conditions of
    a question resampled together)

Standard comparisons when --run-dir is given:
  A  stepwise_no_history            vs sequential_final          (harness/determinism check)
  B  stepwise_answer_history        vs stepwise_no_history       (self-answer commitment)
  C  stepwise_answer_confidence...  vs stepwise_answer_history   (self-reported confidence)
  D  sequential_final_length_matched vs sequential_final         (turn/context structure)
Optionally --exp2/--exp3 add the as-published reference (Exp3 vs Exp2).

Usage:
    python3 analyze_ablation.py --run-dir <run>/2026-08-26_self_history_ablation
    python3 analyze_ablation.py --exp2 <exp2.jsonl> --exp3 <exp3.jsonl>   # as-published check
"""

import argparse
import json
import os
import random
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "analysis"))
from exact_mcnemar import exact_mcnemar_pvalue  # noqa: E402  (reuse tested helper)

OPTION_LABELS = ("A", "B", "C", "D", "E")
N_BOOT = 10000
BOOT_SEED = 12345


def normalize_row(row):
    """Extract a schema-agnostic record from an ablation OR legacy Exp2/3 row."""
    final = row.get("final_prediction", row.get("attack_prediction"))
    initial = row.get("initial_prediction")
    correct = row.get("correct_answer")
    distractor = row.get("distractor")
    valid = row.get("final_valid")
    if valid is None:
        valid = final in OPTION_LABELS
    return {
        "id": row.get("id"),
        "final": final,
        "initial": initial,
        "correct": correct,
        "distractor": distractor,
        "valid": bool(valid),
        "is_correct": final == correct,
        "chose_distractor": final == distractor,
        "changed": (final != initial) if initial is not None else None,
        "initial_correct": (initial == correct) if initial is not None else None,
    }


def load_condition(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = normalize_row(json.loads(line))
            out[r["id"]] = r
    return out


def rate(records, metric):
    vals = [metric(r) for r in records if metric(r) is not None]
    return (sum(vals) / len(vals)) if vals else float("nan")


def harmful_conformity(records):
    """Among items initially correct, fraction that end on the target distractor."""
    denom = [r for r in records if r["initial_correct"]]
    if not denom:
        return float("nan"), 0
    num = sum(1 for r in denom if r["chose_distractor"])
    return num / len(denom), len(denom)


def summarize(name, cond):
    recs = list(cond.values())
    valid = [r for r in recs if r["valid"]]
    hcr, hcr_n = harmful_conformity(valid)
    print(f"  [{name}] N={len(recs)} valid={len(valid)} invalid={len(recs) - len(valid)}")
    print(f"        accuracy={rate(valid, lambda r: r['is_correct']):.4f}  "
          f"distractor_adoption={rate(valid, lambda r: r['chose_distractor']):.4f}  "
          f"flip_rate={rate(valid, lambda r: r['changed']):.4f}  "
          f"harmful_conformity={hcr:.4f} (den={hcr_n})")


def paired_bootstrap_ci(ids, c1, c2, metric, n=N_BOOT, seed=BOOT_SEED):
    rng = random.Random(seed)
    v1 = [1.0 if metric(c1[i]) else 0.0 for i in ids]
    v2 = [1.0 if metric(c2[i]) else 0.0 for i in ids]
    N = len(ids)
    point = (sum(v1) - sum(v2)) / N
    diffs = []
    for _ in range(n):
        s = [rng.randrange(N) for _ in range(N)]  # resample QUESTIONS (both conds together)
        diffs.append((sum(v1[j] for j in s) - sum(v2[j] for j in s)) / N)
    diffs.sort()
    return point, diffs[int(0.025 * n)], diffs[min(n - 1, int(0.975 * n))]


def mcnemar(ids, c1, c2, metric):
    b = sum(1 for i in ids if metric(c1[i]) and not metric(c2[i]))
    c = sum(1 for i in ids if not metric(c1[i]) and metric(c2[i]))
    return b, c, exact_mcnemar_pvalue(b, c)


def compare(label, name1, cond1, name2, cond2):
    ids = sorted(set(cond1) & set(cond2))
    ids = [i for i in ids if cond1[i]["valid"] and cond2[i]["valid"]]
    print(f"\n{'='*88}\n{label}:  {name1}  vs  {name2}\n{'='*88}")
    print(f"  common-valid paired N = {len(ids)}")
    if not ids:
        print("  (no common-valid pairs)")
        return
    for metric_name, metric in [("accuracy", lambda r: r["is_correct"]),
                                ("distractor_adoption", lambda r: r["chose_distractor"])]:
        r1 = sum(metric(cond1[i]) for i in ids) / len(ids)
        r2 = sum(metric(cond2[i]) for i in ids) / len(ids)
        point, lo, hi = paired_bootstrap_ci(ids, cond1, cond2, metric)
        b, c, p = mcnemar(ids, cond1, cond2, metric)
        print(f"  [{metric_name}] {name1}={r1:.4f}  {name2}={r2:.4f}  "
              f"Δ={(r1 - r2) * 100:+.2f}pp  boot95%=[{lo*100:+.2f},{hi*100:+.2f}]pp  "
              f"McNemar b={b} c={c} p={p:.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None, help="ablation run dir with {condition}.jsonl")
    ap.add_argument("--exp2", default=None, help="archived Exp2 jsonl (as-published Sequential-Final)")
    ap.add_argument("--exp3", default=None, help="archived Exp3 jsonl (as-published Stepwise)")
    args = ap.parse_args()

    conds = {}
    if args.run_dir:
        for name in ["sequential_final", "stepwise_no_history", "stepwise_answer_history",
                     "stepwise_answer_confidence_history", "sequential_final_length_matched"]:
            p = os.path.join(args.run_dir, f"{name}.jsonl")
            if os.path.exists(p):
                conds[name] = load_condition(p)
    if args.exp2:
        conds["as_published_sequential_final(exp2)"] = load_condition(args.exp2)
    if args.exp3:
        conds["as_published_stepwise(exp3)"] = load_condition(args.exp3)

    if not conds:
        print("Nothing to analyze. Provide --run-dir and/or --exp2/--exp3.")
        raise SystemExit(2)

    print("PER-CONDITION SUMMARY")
    for name, cond in conds.items():
        summarize(name, cond)

    def have(*names):
        return all(n in conds for n in names)

    if have("stepwise_no_history", "sequential_final"):
        compare("Comparison A", "stepwise_no_history", conds["stepwise_no_history"],
                "sequential_final", conds["sequential_final"])
    if have("stepwise_answer_history", "stepwise_no_history"):
        compare("Comparison B", "stepwise_answer_history", conds["stepwise_answer_history"],
                "stepwise_no_history", conds["stepwise_no_history"])
    if have("stepwise_answer_confidence_history", "stepwise_answer_history"):
        compare("Comparison C", "stepwise_answer_confidence_history", conds["stepwise_answer_confidence_history"],
                "stepwise_answer_history", conds["stepwise_answer_history"])
    if have("sequential_final_length_matched", "sequential_final"):
        compare("Comparison D", "sequential_final_length_matched", conds["sequential_final_length_matched"],
                "sequential_final", conds["sequential_final"])
    if have("as_published_stepwise(exp3)", "as_published_sequential_final(exp2)"):
        compare("As-published reference", "as_published_stepwise(exp3)", conds["as_published_stepwise(exp3)"],
                "as_published_sequential_final(exp2)", conds["as_published_sequential_final(exp2)"])


if __name__ == "__main__":
    main()
