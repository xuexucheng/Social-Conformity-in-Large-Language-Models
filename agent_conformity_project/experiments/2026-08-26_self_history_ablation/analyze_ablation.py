"""Paired analysis for the Self-History Ablation.

The primary mechanism family is evaluated on common-valid paired items:

  A  No-History vs Sequential-Final              harness/determinism check
  B  Answer-History vs No-History                 assistant turns + answer content
  C  Answer+Confidence vs Answer-History          added confidence line
  D  Neutral-Turn vs Sequential-Final             turn/context structure
  E  Answer+Confidence vs Neutral-Turn             history content beyond matched structure
  F  Answer-History vs Neutral-Turn                answer-only diagnostic (not length matched)
  G  Neutral-V2 vs Sequential-Final                wording-robust structure control
  H  Short-Ack vs Sequential-Final                 short assistant-role-marker control
  I  Neutral-V2 vs Neutral-V1                      neutral-wording sensitivity
  J  Answer+Confidence vs Neutral-V2               content beyond alternative neutral control
  K  Neutral-V1 vs Short-Ack                       added neutral text/length beyond role marker

For accuracy, target-distractor adoption, harmful conformity, and flip rate the
script reports paired percentage-point differences, question-level paired
bootstrap confidence intervals, exact McNemar tests, and Holm-adjusted p-values
within each outcome across inferential comparisons B--E and G--K. Comparisons A
and F are checks/diagnostics and are deliberately excluded from multiplicity
correction. Optional archived Exp2/Exp3 results are descriptive references.
"""

import argparse
import json
import os
import random
import statistics
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(PROJECT_DIR, "analysis"))
from exact_mcnemar import exact_mcnemar_pvalue  # noqa: E402

OPTION_LABELS = ("A", "B", "C", "D", "E")
DEFAULT_N_BOOT = 10000
DEFAULT_BOOT_SEED = 12345

NEW_CONDITION_NAMES = [
    "sequential_final",
    "stepwise_no_history",
    "stepwise_answer_history",
    "stepwise_answer_confidence_history",
    "sequential_final_length_matched",
    "sequential_final_neutral_v2",
    "sequential_final_short_ack",
]

METRICS = [
    ("accuracy", lambda r: r["is_correct"], lambda r: True),
    ("distractor_adoption", lambda r: r["chose_distractor"], lambda r: True),
    ("harmful_conformity", lambda r: r["chose_distractor"], lambda r: r["initial_correct"] is True),
    ("flip_rate", lambda r: r["changed"], lambda r: r["changed"] is not None),
]


def normalize_row(row):
    """Extract a schema-agnostic record from an ablation or legacy Exp2/3 row."""
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
        "question": row.get("question"),
        "options": row.get("options"),
        "agent_opinions": row.get("agent_opinions"),
        "raw_initial_output": row.get("raw_initial_output"),
        "step_outputs": row.get("step_outputs", row.get("attack_step_outputs", [])) or [],
        "final_prompt_token_count": row.get("final_prompt_token_count"),
    }


def load_condition(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = normalize_row(json.loads(line))
            item_id = row["id"]
            if item_id is None:
                raise ValueError(f"missing item id in {path}:{line_number}")
            if item_id in out:
                raise ValueError(f"duplicate item id {item_id!r} in {path}:{line_number}")
            out[item_id] = row
    return out


def _rate(records, metric):
    values = [metric(record) for record in records]
    values = [value for value in values if value is not None]
    return (sum(values) / len(values)) if values else float("nan")


def summarize_condition(name, condition):
    records = list(condition.values())
    valid = [record for record in records if record["valid"]]
    initially_correct = [record for record in valid if record["initial_correct"] is True]
    summary = {
        "condition": name,
        "n": len(records),
        "valid_n": len(valid),
        "invalid_n": len(records) - len(valid),
        "accuracy": _rate(valid, lambda r: r["is_correct"]),
        "distractor_adoption": _rate(valid, lambda r: r["chose_distractor"]),
        "flip_rate": _rate(valid, lambda r: r["changed"]),
        "harmful_conformity": _rate(initially_correct, lambda r: r["chose_distractor"]),
        "harmful_conformity_n": len(initially_correct),
    }
    print(
        f"  [{name}] N={summary['n']} valid={summary['valid_n']} "
        f"invalid={summary['invalid_n']}"
    )
    print(
        f"        accuracy={summary['accuracy']:.4f}  "
        f"distractor_adoption={summary['distractor_adoption']:.4f}  "
        f"flip_rate={summary['flip_rate']:.4f}  "
        f"harmful_conformity={summary['harmful_conformity']:.4f} "
        f"(den={summary['harmful_conformity_n']})"
    )
    return summary


def audit_new_conditions(conditions):
    """Reject ID or item-level metadata drift across the new ablation family."""
    available = [name for name in NEW_CONDITION_NAMES if name in conditions]
    if len(available) < 2:
        return
    reference_name = available[0]
    reference = conditions[reference_name]
    fields = [
        "question",
        "options",
        "correct",
        "distractor",
        "agent_opinions",
        "initial",
        "raw_initial_output",
    ]
    for name in available[1:]:
        current = conditions[name]
        if set(current) != set(reference):
            raise ValueError(
                f"ID-set mismatch: {reference_name} has {len(reference)} IDs, "
                f"{name} has {len(current)}"
            )
        mismatches = [
            item_id
            for item_id in reference
            if any(reference[item_id][field] != current[item_id][field] for field in fields)
        ]
        if mismatches:
            raise ValueError(
                f"metadata mismatch between {reference_name} and {name} on "
                f"{len(mismatches)} item(s); first={mismatches[0]!r}"
            )
    print(
        f"INTEGRITY AUDIT: PASS ({len(reference)} identical IDs and item metadata "
        f"across {len(available)} new conditions)"
    )


def _paired_ids(condition1, condition2, eligibility, allowed_ids=None):
    ids = sorted(set(condition1) & set(condition2))
    if allowed_ids is not None:
        ids = [item_id for item_id in ids if item_id in allowed_ids]
    return [
        item_id
        for item_id in ids
        if condition1[item_id]["valid"]
        and condition2[item_id]["valid"]
        and eligibility(condition1[item_id])
        and eligibility(condition2[item_id])
    ]


def paired_bootstrap_ci(ids, condition1, condition2, metric, n, seed):
    values1 = [1.0 if metric(condition1[item_id]) else 0.0 for item_id in ids]
    values2 = [1.0 if metric(condition2[item_id]) else 0.0 for item_id in ids]
    sample_n = len(ids)
    point = (sum(values1) - sum(values2)) / sample_n
    rng = random.Random(seed)
    differences = []
    for _ in range(n):
        sampled = [rng.randrange(sample_n) for _ in range(sample_n)]
        differences.append(
            (
                sum(values1[index] for index in sampled)
                - sum(values2[index] for index in sampled)
            )
            / sample_n
        )
    differences.sort()
    return (
        point,
        differences[int(0.025 * n)],
        differences[min(n - 1, int(0.975 * n))],
    )


def mcnemar(ids, condition1, condition2, metric):
    b = sum(
        1
        for item_id in ids
        if metric(condition1[item_id]) and not metric(condition2[item_id])
    )
    c = sum(
        1
        for item_id in ids
        if not metric(condition1[item_id]) and metric(condition2[item_id])
    )
    return b, c, exact_mcnemar_pvalue(b, c)


def build_comparison(
    label,
    name1,
    condition1,
    name2,
    condition2,
    n_boot,
    seed,
    adjust,
    allowed_ids=None,
    analysis_subset="all_common_valid",
):
    rows = []
    for metric_name, metric, eligibility in METRICS:
        ids = _paired_ids(condition1, condition2, eligibility, allowed_ids=allowed_ids)
        if not ids:
            continue
        rate1 = sum(metric(condition1[item_id]) for item_id in ids) / len(ids)
        rate2 = sum(metric(condition2[item_id]) for item_id in ids) / len(ids)
        point, low, high = paired_bootstrap_ci(
            ids, condition1, condition2, metric, n_boot, seed
        )
        b, c, p_value = mcnemar(ids, condition1, condition2, metric)
        rows.append(
            {
                "comparison": label,
                "condition1": name1,
                "condition2": name2,
                "metric": metric_name,
                "paired_n": len(ids),
                "rate1": rate1,
                "rate2": rate2,
                "delta": point,
                "ci_low": low,
                "ci_high": high,
                "mcnemar_b": b,
                "mcnemar_c": c,
                "p_raw": p_value,
                "p_holm": None,
                "holm_family": "B-E and G-K within metric" if adjust else None,
                "adjust": adjust,
                "analysis_subset": analysis_subset,
            }
        )
    return rows


def holm_adjust(p_values):
    """Return Holm step-down adjusted p-values in original order."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [None] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def apply_holm_within_metric(results):
    for metric_name, _, _ in METRICS:
        family = [
            row
            for row in results
            if row["adjust"] and row["metric"] == metric_name
        ]
        adjusted = holm_adjust([row["p_raw"] for row in family])
        for row, p_value in zip(family, adjusted):
            row["p_holm"] = p_value


def print_comparison(label, rows):
    if not rows:
        return
    first = rows[0]
    print(f"\n{'=' * 96}\n{label}:  {first['condition1']}  vs  {first['condition2']}\n{'=' * 96}")
    for row in rows:
        holm = f" Holm={row['p_holm']:.4g}" if row["p_holm"] is not None else ""
        print(
            f"  [{row['metric']}] paired_N={row['paired_n']}  "
            f"{row['condition1']}={row['rate1']:.4f}  "
            f"{row['condition2']}={row['rate2']:.4f}  "
            f"delta={row['delta'] * 100:+.2f}pp  "
            f"boot95%=[{row['ci_low'] * 100:+.2f},{row['ci_high'] * 100:+.2f}]pp  "
            f"McNemar b={row['mcnemar_b']} c={row['mcnemar_c']} "
            f"p={row['p_raw']:.4g}{holm}"
        )


def print_token_audit(name, condition):
    counts = [
        row["final_prompt_token_count"]
        for row in condition.values()
        if row["final_prompt_token_count"] is not None
    ]
    if not counts:
        print(f"  [{name}] final prompt token counts: MISSING")
        return
    print(
        f"  [{name}] final prompt tokens: n={len(counts)} "
        f"min={min(counts)} median={statistics.median(counts):g} max={max(counts)}"
    )


def build_token_pair_audit(name1, condition1, name2, condition2):
    """Describe paired prompt-token equality and return the exact-match IDs."""
    common_ids = sorted(set(condition1) & set(condition2))
    covered = [
        item_id
        for item_id in common_ids
        if condition1[item_id]["final_prompt_token_count"] is not None
        and condition2[item_id]["final_prompt_token_count"] is not None
    ]
    if not covered:
        return {
            "condition1": name1,
            "condition2": name2,
            "common_n": len(common_ids),
            "covered_n": 0,
            "exact_match_n": 0,
            "exact_match_ids": [],
            "difference_counts": {},
            "mismatches": [],
        }
    differences = {
        item_id: (
            condition1[item_id]["final_prompt_token_count"]
            - condition2[item_id]["final_prompt_token_count"]
        )
        for item_id in covered
    }
    exact_ids = [item_id for item_id in covered if differences[item_id] == 0]
    difference_counts = {}
    for difference in differences.values():
        key = str(difference)
        difference_counts[key] = difference_counts.get(key, 0) + 1
    return {
        "condition1": name1,
        "condition2": name2,
        "common_n": len(common_ids),
        "covered_n": len(covered),
        "exact_match_n": len(exact_ids),
        "exact_match_ids": exact_ids,
        "difference_counts": difference_counts,
        "mismatches": [
            {
                "id": item_id,
                "condition1_tokens": condition1[item_id]["final_prompt_token_count"],
                "condition2_tokens": condition2[item_id]["final_prompt_token_count"],
                "difference": differences[item_id],
            }
            for item_id in covered
            if differences[item_id] != 0
        ],
    }


def print_token_pair_audit(audit):
    print(
        f"  [{audit['condition1']} - {audit['condition2']}] "
        f"covered={audit['covered_n']}/{audit['common_n']} "
        f"exact={audit['exact_match_n']} differences={audit['difference_counts']}"
    )
    if audit["mismatches"]:
        preview = ", ".join(
            f"{row['id']}:{row['difference']:+d}" for row in audit["mismatches"][:10]
        )
        print(f"        mismatches (condition1-condition2; first 10): {preview}")


def print_stepwise_trajectory(name, condition):
    records = list(condition.values())
    n_steps = max((len(record["step_outputs"]) for record in records), default=0)
    if n_steps < 2:
        return
    print(f"  [{name}]")
    for step_index in range(n_steps):
        available = [
            (record, record["step_outputs"][step_index])
            for record in records
            if len(record["step_outputs"]) > step_index
            and record["step_outputs"][step_index].get("prediction") in OPTION_LABELS
        ]
        if not available:
            continue
        accuracy = sum(
            step.get("prediction") == record["correct"] for record, step in available
        ) / len(available)
        adoption = sum(
            step.get("prediction") == record["distractor"] for record, step in available
        ) / len(available)
        initially_correct = [
            (record, step)
            for record, step in available
            if record["initial_correct"] is True
        ]
        harmful = (
            sum(
                step.get("prediction") == record["distractor"]
                for record, step in initially_correct
            )
            / len(initially_correct)
            if initially_correct
            else float("nan")
        )
        logprob_covered = sum(
            bool(step.get("option_logprobs"))
            and any(value is not None for value in step["option_logprobs"].values())
            for _, step in available
        )
        print(
            f"        step={step_index + 1} valid={len(available)} "
            f"accuracy={accuracy:.4f} distractor_adoption={adoption:.4f} "
            f"harmful_conformity={harmful:.4f} logprob_coverage={logprob_covered}/{len(available)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=JSONL",
        help="add a condition file from another run directory (repeatable)",
    )
    parser.add_argument("--exp2", default=None)
    parser.add_argument("--exp3", default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOT_SEED)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    if args.bootstrap_reps < 100:
        raise SystemExit("--bootstrap-reps must be at least 100")

    conditions = {}
    if args.run_dir:
        for name in NEW_CONDITION_NAMES:
            path = os.path.join(args.run_dir, f"{name}.jsonl")
            if os.path.exists(path):
                conditions[name] = load_condition(path)
    for specification in args.condition:
        if "=" not in specification:
            raise SystemExit(f"--condition must be NAME=JSONL, got: {specification!r}")
        name, path = specification.split("=", 1)
        name, path = name.strip(), path.strip()
        if not name or not path:
            raise SystemExit(f"--condition must be NAME=JSONL, got: {specification!r}")
        if name in conditions:
            raise SystemExit(f"condition {name!r} was provided more than once")
        if not os.path.isfile(path):
            raise SystemExit(f"condition file not found: {path}")
        conditions[name] = load_condition(path)
    if args.exp2:
        conditions["as_published_sequential_final(exp2)"] = load_condition(args.exp2)
    if args.exp3:
        conditions["as_published_stepwise(exp3)"] = load_condition(args.exp3)
    if not conditions:
        raise SystemExit(
            "Nothing to analyze. Provide --run-dir, --condition, and/or --exp2/--exp3."
        )

    audit_new_conditions(conditions)
    print("\nPER-CONDITION SUMMARY")
    summaries = [summarize_condition(name, condition) for name, condition in conditions.items()]

    def have(*names):
        return all(name in conditions for name in names)

    specifications = []
    if have("stepwise_no_history", "sequential_final"):
        specifications.append(("Comparison A (harness)", "stepwise_no_history", "sequential_final", False))
    if have("stepwise_answer_history", "stepwise_no_history"):
        specifications.append(("Comparison B (answer turns + content)", "stepwise_answer_history", "stepwise_no_history", True))
    if have("stepwise_answer_confidence_history", "stepwise_answer_history"):
        specifications.append(("Comparison C (added confidence line)", "stepwise_answer_confidence_history", "stepwise_answer_history", True))
    if have("sequential_final_length_matched", "sequential_final"):
        specifications.append(("Comparison D (turn/context structure)", "sequential_final_length_matched", "sequential_final", True))
    if have("stepwise_answer_confidence_history", "sequential_final_length_matched"):
        specifications.append(("Comparison E (history content beyond matched structure)", "stepwise_answer_confidence_history", "sequential_final_length_matched", True))
    if have("stepwise_answer_history", "sequential_final_length_matched"):
        specifications.append(("Comparison F (answer-only diagnostic; length unmatched)", "stepwise_answer_history", "sequential_final_length_matched", False))
    if have("sequential_final_neutral_v2", "sequential_final"):
        specifications.append(("Comparison G (Neutral-V2 structure robustness)", "sequential_final_neutral_v2", "sequential_final", True))
    if have("sequential_final_short_ack", "sequential_final"):
        specifications.append(("Comparison H (short assistant-role marker)", "sequential_final_short_ack", "sequential_final", True))
    if have("sequential_final_neutral_v2", "sequential_final_length_matched"):
        specifications.append(("Comparison I (neutral wording sensitivity)", "sequential_final_neutral_v2", "sequential_final_length_matched", True))
    if have("stepwise_answer_confidence_history", "sequential_final_neutral_v2"):
        specifications.append(("Comparison J (history content beyond Neutral-V2)", "stepwise_answer_confidence_history", "sequential_final_neutral_v2", True))
    if have("sequential_final_length_matched", "sequential_final_short_ack"):
        specifications.append(("Comparison K (neutral text/length beyond role marker)", "sequential_final_length_matched", "sequential_final_short_ack", True))
    if have("as_published_stepwise(exp3)", "as_published_sequential_final(exp2)"):
        specifications.append(("As-published reference", "as_published_stepwise(exp3)", "as_published_sequential_final(exp2)", False))

    results = []
    rows_by_label = {}
    for label, name1, name2, adjust in specifications:
        rows = build_comparison(
            label,
            name1,
            conditions[name1],
            name2,
            conditions[name2],
            args.bootstrap_reps,
            args.seed,
            adjust,
        )
        rows_by_label[label] = rows
        results.extend(rows)

    token_pair_audits = []
    exact_token_specifications = []
    for label, name1, name2 in [
        (
            "Comparison E exact-token sensitivity",
            "stepwise_answer_confidence_history",
            "sequential_final_length_matched",
        ),
        (
            "Comparison J exact-token sensitivity",
            "stepwise_answer_confidence_history",
            "sequential_final_neutral_v2",
        ),
    ]:
        if not have(name1, name2):
            continue
        audit = build_token_pair_audit(
            name1, conditions[name1], name2, conditions[name2]
        )
        token_pair_audits.append(audit)
        if audit["exact_match_ids"]:
            exact_token_specifications.append((label, name1, name2, audit["exact_match_ids"]))
            rows = build_comparison(
                label,
                name1,
                conditions[name1],
                name2,
                conditions[name2],
                args.bootstrap_reps,
                args.seed,
                False,
                allowed_ids=set(audit["exact_match_ids"]),
                analysis_subset="exact_final_prompt_token_match",
            )
            rows_by_label[label] = rows
            results.extend(rows)
    apply_holm_within_metric(results)
    for label, _, _, _ in specifications:
        print_comparison(label, rows_by_label[label])
    for label, _, _, _ in exact_token_specifications:
        print_comparison(label, rows_by_label[label])

    print("\nTOKEN AUDIT")
    for name in NEW_CONDITION_NAMES:
        if name in conditions:
            print_token_audit(name, conditions[name])
    for audit in token_pair_audits:
        print_token_pair_audit(audit)
    print("\nSTEPWISE TRAJECTORIES")
    for name in [
        "stepwise_no_history",
        "stepwise_answer_history",
        "stepwise_answer_confidence_history",
    ]:
        if name in conditions:
            print_stepwise_trajectory(name, conditions[name])

    if args.output_json:
        payload = {
            "bootstrap_reps": args.bootstrap_reps,
            "seed": args.seed,
            "holm_family": "comparisons B-E and G-K, corrected separately within each metric",
            "summaries": summaries,
            "token_pair_audits": token_pair_audits,
            "comparisons": [
                {key: value for key, value in row.items() if key != "adjust"}
                for row in results
            ],
        }
        output_dir = os.path.dirname(os.path.abspath(args.output_json))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nWROTE {args.output_json}")


if __name__ == "__main__":
    main()
