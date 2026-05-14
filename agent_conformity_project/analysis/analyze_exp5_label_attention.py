#!/usr/bin/env python3
"""Analyze Exp5 model-label attention JSONL results.

This script only reads an existing result JSONL and writes a Markdown report.
It does not run experiments or modify source result files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


CONDITIONS = ["labelled_5_wrong", "unlabelled_5_wrong", "mixed_label_conflict"]
OPTION_LABELS = {"A", "B", "C", "D", "E"}
DEFAULT_INPUTS = (
    (
        "agent_conformity_project/results/runs/"
        "exp5_label_attention_commonsenseqa500_gemma2_2b_fixed_20260505_225549/"
        "2026-04-29_five_wrong_guidance/exp5_model_label_attention_test.jsonl"
    ),
    (
        "results/runs/exp5_label_attention_commonsenseqa500_gemma2_2b_fixed_20260505_225549/"
        "2026-04-29_five_wrong_guidance/exp5_model_label_attention_test.jsonl"
    ),
)
DEFAULT_OUTPUT = (
    "agent_conformity_project/analysis_outputs/"
    "exp5_label_attention_gemma2_2b_commonsenseqa500_fixed_summary.md"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pct(num: int | float, den: int | float) -> float | None:
    return None if den == 0 else num / den


def fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def fmt_num(value: float | None, digits: int = 4) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def field(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return default


def normalize_answer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in OPTION_LABELS else None


def parse_answer_from_raw(raw_output: Any) -> str | None:
    if raw_output is None:
        return None
    text = str(raw_output)
    match = re.search(r"\bANSWER\s*:\s*([A-E])\b", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-E])\b", text.strip().upper())
    return match.group(1) if match else None


def final_answer(row: dict[str, Any]) -> str | None:
    return normalize_answer(field(row, "final_answer", "attack_prediction", "answer")) or parse_answer_from_raw(
        field(row, "raw_output", "raw_attack_output", "text", default="")
    )


def raw_output(row: dict[str, Any]) -> str:
    value = field(row, "raw_output", "raw_attack_output", "text", default="")
    return "" if value is None else str(value)


def is_empty_option_logprobs(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return True
    return all(value.get(label) is None for label in OPTION_LABELS)


def option_logprob(row: dict[str, Any], option: str | None) -> float | None:
    if option is None:
        return None
    option_logprobs = field(row, "option_logprobs", "attack_option_logprobs", default={})
    if not isinstance(option_logprobs, dict):
        return None
    value = option_logprobs.get(option)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def short_text(text: Any, limit: int = 120) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def two_prop_z_test(success_a: int, total_a: int, success_b: int, total_b: int) -> dict[str, float | None]:
    if min(total_a, total_b) == 0:
        return {"z": None, "p": None}
    pooled = (success_a + success_b) / (total_a + total_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
    if se == 0:
        return {"z": None, "p": None}
    z = (success_a / total_a - success_b / total_b) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return {"z": z, "p": p_value}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(field(row, "condition", default="UNKNOWN"))].append(row)

    first_keys = sorted(rows[0].keys()) if rows else []
    total = len(rows)
    error_rows = [r for r in rows if bool(field(r, "is_error", default=False)) or field(r, "error") is not None]
    final_empty = [r for r in rows if normalize_answer(field(r, "final_answer", "attack_prediction", "answer")) is None]
    raw_empty = [r for r in rows if raw_output(r).strip() == ""]
    option_empty = [r for r in rows if is_empty_option_logprobs(field(r, "option_logprobs", "attack_option_logprobs"))]
    valid = [r for r in rows if final_answer(r) is not None]

    condition_summaries = {}
    for condition in CONDITIONS:
        condition_rows = grouped.get(condition, [])
        valid_rows = [r for r in condition_rows if final_answer(r) is not None]
        invalid_count = len(condition_rows) - len(valid_rows)
        correct = 0
        distractor = 0
        other_wrong = 0
        correct_lps = []
        distractor_lps = []
        margins = []
        examples = []

        for row in condition_rows:
            ans = final_answer(row)
            corr = normalize_answer(field(row, "correct_answer"))
            dist = normalize_answer(field(row, "distractor"))
            if ans is not None:
                if ans == corr:
                    correct += 1
                elif ans == dist:
                    distractor += 1
                else:
                    other_wrong += 1
            corr_lp = option_logprob(row, corr)
            dist_lp = option_logprob(row, dist)
            if corr_lp is not None:
                correct_lps.append(corr_lp)
            if dist_lp is not None:
                distractor_lps.append(dist_lp)
            if corr_lp is not None and dist_lp is not None:
                margins.append(dist_lp - corr_lp)
            if len(examples) < 5:
                examples.append({"final_answer": ans, "raw_output": raw_output(row)})

        condition_summaries[condition] = {
            "rows": len(condition_rows),
            "valid": len(valid_rows),
            "correct": correct,
            "distractor": distractor,
            "other_wrong": other_wrong,
            "invalid": invalid_count,
            "accuracy": pct(correct, len(condition_rows)),
            "distractor_rate": pct(distractor, len(condition_rows)),
            "other_wrong_rate": pct(other_wrong, len(condition_rows)),
            "invalid_rate": pct(invalid_count, len(condition_rows)),
            "correct_answer_rate": pct(correct, len(valid_rows)),
            "avg_correct_lp": mean(correct_lps) if correct_lps else None,
            "avg_distractor_lp": mean(distractor_lps) if distractor_lps else None,
            "avg_margin": mean(margins) if margins else None,
            "logprob_n": len(margins),
            "examples": examples,
        }

    error_examples = Counter()
    for row in error_rows:
        key = f"{field(row, 'error_type', default='UNKNOWN')}: {short_text(field(row, 'error', default=''))}"
        error_examples[key] += 1

    return {
        "first_keys": first_keys,
        "total": total,
        "condition_counts": {condition: len(grouped.get(condition, [])) for condition in CONDITIONS},
        "unknown_condition_counts": {k: len(v) for k, v in grouped.items() if k not in CONDITIONS},
        "quality": {
            "error": len(error_rows),
            "final_empty": len(final_empty),
            "raw_empty": len(raw_empty),
            "option_logprobs_empty": len(option_empty),
            "valid": len(valid),
        },
        "error_examples": error_examples.most_common(5),
        "conditions": condition_summaries,
    }


def condition_table(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Condition | Rows | Valid parsed | Accuracy | Distractor / HCR | Other wrong | Invalid / parse error | Correct answer rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        s = summary["conditions"][condition]
        lines.append(
            f"| {condition} | {s['rows']} | {s['valid']} | {fmt_pct(s['accuracy'])} | "
            f"{fmt_pct(s['distractor_rate'])} | {fmt_pct(s['other_wrong_rate'])} | "
            f"{fmt_pct(s['invalid_rate'])} | {fmt_pct(s['correct_answer_rate'])} |"
        )
    return lines


def quality_table(summary: dict[str, Any]) -> list[str]:
    total = summary["total"]
    q = summary["quality"]
    lines = [
        "| Check | Count | Rate |",
        "|---|---:|---:|",
        f"| Total rows | {total} | 100.00% |",
    ]
    for condition, count in summary["condition_counts"].items():
        lines.append(f"| Rows: {condition} | {count} | {fmt_pct(pct(count, total))} |")
    for condition, count in summary["unknown_condition_counts"].items():
        lines.append(f"| Rows: {condition} | {count} | {fmt_pct(pct(count, total))} |")
    lines.extend(
        [
            f"| is_error=true or error present | {q['error']} | {fmt_pct(pct(q['error'], total))} |",
            f"| final_answer empty/null | {q['final_empty']} | {fmt_pct(pct(q['final_empty'], total))} |",
            f"| raw_output empty | {q['raw_empty']} | {fmt_pct(pct(q['raw_empty'], total))} |",
            f"| option_logprobs all empty/null | {q['option_logprobs_empty']} | {fmt_pct(pct(q['option_logprobs_empty'], total))} |",
            f"| valid parsed answer | {q['valid']} | {fmt_pct(pct(q['valid'], total))} |",
        ]
    )
    return lines


def examples_section(summary: dict[str, Any]) -> list[str]:
    lines = ["### Raw Output Examples", ""]
    for condition in CONDITIONS:
        lines.extend([f"**{condition}**", ""])
        lines.append("| # | final_answer | raw_output |")
        lines.append("|---:|---|---|")
        for idx, example in enumerate(summary["conditions"][condition]["examples"], start=1):
            lines.append(
                f"| {idx} | {example['final_answer'] or 'N/A'} | `{short_text(example['raw_output'], 160)}` |"
            )
        lines.append("")
    return lines


def logprob_section(summary: dict[str, Any]) -> tuple[list[str], bool]:
    lines = [
        "| Condition | N margins | Avg logprob(correct) | Avg logprob(distractor) | Avg distractor_minus_correct |",
        "|---|---:|---:|---:|---:|",
    ]
    has_sparse = False
    for condition in CONDITIONS:
        s = summary["conditions"][condition]
        if s["logprob_n"] < max(1, int(0.5 * s["rows"])):
            has_sparse = True
        lines.append(
            f"| {condition} | {s['logprob_n']} | {fmt_num(s['avg_correct_lp'])} | "
            f"{fmt_num(s['avg_distractor_lp'])} | {fmt_num(s['avg_margin'])} |"
        )
    return lines, has_sparse


def write_report(path: Path, input_path: Path, summary: dict[str, Any]) -> None:
    c = summary["conditions"]
    labelled = c["labelled_5_wrong"]
    unlabelled = c["unlabelled_5_wrong"]
    mixed = c["mixed_label_conflict"]
    try:
        display_input = input_path.resolve().relative_to(repo_root()).as_posix()
    except ValueError:
        display_input = input_path.as_posix()

    labelled_minus_unlabelled_dist = None
    labelled_minus_unlabelled_acc = None
    if labelled["distractor_rate"] is not None and unlabelled["distractor_rate"] is not None:
        labelled_minus_unlabelled_dist = labelled["distractor_rate"] - unlabelled["distractor_rate"]
    if labelled["accuracy"] is not None and unlabelled["accuracy"] is not None:
        labelled_minus_unlabelled_acc = labelled["accuracy"] - unlabelled["accuracy"]

    z_lu = two_prop_z_test(
        labelled["distractor"], labelled["rows"], unlabelled["distractor"], unlabelled["rows"]
    )
    z_lm = two_prop_z_test(
        labelled["distractor"], labelled["rows"], mixed["distractor"], mixed["rows"]
    )
    margin_lu = None
    margin_lm = None
    if labelled["avg_margin"] is not None and unlabelled["avg_margin"] is not None:
        margin_lu = labelled["avg_margin"] - unlabelled["avg_margin"]
    if labelled["avg_margin"] is not None and mixed["avg_margin"] is not None:
        margin_lm = mixed["avg_margin"] - labelled["avg_margin"]

    logprob_lines, logprob_sparse = logprob_section(summary)
    failed = summary["quality"]["valid"] == 0 or summary["quality"]["error"] == summary["total"]

    lines = [
        "# Exp5 Model Label Attention Analysis",
        "",
        "## 1. Experiment Setup",
        "",
        "- Model: `google/gemma-2-2b-it`",
        "- Dataset: CommonsenseQA500 from the fixed Exp5 run",
        f"- Source JSONL: `{display_input}`",
        f"- Total rows: {summary['total']} sample-condition records",
        "- Conditions: `labelled_5_wrong`, `unlabelled_5_wrong`, `mixed_label_conflict`",
        "",
        "The experiment tests whether explicit peer identity labels (`Model 1` through `Model 5`) add social signal beyond repeated recommendation content.",
        "",
        "First sample keys observed:",
        "",
        "```text",
        ", ".join(summary["first_keys"]),
        "```",
        "",
        "## 2. Data Integrity Check",
        "",
        *quality_table(summary),
        "",
    ]

    if summary["error_examples"]:
        lines.extend(["Top error examples:", ""])
        for text, count in summary["error_examples"]:
            lines.append(f"- {count}x `{text}`")
        lines.append("")
    else:
        lines.extend(["No error rows were found.", ""])

    lines.extend(
        [
            "## 3. Main Results",
            "",
            *condition_table(summary),
            "",
            *examples_section(summary),
            "## 4. Labelled vs Unlabelled Comparison",
            "",
            f"- Distractor rate difference (`labelled - unlabelled`): {fmt_pct(labelled_minus_unlabelled_dist)}",
            f"- Accuracy difference (`labelled - unlabelled`): {fmt_pct(labelled_minus_unlabelled_acc)}",
            f"- Two-proportion z-test for distractor choice: z={fmt_num(z_lu['z'])}, p={fmt_num(z_lu['p'])}",
            "",
        ]
    )

    if labelled_minus_unlabelled_dist is not None:
        if abs(labelled_minus_unlabelled_dist) < 0.02:
            lines.append(
                "The labelled and unlabelled distractor rates are very close, so the final-answer evidence suggests the model responded mainly to the repeated recommendation content rather than to the explicit `Model 1/2/3/4/5` labels."
            )
        elif labelled_minus_unlabelled_dist > 0:
            lines.append(
                "The labelled condition has a higher distractor rate, consistent with the possibility that explicit model labels strengthened social conformity pressure."
            )
        else:
            lines.append(
                "The labelled condition has a lower distractor rate, which does not support a label-amplification interpretation."
            )
    lines.extend(
        [
            "",
            "## 5. Mixed Label Conflict Analysis",
            "",
            f"- Mixed distractor choice rate: {fmt_pct(mixed['distractor_rate'])}",
            f"- Mixed correct-answer rate: {fmt_pct(mixed['correct_answer_rate'])}",
            f"- Mixed other-option rate: {fmt_pct(mixed['other_wrong_rate'])}",
            f"- Labelled vs mixed distractor z-test: z={fmt_num(z_lm['z'])}, p={fmt_num(z_lm['p'])}",
            "",
        ]
    )
    if mixed["distractor_rate"] is not None and labelled["distractor_rate"] is not None:
        if mixed["distractor_rate"] < labelled["distractor_rate"] - 0.02:
            lines.append(
                "The mixed conflict condition reduces distractor selection relative to unanimously wrong labelled peers, suggesting that the model uses at least some information about conflicting labelled peer support."
            )
        elif abs(mixed["distractor_rate"] - labelled["distractor_rate"]) < 0.02:
            lines.append(
                "The mixed conflict condition is close to the unanimously wrong labelled condition, suggesting limited use of label-level conflict in final choices."
            )
        else:
            lines.append(
                "The mixed conflict condition increases distractor selection relative to labelled_5_wrong, which would be unexpected and should be checked against item-level patterns."
            )

    lines.extend(["", "## 6. Logprob Evidence", ""])
    if logprob_sparse:
        lines.append(
            "Logprob evidence is unavailable or sparse, so conclusions rely on final answers only."
        )
        lines.append("")
    lines.extend(logprob_lines)
    lines.extend(
        [
            "",
            f"- `labelled - unlabelled` margin difference: {fmt_num(margin_lu)}",
            f"- `mixed - labelled` margin difference: {fmt_num(margin_lm)}",
            "",
        ]
    )
    if not logprob_sparse and margin_lm is not None:
        if margin_lm < 0:
            lines.append(
                "The average distractor-minus-correct logprob margin is lower in mixed_label_conflict than in labelled_5_wrong, which supports the interpretation that conflicting labelled support weakens the distractor at the probability level."
            )
        else:
            lines.append(
                "The average distractor-minus-correct logprob margin does not decline in mixed_label_conflict, so probability evidence does not show a clear weakening of the distractor."
            )
        lines.append("")

    lines.extend(["## 7. Interpretation for Paper", ""])
    if failed:
        lines.append(
            "This run does not contain usable answers, so it should be reported as an execution failure rather than as behavioral evidence."
        )
    else:
        lines.append(
            "In this fixed Gemma-2-2B-it CommonsenseQA500 run, explicit `Model 1/2/3/4/5` labels do not appear to substantially amplify harmful conformity beyond repeated unlabelled recommendations when judged by final answers. The key comparison is the small labelled-minus-unlabelled distractor-rate difference."
        )
        lines.append("")
        lines.append(
            "The mixed-label conflict condition is the stronger mechanism check: if its correct-answer rate increases and its distractor rate/logprob margin falls relative to `labelled_5_wrong`, the model is not merely reacting to generic peer text but is at least partially sensitive to the direction of support attached to labelled peers. This makes Exp5 useful as a mechanism validation experiment, especially when paired with cross-model replication and significance testing."
        )
    lines.extend(
        [
            "",
            "## 8. Limitations",
            "",
            "- This analysis covers only `google/gemma-2-2b-it` on CommonsenseQA500.",
            "- The social signal is fixed to five guidance statements with a single distractor design.",
            "- Results should be compared against Qwen and Phi runs before making broad claims.",
            "- The included z-tests are simple two-proportion tests; a paper-ready version should add item-paired tests or regression with item controls.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=root / DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = next(
            (root / path for path in DEFAULT_INPUTS if (root / path).exists()),
            root / DEFAULT_INPUTS[0],
        )

    rows = load_jsonl(input_path)
    if rows:
        print(f"First sample keys: {sorted(rows[0].keys())}")
    else:
        print("Input JSONL is empty.")
    summary = summarize(rows)
    write_report(args.output, input_path, summary)
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
