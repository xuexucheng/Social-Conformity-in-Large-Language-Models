#!/usr/bin/env python3
"""Audit invalid outputs and quantify parser/missing-output sensitivity.

The analysis never silently replaces stored predictions. It reports how the
current parser treats the raw text, recomputes paired results under a clearly
labelled reparse-when-raw-available policy, and derives worst-case bounds for
missing final outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.parser import parse_answer_details  # noqa: E402

try:
    from paired_protocol_analysis import (  # noqa: E402
        FINAL_ANSWER_ALIASES,
        INITIAL_ANSWER_ALIASES,
        LoadedCondition,
        OUTCOME_DESCRIPTIONS,
        analyze_pair,
        apply_holm_correction,
        collect_condition_paths,
        exact_mcnemar_pvalue,
        final_answer,
        gold_and_distractor,
        load_condition,
        parse_pairs,
        record_answer,
        stable_seed,
    )
except ImportError:  # pragma: no cover - used when imported as a package
    from .paired_protocol_analysis import (  # noqa: E402
        FINAL_ANSWER_ALIASES,
        INITIAL_ANSWER_ALIASES,
        LoadedCondition,
        OUTCOME_DESCRIPTIONS,
        analyze_pair,
        apply_holm_correction,
        collect_condition_paths,
        exact_mcnemar_pvalue,
        final_answer,
        gold_and_distractor,
        load_condition,
        parse_pairs,
        record_answer,
        stable_seed,
    )


RAW_ALIASES = {
    "initial": ("raw_initial_output", "raw_initial"),
    "final": ("raw_attack_output", "raw_output", "text"),
}
ANSWER_ALIASES = {
    "initial": INITIAL_ANSWER_ALIASES,
    "final": FINAL_ANSWER_ALIASES,
}
ANSWER_LIKE_PATTERN = re.compile(
    r"\b(?:FINAL\s+ANSWER|ANSWER|CHOICE|OPTION)\s*(?:IS|:)?\s*[\(\[]?\s*([A-E])\b",
    flags=re.IGNORECASE,
)
STANDALONE_PATTERN = re.compile(
    r"^[\s>*_`'\"\-]*[\(\[]?\s*([A-E])\s*[\)\]]?[\s.,:;!?*_`'\"\-]*$",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ParserAuditBundle:
    audit_rows: list[dict[str, Any]]
    condition_summary: list[dict[str, Any]]
    method_counts: list[dict[str, Any]]
    review_sample: list[dict[str, Any]]
    reparsed_conditions: dict[tuple[str, str], LoadedCondition]


def first_raw(record: dict[str, Any], phase: str) -> tuple[str | None, str]:
    first_present_field = None
    for alias in RAW_ALIASES[phase]:
        if alias in record:
            if first_present_field is None:
                first_present_field = alias
            value = record.get(alias)
            text = "" if value is None else str(value)
            if text.strip():
                return alias, text
    return first_present_field, ""


def stored_answer(record: dict[str, Any], phase: str) -> str | None:
    if phase == "final" and record.get("is_error") is True:
        return None
    return record_answer(record, ANSWER_ALIASES[phase])


def explicit_candidate_labels(raw_text: str) -> list[str]:
    labels = [match.group(1).upper() for match in ANSWER_LIKE_PATTERN.finditer(raw_text)]
    labels.extend(match.group(1).upper() for match in STANDALONE_PATTERN.finditer(raw_text))
    return sorted(set(labels))


def parser_category(
    stored: str | None,
    reparsed: str | None,
    raw_text: str,
    is_error: bool,
) -> str:
    if is_error:
        return "execution_error"
    if not raw_text.strip():
        return "stored_valid_raw_missing" if stored else "stored_invalid_raw_missing"
    if stored and reparsed == stored:
        return "agreement_valid"
    if stored is None and reparsed:
        return "stored_invalid_reparsed_valid"
    if stored and reparsed is None:
        return "stored_valid_reparsed_invalid"
    if stored and reparsed and stored != reparsed:
        return "stored_reparsed_disagree"
    return "both_invalid"


def audit_record_phase(
    stratum: str,
    condition: str,
    item_id: str,
    record: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    raw_field, raw_text = first_raw(record, phase)
    stored = stored_answer(record, phase)
    details = parse_answer_details(raw_text, options=record.get("options")) if raw_text.strip() else {
        "answer": None,
        "parse_status": "failed",
        "parse_method": None,
        "matched_text": None,
    }
    reparsed = details.get("answer")
    is_error = bool(record.get("is_error") is True)
    if phase == "final" and is_error:
        reparsed = None
    candidates = explicit_candidate_labels(raw_text)
    category = parser_category(stored, reparsed, raw_text, is_error if phase == "final" else False)
    return {
        "stratum": stratum,
        "condition": condition,
        "item_id": item_id,
        "phase": phase,
        "is_error": is_error,
        "stored_answer": stored,
        "raw_field": raw_field,
        "raw_missing": not raw_text.strip(),
        "reparsed_answer": reparsed,
        "parse_status": details.get("parse_status"),
        "parse_method": details.get("parse_method"),
        "matched_text": details.get("matched_text"),
        "explicit_candidate_labels": ",".join(candidates),
        "ambiguous_explicit_candidates": len(candidates) > 1,
        "category": category,
        "raw_preview": " ".join(raw_text.split())[:500],
    }


def reparse_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    for phase in ("initial", "final"):
        _, raw_text = first_raw(record, phase)
        if not raw_text.strip():
            continue
        details = parse_answer_details(raw_text, options=record.get("options"))
        parsed = details.get("answer")
        if phase == "initial":
            updated["initial_prediction"] = parsed
        elif record.get("is_error") is not True:
            updated["attack_prediction"] = parsed
            if "final_answer" in updated:
                updated["final_answer"] = parsed
            if "prediction" in updated and "attack_prediction" not in record:
                updated["prediction"] = parsed
    return updated


def build_parser_audit(
    loaded: dict[tuple[str, str], LoadedCondition],
    sample_per_category: int,
    seed: int,
) -> ParserAuditBundle:
    audit_rows = []
    reparsed_conditions = {}
    for key, condition_data in sorted(loaded.items()):
        reparsed_rows = {}
        for item_id, record in sorted(condition_data.rows.items()):
            for phase in ("initial", "final"):
                audit_rows.append(
                    audit_record_phase(
                        condition_data.stratum,
                        condition_data.name,
                        item_id,
                        record,
                        phase,
                    )
                )
            reparsed_rows[item_id] = reparse_record(record)
        reparsed_conditions[key] = LoadedCondition(
            stratum=condition_data.stratum,
            name=condition_data.name,
            path=condition_data.path,
            rows=reparsed_rows,
            quality=dict(condition_data.quality),
        )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in audit_rows:
        grouped.setdefault((row["stratum"], row["condition"], row["phase"]), []).append(row)

    condition_summary = []
    method_counts = []
    for (stratum, condition, phase), rows in sorted(grouped.items()):
        categories = Counter(str(row["category"]) for row in rows)
        total = len(rows)
        condition_summary.append(
            {
                "stratum": stratum,
                "condition": condition,
                "phase": phase,
                "total": total,
                "stored_valid": sum(row["stored_answer"] is not None for row in rows),
                "stored_invalid": sum(row["stored_answer"] is None for row in rows),
                "stored_invalid_rate": (
                    sum(row["stored_answer"] is None for row in rows) / total if total else None
                ),
                "raw_missing": sum(bool(row["raw_missing"]) for row in rows),
                "reparsed_valid": sum(row["reparsed_answer"] is not None for row in rows),
                "agreement_valid": categories["agreement_valid"],
                "recoverable_stored_invalid": categories["stored_invalid_reparsed_valid"],
                "stored_reparsed_disagree": categories["stored_reparsed_disagree"],
                "stored_valid_reparsed_invalid": categories["stored_valid_reparsed_invalid"],
                "both_invalid": categories["both_invalid"],
                "execution_error": categories["execution_error"],
                "ambiguous_explicit_candidates": sum(
                    bool(row["ambiguous_explicit_candidates"]) for row in rows
                ),
            }
        )
        methods = Counter(str(row["parse_method"] or "NONE") for row in rows)
        for method, count in sorted(methods.items()):
            method_counts.append(
                {
                    "stratum": stratum,
                    "condition": condition,
                    "phase": phase,
                    "parse_method": method,
                    "count": count,
                    "rate": count / total if total else None,
                }
            )

    review_candidates = [
        row
        for row in audit_rows
        if row["category"] != "agreement_valid"
        or bool(row["ambiguous_explicit_candidates"])
    ]
    review_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in review_candidates:
        key = (row["stratum"], row["condition"], row["phase"], row["category"])
        review_groups.setdefault(key, []).append(row)
    review_sample = []
    for key, rows in sorted(review_groups.items()):
        ranked = sorted(
            rows,
            key=lambda row: stable_seed(
                seed,
                "parser_review",
                *key,
                str(row["item_id"]),
            ),
        )
        review_sample.extend(ranked[:sample_per_category])

    return ParserAuditBundle(
        audit_rows=audit_rows,
        condition_summary=condition_summary,
        method_counts=method_counts,
        review_sample=review_sample,
        reparsed_conditions=reparsed_conditions,
    )


def initial_answer_for_bounds(record: dict[str, Any] | None) -> str | None:
    if not record:
        return None
    return record_answer(record, INITIAL_ANSWER_ALIASES)


def resolve_bounds_baseline(
    item_id: str,
    left_record: dict[str, Any],
    right_record: dict[str, Any],
    baseline_condition: LoadedCondition | None,
    initial_policy: str,
    gold: str,
    distractor: str,
) -> tuple[str | None, str]:
    if baseline_condition is not None:
        baseline_record = baseline_condition.rows.get(item_id)
        if baseline_record is None:
            return None, "baseline_item_missing"
        baseline_gold, baseline_distractor = gold_and_distractor(baseline_record)
        if baseline_gold != gold or baseline_distractor != distractor:
            return None, "baseline_metadata_mismatch"
        answer = initial_answer_for_bounds(baseline_record)
        return (answer, "baseline_condition") if answer else (None, "baseline_invalid")

    left_initial = initial_answer_for_bounds(left_record)
    right_initial = initial_answer_for_bounds(right_record)
    if initial_policy == "left":
        return (left_initial, "left") if left_initial else (None, "left_initial_invalid")
    if initial_policy == "right":
        return (right_initial, "right") if right_initial else (None, "right_initial_invalid")
    if left_initial is None or right_initial is None:
        return None, "initial_invalid"
    if left_initial != right_initial:
        return None, "initial_mismatch"
    return left_initial, "initial_equal"


def event_from_answer(
    outcome: str,
    answer: str | None,
    baseline: str | None,
    gold: str,
    distractor: str,
) -> int | None:
    if answer is None:
        return None
    if outcome == "final_accuracy":
        return int(answer == gold)
    if outcome == "target_adoption":
        return int(answer == distractor)
    if outcome in {"conformity_rate", "harmful_conformity"}:
        return int(answer == distractor)
    if outcome == "beneficial_revision":
        return int(answer == gold)
    if outcome == "answer_change":
        return int(answer != baseline)
    raise ValueError(f"Unknown outcome: {outcome}")


def baseline_eligible(
    outcome: str,
    baseline: str | None,
    gold: str,
    distractor: str,
) -> bool:
    if outcome in {"final_accuracy", "target_adoption"}:
        return True
    if baseline is None:
        return False
    if outcome == "conformity_rate":
        return baseline != distractor
    if outcome == "harmful_conformity":
        return baseline == gold
    if outcome == "beneficial_revision":
        return baseline != gold
    if outcome == "answer_change":
        return True
    raise ValueError(f"Unknown outcome: {outcome}")


def missing_data_bounds(
    stratum: str,
    left_condition: LoadedCondition,
    right_condition: LoadedCondition,
    baseline_condition: LoadedCondition | None,
    initial_policy: str,
) -> list[dict[str, Any]]:
    common_ids = sorted(set(left_condition.rows) & set(right_condition.rows))
    pair_name = f"{left_condition.name} vs {right_condition.name}"
    result_rows = []
    for outcome in OUTCOME_DESCRIPTIONS:
        possible_differences = []
        complete_pairs = []
        left_available = []
        right_available = []
        left_missing = 0
        right_missing = 0
        both_missing = 0
        metadata_mismatch = 0
        baseline_excluded = 0

        for item_id in common_ids:
            left_record = left_condition.rows[item_id]
            right_record = right_condition.rows[item_id]
            left_gold, left_distractor = gold_and_distractor(left_record)
            right_gold, right_distractor = gold_and_distractor(right_record)
            if (
                left_gold is None
                or left_distractor is None
                or left_gold != right_gold
                or left_distractor != right_distractor
            ):
                metadata_mismatch += 1
                continue
            baseline, _ = resolve_bounds_baseline(
                item_id,
                left_record,
                right_record,
                baseline_condition,
                initial_policy,
                left_gold,
                left_distractor,
            )
            if not baseline_eligible(outcome, baseline, left_gold, left_distractor):
                baseline_excluded += 1
                continue

            left_event = event_from_answer(
                outcome,
                final_answer(left_record),
                baseline,
                left_gold,
                left_distractor,
            )
            right_event = event_from_answer(
                outcome,
                final_answer(right_record),
                baseline,
                left_gold,
                left_distractor,
            )
            if left_event is None:
                left_missing += 1
            else:
                left_available.append(left_event)
            if right_event is None:
                right_missing += 1
            else:
                right_available.append(right_event)
            if left_event is None and right_event is None:
                both_missing += 1
            if left_event is not None and right_event is not None:
                complete_pairs.append((left_event, right_event))

            left_low, left_high = ((0, 1) if left_event is None else (left_event, left_event))
            right_low, right_high = (
                (0, 1) if right_event is None else (right_event, right_event)
            )
            possible_differences.append((right_low - left_high, right_high - left_low))

        eligible_total = len(possible_differences)
        complete_n = len(complete_pairs)
        complete_left_rate = (
            sum(left for left, _ in complete_pairs) / complete_n if complete_n else None
        )
        complete_right_rate = (
            sum(right for _, right in complete_pairs) / complete_n if complete_n else None
        )
        n10 = sum(left == 1 and right == 0 for left, right in complete_pairs)
        n01 = sum(left == 0 and right == 1 for left, right in complete_pairs)
        lower_bound = (
            sum(lower for lower, _ in possible_differences) / eligible_total
            if eligible_total
            else None
        )
        upper_bound = (
            sum(upper for _, upper in possible_differences) / eligible_total
            if eligible_total
            else None
        )
        left_non_event_rate = (
            sum(left_available) / eligible_total if eligible_total else None
        )
        right_non_event_rate = (
            sum(right_available) / eligible_total if eligible_total else None
        )
        result_rows.append(
            {
                "stratum": stratum,
                "pair": pair_name,
                "left_condition": left_condition.name,
                "right_condition": right_condition.name,
                "outcome": outcome,
                "eligibility_rule": OUTCOME_DESCRIPTIONS[outcome],
                "common_item_n": len(common_ids),
                "metadata_mismatch_n": metadata_mismatch,
                "baseline_excluded_n": baseline_excluded,
                "eligible_including_invalid_n": eligible_total,
                "complete_case_n": complete_n,
                "left_missing_n": left_missing,
                "right_missing_n": right_missing,
                "both_missing_n": both_missing,
                "left_available_n": len(left_available),
                "right_available_n": len(right_available),
                "left_available_rate": (
                    sum(left_available) / len(left_available) if left_available else None
                ),
                "right_available_rate": (
                    sum(right_available) / len(right_available) if right_available else None
                ),
                "complete_case_left_rate": complete_left_rate,
                "complete_case_right_rate": complete_right_rate,
                "complete_case_delta_right_minus_left": (
                    complete_right_rate - complete_left_rate if complete_n else None
                ),
                "invalid_as_non_event_left_rate": left_non_event_rate,
                "invalid_as_non_event_right_rate": right_non_event_rate,
                "invalid_as_non_event_delta_right_minus_left": (
                    right_non_event_rate - left_non_event_rate if eligible_total else None
                ),
                "worst_case_delta_lower": lower_bound,
                "worst_case_delta_upper": upper_bound,
                "discordant_left1_right0_complete": n10,
                "discordant_left0_right1_complete": n01,
                "mcnemar_exact_p": (
                    exact_mcnemar_pvalue(n10, n01) if complete_n else None
                ),
                "holm_adjusted_p": None,
                "holm_family_n": None,
            }
        )
    return result_rows


def paired_results_for_policy(
    conditions_by_stratum: dict[str, dict[str, LoadedCondition]],
    pairs: list[tuple[str, str]],
    baseline_condition_name: str | None,
    initial_policy: str,
    bootstrap_repetitions: int,
    seed: int,
    parse_policy: str,
) -> list[dict[str, Any]]:
    results = []
    for stratum, conditions in sorted(conditions_by_stratum.items()):
        baseline_data = conditions.get(baseline_condition_name) if baseline_condition_name else None
        for left_name, right_name in pairs:
            pair_results, _ = analyze_pair(
                stratum=stratum,
                left_condition=conditions[left_name],
                right_condition=conditions[right_name],
                baseline_condition=baseline_data,
                initial_policy=initial_policy,
                bootstrap_repetitions=bootstrap_repetitions,
                seed=seed,
            )
            for row in pair_results:
                row["parse_policy"] = parse_policy
            results.extend(pair_results)
    apply_holm_correction(results, "outcome")
    return results


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def fmt_pp(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "N/A"
    return "<.001" if value < 0.001 else f"{value:.3f}".lstrip("0")


def build_report(
    condition_summary: list[dict[str, Any]],
    stored_results: list[dict[str, Any]],
    reparsed_results: list[dict[str, Any]],
    bounds_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Invalid Output and Parser Sensitivity Analysis",
        "",
        "## Parser audit",
        "",
        "| Stratum | Condition | Phase | Total | Stored invalid | Invalid % | Reparsed valid | Recoverable invalid | Stored/reparsed disagree | Both invalid | Execution errors | Ambiguous |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in condition_summary:
        lines.append(
            f"| {row['stratum']} | {row['condition']} | {row['phase']} | "
            f"{row['total']} | {row['stored_invalid']} | {fmt_pct(row['stored_invalid_rate'])} | "
            f"{row['reparsed_valid']} | {row['recoverable_stored_invalid']} | "
            f"{row['stored_reparsed_disagree']} | {row['both_invalid']} | "
            f"{row['execution_error']} | {row['ambiguous_explicit_candidates']} |"
        )

    lines.extend(
        [
            "",
            "## Stored versus current-parser complete-case results",
            "",
            "Reparsed results use the current parser only when raw text is available; missing raw text retains the stored prediction. Execution-error rows remain invalid.",
            "",
            "| Policy | Stratum | Comparison | Outcome | N | Left rate | Right rate | Delta pp | 95% CI pp | McNemar p | Holm p |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in [*stored_results, *reparsed_results]:
        if row["outcome"] not in {"final_accuracy", "conformity_rate", "harmful_conformity"}:
            continue
        lines.append(
            f"| {row['parse_policy']} | {row['stratum']} | {row['pair']} | {row['outcome']} | "
            f"{row['eligible_n']} | {fmt_pct(row['left_rate'])} | {fmt_pct(row['right_rate'])} | "
            f"{fmt_pp(row['delta_right_minus_left'])} | "
            f"[{fmt_pp(row['delta_ci_low'])}, {fmt_pp(row['delta_ci_high'])}] | "
            f"{fmt_p(row['mcnemar_exact_p'])} | {fmt_p(row['holm_adjusted_p'])} |"
        )

    lines.extend(
        [
            "",
            "## Missing-output sensitivity bounds",
            "",
            "| Stratum | Comparison | Outcome | Eligible incl. invalid | Complete N | Left missing | Right missing | Complete delta pp | Invalid=non-event delta pp | Worst-case delta pp |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bounds_rows:
        if row["outcome"] not in {"final_accuracy", "conformity_rate", "harmful_conformity"}:
            continue
        lines.append(
            f"| {row['stratum']} | {row['pair']} | {row['outcome']} | "
            f"{row['eligible_including_invalid_n']} | {row['complete_case_n']} | "
            f"{row['left_missing_n']} | {row['right_missing_n']} | "
            f"{fmt_pp(row['complete_case_delta_right_minus_left'])} | "
            f"{fmt_pp(row['invalid_as_non_event_delta_right_minus_left'])} | "
            f"[{fmt_pp(row['worst_case_delta_lower'])}, {fmt_pp(row['worst_case_delta_upper'])}] |"
        )

    lines.extend(
        [
            "",
            "## Reporting guidance",
            "",
            "- Keep stored common-valid results as the primary analysis.",
            "- Treat current-parser results as a parser-version sensitivity analysis, not as silent data repair.",
            "- Report invalid rates for every model-dataset-protocol condition.",
            "- If the worst-case interval is wide enough to reverse the protocol effect, explicitly state that missing outputs limit the comparison.",
            "- Manually review the deterministic cases in `parser_review_sample.csv` before changing any parser rule.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit stored predictions against the current parser and compute invalid-output sensitivity."
        )
    )
    parser.add_argument("--input-dir")
    parser.add_argument("--stratum", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument(
        "--condition", action="append", default=[], metavar="[STRATUM::]NAME=FILE"
    )
    parser.add_argument("--pair", action="append", default=[], metavar="LEFT:RIGHT")
    parser.add_argument("--baseline-condition")
    parser.add_argument(
        "--initial-policy",
        choices=("require-equal", "left", "right"),
        default="require-equal",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-per-category", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be at least 1")
    if args.sample_per_category < 0:
        parser.error("--sample-per-category must be non-negative")

    try:
        condition_paths = collect_condition_paths(args)
        loaded = {
            key: load_condition(key[0], key[1], path)
            for key, path in sorted(condition_paths.items())
        }
        stored_by_stratum: dict[str, dict[str, LoadedCondition]] = {}
        for (stratum, condition), data in loaded.items():
            stored_by_stratum.setdefault(stratum, {})[condition] = data
        pairs = parse_pairs(args.pair, stored_by_stratum, baseline_condition=None)
        if args.baseline_condition:
            for stratum, conditions in stored_by_stratum.items():
                if args.baseline_condition not in conditions:
                    raise ValueError(
                        f"Stratum {stratum!r} lacks baseline condition "
                        f"{args.baseline_condition!r}."
                    )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    audit = build_parser_audit(
        loaded,
        sample_per_category=args.sample_per_category,
        seed=args.seed,
    )
    reparsed_by_stratum: dict[str, dict[str, LoadedCondition]] = {}
    for (stratum, condition), data in audit.reparsed_conditions.items():
        reparsed_by_stratum.setdefault(stratum, {})[condition] = data

    stored_results = paired_results_for_policy(
        stored_by_stratum,
        pairs,
        baseline_condition_name=args.baseline_condition,
        initial_policy=args.initial_policy,
        bootstrap_repetitions=args.bootstrap_reps,
        seed=args.seed,
        parse_policy="stored",
    )
    reparsed_results = paired_results_for_policy(
        reparsed_by_stratum,
        pairs,
        baseline_condition_name=args.baseline_condition,
        initial_policy=args.initial_policy,
        bootstrap_repetitions=args.bootstrap_reps,
        seed=args.seed,
        parse_policy="current_parser_when_raw_available",
    )
    bounds_rows = []
    for stratum, conditions in sorted(stored_by_stratum.items()):
        baseline_data = conditions.get(args.baseline_condition) if args.baseline_condition else None
        for left_name, right_name in pairs:
            bounds_rows.extend(
                missing_data_bounds(
                    stratum,
                    conditions[left_name],
                    conditions[right_name],
                    baseline_condition=baseline_data,
                    initial_policy=args.initial_policy,
                )
            )
    apply_holm_correction(bounds_rows, "outcome")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "parser_condition_summary.csv", audit.condition_summary)
    write_csv(output_dir / "parser_method_counts.csv", audit.method_counts)
    write_csv(output_dir / "parser_item_audit.csv", audit.audit_rows)
    write_csv(output_dir / "parser_review_sample.csv", audit.review_sample)
    write_csv(
        output_dir / "stored_vs_reparsed_pairwise_results.csv",
        [*stored_results, *reparsed_results],
    )
    write_csv(output_dir / "invalid_missing_data_bounds.csv", bounds_rows)
    (output_dir / "invalid_parser_sensitivity_report.md").write_text(
        build_report(
            condition_summary=audit.condition_summary,
            stored_results=stored_results,
            reparsed_results=reparsed_results,
            bounds_rows=bounds_rows,
        ),
        encoding="utf-8",
    )
    metadata = {
        "primary_policy": "stored_predictions_common_valid",
        "parser_sensitivity_policy": (
            "use_current_parser_when_phase_raw_text_is_available; otherwise retain stored; "
            "execution_error_final_rows_remain_invalid"
        ),
        "missing_output_sensitivity": {
            "invalid_as_non_event": "missing binary outcomes assigned event=0",
            "worst_case_bounds": (
                "each missing binary outcome allowed to be 0 or 1; report minimum and "
                "maximum paired mean difference"
            ),
        },
        "initial_policy": args.initial_policy,
        "baseline_condition": args.baseline_condition,
        "bootstrap_repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "sample_per_category": args.sample_per_category,
        "pairs": [{"left": left, "right": right} for left, right in pairs],
        "parser_source": "agent_conformity_project/src/parser.py",
        "conditions": [
            {
                "stratum": stratum,
                "condition": condition,
                "path": str(path.resolve()),
            }
            for (stratum, condition), path in sorted(condition_paths.items())
        ],
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    final_summary = [row for row in audit.condition_summary if row["phase"] == "final"]
    print(
        f"[OK] audited {sum(row['total'] for row in final_summary)} final outputs across "
        f"{len(final_summary)} condition-strata"
    )
    print(
        f"[OK] stored invalid final outputs: "
        f"{sum(row['stored_invalid'] for row in final_summary)}"
    )
    print(
        f"[OK] stored-invalid/current-parser-valid outputs: "
        f"{sum(row['recoverable_stored_invalid'] for row in final_summary)}"
    )
    print(f"[OK] wrote parser summary: {output_dir / 'parser_condition_summary.csv'}")
    print(f"[OK] wrote missing-data bounds: {output_dir / 'invalid_missing_data_bounds.csv'}")
    print(
        f"[OK] wrote Markdown report: "
        f"{output_dir / 'invalid_parser_sensitivity_report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
