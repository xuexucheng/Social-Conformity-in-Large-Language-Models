#!/usr/bin/env python3
"""Analyze whether protocol effects persist after controlling distractor plausibility.

Plausibility is defined before social exposure as:

    log p(target distractor | question) - log p(gold answer | question)

The script freezes that margin from one baseline condition, audits logprob
coverage, creates within-stratum plausibility quartiles, estimates paired
protocol effects inside each quartile, and fits pooled logistic models with
item-clustered robust standard errors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

try:
    from paired_protocol_analysis import (
        LoadedCondition,
        OUTCOME_DESCRIPTIONS,
        apply_holm_correction,
        collect_condition_paths,
        exact_mcnemar_pvalue,
        final_answer,
        gold_and_distractor,
        initial_answer,
        load_condition,
        paired_bootstrap_interval,
        parse_pairs,
        stable_seed,
        wilson_interval,
    )
except ImportError:  # pragma: no cover - used when imported as a package
    from .paired_protocol_analysis import (
        LoadedCondition,
        OUTCOME_DESCRIPTIONS,
        apply_holm_correction,
        collect_condition_paths,
        exact_mcnemar_pvalue,
        final_answer,
        gold_and_distractor,
        initial_answer,
        load_condition,
        paired_bootstrap_interval,
        parse_pairs,
        stable_seed,
        wilson_interval,
    )


PLAUSIBILITY_OUTCOMES = (
    "final_accuracy",
    "target_adoption",
    "conformity_rate",
    "harmful_conformity",
)
INITIAL_LOGPROB_ALIASES = (
    "initial_option_logprobs",
    "initial_logprobs",
    "option_logprobs",
)
QUARTILE_LABELS = {
    1: "Q1_lowest",
    2: "Q2",
    3: "Q3",
    4: "Q4_highest",
}


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def initial_logprob_map(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    for alias in INITIAL_LOGPROB_ALIASES:
        value = record.get(alias)
        if isinstance(value, dict):
            return value
    return {}


def extract_initial_logprobs(
    record: dict[str, Any] | None,
    gold: str | None,
    distractor: str | None,
) -> tuple[float | None, float | None, float | None]:
    if not record or not gold or not distractor:
        return None, None, None
    values = initial_logprob_map(record)
    gold_logprob = finite_float(values.get(gold))
    distractor_logprob = finite_float(values.get(distractor))
    if gold_logprob is None or distractor_logprob is None:
        return gold_logprob, distractor_logprob, None
    return gold_logprob, distractor_logprob, distractor_logprob - gold_logprob


def assign_rank_quartiles(baseline_rows: list[dict[str, Any]]) -> None:
    valid = [
        row
        for row in baseline_rows
        if row["margin"] is not None and row.get("baseline_status", "usable") == "usable"
    ]
    valid.sort(key=lambda row: (float(row["margin"]), str(row["item_id"])))
    total = len(valid)
    for rank, row in enumerate(valid):
        quartile = min(4, (rank * 4) // total + 1)
        row["quartile"] = quartile
        row["quartile_label"] = QUARTILE_LABELS[quartile]


def build_baseline_rows(
    stratum: str,
    baseline_condition: LoadedCondition,
) -> list[dict[str, Any]]:
    rows = []
    for item_id, record in sorted(baseline_condition.rows.items()):
        gold, distractor = gold_and_distractor(record)
        baseline = initial_answer(record)
        gold_logprob, distractor_logprob, margin = extract_initial_logprobs(
            record, gold, distractor
        )
        if gold is None or distractor is None:
            status = "invalid_metadata"
        elif baseline is None:
            status = "invalid_initial_answer"
        elif gold_logprob is None and distractor_logprob is None:
            status = "gold_and_target_logprobs_missing"
        elif gold_logprob is None:
            status = "gold_logprob_missing"
        elif distractor_logprob is None:
            status = "target_logprob_missing"
        else:
            status = "usable"
        rows.append(
            {
                "stratum": stratum,
                "baseline_condition": baseline_condition.name,
                "item_id": item_id,
                "gold": gold,
                "distractor": distractor,
                "baseline_answer": baseline,
                "baseline_correct": None if baseline is None or gold is None else baseline == gold,
                "baseline_is_target": (
                    None if baseline is None or distractor is None else baseline == distractor
                ),
                "gold_logprob": gold_logprob,
                "target_logprob": distractor_logprob,
                "margin": margin,
                "quartile": None,
                "quartile_label": None,
                "baseline_status": status,
            }
        )
    assign_rank_quartiles(rows)
    return rows


def build_long_audit(
    stratum: str,
    baseline_rows: list[dict[str, Any]],
    conditions: dict[str, LoadedCondition],
    analysis_condition_names: list[str],
    initial_policy: str,
    margin_tolerance: float,
) -> list[dict[str, Any]]:
    audit_rows = []
    for baseline_row in baseline_rows:
        item_id = baseline_row["item_id"]
        for condition_name in analysis_condition_names:
            condition_record = conditions[condition_name].rows.get(item_id)
            condition_gold, condition_distractor = gold_and_distractor(condition_record)
            condition_initial = initial_answer(condition_record)
            condition_final = final_answer(condition_record)
            _, _, condition_margin = extract_initial_logprobs(
                condition_record, condition_gold, condition_distractor
            )

            metadata_consistent = bool(
                condition_record
                and baseline_row["gold"] is not None
                and baseline_row["distractor"] is not None
                and condition_gold == baseline_row["gold"]
                and condition_distractor == baseline_row["distractor"]
            )
            initial_consistent = bool(
                condition_initial is not None
                and condition_initial == baseline_row["baseline_answer"]
            )
            margin_consistent = bool(
                condition_margin is None
                or baseline_row["margin"] is None
                or abs(float(condition_margin) - float(baseline_row["margin"])) <= margin_tolerance
            )

            reasons = []
            if baseline_row["baseline_status"] != "usable":
                reasons.append(baseline_row["baseline_status"])
            if condition_record is None:
                reasons.append("condition_item_missing")
            elif not metadata_consistent:
                reasons.append("condition_metadata_mismatch")
            if condition_record is not None and condition_final is None:
                reasons.append("condition_final_invalid")
            if initial_policy == "require-equal" and condition_record is not None and not initial_consistent:
                reasons.append("condition_initial_mismatch")
            if condition_record is not None and not margin_consistent:
                reasons.append("condition_margin_mismatch")

            analysis_valid = not reasons
            audit_rows.append(
                {
                    **baseline_row,
                    "condition": condition_name,
                    "condition_present": condition_record is not None,
                    "condition_is_error": bool(
                        condition_record and condition_record.get("is_error") is True
                    ),
                    "condition_initial": condition_initial,
                    "condition_final": condition_final,
                    "condition_margin": condition_margin,
                    "metadata_consistent": metadata_consistent,
                    "initial_consistent": initial_consistent,
                    "margin_consistent": margin_consistent,
                    "analysis_valid": analysis_valid,
                    "exclusion_reason": ";".join(reasons),
                }
            )
    return audit_rows


def summarize_coverage(
    stratum: str,
    baseline_rows: list[dict[str, Any]],
    long_audit: list[dict[str, Any]],
    analysis_condition_names: list[str],
) -> list[dict[str, Any]]:
    baseline_total = len(baseline_rows)
    baseline_usable = sum(row["baseline_status"] == "usable" for row in baseline_rows)
    missing_gold = sum(row["gold_logprob"] is None for row in baseline_rows)
    missing_target = sum(row["target_logprob"] is None for row in baseline_rows)
    rows = []
    for condition_name in analysis_condition_names:
        condition_rows = [row for row in long_audit if row["condition"] == condition_name]
        valid = sum(bool(row["analysis_valid"]) for row in condition_rows)
        rows.append(
            {
                "stratum": stratum,
                "condition": condition_name,
                "baseline_items": baseline_total,
                "baseline_margin_usable": baseline_usable,
                "baseline_margin_coverage": baseline_usable / baseline_total if baseline_total else None,
                "gold_logprob_missing": missing_gold,
                "target_logprob_missing": missing_target,
                "condition_present": sum(bool(row["condition_present"]) for row in condition_rows),
                "condition_final_valid": sum(row["condition_final"] is not None for row in condition_rows),
                "initial_mismatch": sum(
                    bool(row["condition_present"])
                    and row["baseline_answer"] is not None
                    and row["condition_initial"] is not None
                    and row["baseline_answer"] != row["condition_initial"]
                    for row in condition_rows
                ),
                "margin_mismatch": sum(not bool(row["margin_consistent"]) for row in condition_rows),
                "analysis_valid": valid,
                "analysis_valid_rate": valid / baseline_total if baseline_total else None,
            }
        )
    return rows


def outcome_event(row: dict[str, Any], outcome: str) -> tuple[bool, int | None]:
    final = row["condition_final"]
    baseline = row["baseline_answer"]
    gold = row["gold"]
    distractor = row["distractor"]
    if not row["analysis_valid"]:
        return False, None
    if outcome == "final_accuracy":
        return True, int(final == gold)
    if outcome == "target_adoption":
        return True, int(final == distractor)
    if outcome == "conformity_rate":
        if baseline == distractor:
            return False, None
        return True, int(final == distractor)
    if outcome == "harmful_conformity":
        if baseline != gold:
            return False, None
        return True, int(final == distractor)
    raise ValueError(f"Unsupported plausibility outcome: {outcome}")


def summarize_quartiles(long_audit: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int, str], list[tuple[dict[str, Any], int]]] = {}
    for row in long_audit:
        quartile = row.get("quartile")
        if quartile is None:
            continue
        for outcome in PLAUSIBILITY_OUTCOMES:
            eligible, event = outcome_event(row, outcome)
            if eligible:
                key = (row["stratum"], row["condition"], int(quartile), outcome)
                groups.setdefault(key, []).append((row, int(event)))

    results = []
    for (stratum, condition, quartile, outcome), values in sorted(groups.items()):
        events = [event for _, event in values]
        margins = [float(row["margin"]) for row, _ in values]
        total = len(events)
        event_count = sum(events)
        rate = event_count / total if total else None
        ci_low, ci_high = wilson_interval(event_count, total)
        results.append(
            {
                "stratum": stratum,
                "condition": condition,
                "quartile": quartile,
                "quartile_label": QUARTILE_LABELS[quartile],
                "outcome": outcome,
                "eligible_n": total,
                "event_n": event_count,
                "rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "margin_mean": float(np.mean(margins)),
                "margin_min": min(margins),
                "margin_max": max(margins),
            }
        )
    return results


def summarize_paired_quartiles(
    long_audit: list[dict[str, Any]],
    pairs: list[tuple[str, str]],
    bootstrap_repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    by_key = {
        (row["stratum"], row["condition"], row["item_id"]): row
        for row in long_audit
        if row["analysis_valid"]
    }
    strata = sorted({row["stratum"] for row in long_audit})
    results = []

    for stratum in strata:
        for left_name, right_name in pairs:
            left_ids = {
                item_id
                for row_stratum, condition, item_id in by_key
                if row_stratum == stratum and condition == left_name
            }
            right_ids = {
                item_id
                for row_stratum, condition, item_id in by_key
                if row_stratum == stratum and condition == right_name
            }
            common_ids = sorted(left_ids & right_ids)
            for quartile in range(1, 5):
                quartile_ids = [
                    item_id
                    for item_id in common_ids
                    if by_key[(stratum, left_name, item_id)]["quartile"] == quartile
                ]
                for outcome in PLAUSIBILITY_OUTCOMES:
                    paired_events = []
                    for item_id in quartile_ids:
                        left_row = by_key[(stratum, left_name, item_id)]
                        right_row = by_key[(stratum, right_name, item_id)]
                        left_eligible, left_event = outcome_event(left_row, outcome)
                        right_eligible, right_event = outcome_event(right_row, outcome)
                        if left_eligible and right_eligible:
                            paired_events.append((int(left_event), int(right_event)))

                    left_events = [value[0] for value in paired_events]
                    right_events = [value[1] for value in paired_events]
                    eligible_count = len(paired_events)
                    left_count = sum(left_events)
                    right_count = sum(right_events)
                    left_rate = left_count / eligible_count if eligible_count else None
                    right_rate = right_count / eligible_count if eligible_count else None
                    delta = right_rate - left_rate if eligible_count else None
                    delta_low, delta_high = paired_bootstrap_interval(
                        left_events,
                        right_events,
                        repetitions=bootstrap_repetitions,
                        seed=stable_seed(
                            seed,
                            "plausibility",
                            stratum,
                            left_name,
                            right_name,
                            str(quartile),
                            outcome,
                        ),
                    )
                    n10 = sum(left == 1 and right == 0 for left, right in paired_events)
                    n01 = sum(left == 0 and right == 1 for left, right in paired_events)
                    results.append(
                        {
                            "stratum": stratum,
                            "pair": f"{left_name} vs {right_name}",
                            "left_condition": left_name,
                            "right_condition": right_name,
                            "quartile": quartile,
                            "quartile_label": QUARTILE_LABELS[quartile],
                            "outcome": outcome,
                            "eligible_n": eligible_count,
                            "left_event_n": left_count,
                            "right_event_n": right_count,
                            "left_rate": left_rate,
                            "right_rate": right_rate,
                            "delta_right_minus_left": delta,
                            "delta_ci_low": delta_low,
                            "delta_ci_high": delta_high,
                            "discordant_left1_right0": n10,
                            "discordant_left0_right1": n01,
                            "mcnemar_exact_p": (
                                exact_mcnemar_pvalue(n10, n01) if eligible_count else None
                            ),
                            "holm_adjusted_p": None,
                            "holm_family_n": None,
                        }
                    )
    apply_holm_correction(results, "outcome")
    return results


def balanced_regression_rows(
    long_audit: list[dict[str, Any]],
    analysis_condition_names: list[str],
) -> list[dict[str, Any]]:
    by_item: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in long_audit:
        if row["analysis_valid"]:
            by_item.setdefault((row["stratum"], row["item_id"]), {})[row["condition"]] = row
    required = set(analysis_condition_names)
    balanced = []
    for condition_rows in by_item.values():
        if required.issubset(condition_rows):
            balanced.extend(condition_rows[name] for name in analysis_condition_names)
    return balanced


def within_stratum_margin_z(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float]:
    unique_items: dict[tuple[str, str], float] = {}
    for row in rows:
        unique_items[(row["stratum"], row["item_id"])] = float(row["margin"])
    grouped: dict[str, list[tuple[str, float]]] = {}
    for (stratum, item_id), margin in unique_items.items():
        grouped.setdefault(stratum, []).append((item_id, margin))
    zscores = {}
    for stratum, values in grouped.items():
        margins = np.asarray([margin for _, margin in values], dtype=np.float64)
        mean = float(margins.mean())
        std = float(margins.std(ddof=0))
        for item_id, margin in values:
            zscores[(stratum, item_id)] = 0.0 if std == 0 else (margin - mean) / std
    return zscores


def design_matrix(
    rows: list[dict[str, Any]],
    reference_condition: str,
    include_interactions: bool,
) -> tuple[np.ndarray, list[str], list[str]]:
    conditions = sorted({row["condition"] for row in rows})
    strata = sorted({row["stratum"] for row in rows})
    if reference_condition not in conditions:
        raise ValueError(f"Reference condition {reference_condition!r} is absent from regression data.")
    zscores = within_stratum_margin_z(rows)
    columns = [np.ones(len(rows), dtype=np.float64)]
    names = ["intercept"]
    margin_z = np.asarray(
        [zscores[(row["stratum"], row["item_id"])] for row in rows], dtype=np.float64
    )
    columns.append(margin_z)
    names.append("margin_z_within_stratum")

    baseline_correct = np.asarray([int(row["baseline_correct"]) for row in rows], dtype=np.float64)
    if len(set(baseline_correct.tolist())) > 1:
        columns.append(baseline_correct)
        names.append("baseline_correct")

    condition_columns = {}
    for condition in conditions:
        if condition == reference_condition:
            continue
        values = np.asarray([int(row["condition"] == condition) for row in rows], dtype=np.float64)
        condition_columns[condition] = values
        columns.append(values)
        names.append(f"condition[{condition}]")

    reference_stratum = strata[0]
    for stratum in strata[1:]:
        columns.append(np.asarray([int(row["stratum"] == stratum) for row in rows], dtype=np.float64))
        names.append(f"stratum[{stratum}]")

    if include_interactions:
        for condition, values in condition_columns.items():
            columns.append(values * margin_z)
            names.append(f"condition[{condition}]:margin_z")

    clusters = [f"{row['stratum']}::{row['item_id']}" for row in rows]
    return np.column_stack(columns), names, clusters


def logistic_fit_clustered(
    x: np.ndarray,
    y: np.ndarray,
    clusters: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations, parameters = x.shape
    unique_outcomes = np.unique(y)
    unique_clusters = sorted(set(clusters))
    diagnostics = {
        "observations": observations,
        "clusters": len(unique_clusters),
        "parameters": parameters,
        "converged": False,
        "optimizer_message": "",
        "condition_number": None,
        "separation_warning": False,
    }
    if observations == 0 or len(unique_outcomes) < 2:
        diagnostics["optimizer_message"] = "Outcome has fewer than two observed classes."
        return [], diagnostics

    def objective(beta: np.ndarray) -> float:
        eta = x @ beta
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    def gradient(beta: np.ndarray) -> np.ndarray:
        probability = expit(x @ beta)
        return x.T @ (probability - y)

    result = minimize(
        objective,
        np.zeros(parameters, dtype=np.float64),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    beta = np.asarray(result.x, dtype=np.float64)
    probability = expit(x @ beta)
    weights = np.clip(probability * (1.0 - probability), 1e-10, None)
    information = x.T @ (weights[:, None] * x)
    bread = np.linalg.pinv(information, rcond=1e-12)

    residual = y - probability
    meat = np.zeros((parameters, parameters), dtype=np.float64)
    cluster_array = np.asarray(clusters)
    for cluster in unique_clusters:
        mask = cluster_array == cluster
        score = x[mask].T @ residual[mask]
        meat += np.outer(score, score)
    covariance = bread @ meat @ bread
    if len(unique_clusters) > 1 and observations > parameters:
        covariance *= (len(unique_clusters) / (len(unique_clusters) - 1)) * (
            (observations - 1) / (observations - parameters)
        )
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    diagnostics.update(
        {
            "converged": bool(result.success),
            "optimizer_message": str(result.message),
            "condition_number": float(np.linalg.cond(information)),
            "separation_warning": bool(np.max(np.abs(beta)) > 15.0 or not result.success),
        }
    )
    coefficient_rows = []
    for coefficient, standard_error in zip(beta, standard_errors):
        if standard_error > 0 and math.isfinite(standard_error):
            z_value = coefficient / standard_error
            p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
            lower = coefficient - 1.959963984540054 * standard_error
            upper = coefficient + 1.959963984540054 * standard_error
        else:
            z_value = None
            p_value = None
            lower = None
            upper = None
        coefficient_rows.append(
            {
                "coefficient": float(coefficient),
                "robust_se": float(standard_error),
                "z": z_value,
                "p": p_value,
                "odds_ratio": float(np.exp(np.clip(coefficient, -700, 700))),
                "or_ci_low": (
                    None if lower is None else float(np.exp(np.clip(lower, -700, 700)))
                ),
                "or_ci_high": (
                    None if upper is None else float(np.exp(np.clip(upper, -700, 700)))
                ),
            }
        )
    return coefficient_rows, diagnostics


def regression_analysis(
    balanced_rows: list[dict[str, Any]],
    reference_condition: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coefficient_results = []
    diagnostic_results = []
    for outcome in PLAUSIBILITY_OUTCOMES:
        outcome_rows = []
        y_values = []
        for row in balanced_rows:
            eligible, event = outcome_event(row, outcome)
            if eligible:
                outcome_rows.append(row)
                y_values.append(int(event))

        for model_name, include_interactions in (
            ("adjusted_main", False),
            ("adjusted_interaction", True),
        ):
            if not outcome_rows:
                diagnostic_results.append(
                    {
                        "model": model_name,
                        "outcome": outcome,
                        "reference_condition": reference_condition,
                        "observations": 0,
                        "clusters": 0,
                        "parameters": 0,
                        "converged": False,
                        "optimizer_message": "No eligible complete-case rows.",
                        "condition_number": None,
                        "separation_warning": False,
                    }
                )
                continue
            x, term_names, clusters = design_matrix(
                outcome_rows,
                reference_condition=reference_condition,
                include_interactions=include_interactions,
            )
            coefficients, diagnostics = logistic_fit_clustered(
                x,
                np.asarray(y_values, dtype=np.float64),
                clusters,
            )
            diagnostic_results.append(
                {
                    "model": model_name,
                    "outcome": outcome,
                    "reference_condition": reference_condition,
                    **diagnostics,
                }
            )
            for term_name, values in zip(term_names, coefficients):
                if term_name.startswith("condition[") and ":margin_z" in term_name:
                    term_type = "condition_margin_interaction"
                elif term_name.startswith("condition["):
                    term_type = "condition"
                elif term_name.startswith("stratum["):
                    term_type = "stratum"
                else:
                    term_type = "covariate"
                coefficient_results.append(
                    {
                        "model": model_name,
                        "outcome": outcome,
                        "reference_condition": reference_condition,
                        "term": term_name,
                        "term_type": term_type,
                        **values,
                        "holm_adjusted_p": None,
                        "holm_family_n": None,
                    }
                )

    grouped = {}
    for row in coefficient_results:
        if row["term_type"] != "condition" or row["p"] is None:
            continue
        key = (row["model"], row["outcome"])
        grouped.setdefault(key, []).append(row)
    for family_rows in grouped.values():
        ordered = sorted(family_rows, key=lambda row: float(row["p"]))
        family_size = len(ordered)
        running_max = 0.0
        for index, row in enumerate(ordered):
            adjusted = min(1.0, (family_size - index) * float(row["p"]))
            running_max = max(running_max, adjusted)
            row["holm_adjusted_p"] = running_max
            row["holm_family_n"] = family_size
    return coefficient_results, diagnostic_results


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


def fmt_num(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def fmt_p(value: float | None) -> str:
    if value is None:
        return "N/A"
    return "<.001" if value < 0.001 else f"{value:.3f}".lstrip("0")


def build_report(
    coverage_rows: list[dict[str, Any]],
    quartile_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    baseline_condition_by_stratum: dict[str, str],
    reference_condition: str,
) -> str:
    lines = [
        "# Distractor Plausibility Analysis",
        "",
        "## Specification",
        "",
        "Plausibility margin: `initial log p(target distractor) - initial log p(gold answer)`.",
        "Quartiles are rank-based within each model-dataset stratum, with item ID as the stable tie-break.",
        "Regression margins are z-standardized within stratum and standard errors are clustered by item.",
        f"Regression reference condition: `{reference_condition}`.",
        "",
        "Frozen baseline sources:",
        "",
    ]
    for stratum, condition in sorted(baseline_condition_by_stratum.items()):
        lines.append(f"- `{stratum}`: `{condition}`")

    lines.extend(
        [
            "",
            "## Logprob coverage",
            "",
            "| Stratum | Condition | Baseline items | Usable margins | Margin coverage | Missing gold LP | Missing target LP | Analysis valid |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage_rows:
        lines.append(
            f"| {row['stratum']} | {row['condition']} | {row['baseline_items']} | "
            f"{row['baseline_margin_usable']} | {fmt_pct(row['baseline_margin_coverage'])} | "
            f"{row['gold_logprob_missing']} | {row['target_logprob_missing']} | "
            f"{row['analysis_valid']} |"
        )

    lines.extend(
        [
            "",
            "## Harmful conformity by plausibility quartile",
            "",
            "| Stratum | Condition | Quartile | N | HCR | 95% CI | Mean margin |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in quartile_rows:
        if row["outcome"] != "harmful_conformity":
            continue
        lines.append(
            f"| {row['stratum']} | {row['condition']} | {row['quartile_label']} | "
            f"{row['eligible_n']} | {fmt_pct(row['rate'])} | "
            f"[{fmt_pct(row['ci_low'])}, {fmt_pct(row['ci_high'])}] | "
            f"{fmt_num(row['margin_mean'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired HCR differences within quartile",
            "",
            "| Stratum | Comparison | Quartile | N | Left HCR | Right HCR | Delta pp | 95% CI pp | McNemar p | Holm p |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        if row["outcome"] != "harmful_conformity":
            continue
        delta = row["delta_right_minus_left"]
        lines.append(
            f"| {row['stratum']} | {row['pair']} | {row['quartile_label']} | "
            f"{row['eligible_n']} | {fmt_pct(row['left_rate'])} | {fmt_pct(row['right_rate'])} | "
            f"{fmt_num(None if delta is None else 100.0 * delta, 2)} | "
            f"[{fmt_num(None if row['delta_ci_low'] is None else 100.0 * row['delta_ci_low'], 2)}, "
            f"{fmt_num(None if row['delta_ci_high'] is None else 100.0 * row['delta_ci_high'], 2)}] | "
            f"{fmt_p(row['mcnemar_exact_p'])} | {fmt_p(row['holm_adjusted_p'])} |"
        )

    lines.extend(
        [
            "",
            "## Plausibility-adjusted regression",
            "",
            "Condition odds ratios compare each protocol with the reference condition after controlling for within-stratum margin, baseline correctness when variable, and stratum fixed effects.",
            "",
            "| Model | Outcome | Term | OR | Robust 95% CI | p | Holm p |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in regression_rows:
        if row["term_type"] not in {"condition", "covariate", "condition_margin_interaction"}:
            continue
        lines.append(
            f"| {row['model']} | {row['outcome']} | {row['term']} | "
            f"{fmt_num(row['odds_ratio'])} | "
            f"[{fmt_num(row['or_ci_low'])}, {fmt_num(row['or_ci_high'])}] | "
            f"{fmt_p(row['p'])} | {fmt_p(row['holm_adjusted_p'])} |"
        )

    warning_models = [row for row in diagnostic_rows if row["separation_warning"]]
    lines.extend(["", "## Diagnostics", ""])
    if warning_models:
        lines.append(
            "Some regression fits have convergence or separation warnings. Inspect `regression_diagnostics.csv` before using their coefficients."
        )
    else:
        lines.append("No regression convergence or large-coefficient warnings were detected.")
    lines.extend(
        [
            "",
            "Missing gold/target logprobs are excluded rather than imputed. See `plausibility_item_audit.csv` for every exclusion decision.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze conformity as a function of the target-minus-gold baseline logprob margin."
        )
    )
    parser.add_argument("--input-dir")
    parser.add_argument("--stratum", action="append", default=[], metavar="NAME=DIR")
    parser.add_argument(
        "--condition", action="append", default=[], metavar="[STRATUM::]NAME=FILE"
    )
    parser.add_argument("--pair", action="append", default=[], metavar="LEFT:RIGHT")
    parser.add_argument(
        "--reference-condition",
        help="Regression reference protocol. Defaults to Batch-Single when available.",
    )
    parser.add_argument(
        "--baseline-condition",
        help=(
            "Condition providing frozen initial answers and logprobs. Defaults to the "
            "reference condition. It may also be one of the analyzed protocols."
        ),
    )
    parser.add_argument(
        "--initial-policy",
        choices=("require-equal", "baseline-only"),
        default="require-equal",
    )
    parser.add_argument("--margin-tolerance", type=float, default=1e-8)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def choose_reference_condition(
    requested: str | None,
    pairs: list[tuple[str, str]],
) -> str:
    analysis_conditions = sorted({condition for pair in pairs for condition in pair})
    if requested:
        if requested not in analysis_conditions:
            raise ValueError(
                f"Reference condition {requested!r} is not present in the selected pairs."
            )
        return requested
    if "Batch-Single" in analysis_conditions:
        return "Batch-Single"
    return analysis_conditions[0]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be at least 1")
    if args.margin_tolerance < 0:
        parser.error("--margin-tolerance must be non-negative")

    try:
        condition_paths = collect_condition_paths(args)
        loaded = {
            key: load_condition(key[0], key[1], path)
            for key, path in sorted(condition_paths.items())
        }
        conditions_by_stratum: dict[str, dict[str, LoadedCondition]] = {}
        for (stratum, condition), data in loaded.items():
            conditions_by_stratum.setdefault(stratum, {})[condition] = data
        pairs = parse_pairs(args.pair, conditions_by_stratum, baseline_condition=None)
        reference_condition = choose_reference_condition(args.reference_condition, pairs)
        baseline_condition_name = args.baseline_condition or reference_condition
        for stratum, conditions in conditions_by_stratum.items():
            if baseline_condition_name not in conditions:
                raise ValueError(
                    f"Stratum {stratum!r} lacks baseline condition {baseline_condition_name!r}."
                )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    analysis_condition_names = sorted({condition for pair in pairs for condition in pair})
    all_baseline_rows = []
    all_long_audit = []
    coverage_rows = []
    baseline_condition_by_stratum = {}
    for stratum, conditions in sorted(conditions_by_stratum.items()):
        baseline_condition = conditions[baseline_condition_name]
        baseline_condition_by_stratum[stratum] = baseline_condition.name
        baseline_rows = build_baseline_rows(stratum, baseline_condition)
        long_audit = build_long_audit(
            stratum=stratum,
            baseline_rows=baseline_rows,
            conditions=conditions,
            analysis_condition_names=analysis_condition_names,
            initial_policy=args.initial_policy,
            margin_tolerance=args.margin_tolerance,
        )
        all_baseline_rows.extend(baseline_rows)
        all_long_audit.extend(long_audit)
        coverage_rows.extend(
            summarize_coverage(
                stratum,
                baseline_rows,
                long_audit,
                analysis_condition_names,
            )
        )

    quartile_rows = summarize_quartiles(all_long_audit)
    paired_rows = summarize_paired_quartiles(
        all_long_audit,
        pairs=pairs,
        bootstrap_repetitions=args.bootstrap_reps,
        seed=args.seed,
    )
    balanced_rows = balanced_regression_rows(all_long_audit, analysis_condition_names)
    regression_rows, diagnostic_rows = regression_analysis(
        balanced_rows,
        reference_condition=reference_condition,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "logprob_coverage.csv", coverage_rows)
    write_csv(output_dir / "baseline_margin_audit.csv", all_baseline_rows)
    write_csv(output_dir / "plausibility_item_audit.csv", all_long_audit)
    write_csv(output_dir / "quartile_condition_rates.csv", quartile_rows)
    write_csv(output_dir / "paired_quartile_effects.csv", paired_rows)
    write_csv(output_dir / "plausibility_adjusted_regression.csv", regression_rows)
    write_csv(output_dir / "regression_diagnostics.csv", diagnostic_rows)
    (output_dir / "distractor_plausibility_report.md").write_text(
        build_report(
            coverage_rows=coverage_rows,
            quartile_rows=quartile_rows,
            paired_rows=paired_rows,
            regression_rows=regression_rows,
            diagnostic_rows=diagnostic_rows,
            baseline_condition_by_stratum=baseline_condition_by_stratum,
            reference_condition=reference_condition,
        ),
        encoding="utf-8",
    )
    metadata = {
        "margin_definition": "initial_logprob_target_minus_initial_logprob_gold",
        "quartile_definition": "rank_based_within_stratum_stable_item_id_tiebreak",
        "regression_margin_scaling": "z_standardized_within_stratum",
        "regression_standard_errors": "item_clustered_sandwich",
        "reference_condition": reference_condition,
        "baseline_condition": baseline_condition_name,
        "initial_policy": args.initial_policy,
        "margin_tolerance": args.margin_tolerance,
        "bootstrap_repetitions": args.bootstrap_reps,
        "seed": args.seed,
        "pairs": [{"left": left, "right": right} for left, right in pairs],
        "outcomes": {
            outcome: OUTCOME_DESCRIPTIONS[outcome] for outcome in PLAUSIBILITY_OUTCOMES
        },
        "missing_logprob_policy": "exclude_without_imputation",
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

    print(
        f"[OK] analyzed {len(conditions_by_stratum)} stratum/strata, "
        f"{len(analysis_condition_names)} condition(s), and {len(pairs)} pair(s)"
    )
    print(f"[OK] usable baseline margins: {sum(row['baseline_status'] == 'usable' for row in all_baseline_rows)}")
    print(f"[OK] wrote logprob coverage: {output_dir / 'logprob_coverage.csv'}")
    print(f"[OK] wrote quartile effects: {output_dir / 'paired_quartile_effects.csv'}")
    print(
        f"[OK] wrote adjusted regression: "
        f"{output_dir / 'plausibility_adjusted_regression.csv'}"
    )
    print(
        f"[OK] wrote Markdown report: "
        f"{output_dir / 'distractor_plausibility_report.md'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
