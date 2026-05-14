#!/usr/bin/env python3
"""Independent-batch replication and repeated-subset robustness analysis.

The script first analyzes each non-overlapping result batch separately. It then
merges batches after enforcing unique item IDs, constructs a paired common-valid
pool, repeatedly samples questions without replacement, and reports the observed
variation across overlapping repetitions. Repeated subsets are explicitly not
treated as independent experiments.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from paired_protocol_analysis import (
        KNOWN_PROTOCOL_FILES,
        OUTCOME_DESCRIPTIONS,
        LoadedCondition,
        analyze_pair,
        apply_holm_correction,
        exact_mcnemar_pvalue,
        final_answer,
        gold_and_distractor,
        load_condition,
        outcome_events,
        parse_key_value,
        parse_pairs,
        resolve_baseline,
        stable_seed,
    )
except ImportError:  # pragma: no cover - used when imported as a package
    from .paired_protocol_analysis import (
        KNOWN_PROTOCOL_FILES,
        OUTCOME_DESCRIPTIONS,
        LoadedCondition,
        analyze_pair,
        apply_holm_correction,
        exact_mcnemar_pvalue,
        final_answer,
        gold_and_distractor,
        load_condition,
        outcome_events,
        parse_key_value,
        parse_pairs,
        resolve_baseline,
        stable_seed,
    )


@dataclass
class Batch:
    stratum: str
    name: str
    directory: Path
    conditions: dict[str, LoadedCondition]


def parse_batch_spec(spec: str) -> tuple[str, str, Path]:
    key, value = parse_key_value(spec, "--batch")
    if "::" not in key:
        raise ValueError(
            f"--batch must use STRATUM::BATCH=DIR syntax so independent batches "
            f"have explicit labels: {spec!r}"
        )
    stratum, batch = (part.strip() for part in key.split("::", 1))
    if not stratum or not batch:
        raise ValueError(f"Invalid --batch key: {key!r}")
    return stratum, batch, Path(value)


def load_batch(stratum: str, batch_name: str, directory: Path) -> Batch:
    if not directory.exists():
        raise FileNotFoundError(f"Batch directory does not exist: {directory}")
    conditions = {}
    for condition_name, filename in KNOWN_PROTOCOL_FILES.items():
        path = directory / filename
        if path.exists():
            conditions[condition_name] = load_condition(stratum, condition_name, path)
    if len(conditions) < 2:
        raise ValueError(
            f"Batch {stratum}::{batch_name} has fewer than two standard protocol files "
            f"in {directory}."
        )
    return Batch(
        stratum=stratum,
        name=batch_name,
        directory=directory,
        conditions=conditions,
    )


def collect_batches(args: argparse.Namespace) -> list[Batch]:
    specs = list(args.batch)
    if args.input_dir:
        directory = Path(args.input_dir)
        specs.append(f"{directory.name or 'default'}::single={directory}")
    if not specs:
        raise ValueError("Provide --input-dir or at least one --batch STRATUM::BATCH=DIR.")

    batches = []
    seen = set()
    for spec in specs:
        stratum, batch_name, directory = parse_batch_spec(spec)
        key = (stratum, batch_name)
        if key in seen:
            raise ValueError(f"Batch label specified more than once: {stratum}::{batch_name}")
        seen.add(key)
        batches.append(load_batch(stratum, batch_name, directory))
    return batches


def batches_by_stratum(batches: list[Batch]) -> dict[str, list[Batch]]:
    grouped = {}
    for batch in batches:
        grouped.setdefault(batch.stratum, []).append(batch)
    return {stratum: sorted(values, key=lambda batch: batch.name) for stratum, values in grouped.items()}


def determine_pairs(
    pair_specs: list[str],
    grouped_batches: dict[str, list[Batch]],
) -> list[tuple[str, str]]:
    synthetic = {}
    for stratum, batches in grouped_batches.items():
        common = set.intersection(*(set(batch.conditions) for batch in batches))
        synthetic[stratum] = {
            name: batches[0].conditions[name]
            for name in common
        }
    return parse_pairs(pair_specs, synthetic, baseline_condition=None)


def validate_batch_conditions(
    grouped_batches: dict[str, list[Batch]],
    pairs: list[tuple[str, str]],
    baseline_condition: str | None,
) -> None:
    required = {condition for pair in pairs for condition in pair}
    if baseline_condition:
        required.add(baseline_condition)
    for stratum, batches in grouped_batches.items():
        for batch in batches:
            missing = sorted(required - set(batch.conditions))
            if missing:
                raise ValueError(
                    f"Batch {stratum}::{batch.name} lacks required conditions: "
                    f"{', '.join(missing)}"
                )


def batch_quality_rows(batches: list[Batch]) -> list[dict[str, Any]]:
    rows = []
    for batch in sorted(batches, key=lambda value: (value.stratum, value.name)):
        for condition_name, condition in sorted(batch.conditions.items()):
            rows.append(
                {
                    "stratum": batch.stratum,
                    "batch": batch.name,
                    "condition": condition_name,
                    "directory": str(batch.directory.resolve()),
                    **{
                        key: value
                        for key, value in condition.quality.items()
                        if key not in {"stratum", "condition", "path"}
                    },
                }
            )
    return rows


def independent_batch_results(
    grouped_batches: dict[str, list[Batch]],
    pairs: list[tuple[str, str]],
    baseline_condition: str | None,
    initial_policy: str,
    bootstrap_repetitions: int,
    seed: int,
) -> list[dict[str, Any]]:
    results = []
    for stratum, batches in sorted(grouped_batches.items()):
        for batch in batches:
            baseline_data = batch.conditions.get(baseline_condition) if baseline_condition else None
            for left_name, right_name in pairs:
                pair_results, _ = analyze_pair(
                    stratum=stratum,
                    left_condition=batch.conditions[left_name],
                    right_condition=batch.conditions[right_name],
                    baseline_condition=baseline_data,
                    initial_policy=initial_policy,
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=stable_seed(seed, "independent_batch", stratum, batch.name),
                )
                for row in pair_results:
                    row["batch"] = batch.name
                    row["analysis_type"] = "independent_non_overlapping_batch"
                results.extend(pair_results)
    apply_holm_correction(results, "outcome")
    return results


def merge_conditions(
    grouped_batches: dict[str, list[Batch]],
    required_conditions: set[str],
) -> tuple[
    dict[str, dict[str, LoadedCondition]],
    dict[tuple[str, str, str], str],
]:
    merged_by_stratum = {}
    source_by_item = {}
    for stratum, batches in sorted(grouped_batches.items()):
        merged_conditions = {}
        for condition_name in sorted(required_conditions):
            merged_rows = {}
            duplicate_sources = {}
            for batch in batches:
                condition = batch.conditions[condition_name]
                for item_id, record in condition.rows.items():
                    if item_id in merged_rows:
                        duplicate_sources.setdefault(
                            item_id,
                            [source_by_item[(stratum, condition_name, item_id)]],
                        ).append(batch.name)
                        continue
                    merged_rows[item_id] = record
                    source_by_item[(stratum, condition_name, item_id)] = batch.name
            if duplicate_sources:
                preview = ", ".join(
                    f"{item_id} ({'/'.join(sources)})"
                    for item_id, sources in list(sorted(duplicate_sources.items()))[:10]
                )
                raise ValueError(
                    f"Cross-batch duplicate item IDs in {stratum}::{condition_name}: {preview}. "
                    "Repeated-subset pooling requires genuinely non-overlapping batches."
                )
            merged_conditions[condition_name] = LoadedCondition(
                stratum=stratum,
                name=condition_name,
                path=Path("<merged-batches>"),
                rows=merged_rows,
                quality={"unique_item_ids": len(merged_rows)},
            )
        merged_by_stratum[stratum] = merged_conditions
    return merged_by_stratum, source_by_item


def build_pair_pool(
    stratum: str,
    left_condition: LoadedCondition,
    right_condition: LoadedCondition,
    baseline_condition: LoadedCondition | None,
    initial_policy: str,
    source_by_item: dict[tuple[str, str, str], str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    left_name = left_condition.name
    right_name = right_condition.name
    pair_name = f"{left_name} vs {right_name}"
    union_ids = sorted(set(left_condition.rows) | set(right_condition.rows))
    pool = []
    audit = []
    reason_counts = Counter()

    for item_id in union_ids:
        left_record = left_condition.rows.get(item_id)
        right_record = right_condition.rows.get(item_id)
        reasons = []
        left_source = source_by_item.get((stratum, left_name, item_id))
        right_source = source_by_item.get((stratum, right_name, item_id))
        baseline_source = (
            source_by_item.get((stratum, baseline_condition.name, item_id))
            if baseline_condition
            else None
        )
        left_final = final_answer(left_record)
        right_final = final_answer(right_record)
        left_gold, left_distractor = gold_and_distractor(left_record)
        right_gold, right_distractor = gold_and_distractor(right_record)

        if left_record is None:
            reasons.append("left_item_missing")
        if right_record is None:
            reasons.append("right_item_missing")
        metadata_consistent = bool(
            left_record
            and right_record
            and left_gold is not None
            and left_distractor is not None
            and left_gold == right_gold
            and left_distractor == right_distractor
        )
        if left_record and right_record and not metadata_consistent:
            reasons.append("metadata_mismatch")
        if left_record and left_final is None:
            reasons.append("left_final_invalid")
        if right_record and right_final is None:
            reasons.append("right_final_invalid")
        if left_source and right_source and left_source != right_source:
            reasons.append("condition_batch_mismatch")
        if baseline_condition and baseline_source and left_source and baseline_source != left_source:
            reasons.append("baseline_batch_mismatch")

        baseline = None
        baseline_status = "not_evaluated"
        if metadata_consistent and left_record and right_record:
            baseline, baseline_status = resolve_baseline(
                item_id=item_id,
                left_record=left_record,
                right_record=right_record,
                baseline_condition=baseline_condition,
                initial_policy=initial_policy,
                gold=left_gold,
                distractor=left_distractor,
            )
            if baseline is None:
                reasons.append(baseline_status)

        eligible = not reasons
        for reason in reasons:
            reason_counts[reason] += 1
        source_batch = left_source if left_source == right_source else None
        audit.append(
            {
                "stratum": stratum,
                "pair": pair_name,
                "item_id": item_id,
                "left_condition": left_name,
                "right_condition": right_name,
                "left_batch": left_source,
                "right_batch": right_source,
                "baseline_batch": baseline_source,
                "gold": left_gold or right_gold,
                "distractor": left_distractor or right_distractor,
                "baseline_answer": baseline,
                "baseline_status": baseline_status,
                "left_final": left_final,
                "right_final": right_final,
                "pool_eligible": eligible,
                "exclusion_reason": ";".join(reasons),
            }
        )
        if eligible:
            pool.append(
                {
                    "stratum": stratum,
                    "pair": pair_name,
                    "item_id": item_id,
                    "source_batch": source_batch,
                    "gold": left_gold,
                    "distractor": left_distractor,
                    "baseline": baseline,
                    "left_final": left_final,
                    "right_final": right_final,
                }
            )

    summary = {
        "stratum": stratum,
        "pair": pair_name,
        "left_condition": left_name,
        "right_condition": right_name,
        "union_item_n": len(union_ids),
        "eligible_pool_n": len(pool),
        "excluded_n": len(union_ids) - len(pool),
        "batch_counts": json.dumps(
            dict(sorted(Counter(row["source_batch"] for row in pool).items())),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "exclusion_counts": json.dumps(
            dict(sorted(reason_counts.items())),
            ensure_ascii=False,
            sort_keys=True,
        ),
    }
    return pool, audit, summary


def outcome_values(
    items: list[dict[str, Any]],
    outcome: str,
) -> tuple[list[int], list[int]]:
    left_events = []
    right_events = []
    for item in items:
        eligible, left_event, right_event = outcome_events(
            outcome=outcome,
            left_final=item["left_final"],
            right_final=item["right_final"],
            baseline=item["baseline"],
            gold=item["gold"],
            distractor=item["distractor"],
        )
        if eligible:
            left_events.append(int(left_event))
            right_events.append(int(right_event))
    return left_events, right_events


def describe_subset(
    items: list[dict[str, Any]],
    outcome: str,
) -> dict[str, Any]:
    left_events, right_events = outcome_values(items, outcome)
    eligible_n = len(left_events)
    left_rate = sum(left_events) / eligible_n if eligible_n else None
    right_rate = sum(right_events) / eligible_n if eligible_n else None
    n10 = sum(left == 1 and right == 0 for left, right in zip(left_events, right_events))
    n01 = sum(left == 0 and right == 1 for left, right in zip(left_events, right_events))
    return {
        "eligible_n": eligible_n,
        "left_event_n": sum(left_events),
        "right_event_n": sum(right_events),
        "left_rate": left_rate,
        "right_rate": right_rate,
        "delta_right_minus_left": (
            right_rate - left_rate if eligible_n else None
        ),
        "discordant_left1_right0": n10,
        "discordant_left0_right1": n01,
        "mcnemar_exact_p": exact_mcnemar_pvalue(n10, n01) if eligible_n else None,
    }


def repeated_samples(
    pools: dict[tuple[str, str], list[dict[str, Any]]],
    repetitions: int,
    sample_size: int,
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    run_rows = []
    membership_rows = []
    composition_rows = []
    full_pool_rows = []

    for (stratum, pair_name), pool in sorted(pools.items()):
        if len(pool) <= sample_size:
            raise ValueError(
                f"Pool {stratum}::{pair_name} has {len(pool)} eligible items, but "
                f"--sample-size is {sample_size}. The pool must be larger than the "
                "sample so repetitions can vary; combine non-overlapping batches or "
                "choose a smaller sample size."
            )
        for outcome in OUTCOME_DESCRIPTIONS:
            full_pool_rows.append(
                {
                    "stratum": stratum,
                    "pair": pair_name,
                    "pool_n": len(pool),
                    "outcome": outcome,
                    **describe_subset(pool, outcome),
                }
            )

        pool_array = np.arange(len(pool))
        for repetition in range(1, repetitions + 1):
            repetition_seed = stable_seed(
                seed,
                "repeated_subset",
                stratum,
                pair_name,
                str(repetition),
            )
            rng = np.random.default_rng(repetition_seed)
            selected_indices = rng.choice(pool_array, size=sample_size, replace=False)
            selected_items = [pool[int(index)] for index in selected_indices]
            for item in sorted(selected_items, key=lambda value: str(value["item_id"])):
                membership_rows.append(
                    {
                        "stratum": stratum,
                        "pair": pair_name,
                        "repetition": repetition,
                        "repetition_seed": repetition_seed,
                        "item_id": item["item_id"],
                        "source_batch": item["source_batch"],
                    }
                )
            batch_counts = Counter(item["source_batch"] for item in selected_items)
            for batch, count in sorted(batch_counts.items()):
                composition_rows.append(
                    {
                        "stratum": stratum,
                        "pair": pair_name,
                        "repetition": repetition,
                        "repetition_seed": repetition_seed,
                        "source_batch": batch,
                        "sampled_n": count,
                        "sampled_share": count / sample_size,
                    }
                )
            for outcome in OUTCOME_DESCRIPTIONS:
                run_rows.append(
                    {
                        "stratum": stratum,
                        "pair": pair_name,
                        "repetition": repetition,
                        "repetition_seed": repetition_seed,
                        "sample_n": sample_size,
                        "outcome": outcome,
                        **describe_subset(selected_items, outcome),
                    }
                )
    return run_rows, membership_rows, composition_rows, full_pool_rows


def numeric_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "sd": None,
            "p2_5": None,
            "p97_5": None,
            "min": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p2_5": float(np.quantile(array, 0.025)),
        "p97_5": float(np.quantile(array, 0.975)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_repetitions(
    run_rows: list[dict[str, Any]],
    full_pool_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    full_index = {
        (row["stratum"], row["pair"], row["outcome"]): row
        for row in full_pool_rows
    }
    grouped = {}
    for row in run_rows:
        grouped.setdefault((row["stratum"], row["pair"], row["outcome"]), []).append(row)
    summaries = []
    for key, rows in sorted(grouped.items()):
        eligible = numeric_summary([float(row["eligible_n"]) for row in rows])
        left = numeric_summary(
            [float(row["left_rate"]) for row in rows if row["left_rate"] is not None]
        )
        right = numeric_summary(
            [float(row["right_rate"]) for row in rows if row["right_rate"] is not None]
        )
        delta_values = [
            float(row["delta_right_minus_left"])
            for row in rows
            if row["delta_right_minus_left"] is not None
        ]
        delta = numeric_summary(delta_values)
        full = full_index[key]
        full_delta = full["delta_right_minus_left"]
        if full_delta is None or full_delta == 0:
            same_direction_share = None
        else:
            same_direction_share = sum(
                (value > 0) == (full_delta > 0) for value in delta_values
            ) / len(delta_values) if delta_values else None
        summaries.append(
            {
                "stratum": key[0],
                "pair": key[1],
                "outcome": key[2],
                "repetitions": len(rows),
                "sample_n": rows[0]["sample_n"],
                "full_pool_n": full["pool_n"],
                "full_pool_eligible_n": full["eligible_n"],
                "full_pool_left_rate": full["left_rate"],
                "full_pool_right_rate": full["right_rate"],
                "full_pool_delta": full_delta,
                "eligible_n_mean": eligible["mean"],
                "eligible_n_sd": eligible["sd"],
                "eligible_n_min": eligible["min"],
                "eligible_n_max": eligible["max"],
                "left_rate_mean": left["mean"],
                "left_rate_sd": left["sd"],
                "left_rate_empirical_p2_5": left["p2_5"],
                "left_rate_empirical_p97_5": left["p97_5"],
                "right_rate_mean": right["mean"],
                "right_rate_sd": right["sd"],
                "right_rate_empirical_p2_5": right["p2_5"],
                "right_rate_empirical_p97_5": right["p97_5"],
                "delta_mean": delta["mean"],
                "delta_sd": delta["sd"],
                "delta_empirical_p2_5": delta["p2_5"],
                "delta_empirical_p97_5": delta["p97_5"],
                "delta_min": delta["min"],
                "delta_max": delta["max"],
                "same_direction_as_full_pool_share": same_direction_share,
                "mcnemar_p_below_0_05_share_descriptive": (
                    sum(
                        row["mcnemar_exact_p"] is not None
                        and row["mcnemar_exact_p"] < 0.05
                        for row in rows
                    )
                    / len(rows)
                ),
            }
        )
    return summaries


def overlap_summary(membership_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {}
    for row in membership_rows:
        key = (row["stratum"], row["pair"])
        grouped.setdefault(key, {}).setdefault(int(row["repetition"]), set()).add(row["item_id"])
    results = []
    for (stratum, pair_name), repetitions in sorted(grouped.items()):
        overlaps = []
        jaccards = []
        for left_rep, right_rep in itertools.combinations(sorted(repetitions), 2):
            left = repetitions[left_rep]
            right = repetitions[right_rep]
            intersection = len(left & right)
            union = len(left | right)
            overlaps.append(intersection)
            jaccards.append(intersection / union if union else 0.0)
        overlap_stats = numeric_summary([float(value) for value in overlaps])
        jaccard_stats = numeric_summary(jaccards)
        results.append(
            {
                "stratum": stratum,
                "pair": pair_name,
                "repetitions": len(repetitions),
                "repetition_pairs": len(overlaps),
                "overlap_n_mean": overlap_stats["mean"],
                "overlap_n_sd": overlap_stats["sd"],
                "overlap_n_min": overlap_stats["min"],
                "overlap_n_max": overlap_stats["max"],
                "jaccard_mean": jaccard_stats["mean"],
                "jaccard_min": jaccard_stats["min"],
                "jaccard_max": jaccard_stats["max"],
            }
        )
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


def build_report(
    pool_quality: list[dict[str, Any]],
    independent_results: list[dict[str, Any]],
    repeated_summary: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    repetitions: int,
    sample_size: int,
) -> str:
    lines = [
        "# Repeated-Subset Robustness Analysis",
        "",
        "## Design",
        "",
        f"- Repetitions: `{repetitions}`",
        f"- Questions sampled per repetition: `{sample_size}`",
        "- Sampling within a repetition: without replacement.",
        "- Sampling across repetitions: overlap is allowed and explicitly quantified.",
        "- Independent batch results and overlapping repeated-subset results are reported separately.",
        "- Empirical 2.5–97.5% intervals across repetitions are descriptive robustness ranges, not independent-replication confidence intervals.",
        "",
        "## Pooled pair quality",
        "",
        "| Stratum | Comparison | Union items | Eligible pool | Excluded | Batch counts |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in pool_quality:
        lines.append(
            f"| {row['stratum']} | {row['pair']} | {row['union_item_n']} | "
            f"{row['eligible_pool_n']} | {row['excluded_n']} | `{row['batch_counts']}` |"
        )

    lines.extend(
        [
            "",
            "## Independent non-overlapping batch replication",
            "",
            "| Stratum | Batch | Comparison | Outcome | N | Left rate | Right rate | Delta pp | 95% CI pp |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in independent_results:
        if row["outcome"] not in {"conformity_rate", "harmful_conformity"}:
            continue
        lines.append(
            f"| {row['stratum']} | {row['batch']} | {row['pair']} | {row['outcome']} | "
            f"{row['eligible_n']} | {fmt_pct(row['left_rate'])} | {fmt_pct(row['right_rate'])} | "
            f"{fmt_pp(row['delta_right_minus_left'])} | "
            f"[{fmt_pp(row['delta_ci_low'])}, {fmt_pp(row['delta_ci_high'])}] |"
        )

    lines.extend(
        [
            "",
            "## Repeated-subset summary",
            "",
            "| Stratum | Comparison | Outcome | Mean eligible N | Left mean ± SD | Right mean ± SD | Delta mean ± SD pp | Delta empirical 2.5–97.5 pp | Same direction |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in repeated_summary:
        if row["outcome"] not in {"conformity_rate", "harmful_conformity"}:
            continue
        lines.append(
            f"| {row['stratum']} | {row['pair']} | {row['outcome']} | "
            f"{row['eligible_n_mean']:.1f} | "
            f"{fmt_pct(row['left_rate_mean'])} ± {fmt_pct(row['left_rate_sd'])} | "
            f"{fmt_pct(row['right_rate_mean'])} ± {fmt_pct(row['right_rate_sd'])} | "
            f"{fmt_pp(row['delta_mean'])} ± {fmt_pp(row['delta_sd'])} | "
            f"[{fmt_pp(row['delta_empirical_p2_5'])}, {fmt_pp(row['delta_empirical_p97_5'])}] | "
            f"{fmt_pct(row['same_direction_as_full_pool_share'])} |"
        )

    lines.extend(
        [
            "",
            "## Repetition overlap",
            "",
            "| Stratum | Comparison | Repetition pairs | Mean shared items | Min–max shared | Mean Jaccard |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in overlap_rows:
        lines.append(
            f"| {row['stratum']} | {row['pair']} | {row['repetition_pairs']} | "
            f"{row['overlap_n_mean']:.1f} | {row['overlap_n_min']:.0f}–{row['overlap_n_max']:.0f} | "
            f"{row['jaccard_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The overlap table is included to prevent the repeated subsets from being misinterpreted as independent runs. Use the separate batch estimates as the genuine non-overlapping replication evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze independent result batches, then repeatedly sample paired common-valid items."
        )
    )
    parser.add_argument(
        "--input-dir",
        help="Single-batch shorthand; repeated sampling normally requires multiple --batch inputs.",
    )
    parser.add_argument(
        "--batch",
        action="append",
        default=[],
        metavar="STRATUM::BATCH=DIR",
        help="Add a named non-overlapping result batch; repeat for main500 and new500.",
    )
    parser.add_argument("--pair", action="append", default=[], metavar="LEFT:RIGHT")
    parser.add_argument("--baseline-condition")
    parser.add_argument(
        "--initial-policy",
        choices=("require-equal", "left", "right"),
        default="require-equal",
    )
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if args.sample_size < 1:
        parser.error("--sample-size must be at least 1")
    if args.bootstrap_reps < 1:
        parser.error("--bootstrap-reps must be at least 1")

    try:
        batches = collect_batches(args)
        grouped_batches = batches_by_stratum(batches)
        pairs = determine_pairs(args.pair, grouped_batches)
        validate_batch_conditions(
            grouped_batches,
            pairs,
            baseline_condition=args.baseline_condition,
        )
        required_conditions = {condition for pair in pairs for condition in pair}
        if args.baseline_condition:
            required_conditions.add(args.baseline_condition)
        merged_by_stratum, source_by_item = merge_conditions(
            grouped_batches,
            required_conditions,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    independent_results = independent_batch_results(
        grouped_batches,
        pairs,
        baseline_condition=args.baseline_condition,
        initial_policy=args.initial_policy,
        bootstrap_repetitions=args.bootstrap_reps,
        seed=args.seed,
    )

    pools = {}
    pool_audit = []
    pool_quality = []
    for stratum, conditions in sorted(merged_by_stratum.items()):
        baseline_data = conditions.get(args.baseline_condition) if args.baseline_condition else None
        for left_name, right_name in pairs:
            pool, audit, summary = build_pair_pool(
                stratum,
                conditions[left_name],
                conditions[right_name],
                baseline_condition=baseline_data,
                initial_policy=args.initial_policy,
                source_by_item=source_by_item,
            )
            pools[(stratum, summary["pair"])] = pool
            pool_audit.extend(audit)
            pool_quality.append(summary)

    try:
        run_rows, membership_rows, composition_rows, full_pool_rows = repeated_samples(
            pools,
            repetitions=args.repetitions,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    repeated_summary = summarize_repetitions(run_rows, full_pool_rows)
    overlap_rows = overlap_summary(membership_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "batch_condition_quality.csv", batch_quality_rows(batches))
    write_csv(output_dir / "independent_batch_pairwise_results.csv", independent_results)
    write_csv(output_dir / "pooled_pair_quality.csv", pool_quality)
    write_csv(output_dir / "pooled_item_audit.csv", pool_audit)
    write_csv(output_dir / "full_pool_results.csv", full_pool_rows)
    write_csv(output_dir / "repeated_subset_runs.csv", run_rows)
    write_csv(output_dir / "repetition_item_membership.csv", membership_rows)
    write_csv(output_dir / "repetition_batch_composition.csv", composition_rows)
    write_csv(output_dir / "repeated_subset_summary.csv", repeated_summary)
    write_csv(output_dir / "repetition_overlap_summary.csv", overlap_rows)
    (output_dir / "repeated_subset_report.md").write_text(
        build_report(
            pool_quality=pool_quality,
            independent_results=independent_results,
            repeated_summary=repeated_summary,
            overlap_rows=overlap_rows,
            repetitions=args.repetitions,
            sample_size=args.sample_size,
        ),
        encoding="utf-8",
    )
    metadata = {
        "sampling_unit": "item/question",
        "within_repetition_sampling": "without_replacement",
        "across_repetition_overlap": "allowed_and_reported",
        "repetitions_are_independent_experiments": False,
        "repetitions": args.repetitions,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "bootstrap_repetitions_for_independent_batches": args.bootstrap_reps,
        "initial_policy": args.initial_policy,
        "baseline_condition": args.baseline_condition,
        "pairs": [{"left": left, "right": right} for left, right in pairs],
        "batches": [
            {
                "stratum": batch.stratum,
                "batch": batch.name,
                "directory": str(batch.directory.resolve()),
            }
            for batch in sorted(batches, key=lambda value: (value.stratum, value.name))
        ],
        "empirical_interval_interpretation": (
            "descriptive 2.5-97.5 percentile range across overlapping repeated subsets; "
            "not an independent-replication confidence interval"
        ),
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[OK] analyzed {len(batches)} independent batch(es), "
        f"{len(pools)} pooled pair-stratum(s), and {args.repetitions} repetitions"
    )
    print(f"[OK] wrote independent batch results: {output_dir / 'independent_batch_pairwise_results.csv'}")
    print(f"[OK] wrote repeated-subset summary: {output_dir / 'repeated_subset_summary.csv'}")
    print(f"[OK] wrote membership audit: {output_dir / 'repetition_item_membership.csv'}")
    print(f"[OK] wrote Markdown report: {output_dir / 'repeated_subset_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
