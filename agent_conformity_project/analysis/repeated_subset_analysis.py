#!/usr/bin/env python3
"""Repeated 500-item robustness analysis over cached per-item predictions.

No model inference is performed.  For each dataset, one aligned item sample is
drawn per repetition and reused across every model and condition.  Sampling is
without replacement within a repetition; repetitions may overlap.  Metrics are
computed on each condition's valid rows, while protocol deltas use common-valid
paired rows inside the sampled item set.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from paired_protocol_analysis import (
    METRICS,
    audit_pair,
    condition_summary,
    load_condition,
    load_manifest,
    paired_ids,
    _percentile,
)


def _aggregate(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "repetitions": len(values),
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "empirical_low": _percentile(ordered, 0.025),
        "empirical_high": _percentile(ordered, 0.975),
    }


def analyze_repeated_subsets(
    manifest: dict[str, Any],
    sample_size: int,
    repetitions: int,
    seed_start: int,
) -> dict[str, Any]:
    loaded_cells = []
    for cell in manifest["cells"]:
        loaded_cells.append(
            {
                **cell,
                "records": {
                    name: load_condition(path)
                    for name, path in cell["conditions"].items()
                },
            }
        )

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in loaded_cells:
        by_dataset[cell["dataset"]].append(cell)

    frames: dict[str, list[str]] = {}
    for dataset, cells in by_dataset.items():
        id_sets = [
            set(records)
            for cell in cells
            for records in cell["records"].values()
        ]
        frame = sorted(set.intersection(*id_sets))
        if len(frame) < sample_size:
            raise ValueError(
                f"dataset {dataset!r} has only {len(frame)} aligned attempted IDs; "
                f"cannot sample {sample_size}"
            )
        frames[dataset] = frame

    samples = []
    condition_rows = []
    delta_rows = []
    for repetition in range(repetitions):
        seed = seed_start + repetition
        selected_by_dataset = {}
        for dataset, frame in frames.items():
            rng = random.Random(seed)
            selected_by_dataset[dataset] = sorted(rng.sample(frame, sample_size))
            samples.append(
                {
                    "repetition": repetition + 1,
                    "seed": seed,
                    "dataset": dataset,
                    "frame_n": len(frame),
                    "sample_ids": selected_by_dataset[dataset],
                }
            )

        for cell in loaded_cells:
            allowed = set(selected_by_dataset[cell["dataset"]])
            for condition, records in cell["records"].items():
                summary = condition_summary(records, allowed_ids=allowed)
                for metric_name, _, _ in METRICS:
                    value = summary[metric_name]
                    if value is None:
                        continue
                    condition_rows.append(
                        {
                            "repetition": repetition + 1,
                            "seed": seed,
                            "dataset": cell["dataset"],
                            "model": cell["model"],
                            "condition": condition,
                            "metric": metric_name,
                            "denominator_n": summary[f"{metric_name}_n"],
                            "value": value,
                        }
                    )
            for condition1, condition2 in manifest["comparisons"]:
                records1 = cell["records"][condition1]
                records2 = cell["records"][condition2]
                audit_pair(condition1, records1, condition2, records2)
                for metric_name, outcome, eligibility in METRICS:
                    ids = paired_ids(records1, records2, eligibility, allowed_ids=allowed)
                    if not ids:
                        continue
                    rate1 = sum(outcome(records1[item_id]) for item_id in ids) / len(ids)
                    rate2 = sum(outcome(records2[item_id]) for item_id in ids) / len(ids)
                    delta_rows.append(
                        {
                            "repetition": repetition + 1,
                            "seed": seed,
                            "dataset": cell["dataset"],
                            "model": cell["model"],
                            "condition1": condition1,
                            "condition2": condition2,
                            "metric": metric_name,
                            "paired_n": len(ids),
                            "rate1": rate1,
                            "rate2": rate2,
                            "delta": rate2 - rate1,
                        }
                    )

    condition_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in condition_rows:
        key = (row["dataset"], row["model"], row["condition"], row["metric"])
        condition_groups[key].append(row)
    condition_aggregate = []
    for key, rows in sorted(condition_groups.items()):
        condition_aggregate.append(
            {
                "dataset": key[0],
                "model": key[1],
                "condition": key[2],
                "metric": key[3],
                **_aggregate([row["value"] for row in rows]),
                "mean_denominator_n": statistics.mean(
                    row["denominator_n"] for row in rows
                ),
            }
        )

    delta_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in delta_rows:
        key = (
            row["dataset"],
            row["model"],
            row["condition1"],
            row["condition2"],
            row["metric"],
        )
        delta_groups[key].append(row)
    delta_aggregate = []
    for key, rows in sorted(delta_groups.items()):
        delta_aggregate.append(
            {
                "dataset": key[0],
                "model": key[1],
                "condition1": key[2],
                "condition2": key[3],
                "metric": key[4],
                "delta_definition": "condition2_minus_condition1",
                **_aggregate([row["delta"] for row in rows]),
                "mean_paired_n": statistics.mean(row["paired_n"] for row in rows),
            }
        )

    return {
        "manifest": str(manifest["path"]),
        "sampling": {
            "sample_size": sample_size,
            "repetitions": repetitions,
            "seed_start": seed_start,
            "within_repetition": "without replacement",
            "between_repetitions": "overlap allowed",
            "alignment": "one dataset-level item sample reused across every model and condition",
            "frame_sizes": {name: len(ids) for name, ids in frames.items()},
        },
        "samples": samples,
        "condition_repetitions": condition_rows,
        "delta_repetitions": delta_rows,
        "condition_aggregate": condition_aggregate,
        "delta_aggregate": delta_aggregate,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "repeated_subset_summary.json"
    condition_csv = output_dir / "repeated_subset_conditions.csv"
    delta_csv = output_dir / "repeated_subset_deltas.csv"
    md_path = output_dir / "repeated_subset_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(condition_csv, payload["condition_aggregate"])
    _write_csv(delta_csv, payload["delta_aggregate"])
    sampling = payload["sampling"]
    lines = [
        "# Repeated-subset robustness summary",
        "",
        f"Sample size: {sampling['sample_size']}",
        f"Repetitions: {sampling['repetitions']}",
        f"Seeds: {sampling['seed_start']} through "
        f"{sampling['seed_start'] + sampling['repetitions'] - 1}",
        "Sampling is without replacement within a repetition; repetitions may overlap.",
        "",
        "## Protocol deltas",
        "",
    ]
    for row in payload["delta_aggregate"]:
        lines.append(
            f"- {row['dataset']} / {row['model']} / {row['condition1']} -> "
            f"{row['condition2']} / {row['metric']}: mean delta "
            f"{row['mean'] * 100:+.2f} pp (SD {row['sd'] * 100:.2f}; "
            f"empirical 95% interval [{row['empirical_low'] * 100:+.2f}, "
            f"{row['empirical_high'] * 100:+.2f}])"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "json": json_path,
        "condition_csv": condition_csv,
        "delta_csv": delta_csv,
        "markdown": md_path,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    payload = analyze_repeated_subsets(
        manifest,
        sample_size=args.sample_size,
        repetitions=args.repetitions,
        seed_start=args.seed_start,
    )
    outputs = write_outputs(payload, args.output_dir)
    for label, path in outputs.items():
        print(f"{label.upper()}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
