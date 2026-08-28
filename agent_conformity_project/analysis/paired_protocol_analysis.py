#!/usr/bin/env python3
"""Paper-facing paired analysis for the four main conformity protocols.

The script is deliberately inference-free.  It consumes the archived per-item
JSONL files, fails closed on duplicate identifiers or paired metadata drift,
and evaluates prespecified protocol contrasts on common-valid item sets.

Default contrasts are Exp1 -> Exp2 and Exp2 -> Exp3.  For every model-dataset
cell and outcome it reports both condition rates, condition-2-minus-condition-1
percentage-point differences, a question-level paired bootstrap interval,
exact McNemar counts/p-values, and Holm-adjusted p-values.  Holm correction is
performed separately within each outcome across every prespecified cell and
contrast in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from exact_mcnemar import exact_mcnemar_pvalue  # noqa: E402


OPTION_LABELS = frozenset("ABCDE")
DEFAULT_COMPARISONS = (("exp1", "exp2"), ("exp2", "exp3"))


def _first_present(row: dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row:
            return row.get(name)
    return None


def _label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in OPTION_LABELS else None


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy main-run and revision-run row schemas."""
    item_id = _first_present(row, ("id", "item_id"))
    initial = _label(
        _first_present(
            row,
            ("initial_prediction", "initial_answer", "private_prediction"),
        )
    )
    final = _label(
        _first_present(
            row,
            (
                "final_prediction",
                "attack_prediction",
                "final_answer",
                "prediction",
            ),
        )
    )
    correct = _label(
        _first_present(row, ("correct_answer", "correct", "gold_answer"))
    )
    distractor = _label(
        _first_present(row, ("distractor", "target_distractor", "target_wrong"))
    )
    options = row.get("options")
    option_labels = {
        str(key).strip().upper()
        for key in options
    } if isinstance(options, dict) else OPTION_LABELS
    explicit_valid = _first_present(row, ("final_valid", "valid"))
    structurally_valid = (
        row.get("is_error") is not True
        and initial in option_labels
        and final in option_labels
        and correct in option_labels
        and distractor in option_labels
        and correct != distractor
    )
    valid = structurally_valid and explicit_valid is not False
    return {
        "id": None if item_id is None else str(item_id),
        "initial": initial,
        "final": final,
        "correct": correct,
        "distractor": distractor,
        "valid": bool(valid),
        "question": row.get("question"),
        "options": options,
        "source": row.get("source"),
        "subject": row.get("subject"),
        "raw": row,
    }


def load_condition(path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(path)
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            record = normalize_row(payload)
            item_id = record["id"]
            if item_id is None:
                raise ValueError(f"missing item id at {path}:{line_number}")
            if item_id in records:
                raise ValueError(f"duplicate item id {item_id!r} at {path}:{line_number}")
            records[item_id] = record
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def _resolve_path(base: Path, value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("manifest must contain a non-empty 'cells' list")
    comparisons = manifest.get("comparisons", [list(pair) for pair in DEFAULT_COMPARISONS])
    parsed_comparisons: list[tuple[str, str]] = []
    for pair in comparisons:
        if not isinstance(pair, list) or len(pair) != 2 or not all(pair):
            raise ValueError(f"invalid comparison specification: {pair!r}")
        parsed_comparisons.append((str(pair[0]), str(pair[1])))

    normalized_cells = []
    seen_keys = set()
    for index, cell in enumerate(cells, start=1):
        dataset = str(cell.get("dataset") or "").strip()
        model = str(cell.get("model") or "").strip()
        conditions = cell.get("conditions")
        if not dataset or not model or not isinstance(conditions, dict):
            raise ValueError(
                f"cell {index} requires dataset, model, and a conditions mapping"
            )
        key = (dataset, model)
        if key in seen_keys:
            raise ValueError(f"duplicate manifest cell: dataset={dataset!r}, model={model!r}")
        seen_keys.add(key)
        resolved = {
            str(name): _resolve_path(path.parent, str(condition_path))
            for name, condition_path in conditions.items()
        }
        required = {name for pair in parsed_comparisons for name in pair}
        missing = sorted(required - set(resolved))
        if missing:
            raise ValueError(f"cell {dataset}/{model} is missing conditions: {missing}")
        for name, condition_path in resolved.items():
            if not condition_path.is_file():
                raise FileNotFoundError(
                    f"condition file not found for {dataset}/{model}/{name}: {condition_path}"
                )
        normalized_cells.append(
            {"dataset": dataset, "model": model, "conditions": resolved}
        )
    return {
        "path": path,
        "comparisons": parsed_comparisons,
        "cells": normalized_cells,
    }


def _always(_record: dict[str, Any]) -> bool:
    return True


def _initial_not_target(record: dict[str, Any]) -> bool:
    return record["initial"] != record["distractor"]


def _initial_correct(record: dict[str, Any]) -> bool:
    return record["initial"] == record["correct"]


def _initial_incorrect(record: dict[str, Any]) -> bool:
    return record["initial"] != record["correct"]


METRICS: tuple[
    tuple[str, Callable[[dict[str, Any]], bool], Callable[[dict[str, Any]], bool]],
    ...,
] = (
    ("accuracy", lambda row: row["final"] == row["correct"], _always),
    ("target_adoption", lambda row: row["final"] == row["distractor"], _always),
    ("change_rate", lambda row: row["final"] != row["initial"], _always),
    ("conformity_rate", lambda row: row["final"] == row["distractor"], _initial_not_target),
    ("harmful_conformity", lambda row: row["final"] == row["distractor"], _initial_correct),
    ("beneficial_revision", lambda row: row["final"] == row["correct"], _initial_incorrect),
)


METRIC_LOOKUP = {name: (outcome, eligibility) for name, outcome, eligibility in METRICS}


def condition_summary(
    records: dict[str, dict[str, Any]],
    allowed_ids: set[str] | None = None,
) -> dict[str, Any]:
    selected = [
        row
        for item_id, row in records.items()
        if allowed_ids is None or item_id in allowed_ids
    ]
    valid = [row for row in selected if row["valid"]]
    summary: dict[str, Any] = {
        "attempted_n": len(selected),
        "valid_n": len(valid),
        "invalid_n": len(selected) - len(valid),
    }
    for metric_name, outcome, eligibility in METRICS:
        eligible = [row for row in valid if eligibility(row)]
        summary[f"{metric_name}_n"] = len(eligible)
        summary[metric_name] = (
            sum(1 for row in eligible if outcome(row)) / len(eligible)
            if eligible
            else None
        )
    return summary


AUDIT_FIELDS = ("question", "options", "correct", "distractor", "initial")


def audit_pair(
    name1: str,
    records1: dict[str, dict[str, Any]],
    name2: str,
    records2: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ids1, ids2 = set(records1), set(records2)
    common = sorted(ids1 & ids2)
    mismatches = []
    for item_id in common:
        changed = [
            field
            for field in AUDIT_FIELDS
            if records1[item_id].get(field) != records2[item_id].get(field)
        ]
        if changed:
            mismatches.append({"id": item_id, "fields": changed})
    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"paired metadata drift between {name1} and {name2}: "
            f"{len(mismatches)} item(s); first={first['id']!r}, fields={first['fields']}"
        )
    return {
        "condition1_n": len(ids1),
        "condition2_n": len(ids2),
        "common_id_n": len(common),
        "condition1_only_n": len(ids1 - ids2),
        "condition2_only_n": len(ids2 - ids1),
    }


def paired_ids(
    records1: dict[str, dict[str, Any]],
    records2: dict[str, dict[str, Any]],
    eligibility: Callable[[dict[str, Any]], bool],
    allowed_ids: set[str] | None = None,
) -> list[str]:
    ids = set(records1) & set(records2)
    if allowed_ids is not None:
        ids &= allowed_ids
    return sorted(
        item_id
        for item_id in ids
        if records1[item_id]["valid"]
        and records2[item_id]["valid"]
        and eligibility(records1[item_id])
        and eligibility(records2[item_id])
    )


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _derived_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def paired_bootstrap_ci(
    ids: list[str],
    records1: dict[str, dict[str, Any]],
    records2: dict[str, dict[str, Any]],
    outcome: Callable[[dict[str, Any]], bool],
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    if not ids:
        raise ValueError("paired bootstrap requires at least one item")
    values1 = [1.0 if outcome(records1[item_id]) else 0.0 for item_id in ids]
    values2 = [1.0 if outcome(records2[item_id]) else 0.0 for item_id in ids]
    n = len(ids)
    point = (sum(values2) - sum(values1)) / n
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        difference = 0.0
        for _index in range(n):
            selected = rng.randrange(n)
            difference += values2[selected] - values1[selected]
        samples.append(difference / n)
    samples.sort()
    return point, _percentile(samples, 0.025), _percentile(samples, 0.975)


def exact_mcnemar(
    ids: list[str],
    records1: dict[str, dict[str, Any]],
    records2: dict[str, dict[str, Any]],
    outcome: Callable[[dict[str, Any]], bool],
) -> tuple[int, int, float]:
    # b: condition 1 outcome=1, condition 2 outcome=0; c: reverse.
    b = sum(
        outcome(records1[item_id]) and not outcome(records2[item_id])
        for item_id in ids
    )
    c = sum(
        not outcome(records1[item_id]) and outcome(records2[item_id])
        for item_id in ids
    )
    return int(b), int(c), exact_mcnemar_pvalue(int(b), int(c))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def analyze_manifest(
    manifest: dict[str, Any],
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    condition_summaries: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        loaded = {
            name: load_condition(path)
            for name, path in cell["conditions"].items()
        }
        for name, records in loaded.items():
            condition_summaries.append(
                {
                    "dataset": cell["dataset"],
                    "model": cell["model"],
                    "condition": name,
                    "path": str(cell["conditions"][name]),
                    **condition_summary(records),
                }
            )
        for condition1, condition2 in manifest["comparisons"]:
            records1, records2 = loaded[condition1], loaded[condition2]
            audits.append(
                {
                    "dataset": cell["dataset"],
                    "model": cell["model"],
                    "condition1": condition1,
                    "condition2": condition2,
                    **audit_pair(condition1, records1, condition2, records2),
                }
            )
            for metric_name, outcome, eligibility in METRICS:
                ids = paired_ids(records1, records2, eligibility)
                if not ids:
                    comparisons.append(
                        {
                            "dataset": cell["dataset"],
                            "model": cell["model"],
                            "condition1": condition1,
                            "condition2": condition2,
                            "delta_definition": "condition2_minus_condition1",
                            "metric": metric_name,
                            "paired_n": 0,
                            "rate1": None,
                            "rate2": None,
                            "delta": None,
                            "ci_low": None,
                            "ci_high": None,
                            "mcnemar_b": None,
                            "mcnemar_c": None,
                            "p_raw": None,
                            "p_holm": None,
                        }
                    )
                    continue
                rate1 = sum(outcome(records1[item_id]) for item_id in ids) / len(ids)
                rate2 = sum(outcome(records2[item_id]) for item_id in ids) / len(ids)
                derived_seed = _derived_seed(
                    bootstrap_seed,
                    cell["dataset"],
                    cell["model"],
                    condition1,
                    condition2,
                    metric_name,
                )
                delta, ci_low, ci_high = paired_bootstrap_ci(
                    ids,
                    records1,
                    records2,
                    outcome,
                    bootstrap_reps,
                    derived_seed,
                )
                b, c, p_raw = exact_mcnemar(ids, records1, records2, outcome)
                comparisons.append(
                    {
                        "dataset": cell["dataset"],
                        "model": cell["model"],
                        "condition1": condition1,
                        "condition2": condition2,
                        "delta_definition": "condition2_minus_condition1",
                        "metric": metric_name,
                        "paired_n": len(ids),
                        "rate1": rate1,
                        "rate2": rate2,
                        "delta": delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "mcnemar_b": b,
                        "mcnemar_c": c,
                        "p_raw": p_raw,
                        "p_holm": None,
                    }
                )

    for metric_name, _, _ in METRICS:
        family = [
            row
            for row in comparisons
            if row["metric"] == metric_name and row["p_raw"] is not None
        ]
        adjusted = holm_adjust([row["p_raw"] for row in family])
        for row, p_value in zip(family, adjusted):
            row["p_holm"] = p_value

    return {
        "manifest": str(manifest["path"]),
        "bootstrap_reps": bootstrap_reps,
        "bootstrap_seed": bootstrap_seed,
        "delta_definition": "condition2_minus_condition1",
        "holm_family": (
            "all prespecified model-dataset cells and protocol contrasts, "
            "corrected separately within each outcome"
        ),
        "condition_summaries": condition_summaries,
        "pair_audits": audits,
        "comparisons": comparisons,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_rate(value: float | None) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _fmt_number(value: float | int | None, spec: str = ".6g") -> str:
    return "NA" if value is None else format(value, spec)


def write_outputs(payload: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "main_paired_summary.json"
    csv_path = output_dir / "main_paired_comparisons.csv"
    md_path = output_dir / "main_paired_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, payload["comparisons"])
    lines = [
        "# Main protocol paired analysis",
        "",
        f"Bootstrap repetitions: {payload['bootstrap_reps']}",
        f"Bootstrap seed: {payload['bootstrap_seed']}",
        f"Multiplicity family: {payload['holm_family']}",
        "",
        "All deltas below are condition 2 minus condition 1.",
        "",
    ]
    for row in payload["comparisons"]:
        prefix = (
            f"- {row['dataset']} / {row['model']} / "
            f"{row['condition1']} -> {row['condition2']} / {row['metric']}: "
            f"N={row['paired_n']}, {_fmt_rate(row['rate1'])} -> {_fmt_rate(row['rate2'])}"
        )
        if row["delta"] is None:
            lines.append(prefix + ", no eligible common-valid items")
            continue
        lines.append(
            prefix
            + f", delta={row['delta'] * 100:+.2f} pp, "
            + f"95% CI [{row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f}], "
            + f"McNemar b={row['mcnemar_b']}, c={row['mcnemar_c']}, "
            + f"p={_fmt_number(row['p_raw'])}, Holm p={_fmt_number(row['p_holm'])}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON manifest of model-dataset cells")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args(argv)
    if args.bootstrap_reps < 100:
        parser.error("--bootstrap-reps must be at least 100")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    payload = analyze_manifest(manifest, args.bootstrap_reps, args.seed)
    outputs = write_outputs(payload, args.output_dir)
    for label, path in outputs.items():
        print(f"{label.upper()}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
