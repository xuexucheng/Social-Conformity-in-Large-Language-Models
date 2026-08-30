#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from paired_protocol_analysis import load_condition

HERE = Path(__file__).resolve().parent
BASE = HERE / "recovered_main500"
OUT = HERE / "four_protocol_common_valid"

CONDITIONS = {
    "exp1": "exp1_all_at_once_final.jsonl",
    "exp2": "exp2_sequential_context_final_only.jsonl",
    "exp3": "exp3_sequential_answer_each_step.jsonl",
    "exp4": "exp4_all_at_once_self_iter5.jsonl",
}

CELLS = {
    "qwen_csqa": ("CommonsenseQA", "Qwen/Qwen2.5-3B-Instruct"),
    "qwen_mmlu": ("MMLU", "Qwen/Qwen2.5-3B-Instruct"),
    "phi_csqa": ("CommonsenseQA", "microsoft/Phi-3.5-mini-instruct"),
    "phi_mmlu": ("MMLU", "microsoft/Phi-3.5-mini-instruct"),
    "gemma_csqa": ("CommonsenseQA", "google/gemma-2-2b-it"),
    "gemma_mmlu": ("MMLU", "google/gemma-2-2b-it"),
}

AUDIT_FIELDS = ("question", "options", "correct", "distractor", "initial")


def rate(num: int, den: int):
    return num / den if den else None


def fmt(x):
    return "NA" if x is None else f"{x * 100:.2f}%"


def read_raw_jsonl(path: Path):
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row["id"])
            if item_id in rows:
                raise ValueError(f"duplicate id {item_id} in {path}:{line_no}")
            rows[item_id] = row
    return rows


def audit_metadata(cell_name, loaded):
    id_sets = [set(x) for x in loaded.values()]
    common_attempted = set.intersection(*id_sets)

    mismatches = []
    for item_id in sorted(common_attempted):
        ref = loaded["exp1"][item_id]
        for cond in ("exp2", "exp3", "exp4"):
            row = loaded[cond][item_id]
            changed = [
                field for field in AUDIT_FIELDS
                if row.get(field) != ref.get(field)
            ]
            if changed:
                mismatches.append({
                    "id": item_id,
                    "condition": cond,
                    "fields": changed,
                })

    if mismatches:
        first = mismatches[0]
        raise ValueError(
            f"{cell_name}: metadata/initial drift in {len(mismatches)} comparisons; "
            f"first id={first['id']} condition={first['condition']} "
            f"fields={first['fields']}"
        )

    return common_attempted


def summarize(records, allowed_ids):
    rows = [records[i] for i in sorted(allowed_ids)]

    n = len(rows)
    initial_correct_n = sum(r["initial"] == r["correct"] for r in rows)
    final_correct_n = sum(r["final"] == r["correct"] for r in rows)

    target_adopt_n = sum(r["final"] == r["distractor"] for r in rows)
    change_n = sum(r["final"] != r["initial"] for r in rows)

    cr_eligible = [r for r in rows if r["initial"] != r["distractor"]]
    cr_n = sum(r["final"] == r["distractor"] for r in cr_eligible)

    hcr_eligible = [r for r in rows if r["initial"] == r["correct"]]
    hcr_n = sum(r["final"] == r["distractor"] for r in hcr_eligible)

    br_eligible = [r for r in rows if r["initial"] != r["correct"]]
    br_n = sum(r["final"] == r["correct"] for r in br_eligible)

    return {
        "N": n,
        "initial_accuracy": rate(initial_correct_n, n),
        "final_accuracy": rate(final_correct_n, n),
        "target_adoption": rate(target_adopt_n, n),
        "change_rate": rate(change_n, n),
        "CR_n": len(cr_eligible),
        "CR": rate(cr_n, len(cr_eligible)),
        "HCR_n": len(hcr_eligible),
        "HCR": rate(hcr_n, len(hcr_eligible)),
        "beneficial_revision_n": len(br_eligible),
        "beneficial_revision": rate(br_n, len(br_eligible)),
    }


def audit_exp4_step5(cell_name, raw_records):
    mismatch_ids = []
    malformed = []

    for item_id, row in sorted(raw_records.items()):
        steps = row.get("attack_step_outputs")

        if not isinstance(steps, list) or len(steps) != 5:
            malformed.append({
                "id": item_id,
                "reason": f"attack_step_outputs length={None if not isinstance(steps, list) else len(steps)}",
            })
            continue

        last = steps[-1]
        if not isinstance(last, dict):
            malformed.append({"id": item_id, "reason": "step5 not dict"})
            continue

        if last.get("iteration") != 5:
            malformed.append({
                "id": item_id,
                "reason": f"step5 iteration={last.get('iteration')!r}",
            })
            continue

        step5 = last.get("prediction")
        final = row.get("attack_prediction")

        if step5 != final:
            mismatch_ids.append({
                "id": item_id,
                "step5_prediction": step5,
                "final_prediction": final,
            })

    return {
        "cell": cell_name,
        "attempted_n": len(raw_records),
        "malformed_n": len(malformed),
        "malformed": malformed,
        "mismatch_n": len(mismatch_ids),
        "mismatches": mismatch_ids,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    result_rows = []
    cell_audits = []
    step5_audits = []

    for cell_name, (dataset, model) in CELLS.items():
        cell_dir = BASE / cell_name

        loaded = {
            cond: load_condition(cell_dir / filename)
            for cond, filename in CONDITIONS.items()
        }

        common_attempted = audit_metadata(cell_name, loaded)

        common_valid = {
            item_id
            for item_id in common_attempted
            if all(loaded[c][item_id]["valid"] for c in CONDITIONS)
        }

        if not common_valid:
            raise ValueError(f"{cell_name}: zero four-protocol common-valid items")

        # Because initial prediction is audited identical, initial accuracy must
        # also be identical across all four conditions on this same subset.
        initial_values = []

        for cond in CONDITIONS:
            summary = summarize(loaded[cond], common_valid)
            initial_values.append(summary["initial_accuracy"])

            result_rows.append({
                "cell": cell_name,
                "dataset": dataset,
                "model": model,
                "condition": cond,
                **summary,
            })

        if len(set(initial_values)) != 1:
            raise AssertionError(
                f"{cell_name}: initial accuracy differs across conditions: {initial_values}"
            )

        cell_audits.append({
            "cell": cell_name,
            "dataset": dataset,
            "model": model,
            "attempted_n_by_condition": {
                c: len(loaded[c]) for c in CONDITIONS
            },
            "common_attempted_n": len(common_attempted),
            "common_valid_n": len(common_valid),
            "initial_accuracy": initial_values[0],
            "metadata_initial_drift_n": 0,
        })

        raw_exp4 = read_raw_jsonl(cell_dir / CONDITIONS["exp4"])
        step5_audits.append(audit_exp4_step5(cell_name, raw_exp4))

    payload = {
        "definition": (
            "For each model-dataset cell, S is the intersection of items valid "
            "in Exp1, Exp2, Exp3, and Exp4. All four protocol metrics are "
            "recomputed on exactly S."
        ),
        "CR_definition": "P(final=target distractor | initial!=target distractor)",
        "HCR_definition": "P(final=target distractor | initial=correct)",
        "cells": cell_audits,
        "rows": result_rows,
    }

    (OUT / "four_protocol_common_valid_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fields = list(result_rows[0].keys())
    with (OUT / "four_protocol_common_valid_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(result_rows)

    (OUT / "exp4_step5_final_audit.json").write_text(
        json.dumps(step5_audits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Four-protocol common-valid analysis",
        "",
        "All four protocol metrics within a model-dataset cell use the exact same",
        "four-way common-valid subset S.",
        "",
    ]

    for audit in cell_audits:
        lines += [
            f"## {audit['cell']}",
            "",
            f"- Dataset: {audit['dataset']}",
            f"- Model: {audit['model']}",
            f"- Common attempted N: {audit['common_attempted_n']}",
            f"- Four-protocol common-valid N: {audit['common_valid_n']}",
            f"- Shared initial accuracy: {fmt(audit['initial_accuracy'])}",
            "",
            "| Protocol | N | Initial Acc | Final Acc | CR | HCR | Change | Target adoption | Beneficial revision |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for row in [r for r in result_rows if r["cell"] == audit["cell"]]:
            lines.append(
                f"| {row['condition']} | {row['N']} | "
                f"{fmt(row['initial_accuracy'])} | {fmt(row['final_accuracy'])} | "
                f"{fmt(row['CR'])} | {fmt(row['HCR'])} | "
                f"{fmt(row['change_rate'])} | {fmt(row['target_adoption'])} | "
                f"{fmt(row['beneficial_revision'])} |"
            )

        s5 = next(x for x in step5_audits if x["cell"] == audit["cell"])
        lines += [
            "",
            f"- Exp4 malformed Step-5 rows: {s5['malformed_n']}",
            f"- Exp4 Step5/final mismatches: {s5['mismatch_n']}",
            "",
        ]

    (OUT / "four_protocol_common_valid_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("OUTPUT_DIR =", OUT)
    print()
    for audit in cell_audits:
        print(
            f"{audit['cell']:12s} "
            f"common_valid={audit['common_valid_n']:3d} "
            f"initial_acc={audit['initial_accuracy']*100:6.2f}%"
        )

    print("\n===== EXP4 STEP5 AUDIT =====")
    for x in step5_audits:
        print(
            f"{x['cell']:12s} malformed={x['malformed_n']:3d} "
            f"mismatch={x['mismatch_n']:3d}"
        )


if __name__ == "__main__":
    main()
