#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"

CELLS = (
    ("qwen_csqa", "Qwen2.5-3B", "CommonsenseQA"),
    ("qwen_mmlu", "Qwen2.5-3B", "MMLU"),
    ("phi_csqa", "Phi-3.5-mini", "CommonsenseQA"),
    ("phi_mmlu", "Phi-3.5-mini", "MMLU"),
    ("gemma_csqa", "Gemma-2-2B", "CommonsenseQA"),
    ("gemma_mmlu", "Gemma-2-2B", "MMLU"),
)

CONDS = ("exp1", "exp2", "exp3", "exp4")
BOOTSTRAP_REPS = 10_000
BASE_SEED = 12345
LABELS = set("ABCDE")


def first(row, names):
    for name in names:
        if name in row:
            return row.get(name)
    return None


def label(x):
    if x is None:
        return None
    x = str(x).strip().upper()
    return x if x in LABELS else None


def normalize(row):
    item_id = first(row, ("id", "item_id"))
    initial = label(first(
        row,
        ("initial_prediction", "initial_answer", "private_prediction"),
    ))
    final = label(first(
        row,
        ("final_prediction", "attack_prediction",
         "final_answer", "prediction"),
    ))
    correct = label(first(
        row,
        ("correct_answer", "correct", "gold_answer"),
    ))
    distractor = label(first(
        row,
        ("distractor", "target_distractor", "target_wrong"),
    ))

    options = row.get("options")
    option_labels = (
        {str(k).strip().upper() for k in options}
        if isinstance(options, dict)
        else LABELS
    )

    explicit_valid = first(row, ("final_valid", "valid"))

    valid = (
        row.get("is_error") is not True
        and initial in option_labels
        and final in option_labels
        and correct in option_labels
        and distractor in option_labels
        and correct != distractor
        and explicit_valid is not False
    )

    return {
        "id": str(item_id),
        "initial": initial,
        "final": final,
        "correct": correct,
        "distractor": distractor,
        "question": row.get("question"),
        "options": options,
        "valid": bool(valid),
    }


def load_condition(path):
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = normalize(json.loads(line))
            if row["id"] in out:
                raise ValueError(
                    f"duplicate id {row['id']} in {path}:{line_no}"
                )
            out[row["id"]] = row
    return out


def load_scores(path):
    out = {}
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            item_id = str(row["id"])
            if item_id in out:
                raise ValueError(
                    f"duplicate score id {item_id} in {path}:{line_no}"
                )

            p = float(row["target_probability"])
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ValueError(
                    f"invalid target_probability id={item_id}: {p}"
                )

            labels = row["candidate_labels"]
            prob_sum = sum(float(row[f"prob_{x}"]) for x in labels)
            if abs(prob_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"probabilities do not sum to 1 id={item_id}: "
                    f"{prob_sum}"
                )

            out[item_id] = row

    if len(out) != 500:
        raise ValueError(
            f"{path}: expected 500 score rows, got {len(out)}"
        )
    return out


def percentile(values, p):
    values = sorted(values)
    pos = p * (len(values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    w = pos - lo
    return values[lo] * (1 - w) + values[hi] * w


def derived_seed(*parts):
    digest = hashlib.sha256(
        "\x1f".join(parts).encode("utf-8")
    ).digest()
    return BASE_SEED + int.from_bytes(digest[:4], "big")


def bootstrap_delta(ids, a, b, seed):
    # Paired CR difference: condition b - condition a.
    if not ids:
        return None, None, None

    diffs = [
        int(b[i]["final"] == b[i]["distractor"])
        - int(a[i]["final"] == a[i]["distractor"])
        for i in ids
    ]

    point = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    samples = []

    n = len(diffs)
    for _ in range(BOOTSTRAP_REPS):
        samples.append(
            sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        )

    samples.sort()
    lo = percentile(samples, 0.025)
    hi = percentile(samples, 0.975)
    return point, lo, hi


def rate(rows, ids, kind):
    if kind == "cr":
        eligible = [
            i for i in ids
            if rows[i]["initial"] != rows[i]["distractor"]
        ]
    elif kind == "hcr":
        eligible = [
            i for i in ids
            if rows[i]["initial"] == rows[i]["correct"]
        ]
    else:
        raise ValueError(kind)

    if not eligible:
        return 0, None

    hits = sum(
        rows[i]["final"] == rows[i]["distractor"]
        for i in eligible
    )
    return len(eligible), hits / len(eligible)


def fmt_pct(x):
    return "NA" if x is None else f"{100*x:.2f}"


def main():
    outdir = ANALYSIS / "forced_option_terciles"
    outdir.mkdir(parents=True, exist_ok=True)

    threshold_rows = []
    metric_rows = []
    contrast_rows = []
    md = [
        "# P3 Forced-option distractor plausibility stratification",
        "",
        "Terciles are defined within each model×dataset cell "
        "on the four-protocol common-valid item set using the "
        "pre-social target-option probability.",
        "",
        "Low: p <= q33; Medium: q33 < p <= q67; High: p > q67.",
        "",
        f"Paired bootstrap: {BOOTSTRAP_REPS:,} resamples, "
        f"base seed {BASE_SEED}.",
        "",
    ]

    for cell, model, dataset in CELLS:
        conds = {
            c: load_condition(
                ANALYSIS / "recovered_main500"
                / cell
                / {"exp1": "exp1_all_at_once_final.jsonl", "exp2": "exp2_sequential_context_final_only.jsonl", "exp3": "exp3_sequential_answer_each_step.jsonl", "exp4": "exp4_all_at_once_self_iter5.jsonl"}[c]
            )
            for c in CONDS
        }

        # Strict four-protocol common-valid set.
        common = set.intersection(
            *(set(conds[c]) for c in CONDS)
        )
        common = {
            i for i in common
            if all(conds[c][i]["valid"] for c in CONDS)
        }

        # Fail closed on metadata drift.
        audit_fields = (
            "question", "options", "correct",
            "distractor", "initial",
        )
        for item_id in sorted(common):
            ref = conds["exp1"][item_id]
            for c in ("exp2", "exp3", "exp4"):
                changed = [
                    f for f in audit_fields
                    if conds[c][item_id][f] != ref[f]
                ]
                if changed:
                    raise ValueError(
                        f"{cell}/{item_id}: metadata drift "
                        f"exp1 vs {c}: {changed}"
                    )

        scores = load_scores(
            ANALYSIS
            / "forced_option_plausibility"
            / f"{cell}.jsonl"
        )

        missing = common - set(scores)
        if missing:
            raise ValueError(
                f"{cell}: {len(missing)} common-valid ids "
                "missing scores"
            )

        # Score metadata must agree with the recovered experiment.
        for item_id in common:
            s = scores[item_id]
            r = conds["exp1"][item_id]

            if label(s["correct_answer"]) != r["correct"]:
                raise ValueError(
                    f"{cell}/{item_id}: score correct mismatch"
                )
            if label(s["target_distractor"]) != r["distractor"]:
                raise ValueError(
                    f"{cell}/{item_id}: score distractor mismatch"
                )

        probs = [
            float(scores[i]["target_probability"])
            for i in common
        ]
        q33 = percentile(probs, 1 / 3)
        q67 = percentile(probs, 2 / 3)

        strata = {"Low": [], "Medium": [], "High": []}

        for item_id in sorted(common):
            p = float(scores[item_id]["target_probability"])
            if p <= q33:
                strata["Low"].append(item_id)
            elif p <= q67:
                strata["Medium"].append(item_id)
            else:
                strata["High"].append(item_id)

        threshold_rows.append({
            "cell": cell,
            "model": model,
            "dataset": dataset,
            "common_valid_n": len(common),
            "q33": q33,
            "q67": q67,
            "low_n": len(strata["Low"]),
            "medium_n": len(strata["Medium"]),
            "high_n": len(strata["High"]),
        })

        md += [
            f"## {model} — {dataset}",
            "",
            f"Common-valid N = {len(common)}; "
            f"q33 = {q33:.8g}; q67 = {q67:.8g}.",
            "",
            "| Stratum | N | Exp1 CR/HCR | Exp2 CR/HCR | "
            "Exp3 CR/HCR | Exp4 CR/HCR |",
            "|---|---:|---:|---:|---:|---:|",
        ]

        table_cache = {}

        for stratum in ("Low", "Medium", "High"):
            ids = strata[stratum]
            ps = sorted(
                float(scores[i]["target_probability"])
                for i in ids
            )

            row_cache = {}

            for c in CONDS:
                cr_n, cr = rate(conds[c], ids, "cr")
                hcr_n, hcr = rate(conds[c], ids, "hcr")

                metric_rows.append({
                    "cell": cell,
                    "model": model,
                    "dataset": dataset,
                    "stratum": stratum,
                    "stratum_n": len(ids),
                    "target_probability_min": min(ps),
                    "target_probability_max": max(ps),
                    "condition": c,
                    "cr_n": cr_n,
                    "cr": cr,
                    "hcr_n": hcr_n,
                    "hcr": hcr,
                })

                row_cache[c] = (cr, hcr)

            table_cache[stratum] = row_cache

            md.append(
                f"| {stratum} | {len(ids)} | "
                f"{fmt_pct(row_cache['exp1'][0])}/"
                f"{fmt_pct(row_cache['exp1'][1])} | "
                f"{fmt_pct(row_cache['exp2'][0])}/"
                f"{fmt_pct(row_cache['exp2'][1])} | "
                f"{fmt_pct(row_cache['exp3'][0])}/"
                f"{fmt_pct(row_cache['exp3'][1])} | "
                f"{fmt_pct(row_cache['exp4'][0])}/"
                f"{fmt_pct(row_cache['exp4'][1])} |"
            )

        md += [
            "",
            "CR/HCR values above are percentages.",
            "",
            "| Stratum | Contrast | N(CR) | ΔCR (pp) | 95% CI (pp) |",
            "|---|---|---:|---:|---:|",
        ]

        for stratum in ("Low", "Medium", "High"):
            ids = strata[stratum]

            cr_ids = [
                i for i in ids
                if conds["exp1"][i]["initial"]
                != conds["exp1"][i]["distractor"]
            ]

            for a, b in (
                ("exp1", "exp2"),
                ("exp2", "exp3"),
            ):
                point, lo, hi = bootstrap_delta(
                    cr_ids,
                    conds[a],
                    conds[b],
                    derived_seed(cell, stratum, a, b),
                )

                contrast_rows.append({
                    "cell": cell,
                    "model": model,
                    "dataset": dataset,
                    "stratum": stratum,
                    "condition1": a,
                    "condition2": b,
                    "cr_n": len(cr_ids),
                    "delta_cr": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "bootstrap_reps": BOOTSTRAP_REPS,
                })

                md.append(
                    f"| {stratum} | {a}→{b} | {len(cr_ids)} | "
                    f"{fmt_pct(point)} | "
                    f"[{fmt_pct(lo)}, {fmt_pct(hi)}] |"
                )

        md.append("")

        print(
            f"{cell}: common_valid={len(common)} "
            f"q33={q33:.8g} q67={q67:.8g} "
            f"N=({len(strata['Low'])},"
            f"{len(strata['Medium'])},"
            f"{len(strata['High'])})"
        )

    def write_csv(name, rows):
        path = outdir / name
        with path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)

    write_csv("thresholds.csv", threshold_rows)
    write_csv("condition_metrics.csv", metric_rows)
    write_csv("contrasts.csv", contrast_rows)

    (outdir / "summary.md").write_text(
        "\n".join(md).rstrip() + "\n",
        encoding="utf-8",
    )

    print()
    print("P3_TERCILE_ANALYSIS_OK")
    print("output:", outdir)


if __name__ == "__main__":
    main()
