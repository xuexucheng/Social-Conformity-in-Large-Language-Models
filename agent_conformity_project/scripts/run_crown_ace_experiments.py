import json
import os
import shutil
import sys
import traceback
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import RESULT_DIR, RUN_DIR
from src.crown_ace import (
    apply_crown_ace,
    apply_no_defense,
    apply_prompt_defense,
    apply_rollback_baseline,
    build_peer_answers,
    build_peer_group,
    run_private_baseline,
)
from src.dataset_loader import load_dataset
from src.experiment_io import write_csv, write_json, write_jsonl


EXPERIMENT_DIR = "crown_ace"


def safe_div(a, b):
    return a / b if b else 0.0


def infer_peer_truth(row):
    if row["peer_pattern"] == "unanimous_right":
        return "right"
    return "wrong"


def finalize_row(experiment_name, item, baseline, opinions, peer_pattern, method_name, result):
    final_answer = result["final_answer"]
    initial_answer = baseline["answer"]
    correct_answer = item["correct_answer"]
    peer_answers = [opinion["answer"] for opinion in opinions]

    accept = bool(result.get("accepted"))
    changed = final_answer != initial_answer
    initial_correct = initial_answer == correct_answer
    final_correct = final_answer == correct_answer
    peer_truth = infer_peer_truth({"peer_pattern": peer_pattern})

    row = {
        "experiment_name": experiment_name,
        "item_id": item["id"],
        "question": item["question"],
        "correct_answer": correct_answer,
        "distractor": item["distractor"],
        "method": method_name,
        "peer_pattern": peer_pattern,
        "peer_truth": peer_truth,
        "peer_answers": peer_answers,
        "initial_answer": initial_answer,
        "initial_confidence": baseline["confidence"],
        "initial_probability": baseline["probability"],
        "initial_entropy": round(baseline["entropy"], 6),
        "initial_reasoning": baseline["reasoning"],
        "final_answer": final_answer,
        "final_confidence": result["final_confidence"],
        "final_probability": result.get("final_probability", result.get("final_confidence", 0)),
        "final_entropy": round(result["final_entropy"], 6),
        "final_reasoning": result["final_reasoning"],
        "accept": accept,
        "changed": changed,
        "initial_correct": initial_correct,
        "final_correct": final_correct,
        "wrong_conformity": initial_correct and final_answer in peer_answers and not final_correct,
        "corrected": (not initial_correct) and final_correct,
        "collaborative_correction": (not initial_correct) and peer_truth == "right" and final_correct,
        "resilient_self_correction": (not initial_correct) and peer_truth == "wrong" and final_correct,
        "delta_p": result.get("delta_confidence", 0),
        "delta_h": round(result.get("delta_entropy", 0.0), 6),
        "reasoning_score": result.get("audit", {}).get("reasoning_score", result.get("reasoning_score")),
        "counterevidence_score": result.get("audit", {}).get(
            "counterevidence_score",
            result.get("counterevidence_score"),
        ),
        "logic_echo_strength": (
            result.get("audit", {}).get("logic_echo_strength")
            if result.get("audit")
            else None
        ),
        "audit_decision": result.get("audit", {}).get("decision", "ACCEPT" if accept else "REJECT"),
        "forward_audit": result.get("audit", {}).get("forward_audit", ""),
        "logic_echo": result.get("audit", {}).get("logic_echo", result.get("devil_advocate", "")),
        "evidence": result.get("evidence_result", {}).get("evidence", []),
        "evidence_steps": result.get("evidence_result", {}).get("steps", []),
        "raw_payload": {
            "baseline_raw": baseline["raw_text"],
            "method_raw": result.get("raw_text", ""),
            "proposed_final_raw": result.get("proposed_final", {}).get("raw_text", ""),
            "audit_raw": result.get("audit", {}).get("raw_text", ""),
            "logic_echo_judge_raw": result.get("audit", {}).get("judge_raw_text", ""),
        },
    }
    return row


def compute_group_metrics(rows):
    total = len(rows)
    accepted_changed = [row for row in rows if row["accept"] and row["changed"]]
    wcr_den = sum(1 for row in rows if row["initial_correct"])
    cr_den = sum(1 for row in rows if not row["initial_correct"])
    peer_right_den = sum(1 for row in rows if (not row["initial_correct"]) and row["peer_truth"] == "right")
    peer_wrong_den = sum(1 for row in rows if (not row["initial_correct"]) and row["peer_truth"] == "wrong")

    return {
        "total": total,
        "accuracy": safe_div(sum(1 for row in rows if row["final_correct"]), total),
        "WCR": safe_div(sum(1 for row in rows if row["wrong_conformity"]), wcr_den),
        "CR": safe_div(sum(1 for row in rows if row["corrected"]), cr_den),
        "SAR": safe_div(sum(1 for row in accepted_changed if row["final_correct"]), len(accepted_changed)),
        "CR_coll": safe_div(sum(1 for row in rows if row["collaborative_correction"]), peer_right_den),
        "CR_res": safe_div(sum(1 for row in rows if row["resilient_self_correction"]), peer_wrong_den),
        "avg_delta_p": safe_div(sum(row["delta_p"] for row in rows), total),
        "avg_delta_h": safe_div(sum(row["delta_h"] for row in rows), total),
        "avg_reasoning_score": safe_div(
            sum(row["reasoning_score"] for row in rows if row["reasoning_score"] is not None),
            sum(1 for row in rows if row["reasoning_score"] is not None),
        ),
        "avg_logic_echo_strength": safe_div(
            sum(row["logic_echo_strength"] for row in rows if row["logic_echo_strength"] is not None),
            sum(1 for row in rows if row["logic_echo_strength"] is not None),
        ),
        "accept_rate": safe_div(sum(1 for row in rows if row["accept"]), total),
    }


def add_dnr(summary_rows):
    grouped = defaultdict(dict)
    for row in summary_rows:
        key = (row["experiment_name"], row["peer_pattern"])
        grouped[key][row["method"]] = row

    for variants in grouped.values():
        nd = variants.get("ND")
        if not nd:
            continue
        nd_wcr = nd["WCR"]
        for row in variants.values():
            row["DNR"] = 1.0 - safe_div(row["WCR"], nd_wcr) if nd_wcr else 0.0


def run_method(item, baseline, opinions, method_name, exposure_mode=None):
    if method_name == "ND":
        return apply_no_defense(item, baseline, opinions)
    if method_name == "PD":
        pd_result = apply_prompt_defense(item, baseline, opinions)
        return {
            "final_answer": pd_result["answer"],
            "final_confidence": pd_result["confidence"],
            "final_reasoning": pd_result["reasoning"],
            "final_entropy": pd_result["entropy"],
            "initial_probability": baseline["probability"],
            "final_probability": pd_result["probability"],
            "accepted": pd_result["accepted"],
            "delta_confidence": pd_result["probability"] - baseline["probability"],
            "delta_entropy": baseline["entropy"] - pd_result["entropy"],
            "devil_advocate": pd_result["devil_advocate"],
            "raw_text": pd_result["raw_text"],
        }
    if method_name == "CROWN_ACE":
        return apply_crown_ace(item, baseline, opinions, exposure_mode=exposure_mode or "sequential")
    if method_name == "ROLLBACK":
        seed = apply_crown_ace(item, baseline, opinions, exposure_mode=exposure_mode or "single")
        return apply_rollback_baseline(
            baseline,
            candidate_answer=seed["proposed_final"]["answer"],
            candidate_confidence=seed["proposed_final"]["confidence"],
        )
    raise ValueError(f"Unsupported method: {method_name}")


def experiment_specs():
    return [
        {
            "name": "experiment_1_sequential_pressure_test",
            "description": "Single vs sequential exposure under wrong-peer pressure across unanimous, diverse, and devil's-advocate structures.",
            "eligibility": "initial_correct",
            "cases": [
                {"method": "CROWN_ACE", "exposure_mode": "single", "peer_pattern": "unanimous_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "unanimous_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "single", "peer_pattern": "diverse_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "diverse_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "single", "peer_pattern": "devils_advocate_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "devils_advocate_wrong"},
            ],
        },
        {
            "name": "experiment_2_sequential_correction_test",
            "description": "Single vs sequential exposure when peers are unanimously correct.",
            "eligibility": "initial_wrong",
            "cases": [
                {"method": "CROWN_ACE", "exposure_mode": "single", "peer_pattern": "unanimous_right"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "unanimous_right"},
            ],
        },
        {
            "name": "experiment_3_algorithm_only_correction_test",
            "description": "Single-input prompt defense vs single-input CROWN-Ace algorithm vs rollback baseline on unanimously correct peer advice.",
            "eligibility": "initial_wrong",
            "cases": [
                {"method": "PD", "exposure_mode": "single", "peer_pattern": "unanimous_right"},
                {"method": "CROWN_ACE", "exposure_mode": "single", "peer_pattern": "unanimous_right"},
                {"method": "ROLLBACK", "exposure_mode": "single", "peer_pattern": "unanimous_right"},
            ],
        },
        {
            "name": "experiment_4_full_system_comparison",
            "description": "ND vs PD vs full CROWN-Ace across Zhu-style wrong-signal environments.",
            "eligibility": "all",
            "cases": [
                {"method": "ND", "peer_pattern": "unanimous_wrong"},
                {"method": "PD", "peer_pattern": "unanimous_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "unanimous_wrong"},
                {"method": "ND", "peer_pattern": "diverse_wrong"},
                {"method": "PD", "peer_pattern": "diverse_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "diverse_wrong"},
                {"method": "ND", "peer_pattern": "devils_advocate_wrong"},
                {"method": "PD", "peer_pattern": "devils_advocate_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "devils_advocate_wrong"},
            ],
        },
    ]


def summarize_experiment(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["experiment_name"], row["method"], row["peer_pattern"])].append(row)

    summary_rows = []
    for (experiment_name, method, peer_pattern), group_rows in sorted(grouped.items()):
        metrics = compute_group_metrics(group_rows)
        peer_group = peer_pattern.rsplit("_", 1)[0]
        peer_truth = group_rows[0]["peer_truth"] if group_rows else ""
        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "method": method,
                "peer_pattern": peer_pattern,
                "peer_group": peer_group,
                "peer_truth": peer_truth,
                **metrics,
            }
        )

    add_dnr(summary_rows)

    indexed = {
        (row["experiment_name"], row["method"], row["peer_group"], row["peer_truth"]): row
        for row in summary_rows
    }
    for peer_group in ["unanimous", "diverse", "devils_advocate"]:
        single = indexed.get(("experiment_1_sequential_pressure_test", "CROWN_ACE_SINGLE", peer_group, "wrong"))
        sequential = indexed.get(("experiment_1_sequential_pressure_test", "CROWN_ACE_SEQUENTIAL", peer_group, "wrong"))
        if single and sequential:
            sequential["ES"] = single["WCR"] - sequential["WCR"]

    single = indexed.get(("experiment_2_sequential_correction_test", "CROWN_ACE_SINGLE", "unanimous", "right"))
    sequential = indexed.get(("experiment_2_sequential_correction_test", "CROWN_ACE_SEQUENTIAL", "unanimous", "right"))
    if single and sequential:
        sequential["delta_CR"] = sequential["CR"] - single["CR"]

    rollback = indexed.get(("experiment_3_algorithm_only_correction_test", "ROLLBACK", "unanimous", "right"))
    ace = indexed.get(("experiment_3_algorithm_only_correction_test", "CROWN_ACE_SINGLE", "unanimous", "right"))
    if rollback and ace:
        ace["delta_CR_vs_rollback"] = ace["CR"] - rollback["CR"]

    return summary_rows


def is_eligible(spec, baseline, item):
    initial_correct = baseline["answer"] == item["correct_answer"]
    if spec.get("eligibility") == "initial_correct":
        return initial_correct
    if spec.get("eligibility") == "initial_wrong":
        return not initial_correct
    return True


def split_case_name(case):
    if case["method"] == "CROWN_ACE":
        if case.get("exposure_mode") == "single":
            return "CROWN_ACE_SINGLE"
        if case.get("exposure_mode") == "sequential":
            return "CROWN_ACE_SEQUENTIAL"
    return case["method"]


def run_experiments():
    dataset = load_dataset()
    experiment_root = os.path.join(RESULT_DIR, EXPERIMENT_DIR)
    run_root = os.path.join(RUN_DIR, EXPERIMENT_DIR)
    os.makedirs(experiment_root, exist_ok=True)
    os.makedirs(run_root, exist_ok=True)

    all_rows = []
    manifest = {"status": "completed", "errors": []}

    for spec in experiment_specs():
        experiment_rows = []
        for item in dataset:
            try:
                baseline = run_private_baseline(item)
                if not is_eligible(spec, baseline, item):
                    continue
                for case in spec["cases"]:
                    opinions = build_peer_group(
                        item,
                        build_peer_answers(item, case["peer_pattern"]),
                        pattern=case["peer_pattern"],
                    )
                    result = run_method(
                        item,
                        baseline,
                        opinions,
                        method_name=case["method"],
                        exposure_mode=case.get("exposure_mode"),
                    )
                    row = finalize_row(
                        spec["name"],
                        item,
                        baseline,
                        opinions,
                        case["peer_pattern"],
                        split_case_name(case),
                        result,
                    )
                    experiment_rows.append(row)
                    all_rows.append(row)
            except Exception as exc:
                manifest["status"] = "partial_failure"
                manifest["errors"].append(
                    {
                        "experiment_name": spec["name"],
                        "item_id": item.get("id"),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        summary_rows = summarize_experiment(experiment_rows)
        exp_dir = os.path.join(experiment_root, spec["name"])
        os.makedirs(exp_dir, exist_ok=True)
        write_jsonl(os.path.join(exp_dir, "rows.jsonl"), experiment_rows)
        write_csv(
            os.path.join(exp_dir, "rows.csv"),
            [
                {
                    k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
                for row in experiment_rows
            ],
        )
        write_json(os.path.join(exp_dir, "summary.json"), summary_rows)
        write_csv(os.path.join(exp_dir, "summary.csv"), summary_rows)
        write_json(
            os.path.join(exp_dir, "metadata.json"),
            {
                "experiment_name": spec["name"],
                "description": spec["description"],
                "n_rows": len(experiment_rows),
            },
        )

    overall_summary = summarize_experiment(all_rows)
    write_json(os.path.join(experiment_root, "overall_summary.json"), overall_summary)
    write_csv(os.path.join(experiment_root, "overall_summary.csv"), overall_summary)
    write_json(os.path.join(experiment_root, "run_manifest.json"), manifest)

    shutil.copytree(experiment_root, run_root, dirs_exist_ok=True)
    return experiment_root, run_root, manifest


def write_blocked_manifest(message):
    experiment_root = os.path.join(RESULT_DIR, EXPERIMENT_DIR)
    run_root = os.path.join(RUN_DIR, EXPERIMENT_DIR)
    os.makedirs(experiment_root, exist_ok=True)
    os.makedirs(run_root, exist_ok=True)
    manifest = {"status": "blocked", "message": message}
    write_json(os.path.join(experiment_root, "run_manifest.json"), manifest)
    shutil.copytree(experiment_root, run_root, dirs_exist_ok=True)
    return experiment_root, run_root, manifest


def main():
    try:
        experiment_root, run_root, manifest = run_experiments()
    except FileNotFoundError as exc:
        experiment_root, run_root, manifest = write_blocked_manifest(str(exc))

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[OK] CROWN-Ace artifacts saved to {experiment_root}")
    print(f"[OK] CROWN-Ace artifacts archived to {run_root}")


if __name__ == "__main__":
    main()
