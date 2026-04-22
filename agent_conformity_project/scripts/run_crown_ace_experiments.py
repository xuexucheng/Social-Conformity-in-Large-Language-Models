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
from src.selective_gate import SelectiveGate


EXPERIMENT_DIR = "crown_ace"


def safe_div(a, b):
    return a / b if b else 0.0


def infer_peer_truth(row):
    if row["peer_pattern"] == "unanimous_right":
        return "right"
    return "wrong"


def default_ablation_config():
    return {"name": "FULL"}


def ablation_specs():
    return [
        {"name": "FULL"},
        {"name": "NO_PEER_CONSTRAINT", "use_peer_constraint": False},
        {"name": "NO_OPTIONWISE_VERIFIER", "use_optionwise_verifier": False},
        {"name": "NO_CORRECTION_AUDIT", "use_correction_audit": False},
        {"name": "NO_LOGIC_ECHO", "use_logic_echo": False},
        {"name": "NO_LEARNED_GATE", "use_learned_gate": False, "gate_mode": "rule_based"},
        {"name": "UNCERTAINTY_ONLY_GATE", "use_learned_gate": False, "gate_mode": "uncertainty_only"},
    ]


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
        "ablation_name": result.get("ablation_config", {}).get("name", "NA"),
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
        "beneficial_revision_score": result.get("correction_audit", {}).get("beneficial_revision_score"),
        "harmful_conformity_risk": result.get("conformity_audit", {}).get("harmful_conformity_risk"),
        "candidate_evidence_strength": result.get("correction_audit", {}).get("candidate_evidence_strength"),
        "baseline_error_score": result.get("optionwise_verification", {}).get("baseline_error_score"),
        "best_option_margin": result.get("optionwise_verification", {}).get("best_option_margin"),
        "peer_consistency_score": result.get("evidence_state", {}).get("peer_consistency_score"),
        "evidence_diversity_score": result.get("evidence_state", {}).get("evidence_diversity_score"),
        "evidence_independence_score": result.get("evidence_state", {}).get("evidence_independence_score"),
        "gate_accept_prob": result.get("gate_result", {}).get("accept_revision_probability"),
        "gate_reject_prob": result.get("gate_result", {}).get("reject_revision_probability"),
        "gate_strict_prob": result.get("gate_result", {}).get("need_strict_verification_probability"),
        "decision_type": result.get("decision_type", ""),
        "gate_training_examples": result.get("gate_result", {}).get("training_examples"),
        "evidence": result.get("evidence_result", {}).get("evidence", []),
        "evidence_steps": result.get("evidence_result", {}).get("steps", []),
        "structured_peers": result.get("structured_peers", []),
        "decision_trace": result.get("decision_trace", {}),
        "raw_payload": {
            "baseline_raw": baseline["raw_text"],
            "method_raw": result.get("raw_text", ""),
            "proposed_final_raw": result.get("proposed_final", {}).get("raw_text", ""),
            "audit_raw": result.get("audit", {}).get("raw_text", ""),
            "logic_echo_judge_raw": result.get("audit", {}).get("judge_raw_text", ""),
            "correction_audit_raw": result.get("correction_audit", {}).get("raw_text", ""),
            "conformity_audit_raw": result.get("conformity_audit", {}).get("raw_text", ""),
            "strict_verifier_raw": result.get("strict_verifier", {}).get("raw_text", "") if result.get("strict_verifier") else "",
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
        "avg_beneficial_revision_score": safe_div(
            sum(row["beneficial_revision_score"] for row in rows if row["beneficial_revision_score"] is not None),
            sum(1 for row in rows if row["beneficial_revision_score"] is not None),
        ),
        "avg_harmful_conformity_risk": safe_div(
            sum(row["harmful_conformity_risk"] for row in rows if row["harmful_conformity_risk"] is not None),
            sum(1 for row in rows if row["harmful_conformity_risk"] is not None),
        ),
        "avg_gate_accept_prob": safe_div(
            sum(row["gate_accept_prob"] for row in rows if row["gate_accept_prob"] is not None),
            sum(1 for row in rows if row["gate_accept_prob"] is not None),
        ),
        "avg_evidence_independence_score": safe_div(
            sum(row["evidence_independence_score"] for row in rows if row["evidence_independence_score"] is not None),
            sum(1 for row in rows if row["evidence_independence_score"] is not None),
        ),
    }


def compute_selective_metrics(summary_rows):
    rows = []
    methods = sorted({row["method"] for row in summary_rows})
    for method in methods:
        wrong_rows = [row for row in summary_rows if row["method"] == method and row["peer_truth"] == "wrong"]
        right_rows = [row for row in summary_rows if row["method"] == method and row["peer_truth"] == "right"]
        avg_wcr_wrong = safe_div(sum(row["WCR"] for row in wrong_rows), len(wrong_rows))
        avg_cr_right = safe_div(sum(row["CR"] for row in right_rows), len(right_rows))
        avg_acc_wrong = safe_div(sum(row["accuracy"] for row in wrong_rows), len(wrong_rows))
        avg_acc_right = safe_div(sum(row["accuracy"] for row in right_rows), len(right_rows))
        benefit_harm_tradeoff = avg_cr_right - avg_wcr_wrong
        rows.append(
            {
                "method": method,
                "avg_WCR_wrong": round(avg_wcr_wrong, 6),
                "avg_CR_right": round(avg_cr_right, 6),
                "avg_accuracy_wrong": round(avg_acc_wrong, 6),
                "avg_accuracy_right": round(avg_acc_right, 6),
                "benefit_harm_tradeoff": round(benefit_harm_tradeoff, 6),
                "harm_axis": round(avg_wcr_wrong, 6),
                "benefit_axis": round(avg_cr_right, 6),
            }
        )

    nd = next((row for row in rows if row["method"] == "ND"), None)
    pd = next((row for row in rows if row["method"] == "PD"), None)
    for row in rows:
        if nd:
            row["selective_gain_vs_ND"] = round(
                (nd["avg_WCR_wrong"] - row["avg_WCR_wrong"]) + (row["avg_CR_right"] - nd["avg_CR_right"]),
                6,
            )
        if pd:
            row["selective_gain_vs_PD"] = round(
                (pd["avg_WCR_wrong"] - row["avg_WCR_wrong"]) + (row["avg_CR_right"] - pd["avg_CR_right"]),
                6,
            )
    return rows


def build_benefit_harm_rows(summary_rows):
    rows = []
    for row in summary_rows:
        rows.append(
            {
                "experiment_name": row["experiment_name"],
                "method": row["method"],
                "peer_pattern": row["peer_pattern"],
                "peer_truth": row["peer_truth"],
                "harm_axis_WCR": row["WCR"],
                "benefit_axis_CR": row["CR"],
                "accuracy": row["accuracy"],
                "benefit_harm_tradeoff": round(row["CR"] - row["WCR"], 6),
            }
        )
    return rows


def build_gate_export_rows(rows):
    export_rows = []
    for row in rows:
        if row["method"] != "CROWN_ACE_SEQUENTIAL":
            continue
        export_rows.append(
            {
                "experiment_name": row["experiment_name"],
                "item_id": row["item_id"],
                "peer_pattern": row["peer_pattern"],
                "peer_truth": row["peer_truth"],
                "ablation_name": row["ablation_name"],
                "initial_answer": row["initial_answer"],
                "final_answer": row["final_answer"],
                "correct_answer": row["correct_answer"],
                "changed": row["changed"],
                "corrected": row["corrected"],
                "wrong_conformity": row["wrong_conformity"],
                "initial_probability": row["initial_probability"],
                "initial_entropy": row["initial_entropy"],
                "delta_p": row["delta_p"],
                "delta_h": row["delta_h"],
                "reasoning_score": row["reasoning_score"],
                "counterevidence_score": row["counterevidence_score"],
                "logic_echo_strength": row["logic_echo_strength"],
                "beneficial_revision_score": row["beneficial_revision_score"],
                "harmful_conformity_risk": row["harmful_conformity_risk"],
                "candidate_evidence_strength": row["candidate_evidence_strength"],
                "baseline_error_score": row["baseline_error_score"],
                "best_option_margin": row["best_option_margin"],
                "peer_consistency_score": row["peer_consistency_score"],
                "evidence_diversity_score": row["evidence_diversity_score"],
                "evidence_independence_score": row["evidence_independence_score"],
                "gate_accept_prob": row["gate_accept_prob"],
                "gate_reject_prob": row["gate_reject_prob"],
                "gate_strict_prob": row["gate_strict_prob"],
                "decision_type": row["decision_type"],
            }
        )
    return export_rows


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


def run_method(item, baseline, opinions, method_name, exposure_mode=None, gate=None, peer_pattern=None, ablation_config=None):
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
        return apply_crown_ace(
            item,
            baseline,
            opinions,
            exposure_mode=exposure_mode or "sequential",
            gate=gate,
            peer_pattern=peer_pattern or "unknown",
            ablation_config=ablation_config,
        )
    if method_name == "ROLLBACK":
        seed = apply_crown_ace(
            item,
            baseline,
            opinions,
            exposure_mode=exposure_mode or "sequential",
            gate=gate,
            peer_pattern=peer_pattern or "unknown",
            ablation_config=ablation_config,
        )
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
            "description": "Sequential selective defense under wrong-peer pressure across unanimous, diverse, and devil's-advocate structures.",
            "eligibility": "initial_correct",
            "cases": [
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "unanimous_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "diverse_wrong"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "devils_advocate_wrong"},
            ],
        },
        {
            "name": "experiment_2_sequential_correction_test",
            "description": "Sequential selective correction when peers are unanimously correct.",
            "eligibility": "initial_wrong",
            "cases": [
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "unanimous_right"},
            ],
        },
        {
            "name": "experiment_3_algorithm_only_correction_test",
            "description": "Sequential prompt defense vs sequential selective CROWN-Ace vs rollback baseline on unanimously correct peer advice.",
            "eligibility": "initial_wrong",
            "cases": [
                {"method": "PD", "exposure_mode": "sequential", "peer_pattern": "unanimous_right"},
                {"method": "CROWN_ACE", "exposure_mode": "sequential", "peer_pattern": "unanimous_right"},
                {"method": "ROLLBACK", "exposure_mode": "sequential", "peer_pattern": "unanimous_right"},
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
        {
            "name": "experiment_5_ablation_study",
            "description": "Ablation study for sequential selective defense on matched wrong-peer and right-peer settings.",
            "eligibility": "all",
            "cases": [
                *[
                    {
                        "method": "CROWN_ACE",
                        "exposure_mode": "sequential",
                        "peer_pattern": pattern,
                        "ablation_config": spec,
                    }
                    for pattern in ["unanimous_wrong", "diverse_wrong", "devils_advocate_wrong", "unanimous_right"]
                    for spec in ablation_specs()
                ]
            ],
        },
    ]


def summarize_experiment(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["experiment_name"], row["method"], row.get("ablation_name", "NA"), row["peer_pattern"])].append(row)

    summary_rows = []
    for (experiment_name, method, ablation_name, peer_pattern), group_rows in sorted(grouped.items()):
        metrics = compute_group_metrics(group_rows)
        peer_group = peer_pattern.rsplit("_", 1)[0]
        peer_truth = group_rows[0]["peer_truth"] if group_rows else ""
        summary_rows.append(
            {
                "experiment_name": experiment_name,
                "method": method,
                "ablation_name": ablation_name,
                "peer_pattern": peer_pattern,
                "peer_group": peer_group,
                "peer_truth": peer_truth,
                **metrics,
            }
        )

    add_dnr(summary_rows)

    indexed = {
        (row["experiment_name"], row["method"], row.get("ablation_name", "NA"), row["peer_group"], row["peer_truth"]): row
        for row in summary_rows
    }

    rollback = indexed.get(("experiment_3_algorithm_only_correction_test", "ROLLBACK", "NA", "unanimous", "right"))
    ace = indexed.get(("experiment_3_algorithm_only_correction_test", "CROWN_ACE_SEQUENTIAL", "FULL", "unanimous", "right"))
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
        if case.get("exposure_mode") == "sequential":
            return "CROWN_ACE_SEQUENTIAL"
    return case["method"]


def write_paper_artifacts(experiment_root, overall_rows, overall_summary):
    paper_dir = os.path.join(experiment_root, "paper_artifacts")
    os.makedirs(paper_dir, exist_ok=True)

    selective_rows = compute_selective_metrics(
        [row for row in overall_summary if row.get("ablation_name") in {"NA", "FULL"}]
    )
    tradeoff_rows = build_benefit_harm_rows(
        [row for row in overall_summary if row.get("ablation_name") in {"NA", "FULL"}]
    )
    ablation_rows = [row for row in overall_summary if row.get("experiment_name") == "experiment_5_ablation_study"]
    gate_rows = build_gate_export_rows(overall_rows)

    write_json(os.path.join(paper_dir, "selective_gain_summary.json"), selective_rows)
    write_csv(os.path.join(paper_dir, "selective_gain_summary.csv"), selective_rows)
    write_json(os.path.join(paper_dir, "benefit_harm_tradeoff.json"), tradeoff_rows)
    write_csv(os.path.join(paper_dir, "benefit_harm_tradeoff.csv"), tradeoff_rows)
    write_json(os.path.join(paper_dir, "ablation_summary.json"), ablation_rows)
    write_csv(os.path.join(paper_dir, "ablation_summary.csv"), ablation_rows)
    write_jsonl(os.path.join(paper_dir, "gate_examples.jsonl"), gate_rows)
    write_csv(os.path.join(paper_dir, "gate_examples.csv"), gate_rows)


def run_experiments():
    dataset = load_dataset()
    experiment_root = os.path.join(RESULT_DIR, EXPERIMENT_DIR)
    run_root = os.path.join(RUN_DIR, EXPERIMENT_DIR)
    os.makedirs(experiment_root, exist_ok=True)
    os.makedirs(run_root, exist_ok=True)
    gate = SelectiveGate()

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
                        gate=gate,
                        peer_pattern=case.get("peer_pattern"),
                        ablation_config=case.get("ablation_config"),
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
                    if (
                        case["method"] == "CROWN_ACE"
                        and result.get("gate_features")
                        and result.get("ablation_config", {}).get("name", "FULL") == "FULL"
                    ):
                        candidate_answer = result.get("proposed_final", {}).get("answer")
                        label = int(
                            candidate_answer is not None
                            and candidate_answer != baseline["answer"]
                            and candidate_answer == item["correct_answer"]
                            and baseline["answer"] != item["correct_answer"]
                        )
                        gate.update(result["gate_features"], label)
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
                "gate_training_examples": gate.training_examples,
                "ablation_variants": sorted({row.get("ablation_name", "NA") for row in experiment_rows}),
            },
        )

    overall_summary = summarize_experiment(all_rows)
    write_json(os.path.join(experiment_root, "overall_summary.json"), overall_summary)
    write_csv(os.path.join(experiment_root, "overall_summary.csv"), overall_summary)
    write_paper_artifacts(experiment_root, all_rows, overall_summary)
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
