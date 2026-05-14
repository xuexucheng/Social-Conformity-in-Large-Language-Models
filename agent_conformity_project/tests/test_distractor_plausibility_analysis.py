import csv
import json
import tempfile
import unittest
from pathlib import Path

from agent_conformity_project.analysis.distractor_plausibility_analysis import (
    assign_rank_quartiles,
    extract_initial_logprobs,
    main,
)


LEFT_FILE = "exp1_all_at_once_final.jsonl"
RIGHT_FILE = "exp2_sequential_context_final_only.jsonl"


def result_row(item_id, margin, final, *, missing_target=False):
    logprobs = {"A": -5.0, "B": None if missing_target else -5.0 + margin, "C": -8.0}
    return {
        "id": str(item_id),
        "options": {"A": "gold", "B": "target", "C": "other"},
        "correct_answer": "A",
        "distractor": "B",
        "initial_prediction": "A",
        "attack_prediction": final,
        "initial_option_logprobs": logprobs,
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class DistractorPlausibilityAnalysisTests(unittest.TestCase):
    def test_extract_margin(self):
        row = result_row("1", -2.5, "A")
        gold, target, margin = extract_initial_logprobs(row, "A", "B")
        self.assertEqual(gold, -5.0)
        self.assertEqual(target, -7.5)
        self.assertEqual(margin, -2.5)

    def test_missing_target_is_not_imputed(self):
        row = result_row("1", -2.5, "A", missing_target=True)
        gold, target, margin = extract_initial_logprobs(row, "A", "B")
        self.assertEqual(gold, -5.0)
        self.assertIsNone(target)
        self.assertIsNone(margin)

    def test_rank_quartiles_are_balanced(self):
        rows = [
            {"item_id": str(index), "margin": float(index), "quartile": None, "quartile_label": None}
            for index in range(16)
        ]
        assign_rank_quartiles(rows)
        counts = {quartile: 0 for quartile in range(1, 5)}
        for row in rows:
            counts[row["quartile"]] += 1
        self.assertEqual(counts, {1: 4, 2: 4, 3: 4, 4: 4})

    def test_end_to_end_outputs_quartiles_and_adjusted_regression(self):
        left_rows = []
        right_rows = []
        for index in range(20):
            margin = -4.0 + index * 0.25
            # Non-perfect patterns avoid deterministic separation in the test regression.
            left_final = "B" if index in {3, 7, 11, 14, 16, 18} else "A"
            right_final = "B" if index in {5, 9, 12, 15, 17, 19} else "A"
            left_rows.append(result_row(index, margin, left_final))
            right_rows.append(result_row(index, margin, right_final))
        left_rows.append(result_row("missing", -1.0, "A", missing_target=True))
        right_rows.append(result_row("missing", -1.0, "A", missing_target=True))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            write_jsonl(input_dir / LEFT_FILE, left_rows)
            write_jsonl(input_dir / RIGHT_FILE, right_rows)

            exit_code = main(
                [
                    "--input-dir",
                    str(input_dir),
                    "--pair",
                    "Batch-Single:Sequential-Final",
                    "--reference-condition",
                    "Batch-Single",
                    "--bootstrap-reps",
                    "200",
                    "--seed",
                    "9",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            with (output_dir / "logprob_coverage.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                coverage = list(csv.DictReader(handle))
            self.assertEqual(len(coverage), 2)
            self.assertEqual(int(coverage[0]["baseline_items"]), 21)
            self.assertEqual(int(coverage[0]["baseline_margin_usable"]), 20)
            self.assertEqual(int(coverage[0]["target_logprob_missing"]), 1)

            with (output_dir / "quartile_condition_rates.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                quartiles = list(csv.DictReader(handle))
            observed_quartiles = {
                int(row["quartile"])
                for row in quartiles
                if row["outcome"] == "harmful_conformity"
            }
            self.assertEqual(observed_quartiles, {1, 2, 3, 4})

            with (output_dir / "plausibility_adjusted_regression.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                regression = list(csv.DictReader(handle))
            self.assertTrue(
                any(
                    row["model"] == "adjusted_main"
                    and row["outcome"] == "harmful_conformity"
                    and row["term"] == "condition[Sequential-Final]"
                    for row in regression
                )
            )

            expected_files = {
                "logprob_coverage.csv",
                "baseline_margin_audit.csv",
                "plausibility_item_audit.csv",
                "quartile_condition_rates.csv",
                "paired_quartile_effects.csv",
                "plausibility_adjusted_regression.csv",
                "regression_diagnostics.csv",
                "distractor_plausibility_report.md",
                "analysis_metadata.json",
            }
            self.assertTrue(all((output_dir / name).exists() for name in expected_files))


if __name__ == "__main__":
    unittest.main()
