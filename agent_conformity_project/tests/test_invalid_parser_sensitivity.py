import csv
import json
import tempfile
import unittest
from pathlib import Path

from agent_conformity_project.analysis.invalid_parser_sensitivity import main


LEFT_FILE = "exp1_all_at_once_final.jsonl"
RIGHT_FILE = "exp2_sequential_context_final_only.jsonl"


def result_row(item_id, stored_final, raw_final):
    return {
        "id": str(item_id),
        "options": {"A": "gold", "B": "target", "C": "other"},
        "correct_answer": "A",
        "distractor": "B",
        "initial_prediction": "A",
        "attack_prediction": stored_final,
        "raw_initial_output": "ANSWER: A",
        "raw_attack_output": raw_final,
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class InvalidParserSensitivityTests(unittest.TestCase):
    def test_end_to_end_audit_reparse_and_bounds(self):
        left_rows = [
            result_row("1", "A", "ANSWER: A"),
            result_row("2", None, "The answer is B."),
            result_row("3", "A", "ANSWER: B"),
            result_row("4", None, ""),
            result_row("5", "B", "ANSWER: A\nANSWER: B"),
        ]
        right_rows = [
            result_row("1", "A", "ANSWER: A"),
            result_row("2", "B", "ANSWER: B"),
            result_row("3", "A", "ANSWER: A"),
            result_row("4", "A", "ANSWER: A"),
            result_row("5", "A", "ANSWER: A"),
        ]

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
                    "--bootstrap-reps",
                    "200",
                    "--seed",
                    "11",
                    "--sample-per-category",
                    "10",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            with (output_dir / "parser_condition_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                summaries = list(csv.DictReader(handle))
            left_final = next(
                row
                for row in summaries
                if row["condition"] == "Batch-Single" and row["phase"] == "final"
            )
            self.assertEqual(int(left_final["stored_invalid"]), 2)
            self.assertEqual(int(left_final["recoverable_stored_invalid"]), 1)
            self.assertEqual(int(left_final["stored_reparsed_disagree"]), 1)
            self.assertEqual(int(left_final["ambiguous_explicit_candidates"]), 1)

            with (output_dir / "stored_vs_reparsed_pairwise_results.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                paired = list(csv.DictReader(handle))
            stored_hcr = next(
                row
                for row in paired
                if row["parse_policy"] == "stored"
                and row["outcome"] == "harmful_conformity"
            )
            reparsed_hcr = next(
                row
                for row in paired
                if row["parse_policy"] == "current_parser_when_raw_available"
                and row["outcome"] == "harmful_conformity"
            )
            self.assertEqual(int(stored_hcr["eligible_n"]), 3)
            self.assertEqual(int(reparsed_hcr["eligible_n"]), 4)

            with (output_dir / "invalid_missing_data_bounds.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                bounds = list(csv.DictReader(handle))
            hcr_bounds = next(row for row in bounds if row["outcome"] == "harmful_conformity")
            self.assertEqual(int(hcr_bounds["eligible_including_invalid_n"]), 5)
            self.assertEqual(int(hcr_bounds["complete_case_n"]), 3)
            self.assertEqual(int(hcr_bounds["left_missing_n"]), 2)
            self.assertEqual(int(hcr_bounds["right_missing_n"]), 0)
            self.assertLessEqual(
                float(hcr_bounds["worst_case_delta_lower"]),
                float(hcr_bounds["worst_case_delta_upper"]),
            )

            with (output_dir / "parser_review_sample.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                review = list(csv.DictReader(handle))
            self.assertTrue(any(row["item_id"] == "2" for row in review))
            self.assertTrue(any(row["item_id"] == "5" for row in review))

            expected_files = {
                "parser_condition_summary.csv",
                "parser_method_counts.csv",
                "parser_item_audit.csv",
                "parser_review_sample.csv",
                "stored_vs_reparsed_pairwise_results.csv",
                "invalid_missing_data_bounds.csv",
                "invalid_parser_sensitivity_report.md",
                "analysis_metadata.json",
            }
            self.assertTrue(all((output_dir / name).exists() for name in expected_files))


if __name__ == "__main__":
    unittest.main()
