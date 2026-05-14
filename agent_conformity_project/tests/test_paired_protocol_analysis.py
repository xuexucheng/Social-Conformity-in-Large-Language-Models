import csv
import json
import tempfile
import unittest
from pathlib import Path

from agent_conformity_project.analysis.paired_protocol_analysis import (
    apply_holm_correction,
    exact_mcnemar_pvalue,
    main,
    wilson_interval,
)


LEFT_FILE = "exp1_all_at_once_final.jsonl"
RIGHT_FILE = "exp2_sequential_context_final_only.jsonl"


def result_row(
    item_id,
    initial,
    final,
    gold="A",
    distractor="B",
    *,
    is_error=False,
):
    return {
        "id": item_id,
        "options": {"A": "gold", "B": "target", "C": "other"},
        "correct_answer": gold,
        "distractor": distractor,
        "initial_prediction": initial,
        "attack_prediction": final,
        "is_error": is_error,
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


class PairedProtocolAnalysisTests(unittest.TestCase):
    def test_exact_mcnemar_known_counts(self):
        self.assertAlmostEqual(exact_mcnemar_pvalue(1, 5), 0.21875)
        self.assertEqual(exact_mcnemar_pvalue(0, 0), 1.0)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(4, 10)
        self.assertLess(low, 0.4)
        self.assertGreater(high, 0.4)

    def test_holm_is_monotone_and_bounded(self):
        rows = [
            {"outcome": "x", "mcnemar_exact_p": 0.01, "holm_adjusted_p": None, "holm_family_n": None},
            {"outcome": "x", "mcnemar_exact_p": 0.04, "holm_adjusted_p": None, "holm_family_n": None},
            {"outcome": "x", "mcnemar_exact_p": 0.03, "holm_adjusted_p": None, "holm_family_n": None},
        ]
        apply_holm_correction(rows, "outcome")
        ordered = sorted(rows, key=lambda row: row["mcnemar_exact_p"])
        adjusted = [row["holm_adjusted_p"] for row in ordered]
        self.assertEqual(adjusted, [0.03, 0.06, 0.06])
        self.assertTrue(all(row["holm_family_n"] == 3 for row in rows))

    def test_end_to_end_common_valid_and_shared_baseline(self):
        left_rows = [
            result_row("1", "A", "A"),
            result_row("2", "A", "B"),
            result_row("3", "C", "B"),
            result_row("4", "B", "B"),
            result_row("5", "A", "A"),
            result_row("6", "A", "A"),
        ]
        right_rows = [
            result_row("1", "A", "B"),
            result_row("2", "A", "A"),
            result_row("3", "C", "A"),
            result_row("4", "B", "A"),
            result_row("5", "A", None, is_error=True),
            result_row("6", "C", "B"),
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
                    "7",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            with (output_dir / "pairwise_results.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                results = {row["outcome"]: row for row in csv.DictReader(handle)}

            self.assertEqual(int(results["final_accuracy"]["eligible_n"]), 5)
            self.assertEqual(int(results["final_accuracy"]["left_event_n"]), 2)
            self.assertEqual(int(results["final_accuracy"]["right_event_n"]), 3)

            # Item 5 is not common-valid; item 6 has mismatched initial answers.
            self.assertEqual(int(results["harmful_conformity"]["eligible_n"]), 2)
            self.assertEqual(int(results["harmful_conformity"]["left_event_n"]), 1)
            self.assertEqual(int(results["harmful_conformity"]["right_event_n"]), 1)

            # Item 4 started at the target and is excluded from conformity_rate.
            self.assertEqual(int(results["conformity_rate"]["eligible_n"]), 3)
            self.assertEqual(int(results["conformity_rate"]["left_event_n"]), 2)
            self.assertEqual(int(results["conformity_rate"]["right_event_n"]), 1)

            # Initially wrong items 3 and 4 are eligible for beneficial revision.
            self.assertEqual(int(results["beneficial_revision"]["eligible_n"]), 2)
            self.assertEqual(int(results["beneficial_revision"]["left_event_n"]), 0)
            self.assertEqual(int(results["beneficial_revision"]["right_event_n"]), 2)

            self.assertTrue((output_dir / "condition_quality.csv").exists())
            self.assertTrue((output_dir / "paired_item_audit.csv").exists())
            self.assertTrue((output_dir / "paired_protocol_report.md").exists())
            self.assertTrue((output_dir / "analysis_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
