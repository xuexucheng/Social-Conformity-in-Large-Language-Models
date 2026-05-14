import csv
import json
import tempfile
import unittest
from pathlib import Path

from agent_conformity_project.analysis.repeated_subset_robustness import main


LEFT_FILE = "exp1_all_at_once_final.jsonl"
RIGHT_FILE = "exp2_sequential_context_final_only.jsonl"


def result_row(item_id, final):
    return {
        "id": str(item_id),
        "options": {"A": "gold", "B": "target", "C": "other"},
        "correct_answer": "A",
        "distractor": "B",
        "initial_prediction": "A",
        "attack_prediction": final,
    }


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def build_batch(directory, prefix, offset):
    left_rows = []
    right_rows = []
    for index in range(12):
        item_id = f"{prefix}_{index}"
        left_final = "B" if (index + offset) % 4 == 0 else "A"
        right_final = "B" if (index + offset) % 3 == 0 else "A"
        left_rows.append(result_row(item_id, left_final))
        right_rows.append(result_row(item_id, right_final))
    write_jsonl(directory / LEFT_FILE, left_rows)
    write_jsonl(directory / RIGHT_FILE, right_rows)


class RepeatedSubsetRobustnessTests(unittest.TestCase):
    def test_two_batches_repeated_sampling_and_membership_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main_dir = root / "main"
            new_dir = root / "new"
            output_dir = root / "output"
            main_dir.mkdir()
            new_dir.mkdir()
            build_batch(main_dir, "main", 0)
            build_batch(new_dir, "new", 1)

            exit_code = main(
                [
                    "--batch",
                    f"synthetic::main500={main_dir}",
                    "--batch",
                    f"synthetic::new500={new_dir}",
                    "--pair",
                    "Batch-Single:Sequential-Final",
                    "--repetitions",
                    "5",
                    "--sample-size",
                    "10",
                    "--bootstrap-reps",
                    "200",
                    "--seed",
                    "13",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            with (output_dir / "independent_batch_pairwise_results.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                independent = list(csv.DictReader(handle))
            self.assertEqual(len(independent), 2 * 6)
            self.assertEqual({row["batch"] for row in independent}, {"main500", "new500"})

            with (output_dir / "pooled_pair_quality.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                pool_quality = list(csv.DictReader(handle))
            self.assertEqual(len(pool_quality), 1)
            self.assertEqual(int(pool_quality[0]["eligible_pool_n"]), 24)

            with (output_dir / "repeated_subset_runs.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                runs = list(csv.DictReader(handle))
            self.assertEqual(len(runs), 5 * 6)
            self.assertEqual({int(row["sample_n"]) for row in runs}, {10})

            with (output_dir / "repetition_item_membership.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                membership = list(csv.DictReader(handle))
            self.assertEqual(len(membership), 5 * 10)
            for repetition in range(1, 6):
                ids = {
                    row["item_id"]
                    for row in membership
                    if int(row["repetition"]) == repetition
                }
                self.assertEqual(len(ids), 10)

            with (output_dir / "repeated_subset_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                summaries = list(csv.DictReader(handle))
            self.assertEqual(len(summaries), 6)
            hcr = next(row for row in summaries if row["outcome"] == "harmful_conformity")
            self.assertEqual(int(hcr["repetitions"]), 5)
            self.assertEqual(int(hcr["full_pool_n"]), 24)

            with (output_dir / "repetition_overlap_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                overlap = list(csv.DictReader(handle))
            self.assertEqual(len(overlap), 1)
            self.assertEqual(int(overlap[0]["repetition_pairs"]), 10)

            expected_files = {
                "batch_condition_quality.csv",
                "independent_batch_pairwise_results.csv",
                "pooled_pair_quality.csv",
                "pooled_item_audit.csv",
                "full_pool_results.csv",
                "repeated_subset_runs.csv",
                "repetition_item_membership.csv",
                "repetition_batch_composition.csv",
                "repeated_subset_summary.csv",
                "repetition_overlap_summary.csv",
                "repeated_subset_report.md",
                "analysis_metadata.json",
            }
            self.assertTrue(all((output_dir / name).exists() for name in expected_files))


if __name__ == "__main__":
    unittest.main()
