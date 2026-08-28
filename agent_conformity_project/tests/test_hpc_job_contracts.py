#!/usr/bin/env python3
"""Static fail-fast contracts for the reviewer add-on Slurm jobs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def text(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_qwen_mmlu_job_is_first500_and_seven_condition():
    script = text("hpc_self_history_qwen3b_mmlu500.slurm")
    assert "mmlu_all_validation.json" in script
    assert 'EXPECTED_MD5="7be4bb95c4e8e12531d7004fea12f36d"' in script
    assert "--limit 500" in script
    assert "sequential_final_neutral_v2" in script
    assert "sequential_final_short_ack" in script
    assert "--target-tokens 13" in script


def test_gemma_job_selects_controls_from_observed_history():
    script = text("hpc_self_history_gemma2b_cqa500.slurm")
    history = script.index("STAGE 1: OBSERVED SELF-HISTORY CONDITIONS")
    selection = script.index("TOKENIZER-SPECIFIC NEUTRAL SELECTION")
    controls = script.index("STAGE 2: MATCHED NEUTRAL CONTROLS")
    assert history < selection < controls
    assert "select_neutral_controls.py" in script
    assert "--neutral-selection-json" in script
    assert "--reference-jsonl \"$BASE_DIR/sequential_final.jsonl\"" in script


def test_slurm_files_do_not_duplicate_neutral_wording():
    forbidden = (
        "Acknowledged. Message received. Proceeding to the next message.",
        "Message received. Acknowledged. Moving to the following message.",
    )
    for name in (
        "hpc_self_history_qwen3b_mmlu500.slurm",
        "hpc_self_history_gemma2b_cqa500.slurm",
    ):
        script = text(name)
        assert all(value not in script for value in forbidden), name


def test_clean_label_job_requires_archived_reference():
    script = text("hpc_clean_social_label.slurm")
    assert "REFERENCE_JSONL:?" in script
    assert "select_neutral_label.py" in script
    assert "test_conditions.py" in script
    assert "--reference-jsonl \"$REFERENCE_JSONL\"" in script


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
