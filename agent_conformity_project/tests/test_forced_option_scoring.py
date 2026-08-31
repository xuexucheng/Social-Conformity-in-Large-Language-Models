import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "forced_option_scoring.py"
)

spec = importlib.util.spec_from_file_location(
    "forced_option_scoring",
    SCRIPT,
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_candidate_labels_real_option_sets():
    assert M.candidate_labels(
        {"A": "a", "B": "b", "C": "c", "D": "d"}
    ) == ("A", "B", "C", "D")

    assert M.candidate_labels(
        {"A": "a", "B": "b", "C": "c", "D": "d", "E": "e"}
    ) == ("A", "B", "C", "D", "E")


def test_rejects_noncanonical_option_set():
    with pytest.raises(ValueError):
        M.candidate_labels({"A": "a", "C": "c"})


def test_softmax_normalizes():
    p = M.stable_softmax([-3.0, -2.0, -1.0, -4.0])
    assert sum(p) == pytest.approx(1.0)
    assert p[2] == max(p)


def test_sequence_logprob_uses_logits_immediately_before_candidate_tokens():
    # Sequence:
    # prompt tokens = positions 0,1
    # candidate tokens = ids 3,4 at sequence positions 2,3
    #
    # Therefore:
    # candidate id 3 must use logits[1]
    # candidate id 4 must use logits[2]
    vocab = 6
    logits = torch.zeros((4, vocab), dtype=torch.float32)

    logits[1, 3] = 5.0
    logits[2, 4] = 7.0

    expected = (
        torch.log_softmax(logits[1], dim=-1)[3]
        + torch.log_softmax(logits[2], dim=-1)[4]
    ).item()

    actual = M.sequence_logprob_from_logits(
        logits,
        prompt_len=2,
        candidate_ids=[3, 4],
    )

    assert actual == pytest.approx(expected)


def test_sequence_logprob_rejects_empty_candidate():
    logits = torch.zeros((3, 5))
    with pytest.raises(ValueError):
        M.sequence_logprob_from_logits(
            logits,
            prompt_len=2,
            candidate_ids=[],
        )
