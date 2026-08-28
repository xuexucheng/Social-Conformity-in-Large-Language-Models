#!/usr/bin/env python3
"""Offline invariance checks for the clean social-label experiment."""

from __future__ import annotations

import json

import conditions as C
import run_experiment as R


ITEM = {
    "id": "x",
    "question": "Question?",
    "options": {"A": "one", "B": "two", "C": "three"},
    "correct_answer": "A",
    "distractor": "B",
}
INITIAL = {
    "messages": C.build_initial_messages(ITEM),
    "text": "ANSWER: A\nCONFIDENCE: 100",
}


def test_primary_conditions_change_only_label_word():
    neutral = C.build_final_messages(ITEM, INITIAL, "neutral_source_label", "Source")
    model = C.build_final_messages(ITEM, INITIAL, "model_source_label", "Source")
    assert neutral[:-1] == model[:-1]
    neutral_text = neutral[-1]["content"]
    model_text = model[-1]["content"]
    assert C.replace_primary_label(neutral_text, "Source", "Model") == model_text
    assert neutral_text.count("Source ") == model_text.count("Model ") == 5


def test_all_conditions_share_target_count_order_and_instruction():
    messages = {
        condition: C.build_final_messages(ITEM, INITIAL, condition, "Source")
        for condition in C.CONDITIONS
    }
    for condition, value in messages.items():
        assert value[:3] == messages[C.CONDITIONS[0]][:3], condition
        final = value[-1]["content"]
        assert final.count(ITEM["distractor"]) == C.N_SOURCES
        assert final.endswith(C.FINAL_DECISION_INSTRUCTION)
        assert ITEM["correct_answer"] not in "\n".join(C.guidance_lines(ITEM, condition, "Source"))


def test_builders_are_deterministic():
    one = C.build_final_messages(ITEM, INITIAL, "model_source_label")
    two = C.build_final_messages(ITEM, INITIAL, "model_source_label")
    assert one == two
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


def test_reference_replay_skips_private_baseline_call():
    calls = []

    def fake_model_call(messages, options):
        calls.append(messages)
        return {
            "text": "ANSWER: A\nCONFIDENCE: 100",
            "prediction": "A",
            "confidence": 100,
            "valid": True,
        }

    original = R.model_call
    R.model_call = fake_model_call
    try:
        rows = R.run_item(
            ITEM,
            list(C.CONDITIONS),
            "Source",
            {
                **ITEM,
                "raw_initial_output": "ANSWER: A\nCONFIDENCE: 100",
                "initial_prediction": "A",
            },
        )
    finally:
        R.model_call = original
    assert len(calls) == len(C.CONDITIONS)
    assert set(rows) == set(C.CONDITIONS)
    assert all(row["initial_answer_source"] == "reference_jsonl" for row in rows.values())


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
