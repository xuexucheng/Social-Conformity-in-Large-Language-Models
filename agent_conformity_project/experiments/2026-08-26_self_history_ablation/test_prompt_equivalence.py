"""Static prompt-equivalence tests for the Self-History Ablation.

No model or server required — these operate purely on the message builders in
`conditions.py`, so they can gate the experiment before any GPU time is spent.

Run:  python3 test_prompt_equivalence.py     (or: pytest -q test_prompt_equivalence.py)

Covers the reviewer-facing invariants:
  Test 1  stepwise_no_history step-5 messages  ==  sequential_final messages
  Test 2  answer_history       adds ONLY previous normalized 'ANSWER: X' turns
  Test 3  answer_confidence    adds ONLY the CONFIDENCE line on top of answer_history
  Test 4  length_matched       adds ONLY 4 neutral turns after Agent 1..4
          (and Agent 5 is followed directly by FINAL_DECISION_INSTRUCTION)
  Test 5  question / options / correct / distractor / peer sequence & order
          are identical across every condition
plus: legacy-primitive byte-identity, legacy-logprob-helper identity,
      decoding-config lock, No-History no-leakage at every step, output-path guard,
      step-logprob retention, token-audit fields, strict resume checkpoints, Holm.
"""

import importlib.util
import json
import os
import sys
import tempfile

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for p in (CURRENT_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import conditions as C  # noqa: E402
import analyze_ablation as A  # noqa: E402
import run_ablation as R  # noqa: E402
import verify_neutral_token_match as V  # noqa: E402
import select_neutral_controls as S  # noqa: E402
from src.config import MAX_TOKENS, TEMPERATURE  # noqa: E402


# --- load the archived (legacy) experiment module for byte-identity checks ----
_legacy_path = os.path.join(PROJECT_DIR, "experiments", "2026-04-29_five_wrong_guidance", "run_experiments.py")
_spec = importlib.util.spec_from_file_location("legacy_runexp", _legacy_path)
LEGACY = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEGACY)


# --- fixtures -----------------------------------------------------------------
ITEM = {
    "id": "itemX",
    "source": "commonsenseqa",
    "question": "A revolving door serves as a security measure at a what?",
    "options": {"A": "bank", "B": "library", "C": "department store", "D": "mall", "E": "new york"},
    "correct_answer": "A",
    "distractor": "B",
}
OPINIONS = [ITEM["distractor"]] * C.N_WRONG_GUIDES
INITIAL_RESULT = {"messages": C.build_initial_messages(ITEM), "text": "ANSWER: A\nCONFIDENCE: 100"}

# synthetic per-step outputs (steps 1..4) for the history builders
PRIOR_PREDS = ["A", "B", "C", "D"]
PRIOR_TEXTS = [f"ANSWER: {p}\nCONFIDENCE: {90 + i}" for i, p in enumerate(PRIOR_PREDS)]


# --- helpers ------------------------------------------------------------------
def injected_assistant_turns(messages):
    """Assistant-role messages that appear AFTER the initial answer (index 2)."""
    return [m for m in messages[3:] if m.get("role") == "assistant"]


def without_injected_assistants(messages):
    """messages with every post-initial assistant turn removed."""
    return messages[:3] + [m for m in messages[3:] if m.get("role") != "assistant"]


def agent_user_turns(messages):
    return [m["content"] for m in messages if m.get("role") == "user" and m["content"].startswith("Agent ")]


def eq(a, b):
    return a == b and json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


# --- guard tests --------------------------------------------------------------
def test_legacy_primitive_identity():
    assert C.FINAL_DECISION_INSTRUCTION == LEGACY.FINAL_DECISION_INSTRUCTION
    assert C.format_agent_line(3, "B") == LEGACY.format_agent_line(3, "B")
    assert eq(C.build_initial_answer_context(INITIAL_RESULT),
              LEGACY.build_initial_answer_context(INITIAL_RESULT))
    # Condition 1 builder must equal the archived Exp2 builder exactly.
    assert eq(C.build_sequential_final_messages(INITIAL_RESULT, OPINIONS),
              LEGACY.build_sequential_context_final_messages(INITIAL_RESULT, OPINIONS))


def test_legacy_logprob_helper_identity():
    top = [{"token": "A", "logprob": -0.1}, {"token": "b", "logprob": -2.0}, {"token": "C", "logprob": -3.0}]
    assert R.option_logprobs_from_top_logprobs(top) == LEGACY.option_logprobs_from_top_logprobs(top)
    result = {"content_logprobs": [{"token": "A", "logprob": -0.1, "top_logprobs": top}]}
    assert R.option_logprobs_from_answer_token(result, "A") == LEGACY.option_logprobs_from_answer_token(result, "A")


def test_decoding_config_locked():
    assert TEMPERATURE == 0, f"temperature must be 0 (greedy); got {TEMPERATURE}"
    assert MAX_TOKENS == 128, f"max_tokens must be 128; got {MAX_TOKENS}"


# --- Test 1 -------------------------------------------------------------------
def test1_no_history_step5_equals_sequential_final():
    seqfinal = C.build_sequential_final_messages(INITIAL_RESULT, OPINIONS)
    nh5 = C.build_stepwise_request_messages(
        INITIAL_RESULT, OPINIONS, step=5, prior_step_texts=[], prior_step_predictions=[],
        history_mode=C.HISTORY_NONE,
    )
    assert eq(nh5, seqfinal), "No-History step-5 must be byte-identical to Sequential-Final"


# --- Test 2 -------------------------------------------------------------------
def test2_answer_history_adds_only_normalized_answers():
    nh5 = C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, 5, history_mode=C.HISTORY_NONE)
    ah5 = C.build_stepwise_request_messages(
        INITIAL_RESULT, OPINIONS, 5, PRIOR_TEXTS, PRIOR_PREDS, history_mode=C.HISTORY_ANSWER)
    # removing the injected assistant turns recovers No-History exactly
    assert eq(without_injected_assistants(ah5), nh5)
    # and those injected turns are ONLY 'ANSWER: X' (no confidence, no reasoning)
    injected = injected_assistant_turns(ah5)
    assert [m["content"] for m in injected] == [f"ANSWER: {p}" for p in PRIOR_PREDS]


# --- Test 3 -------------------------------------------------------------------
def test3_answer_confidence_adds_only_confidence():
    ah5 = C.build_stepwise_request_messages(
        INITIAL_RESULT, OPINIONS, 5, PRIOR_TEXTS, PRIOR_PREDS, history_mode=C.HISTORY_ANSWER)
    ac5 = C.build_stepwise_request_messages(
        INITIAL_RESULT, OPINIONS, 5, PRIOR_TEXTS, PRIOR_PREDS, history_mode=C.HISTORY_ANSWER_CONFIDENCE)
    # non-assistant structure identical
    assert eq(without_injected_assistants(ac5), without_injected_assistants(ah5))
    ah_inj = [m["content"] for m in injected_assistant_turns(ah5)]
    ac_inj = [m["content"] for m in injected_assistant_turns(ac5)]
    assert len(ah_inj) == len(ac_inj) == 4
    for ans_only, raw in zip(ah_inj, ac_inj):
        assert raw != ans_only, "confidence condition must add something"
        assert C.strip_confidence_line(raw) == ans_only, "difference must be ONLY the CONFIDENCE line"
        assert raw.startswith(ans_only + "\nCONFIDENCE:")


# --- Test 4 -------------------------------------------------------------------
def test4_length_matched_adds_only_neutral_turns():
    seqfinal = C.build_sequential_final_messages(INITIAL_RESULT, OPINIONS)
    lm = C.build_length_matched_messages(INITIAL_RESULT, OPINIONS, C.DEFAULT_NEUTRAL_TURN)
    assert eq(without_injected_assistants(lm), seqfinal)
    injected = injected_assistant_turns(lm)
    assert [m["content"] for m in injected] == [C.DEFAULT_NEUTRAL_TURN] * 4, "exactly 4 neutral turns (after Agent 1..4)"
    # Agent 5 must be followed DIRECTLY by FINAL_DECISION_INSTRUCTION (no 5th neutral)
    idx5 = next(i for i, m in enumerate(lm) if m["content"] == "Agent 5: B")
    assert lm[idx5 + 1]["role"] == "user" and lm[idx5 + 1]["content"] == C.FINAL_DECISION_INSTRUCTION


def test4b_new_controls_change_only_inserted_assistant_text():
    seqfinal = C.build_sequential_final_messages(INITIAL_RESULT, OPINIONS)
    for turn in (C.DEFAULT_NEUTRAL_V2_TURN, C.DEFAULT_SHORT_ACK_TURN):
        controlled = C.build_length_matched_messages(INITIAL_RESULT, OPINIONS, turn)
        assert eq(without_injected_assistants(controlled), seqfinal)
        assert [m["content"] for m in injected_assistant_turns(controlled)] == [turn] * 4


def test4c_neutral_preflight_uses_conditions_as_single_source_of_truth():
    assert C.DEFAULT_NEUTRAL_V2_TURN == (
        "Message received. Acknowledged. Moving to the following message."
    )
    assert C.DEFAULT_NEUTRAL_V2_TURN != C.DEFAULT_NEUTRAL_TURN

    class ExactCountStub:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            assert text in (C.DEFAULT_NEUTRAL_TURN, C.DEFAULT_NEUTRAL_V2_TURN)
            return list(range(13))

    audit = V.count_control_tokens(
        ExactCountStub(), "Qwen/Qwen2.5-3B-Instruct"
    )
    assert audit["neutral_v1"] == C.DEFAULT_NEUTRAL_TURN
    assert audit["neutral_v2"] == C.DEFAULT_NEUTRAL_V2_TURN
    assert audit["neutral_v1_tokens"] == audit["neutral_v2_tokens"] == 13

    slurm_path = os.path.join(
        os.path.dirname(PROJECT_DIR), "hpc_self_history_qwen3b_neutral_controls.slurm"
    )
    with open(slurm_path, encoding="utf-8") as f:
        slurm = f.read()
    assert "verify_neutral_token_match.py" in slurm
    assert C.DEFAULT_NEUTRAL_TURN not in slurm
    assert C.DEFAULT_NEUTRAL_V2_TURN not in slurm


def test4d_tokenizer_specific_neutral_selection_uses_observed_mode():
    class WordTokenizer:
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return str(text).split()

    observed = ["one two three"] * 7 + ["one two"] * 2
    candidates = ["red blue green", "cat dog bird", "too short"]
    selected = S.select_candidates(WordTokenizer(), observed, candidates)
    assert selected["target_tokens"] == 3
    assert selected["neutral_v1"] == "red blue green"
    assert selected["neutral_v2"] == "cat dog bird"
    audit = V.count_control_tokens(
        WordTokenizer(),
        "unknown-model",
        neutral_turn=selected["neutral_v1"],
        neutral_v2_turn=selected["neutral_v2"],
    )
    assert audit["neutral_v1_tokens"] == audit["neutral_v2_tokens"] == 3


# --- Test 5 -------------------------------------------------------------------
def _final_messages_all_conditions():
    return {
        "sequential_final": C.build_sequential_final_messages(INITIAL_RESULT, OPINIONS),
        "stepwise_no_history": C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, 5, history_mode=C.HISTORY_NONE),
        "stepwise_answer_history": C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, 5, PRIOR_TEXTS, PRIOR_PREDS, C.HISTORY_ANSWER),
        "stepwise_answer_confidence_history": C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, 5, PRIOR_TEXTS, PRIOR_PREDS, C.HISTORY_ANSWER_CONFIDENCE),
        "sequential_final_length_matched": C.build_length_matched_messages(INITIAL_RESULT, OPINIONS, C.DEFAULT_NEUTRAL_TURN),
        "sequential_final_neutral_v2": C.build_length_matched_messages(INITIAL_RESULT, OPINIONS, C.DEFAULT_NEUTRAL_V2_TURN),
        "sequential_final_short_ack": C.build_length_matched_messages(INITIAL_RESULT, OPINIONS, C.DEFAULT_SHORT_ACK_TURN),
    }


def test5_shared_invariants_across_conditions():
    allmsg = _final_messages_all_conditions()
    expected_agents = [f"Agent {i}: {ITEM['distractor']}" for i in range(1, 6)]
    prefixes, agentlists = [], []
    for cond, msgs in allmsg.items():
        # same initial [system, user(question), assistant(initial)] prefix
        prefixes.append(msgs[:3])
        # same peer sequence AND order
        agentlists.append(agent_user_turns(msgs))
        assert agent_user_turns(msgs) == expected_agents, f"{cond} peer sequence/order changed"
    assert all(eq(prefixes[0], p) for p in prefixes[1:]), "initial context differs across conditions"
    assert all(a == agentlists[0] for a in agentlists[1:]), "peer turns differ across conditions"
    # question/options/correct/distractor come from a single item; peers == distractor x5
    assert OPINIONS == [ITEM["distractor"]] * 5
    assert ITEM["correct_answer"] == "A" and ITEM["distractor"] == "B"


# --- extra: No-History leakage-free at EVERY step -----------------------------
def test_no_history_no_leakage_all_steps():
    for k in range(1, 6):
        msgs = C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, k, PRIOR_TEXTS, PRIOR_PREDS, C.HISTORY_NONE)
        assert C.assistant_turns_after_initial(msgs) == [], f"No-History step {k} leaked assistant history"


def test_answer_history_leakage_shape_all_steps():
    # at step k, answer-history should carry exactly k-1 'ANSWER: X' turns, no confidence
    for k in range(1, 6):
        msgs = C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, k, PRIOR_TEXTS, PRIOR_PREDS, C.HISTORY_ANSWER)
        hist = C.assistant_turns_after_initial(msgs)
        assert len(hist) == k - 1
        assert all(h.startswith("ANSWER: ") and "CONFIDENCE" not in h for h in hist)


def test_output_path_guard():
    assert R.OUTPUT_SUBDIR != R.LEGACY_SUBDIR
    assert "2026-04-29" not in R.OUTPUT_SUBDIR


def _logprob_stub(_messages):
    top = [
        {"token": "A", "logprob": -0.1},
        {"token": "B", "logprob": -2.0},
    ]
    return {
        "text": "ANSWER: A\nCONFIDENCE: 100",
        "content_logprobs": [
            {"token": "A", "logprob": -0.1, "top_logprobs": top}
        ],
        "raw": {},
    }


def test_step_records_retain_logprobs():
    rows = R.run_item(
        ITEM,
        _logprob_stub,
        ["stepwise_no_history"],
        "Qwen/Qwen2.5-3B-Instruct",
        "test-run",
        "20260828_000000",
        C.DEFAULT_NEUTRAL_TURN,
    )
    steps = rows["stepwise_no_history"]["step_outputs"]
    assert len(steps) == 5
    assert all(step["option_logprobs"]["A"] == -0.1 for step in steps)
    assert all(step["selected_logprob"] == -0.1 for step in steps)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(range(len((text or "").split())))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        assert tokenize and add_generation_prompt
        token_count = sum(len((message.get("content") or "").split()) + 1 for message in messages)
        return list(range(token_count))


def test_token_audit_is_populated_when_requested():
    rows = R.run_item(
        ITEM,
        _logprob_stub,
        ["sequential_final_length_matched"],
        "Qwen/Qwen2.5-3B-Instruct",
        "test-run",
        "20260828_000000",
        C.DEFAULT_NEUTRAL_TURN,
    )
    row = rows["sequential_final_length_matched"]
    R._add_token_audit(row, _FakeTokenizer(), "Qwen/Qwen2.5-3B-Instruct")
    assert isinstance(row["final_prompt_token_count"], int)
    step = row["step_outputs"][0]
    assert len(step["history_inserted_token_counts"]) == 4
    assert step["history_inserted_total_token_count"] == sum(
        step["history_inserted_token_counts"]
    )
    assert isinstance(step["request_prompt_token_count"], int)


def _write_test_jsonl(path, ids):
    with open(path, "w", encoding="utf-8") as f:
        for item_id in ids:
            f.write(json.dumps({"id": item_id}) + "\n")


def test_resume_requires_aligned_condition_ids():
    with tempfile.TemporaryDirectory() as temp_dir:
        _write_test_jsonl(os.path.join(temp_dir, "a.jsonl"), ["1", "2"])
        _write_test_jsonl(os.path.join(temp_dir, "b.jsonl"), ["1", "2"])
        assert R._resume_completed_ids(temp_dir, ["a", "b"]) == {"1", "2"}
        _write_test_jsonl(os.path.join(temp_dir, "b.jsonl"), ["1"])
        try:
            R._resume_completed_ids(temp_dir, ["a", "b"])
        except SystemExit as exc:
            assert "unaligned checkpoint" in str(exc)
        else:
            raise AssertionError("unaligned resume checkpoint must fail closed")


def test_reference_replay_skips_initial_model_call():
    calls = []

    def counted_stub(messages):
        calls.append(messages)
        return _logprob_stub(messages)

    reference = {
        "raw_initial_output": "ANSWER: B\nCONFIDENCE: 87",
        "initial_prediction": "B",
    }
    rows = R.run_item(
        ITEM,
        counted_stub,
        ["sequential_final_neutral_v2"],
        "Qwen/Qwen2.5-3B-Instruct",
        "test-run",
        "20260828_000000",
        C.DEFAULT_NEUTRAL_TURN,
        initial_reference=reference,
        reference_jsonl="prior.jsonl",
    )
    assert len(calls) == 1, "reference replay must skip the initial model call"
    row = rows["sequential_final_neutral_v2"]
    assert row["raw_initial_output"] == reference["raw_initial_output"]
    assert row["initial_prediction"] == "B"
    assert row["initial_answer_source"] == "reference_jsonl"
    assert row["initial_answer_reference_jsonl"] == "prior.jsonl"


def test_reference_loader_rejects_duplicates_and_metadata_drift():
    reference_row = {
        **ITEM,
        "raw_initial_output": "ANSWER: A\nCONFIDENCE: 100",
        "initial_prediction": "A",
    }
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "reference.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(reference_row) + "\n")
        references = R._load_initial_answer_references(path)
        R._validate_initial_answer_references(references, [ITEM])

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(reference_row) + "\n")
        try:
            R._load_initial_answer_references(path)
        except SystemExit as exc:
            assert "duplicate reference item id" in str(exc)
        else:
            raise AssertionError("duplicate references must fail closed")

        drifted = {**ITEM, "question": "different question"}
        try:
            R._validate_initial_answer_references({ITEM["id"]: reference_row}, [drifted])
        except SystemExit as exc:
            assert "reference metadata mismatch" in str(exc)
        else:
            raise AssertionError("reference metadata drift must fail closed")


def test_exact_token_pair_audit_and_restricted_comparison():
    def record(final, token_count):
        return {
            "valid": True,
            "is_correct": final == "A",
            "chose_distractor": final == "B",
            "initial_correct": True,
            "changed": final != "A",
            "final_prompt_token_count": token_count,
        }

    condition1 = {"one": record("A", 100), "two": record("B", 103)}
    condition2 = {"one": record("B", 100), "two": record("A", 100)}
    audit = A.build_token_pair_audit("c1", condition1, "c2", condition2)
    assert audit["exact_match_ids"] == ["one"]
    assert audit["difference_counts"] == {"0": 1, "3": 1}
    rows = A.build_comparison(
        "exact",
        "c1",
        condition1,
        "c2",
        condition2,
        100,
        1,
        False,
        allowed_ids={"one"},
        analysis_subset="exact_final_prompt_token_match",
    )
    assert rows and all(row["paired_n"] == 1 for row in rows)
    assert all(row["analysis_subset"] == "exact_final_prompt_token_match" for row in rows)


def test_holm_adjustment_step_down_monotonicity():
    adjusted = A.holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


# --- manual runner (no pytest needed) -----------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    ok = _run_all()
    sys.exit(0 if ok else 1)
