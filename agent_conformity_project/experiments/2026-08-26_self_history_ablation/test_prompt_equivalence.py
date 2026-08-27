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
      decoding-config lock, No-History no-leakage at every step, output-path guard.
"""

import importlib.util
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for p in (CURRENT_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import conditions as C  # noqa: E402
import run_ablation as R  # noqa: E402
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


# --- Test 5 -------------------------------------------------------------------
def _final_messages_all_conditions():
    return {
        "sequential_final": C.build_sequential_final_messages(INITIAL_RESULT, OPINIONS),
        "stepwise_no_history": C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, 5, history_mode=C.HISTORY_NONE),
        "stepwise_answer_history": C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, 5, PRIOR_TEXTS, PRIOR_PREDS, C.HISTORY_ANSWER),
        "stepwise_answer_confidence_history": C.build_stepwise_request_messages(INITIAL_RESULT, OPINIONS, 5, PRIOR_TEXTS, PRIOR_PREDS, C.HISTORY_ANSWER_CONFIDENCE),
        "sequential_final_length_matched": C.build_length_matched_messages(INITIAL_RESULT, OPINIONS, C.DEFAULT_NEUTRAL_TURN),
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
