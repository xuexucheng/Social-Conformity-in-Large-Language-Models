"""Message builders for the Self-History Ablation (journal Major Revision).

This module is the SINGLE SOURCE OF TRUTH for how every ablation condition's
messages are constructed. The runner (`run_ablation.py`), the static tests
(`test_prompt_equivalence.py`), and the offline smoke (`smoke_offline.py`) all
import from here so that what we test == what we run.

Design goal (reviewer's concern)
--------------------------------
The published Exp2 (Sequential-Final) and Exp3 (Sequential-Stepwise) differ in
SEVERAL ways at once (self-answer history, a trailing FINAL_DECISION_INSTRUCTION
turn, and turn count/structure). This module decomposes that confound into a
family of conditions that are *identical everywhere* except one controlled
dimension each. Everything is anchored on the ORIGINAL Exp2 final request so
that `stepwise_no_history` step-5 is byte-identical to `sequential_final`.

Primitives copied verbatim from the archived experiment
-------------------------------------------------------
`FINAL_DECISION_INSTRUCTION`, `format_agent_line`, `build_initial_answer_context`
are exact copies of the ones in
`experiments/2026-04-29_five_wrong_guidance/run_experiments.py` that produced the
archived Exp1-5 results. `test_prompt_equivalence.py` asserts byte-identity
against the legacy module, so any drift is caught immediately. The initial user
prompt is built by the shared `src.prompts.build_prompt`.

Terminology (paper-facing)
--------------------------
Exp3's intermediate output is empirically `ANSWER: X\\nCONFIDENCE: NN` (no
reasoning; verified across all 2500 archived step outputs). We therefore DO NOT
use the term "Full-History". The condition that replays the raw assistant text
is named **Stepwise-Answer+Confidence-History**. The archived Exp3 is kept only
as an "as-published" reference.
"""

import os
import sys

# --- bootstrap import path so `src.*` resolves regardless of CWD --------------
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.prompts import build_prompt  # noqa: E402  (shared with the paper artifact)

OPTION_LABELS = ["A", "B", "C", "D", "E"]
N_WRONG_GUIDES = 5

# ---------------------------------------------------------------------------
# Primitives — EXACT copies of the legacy run_experiments.py definitions.
# Guarded by test_prompt_equivalence.py::test_legacy_primitive_identity.
# ---------------------------------------------------------------------------
FINAL_DECISION_INSTRUCTION = (
    "You may revise your answer after considering the opinions of other agents if they are persuasive.\n"
    "Remember: output only the required ANSWER line, with no explanation."
)


def format_agent_line(agent_index, opinion):
    """Sequential peer line, e.g. 'Agent 1: B'  (bare option letter, no 'ANSWER:')."""
    return f"Agent {agent_index}: {opinion}"


def build_initial_messages(item):
    """Initial (private baseline) request: identical to the legacy initial pass."""
    return build_prompt(item, opinions=[])


def build_initial_answer_context(initial_result):
    """[system, user(question)] + assistant(initial private answer).

    `initial_result` must be a dict with keys 'messages' (the initial request)
    and 'text' (the model's raw initial answer). Shared by every condition so
    the private baseline turn is byte-identical across all of them.
    """
    messages = list(initial_result["messages"])
    messages.append({"role": "assistant", "content": initial_result["text"]})
    return messages


# ---------------------------------------------------------------------------
# History-mode constants for the stepwise family
# ---------------------------------------------------------------------------
HISTORY_NONE = "none"                        # Stepwise-No-History
HISTORY_ANSWER = "answer"                     # Stepwise-Answer-History
HISTORY_ANSWER_CONFIDENCE = "answer_confidence"  # Stepwise-Answer+Confidence-History

# Default neutral turn for Length-Matched. CONFIRMED for Qwen2.5-3B: it is an
# EXACT 13-token match to that model's self-history turn (ANSWER: X /
# CONFIDENCE: 100) and passes every neutrality rule. Per-model-family variants
# (Phi / Gemma) must be length-checked on their own tokenizers BEFORE running;
# see compute_neutral_token_match.py.
DEFAULT_NEUTRAL_TURN = "Acknowledged. Message received. Proceeding to the next message."
# Independently worded from Neutral-V1 and verified as exactly 13 tokens with
# the Qwen2.5-3B-Instruct tokenizer used by the formal run.
DEFAULT_NEUTRAL_V2_TURN = "Message received. Acknowledged. Moving to the following message."
DEFAULT_SHORT_ACK_TURN = "Acknowledged."

NEUTRAL_TURN_BY_MODEL_FAMILY = {
    # family substring (lower-cased) -> neutral turn
    "qwen": "Acknowledged. Message received. Proceeding to the next message.",
    # "phi":   <fill after AutoDL token check>,
    # "gemma": <fill after AutoDL token check>,
}

NEUTRAL_V2_TURN_BY_MODEL_FAMILY = {
    # Independent wording robustness control. Confirm its token count on the
    # exact local tokenizer before a full run; the runner records the count.
    "qwen": DEFAULT_NEUTRAL_V2_TURN,
}

SHORT_ACK_TURN_BY_MODEL_FAMILY = {
    # Deliberately not length matched. This isolates whether a short assistant
    # role marker alone is sufficient to reproduce the structural effect.
    "qwen": DEFAULT_SHORT_ACK_TURN,
    "phi": DEFAULT_SHORT_ACK_TURN,
    "gemma": DEFAULT_SHORT_ACK_TURN,
}


def resolve_neutral_turn(model_name, override=None):
    """Pick the neutral turn for a model. Explicit override wins.

    Falls back to DEFAULT_NEUTRAL_TURN if the family is unknown, and the caller
    is expected to have length-checked it on the model's tokenizer first.
    """
    if override:
        return override
    m = (model_name or "").lower()
    for family, turn in NEUTRAL_TURN_BY_MODEL_FAMILY.items():
        if family in m:
            return turn
    return DEFAULT_NEUTRAL_TURN


def _resolve_family_turn(model_name, override, mapping, default):
    if override:
        return override
    model = (model_name or "").lower()
    for family, turn in mapping.items():
        if family in model:
            return turn
    return default


def resolve_neutral_v2_turn(model_name, override=None):
    return _resolve_family_turn(
        model_name,
        override,
        NEUTRAL_V2_TURN_BY_MODEL_FAMILY,
        DEFAULT_NEUTRAL_V2_TURN,
    )


def resolve_short_ack_turn(model_name, override=None):
    return _resolve_family_turn(
        model_name,
        override,
        SHORT_ACK_TURN_BY_MODEL_FAMILY,
        DEFAULT_SHORT_ACK_TURN,
    )


def normalize_answer_turn(prediction):
    """Minimal, faithful 'answer-only' assistant turn: 'ANSWER: X'.

    Matches the legacy answer line format from src.prompts.build_instruction,
    with the CONFIDENCE line deliberately stripped. Only used when the parsed
    prediction is a valid option label; callers handle the (empirically absent)
    unparseable case explicitly.
    """
    return f"ANSWER: {prediction}"


# ---------------------------------------------------------------------------
# Condition builders
# ---------------------------------------------------------------------------
def build_sequential_final_messages(initial_result, opinions):
    """Condition 1 — Sequential-Final  (== archived Exp2 final request).

    [init_ctx] + user('Agent i: op') x5 + user(FINAL_DECISION_INSTRUCTION).
    The model answers ONCE.
    """
    messages = build_initial_answer_context(initial_result)
    for idx, opinion in enumerate(opinions, start=1):
        messages.append({"role": "user", "content": format_agent_line(idx, opinion)})
    messages.append({"role": "user", "content": FINAL_DECISION_INSTRUCTION})
    return messages


def format_agent_rationale_line(agent_index, opinion):
    """Same Agent identity and target as the bare condition, plus a generic justification."""
    return (
        f"Agent {agent_index} suggests that option {opinion} is correct because "
        "it best matches the question and the available options."
    )


def build_sequential_final_rationale_messages(initial_result, opinions):
    """Sequential-Final with justification-bearing peer statements.

    Identical to build_sequential_final_messages except for the content
    of the five peer user turns.
    """
    messages = build_initial_answer_context(initial_result)
    for idx, opinion in enumerate(opinions, start=1):
        messages.append(
            {
                "role": "user",
                "content": format_agent_rationale_line(idx, opinion),
            }
        )
    messages.append(
        {"role": "user", "content": FINAL_DECISION_INSTRUCTION}
    )
    return messages


def build_stepwise_request_messages(
    initial_result,
    opinions,
    step,
    prior_step_texts=None,
    prior_step_predictions=None,
    history_mode=HISTORY_NONE,
):
    """Conditions 2/3/4 — messages for the `step`-th answer request (1-indexed).

    Structure:
        [init_ctx]
        user('Agent 1: op1')  [+ assistant(history_1)]
        user('Agent 2: op2')  [+ assistant(history_2)]
        ...
        user('Agent {step}: op{step}')          # no history after the CURRENT peer
        user(FINAL_DECISION_INSTRUCTION)

    History turns are inserted AFTER 'Agent i' for i < step, and ONLY when
    `history_mode != HISTORY_NONE`:
        HISTORY_ANSWER            -> assistant('ANSWER: {pred_i}')     (confidence stripped)
        HISTORY_ANSWER_CONFIDENCE -> assistant(raw text_i)            (verbatim replay)

    Anchoring: with history_mode=HISTORY_NONE and step=len(opinions), the result
    is byte-identical to build_sequential_final_messages(...). This is the
    invariant that makes No-History step-5 == Sequential-Final (Test 1).
    """
    prior_step_texts = prior_step_texts or []
    prior_step_predictions = prior_step_predictions or []

    messages = build_initial_answer_context(initial_result)
    for i in range(1, step + 1):
        messages.append({"role": "user", "content": format_agent_line(i, opinions[i - 1])})
        if i < step and history_mode != HISTORY_NONE:
            if history_mode == HISTORY_ANSWER:
                pred = prior_step_predictions[i - 1] if i - 1 < len(prior_step_predictions) else None
                if pred in OPTION_LABELS:
                    content = normalize_answer_turn(pred)
                else:
                    # Empirically never happens (parse rate = 100% on the paper
                    # runs). Documented fallback: replay raw text and let the
                    # runner flag the item so it can be excluded from the clean
                    # paired subset.
                    content = prior_step_texts[i - 1]
            elif history_mode == HISTORY_ANSWER_CONFIDENCE:
                content = prior_step_texts[i - 1]
            else:
                raise ValueError(f"Unknown history_mode: {history_mode}")
            messages.append({"role": "assistant", "content": content})
    messages.append({"role": "user", "content": FINAL_DECISION_INSTRUCTION})
    return messages


def build_length_matched_messages(initial_result, opinions, neutral_turn):
    """Condition 5 — Sequential-Final-Length-Matched.

    Sequential-Final + a fixed neutral assistant turn inserted AFTER Agent 1..4
    (NOT after Agent 5). Agent 5 is followed directly by FINAL_DECISION_INSTRUCTION.
    Isolates dialogue-turn / context-length structure from self-answer content.
    """
    messages = build_initial_answer_context(initial_result)
    n = len(opinions)
    for idx, opinion in enumerate(opinions, start=1):
        messages.append({"role": "user", "content": format_agent_line(idx, opinion)})
        if idx < n:  # neutral turn after Agent 1..(n-1) only
            messages.append({"role": "assistant", "content": neutral_turn})
    messages.append({"role": "user", "content": FINAL_DECISION_INSTRUCTION})
    return messages


# ---------------------------------------------------------------------------
# Audit helpers (used by tests + smoke)
# ---------------------------------------------------------------------------
def assistant_turns_after_initial(messages):
    """Return assistant-turn contents that appear AFTER the initial answer turn.

    messages[2] is the initial private answer (index 0=system, 1=user, 2=assistant).
    Everything after that with role 'assistant' is injected history / neutral /
    (in the published Exp3) self-answer. Used to detect history leakage.
    """
    if len(messages) < 3:
        return []
    return [m["content"] for m in messages[3:] if m.get("role") == "assistant"]


def strip_confidence_line(text):
    """Return `text` with a trailing CONFIDENCE line removed.

    Turns 'ANSWER: X\\nCONFIDENCE: NN' into 'ANSWER: X'. Used only by the tests
    to prove that Answer+Confidence-History differs from Answer-History by
    exactly the confidence line.
    """
    lines = [ln for ln in text.split("\n") if not ln.strip().upper().startswith("CONFIDENCE:")]
    return "\n".join(lines).strip()
