"""Offline mock smoke for the Self-History Ablation (no GPU / no server).

Drives the REAL `run_ablation.run_item` engine with a deterministic stub model
(a pure function of the messages), so it validates the full pipeline — message
construction, history feed-back, parsing, row assembly, and the no-leakage /
identity invariants — without any model weights.

It prints a fully human-auditable trace for one complete sample across every
condition (all final messages + every step's output + parsed choice), then runs
assertions over a few samples.

The stub simulates a plausible-but-fake dynamic so the conditions produce
DIFFERENT final answers in the expected pattern:
  * with NO self-answer in context  -> swayed by the peer majority (conforms)
  * with prior self-answers present -> anchors on its own last commitment (resists)
  * neutral (Length-Matched) turns carry no answer -> behave like Sequential-Final
This makes the audit meaningful (history visibly matters) while staying a pure
function of the messages, so No-History step-5 == Sequential-Final still holds.

Run:  python3 smoke_offline.py
"""

import os
import re
import sys
from collections import Counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for p in (CURRENT_DIR, PROJECT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import conditions as C  # noqa: E402
import run_ablation as R  # noqa: E402

ANSWER_RE = re.compile(r"ANSWER:\s*([A-E])", re.I)

SMOKE_ITEMS = [
    {"id": "smoke1", "source": "commonsenseqa",
     "question": "A revolving door serves as a security measure at a what?",
     "options": {"A": "bank", "B": "library", "C": "department store", "D": "mall", "E": "new york"},
     "correct_answer": "A", "distractor": "B"},
    {"id": "smoke2", "source": "commonsenseqa",
     "question": "Where do you put grapes just before checking out?",
     "options": {"A": "mouth", "B": "grocery cart", "C": "super market", "D": "fruit basket", "E": "fridge"},
     "correct_answer": "B", "distractor": "C"},
    {"id": "smoke3", "source": "commonsenseqa",
     "question": "What is a place that someone can go buy a teddy bear?",
     "options": {"A": "toy store", "B": "child's room", "C": "shelf", "D": "kmart", "E": "closet"},
     "correct_answer": "A", "distractor": "D"},
]


def stub_answer_fn(messages):
    """Deterministic, pure function of `messages`. Returns a legacy-format text."""
    # initial private baseline = first assistant turn (index 2), if any
    self0 = None
    if len(messages) >= 3 and messages[2].get("role") == "assistant":
        m = ANSWER_RE.search(messages[2]["content"])
        self0 = m.group(1).upper() if m else None
    # peer letters from 'Agent i: L' user lines
    user_text = "\n".join(m["content"] for m in messages if m.get("role") == "user")
    peers = [p.upper() for p in re.findall(r"Agent \d+:\s*([A-E])\b", user_text)]
    # prior self-answer commitments = assistant turns AFTER the initial one
    selfhist = []
    for m in messages[3:]:
        if m.get("role") == "assistant":
            mm = ANSWER_RE.search(m["content"])
            if mm:
                selfhist.append(mm.group(1).upper())
    # decision rule
    if selfhist:                       # anchor on own last commitment (resist)
        choice, conf = selfhist[-1], 100
    elif len(peers) >= 3:              # swayed by peer majority (conform)
        choice, conf = Counter(peers).most_common(1)[0][0], 90
    else:
        choice, conf = (self0 or "A"), 100
    return {"text": f"ANSWER: {choice}\nCONFIDENCE: {conf}", "content_logprobs": [], "raw": {}}


def show_messages(messages, indent="      "):
    print(f"{indent}({len(messages)} msgs, roles=[{','.join(m['role'][0] for m in messages)}])")
    for i, m in enumerate(messages):
        c = m["content"].replace("\n", "\\n")
        if len(c) > 96:
            c = c[:96] + f"..<+{len(c) - 96}>"
        print(f"{indent}[{i:2d}] {m['role']:9s}| {c}")


def audit_no_leakage(rows):
    """Assert No-History never leaks self-answers; report per-condition history shape."""
    ok = True
    nh = rows.get("stepwise_no_history")
    if nh:
        for step in nh["step_outputs"]:
            hist = step["history_inserted_before_step"]
            if hist:  # must be empty at every step
                print(f"      !! LEAK: No-History step {step['step']} saw assistant history: {hist}")
                ok = False
        if nh.get("no_history_step5_equals_sequential_final") is not True:
            print("      !! No-History step-5 messages != Sequential-Final messages")
            ok = False
    return ok


def main():
    conditions = R.ALL_CONDITIONS
    neutral = C.resolve_neutral_turn("Qwen/Qwen2.5-3B-Instruct")
    print("=" * 96)
    print("OFFLINE MOCK SMOKE — deterministic stub model (no GPU). Neutral turn:")
    print(f"   {neutral!r}")
    print("=" * 96)

    all_ok = True
    for idx, item in enumerate(SMOKE_ITEMS):
        rows = R.run_item(item, stub_answer_fn, conditions,
                          model_name="Qwen/Qwen2.5-3B-Instruct(STUB)",
                          run_name="smoke_offline", timestamp="SMOKE", neutral_turn=neutral)
        full = (idx == 0)  # full trace for the first sample only
        print(f"\n{'#'*96}\n# SAMPLE {item['id']}: correct={item['correct_answer']} "
              f"distractor={item['distractor']}  initial={rows['sequential_final']['initial_prediction']}\n{'#'*96}")

        for cond in conditions:
            r = rows[cond]
            tag = f"final={r['final_prediction']} conf={r['final_confidence']} " \
                  f"correct={r['is_correct']} chose_distractor={r['chose_distractor']} " \
                  f"changed={r['changed_from_initial']}"
            print(f"\n  --- {cond} --- {tag}")
            if cond.startswith("stepwise"):
                for step in r["step_outputs"]:
                    print(f"      step{step['step']}: peer='Agent {step['step']}: {step['opinion']}'  "
                          f"history_before={step['history_inserted_before_step']}  "
                          f"-> {step['raw_text']!r} (parsed {step['prediction']})")
            if full:
                print("     final_messages:")
                show_messages(r["final_messages"])

        leak_ok = audit_no_leakage(rows)
        # live Test-1 identity (independent of the flag)
        ir_txt = rows["sequential_final"]["raw_initial_output"]
        # rebuild to double-check equality here as well
        print(f"\n  [audit] No-History step-5 == Sequential-Final : "
              f"{rows['stepwise_no_history'].get('no_history_step5_equals_sequential_final')}")
        print(f"  [audit] No-History leakage-free at all steps  : {leak_ok}")
        # expected mock pattern: seq-final conforms to distractor; answer-history resists
        sf = rows["sequential_final"]["final_prediction"]
        nh = rows["stepwise_no_history"]["final_prediction"]
        ah = rows["stepwise_answer_history"]["final_prediction"]
        lm = rows["sequential_final_length_matched"]["final_prediction"]
        print(f"  [pattern] sequential_final={sf}  no_history={nh}  answer_history={ah}  length_matched={lm}")
        if sf != nh:
            print("  !! sequential_final != no_history (identity broken under stub)"); all_ok = False
        if lm != sf:
            print("  !! length_matched != sequential_final (neutral turn changed behavior!)"); all_ok = False
        all_ok = all_ok and leak_ok

    print("\n" + "=" * 96)
    print("SMOKE RESULT:", "ALL AUDITS PASSED ✓" if all_ok else "AUDIT FAILURES ✗")
    print("=" * 96)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
