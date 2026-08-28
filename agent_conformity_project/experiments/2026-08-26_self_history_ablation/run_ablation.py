"""Self-History Ablation runner (journal Major Revision).

Runs the matched ablation family on the SAME 500 items / same decoding / same
parser / same peer signals as the archived paper experiment, changing ONLY the
self-history / turn-structure dimension per condition.

Conditions
----------
  sequential_final                      Condition 1  (== archived Exp2; 1 model call)
  stepwise_no_history                   Condition 2  (5 calls; no self-answer fed back)
  stepwise_answer_history               Condition 3  (5 calls; prev 'ANSWER: X' fed back)
  stepwise_answer_confidence_history    Condition 4  (5 calls; prev raw 'ANSWER: X\\nCONFIDENCE: N' fed back)
  sequential_final_length_matched       Condition 5  (1 call; 4 neutral assistant turns)
  sequential_final_neutral_v2           Wording robustness control (1 call; 4 alternative neutral turns)
  sequential_final_short_ack            Role-marker control (1 call; 4 short acknowledgement turns)

Key guarantees
--------------
* Decoding is byte-for-byte the legacy setup: temperature=0, max_tokens=128,
  top_logprobs=20, same call_llm_with_logprobs, same parser.
* The private baseline answer (`initial_result`) is computed ONCE per item, or
  replayed verbatim from a validated prior JSONL for add-on controls, and is
  shared across conditions so the initial turn is identical everywhere.
* Per item we assert stepwise_no_history's step-5 request is byte-identical to
  the sequential_final request (Section-9 harness-integrity check).
* Output goes to a NEW run dir (run_name + timestamp); the runner REFUSES to
  write anything under the legacy 2026-04-29 tree or to overwrite existing files.

The model call is injected (`answer_fn`) so `smoke_offline.py` can drive the same
`run_item` engine with a deterministic stub instead of a live vLLM server.
"""

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.config import (  # noqa: E402
    DATA_PATH,
    MAX_TOKENS,
    MODEL_NAME,
    RESULT_DIR,
    TEMPERATURE,
)
from src.parser import parse_answer, parse_confidence  # noqa: E402

import conditions as C  # noqa: E402  (local module; single source of truth)

OPTION_LABELS = C.OPTION_LABELS
EXPERIMENT_NAME = "self_history_ablation"
OUTPUT_SUBDIR = "2026-08-26_self_history_ablation"
LEGACY_SUBDIR = "2026-04-29_five_wrong_guidance"  # NEVER write here

ALL_CONDITIONS = [
    "sequential_final",
    "stepwise_no_history",
    "stepwise_answer_history",
    "stepwise_answer_confidence_history",
    "sequential_final_length_matched",
    "sequential_final_neutral_v2",
    "sequential_final_short_ack",
]
STEPWISE_MODES = {
    "stepwise_no_history": C.HISTORY_NONE,
    "stepwise_answer_history": C.HISTORY_ANSWER,
    "stepwise_answer_confidence_history": C.HISTORY_ANSWER_CONFIDENCE,
}


# ---------------------------------------------------------------------------
# Logprob helpers — EXACT copies of legacy run_experiments.py.
# Guarded by test_prompt_equivalence.py::test_legacy_logprob_helper_identity.
# ---------------------------------------------------------------------------
def option_logprobs_from_top_logprobs(top_logprobs):
    logprobs = {label: None for label in OPTION_LABELS}
    for entry in top_logprobs or []:
        token = str(entry.get("token", "")).strip().upper()
        if token not in logprobs or logprobs[token] is not None:
            continue
        value = entry.get("logprob")
        if value is not None:
            logprobs[token] = float(value)
    return logprobs


def option_logprobs_from_answer_token(result, prediction):
    if prediction not in OPTION_LABELS:
        return {label: None for label in OPTION_LABELS}, None
    for entry in result.get("content_logprobs", []) or []:
        token = str(entry.get("token", "")).strip().upper()
        if token != prediction:
            continue
        option_logprobs = option_logprobs_from_top_logprobs(entry.get("top_logprobs", []))
        selected_logprob = entry.get("logprob")
        return (
            option_logprobs,
            float(selected_logprob) if selected_logprob is not None else None,
        )
    return {label: None for label in OPTION_LABELS}, None


def standardize_from_call(call_result):
    """Parse a raw call result (from answer_fn) into the standard answer dict.

    Identical parsing to legacy call_legacy_answer_with_logprobs: strip text,
    parse_answer(text), parse_confidence(text), extract option logprobs at the
    answer token. Works for both the live model and the offline stub (the stub
    simply supplies empty content_logprobs).
    """
    text = (call_result.get("text") or "").strip()
    prediction = parse_answer(text)
    confidence = parse_confidence(text)
    option_logprobs, selected_logprob = option_logprobs_from_answer_token(call_result, prediction)
    return {
        "text": text,
        "prediction": prediction,
        "confidence": confidence,
        "valid": prediction in OPTION_LABELS,
        "option_logprobs": option_logprobs,
        "selected_logprob": selected_logprob,
        "raw": call_result.get("raw", {}),
    }


def make_real_answer_fn():
    """Live model call — identical decoding to the archived paper runs."""
    from src.llm_api import call_llm_with_logprobs  # local import: no server needed to import module

    def answer_fn(messages):
        return call_llm_with_logprobs(
            messages, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_logprobs=20
        )

    return answer_fn


# ---------------------------------------------------------------------------
# Per-item engine (model call injected)
# ---------------------------------------------------------------------------
def _prompt_char_count(messages):
    return sum(len(m.get("content") or "") for m in messages)


def _base_row(
    item,
    condition,
    model_name,
    run_name,
    timestamp,
    control_turn,
    initial_answer_source,
    reference_jsonl,
):
    return {
        "schema_version": "1.0",
        "experiment": EXPERIMENT_NAME,
        "condition": condition,
        "run_name": run_name,
        "timestamp": timestamp,
        "model_name": model_name,
        "dataset_path": DATA_PATH,
        "decoding": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "top_logprobs": 20},
        "neutral_turn": control_turn,
        "initial_answer_source": initial_answer_source,
        "initial_answer_reference_jsonl": reference_jsonl,
        "id": item.get("id"),
        "question": item.get("question"),
        "options": item.get("options"),
        "correct_answer": item.get("correct_answer"),
        "distractor": item.get("distractor"),
        "peer_order": list(range(1, C.N_WRONG_GUIDES + 1)),
    }


def _finalize_row(row, initial_std, final_std, final_messages, step_records, opinions):
    row["agent_opinions"] = list(opinions)
    row["initial_prediction"] = initial_std["prediction"]
    row["initial_confidence"] = initial_std["confidence"]
    row["raw_initial_output"] = initial_std["text"]
    row["final_prediction"] = final_std["prediction"]
    row["final_confidence"] = final_std["confidence"]
    row["raw_final_output"] = final_std["text"]
    row["final_valid"] = final_std["valid"]
    row["option_logprobs"] = final_std["option_logprobs"]
    row["selected_logprob"] = final_std["selected_logprob"]
    row["is_correct"] = final_std["prediction"] == row["correct_answer"]
    row["chose_distractor"] = final_std["prediction"] == row["distractor"]
    row["changed_from_initial"] = final_std["prediction"] != initial_std["prediction"]
    row["step_outputs"] = step_records
    row["final_messages"] = final_messages
    row["final_prompt_char_count"] = _prompt_char_count(final_messages)
    row["final_prompt_token_count"] = None  # filled best-effort below / on AutoDL
    return row


def _run_single_call_condition(build_messages, initial_result, opinions, answer_fn):
    messages = build_messages()
    std = standardize_from_call(answer_fn(messages))
    step = {
        "step": 1,
        "opinion": None,
        "history_inserted_before_step": C.assistant_turns_after_initial(messages),
        "raw_text": std["text"],
        "prediction": std["prediction"],
        "confidence": std["confidence"],
        "valid": std["valid"],
        "option_logprobs": std["option_logprobs"],
        "selected_logprob": std["selected_logprob"],
        "request_messages": messages,
    }
    # For single-call conditions the peer opinions are all present at once; record them.
    return std, messages, [step]


def _run_stepwise_condition(initial_result, opinions, history_mode, answer_fn):
    prior_texts, prior_preds = [], []
    step_records = []
    final_std = None
    final_messages = None
    for step in range(1, len(opinions) + 1):
        messages = C.build_stepwise_request_messages(
            initial_result, opinions, step,
            prior_step_texts=prior_texts,
            prior_step_predictions=prior_preds,
            history_mode=history_mode,
        )
        std = standardize_from_call(answer_fn(messages))
        history_before = C.assistant_turns_after_initial(messages)  # what the model saw as self/neutral history
        step_records.append({
            "step": step,
            "opinion": opinions[step - 1],
            "history_inserted_before_step": history_before,
            "raw_text": std["text"],
            "prediction": std["prediction"],
            "confidence": std["confidence"],
            "valid": std["valid"],
            "option_logprobs": std["option_logprobs"],
            "selected_logprob": std["selected_logprob"],
            "request_messages": messages,
        })
        prior_texts.append(std["text"])
        prior_preds.append(std["prediction"])
        final_std, final_messages = std, messages
    return final_std, final_messages, step_records


def run_item(
    item,
    answer_fn,
    conditions,
    model_name,
    run_name,
    timestamp,
    neutral_turn,
    neutral_v2_turn=None,
    short_ack_turn=None,
    initial_reference=None,
    reference_jsonl=None,
):
    """Run all requested conditions for one item. Returns {condition: row}.

    `answer_fn(messages) -> {text, content_logprobs?, raw?}` is injected.
    """
    neutral_v2_turn = neutral_v2_turn or C.resolve_neutral_v2_turn(model_name)
    short_ack_turn = short_ack_turn or C.resolve_short_ack_turn(model_name)
    initial_messages = C.build_initial_messages(item)
    if initial_reference is None:
        initial_std = standardize_from_call(answer_fn(initial_messages))
        initial_answer_source = "model_call"
    else:
        initial_std = standardize_from_call(
            {"text": initial_reference["raw_initial_output"], "content_logprobs": []}
        )
        initial_answer_source = "reference_jsonl"
    initial_result = {"messages": initial_messages, "text": initial_std["text"]}
    opinions = [item["distractor"] for _ in range(C.N_WRONG_GUIDES)]

    # Reference sequential-final messages, always built for the Section-9 audit
    seqfinal_messages = C.build_sequential_final_messages(initial_result, opinions)

    rows = {}
    for cond in conditions:
        control_turn = {
            "sequential_final_length_matched": neutral_turn,
            "sequential_final_neutral_v2": neutral_v2_turn,
            "sequential_final_short_ack": short_ack_turn,
        }.get(cond)
        row = _base_row(
            item,
            cond,
            model_name,
            run_name,
            timestamp,
            control_turn,
            initial_answer_source,
            reference_jsonl,
        )
        if cond == "sequential_final":
            final_std, final_messages, steps = _run_single_call_condition(
                lambda: seqfinal_messages, initial_result, opinions, answer_fn
            )
            steps[0]["opinion"] = "|".join(opinions)  # all peers present at once
        elif cond in STEPWISE_MODES:
            final_std, final_messages, steps = _run_stepwise_condition(
                initial_result, opinions, STEPWISE_MODES[cond], answer_fn
            )
            if cond == "stepwise_no_history":
                # Harness-integrity: step-5 request MUST equal sequential_final request.
                row["no_history_step5_equals_sequential_final"] = (final_messages == seqfinal_messages)
        elif cond == "sequential_final_length_matched":
            final_std, final_messages, steps = _run_single_call_condition(
                lambda: C.build_length_matched_messages(initial_result, opinions, neutral_turn),
                initial_result, opinions, answer_fn,
            )
            steps[0]["opinion"] = "|".join(opinions)
        elif cond == "sequential_final_neutral_v2":
            final_std, final_messages, steps = _run_single_call_condition(
                lambda: C.build_length_matched_messages(initial_result, opinions, neutral_v2_turn),
                initial_result, opinions, answer_fn,
            )
            steps[0]["opinion"] = "|".join(opinions)
        elif cond == "sequential_final_short_ack":
            final_std, final_messages, steps = _run_single_call_condition(
                lambda: C.build_length_matched_messages(initial_result, opinions, short_ack_turn),
                initial_result, opinions, answer_fn,
            )
            steps[0]["opinion"] = "|".join(opinions)
        else:
            raise ValueError(f"Unknown condition: {cond}")
        _finalize_row(row, initial_std, final_std, final_messages, steps, opinions)
        rows[cond] = row
    return rows


# ---------------------------------------------------------------------------
# Output plumbing
# ---------------------------------------------------------------------------
def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit():
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True
        ).stdout.strip() or None
    except Exception:
        return None


def _load_local_tokenizer(tokenizer_path):
    """Load the exact local tokenizer used for auditable prompt-length counts.

    Token auditing is opt-in, but once requested it is strict: callers must not
    silently continue with null counts because the length-matched control then
    becomes impossible to verify from the result artifact.
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def _prompt_token_count(messages, tokenizer, model_name):
    from src.llm_api import normalize_messages_for_model

    normalized = normalize_messages_for_model(messages, model_name)
    ids = tokenizer.apply_chat_template(
        normalized, tokenize=True, add_generation_prompt=True
    )
    return len(ids)


def _text_token_count(text, tokenizer):
    return len(tokenizer.encode(text or "", add_special_tokens=False))


def _add_token_audit(row, tokenizer, model_name):
    """Attach prompt and inserted-history token counts to one completed row."""
    row["final_prompt_token_count"] = _prompt_token_count(
        row["final_messages"], tokenizer, model_name
    )
    for step in row.get("step_outputs", []):
        history = step.get("history_inserted_before_step") or []
        counts = [_text_token_count(text, tokenizer) for text in history]
        step["history_inserted_token_counts"] = counts
        step["history_inserted_total_token_count"] = sum(counts)
        step["request_prompt_token_count"] = _prompt_token_count(
            step.get("request_messages") or [], tokenizer, model_name
        )


def _read_jsonl_ids(path):
    """Return IDs from a complete JSONL, rejecting corruption and duplicates."""
    ids = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"[FATAL] cannot resume: invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            item_id = row.get("id")
            if item_id is None:
                raise SystemExit(f"[FATAL] cannot resume: missing id in {path}:{line_number}")
            ids.append(item_id)
    if len(ids) != len(set(ids)):
        raise SystemExit(f"[FATAL] cannot resume: duplicate item IDs in {path}")
    return set(ids)


REFERENCE_MATCH_FIELDS = (
    "question",
    "options",
    "correct_answer",
    "distractor",
)


def _load_initial_answer_references(path):
    """Load exact initial answers from a prior condition file, fail closed.

    The add-on controls must inherit the already-run baseline answer verbatim;
    even a greedy re-call can vary at the floating-point/kernel level.  IDs and
    item metadata are checked separately against the current dataset below.
    """
    references = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"[FATAL] invalid reference JSON in {path}:{line_number}: {exc}"
                ) from exc
            item_id = row.get("id")
            if item_id is None:
                raise SystemExit(f"[FATAL] reference row lacks id in {path}:{line_number}")
            if item_id in references:
                raise SystemExit(f"[FATAL] duplicate reference item id {item_id!r} in {path}")
            raw_initial_output = row.get("raw_initial_output")
            if not isinstance(raw_initial_output, str) or not raw_initial_output.strip():
                raise SystemExit(
                    f"[FATAL] reference row {item_id!r} lacks a non-empty raw_initial_output"
                )
            parsed = standardize_from_call(
                {"text": raw_initial_output, "content_logprobs": []}
            )["prediction"]
            recorded = row.get("initial_prediction")
            if recorded is not None and parsed != recorded:
                raise SystemExit(
                    f"[FATAL] reference row {item_id!r} initial answer mismatch: "
                    f"raw parses as {parsed!r}, recorded={recorded!r}"
                )
            references[item_id] = row
    if not references:
        raise SystemExit(f"[FATAL] reference JSONL is empty: {path}")
    return references


def _validate_initial_answer_references(references, dataset):
    """Require one metadata-identical reference row for every selected item."""
    dataset_ids = []
    for item in dataset:
        item_id = item.get("id")
        if item_id is None:
            raise SystemExit("[FATAL] dataset item lacks id")
        dataset_ids.append(item_id)
        reference = references.get(item_id)
        if reference is None:
            raise SystemExit(f"[FATAL] no initial-answer reference for item {item_id!r}")
        mismatches = [
            field for field in REFERENCE_MATCH_FIELDS
            if reference.get(field) != item.get(field)
        ]
        if mismatches:
            raise SystemExit(
                f"[FATAL] reference metadata mismatch for item {item_id!r}: "
                + ", ".join(mismatches)
            )
    if len(dataset_ids) != len(set(dataset_ids)):
        raise SystemExit("[FATAL] duplicate item IDs in selected dataset")


def _resume_completed_ids(out_dir, conditions):
    """Require an aligned checkpoint across conditions and return completed IDs."""
    id_sets = {}
    for cond in conditions:
        path = os.path.join(out_dir, f"{cond}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"[FATAL] cannot resume: missing condition file {path}")
        id_sets[cond] = _read_jsonl_ids(path)
    reference_name = conditions[0]
    reference = id_sets[reference_name]
    for cond in conditions[1:]:
        if id_sets[cond] != reference:
            only_reference = len(reference - id_sets[cond])
            only_condition = len(id_sets[cond] - reference)
            raise SystemExit(
                "[FATAL] cannot resume an unaligned checkpoint: "
                f"{reference_name} vs {cond} differ "
                f"({only_reference} only in reference, {only_condition} only in condition)"
            )
    return reference


def _validate_resume_manifest(existing, expected):
    """Fail closed if a resumed run would mix incompatible configurations."""
    keys = [
        "experiment",
        "output_subdir",
        "run_name",
        "model_name",
        "decoding",
        "conditions",
        "neutral_turn",
        "neutral_v2_turn",
        "short_ack_turn",
        "dataset_md5",
        "n_samples",
        "n_wrong_guides",
        "conditions_module_md5",
        "compute_tokens",
        "tokenizer_path",
        "reference_jsonl",
        "reference_jsonl_md5",
        "initial_answer_source",
    ]
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        details = ", ".join(
            f"{key}: existing={existing.get(key)!r} expected={expected.get(key)!r}"
            for key in mismatches
        )
        raise SystemExit(f"[FATAL] resume manifest mismatch: {details}")


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="Self-History Ablation runner")
    ap.add_argument("--conditions", default=os.getenv("CONDITIONS", ",".join(ALL_CONDITIONS)),
                    help="comma-separated subset of: " + ",".join(ALL_CONDITIONS))
    ap.add_argument("--limit", type=int, default=int(os.getenv("SMOKE_ITEMS", "0")) or None,
                    help="run only the first N items (0/unset = all)")
    ap.add_argument("--run-name", default=os.getenv("RUN_NAME", ""))
    ap.add_argument("--neutral-turn", default=os.getenv("NEUTRAL_TURN", ""))
    ap.add_argument("--neutral-v2-turn", default=os.getenv("NEUTRAL_V2_TURN", ""))
    ap.add_argument("--short-ack-turn", default=os.getenv("SHORT_ACK_TURN", ""))
    ap.add_argument(
        "--reference-jsonl",
        default=os.getenv("REFERENCE_JSONL", ""),
        help=(
            "prior condition JSONL whose raw_initial_output is replayed exactly; "
            "IDs and item metadata must match the selected dataset"
        ),
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", default=os.getenv("FORCE_OVERWRITE", "0") == "1")
    mode.add_argument("--resume", action="store_true", default=os.getenv("RESUME_RUN", "0") == "1")
    ap.add_argument("--compute-tokens", action="store_true",
                    default=os.getenv("COMPUTE_TOKENS", "0") == "1")
    ap.add_argument(
        "--tokenizer-path",
        default=os.getenv("TOKENIZER_PATH", ""),
        help="local tokenizer path used when --compute-tokens is enabled",
    )
    args = ap.parse_args()

    if args.force and args.resume:
        raise SystemExit("[FATAL] --force and --resume are mutually exclusive")
    if args.resume and not args.run_name:
        raise SystemExit("[FATAL] --resume requires an explicit --run-name")

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in ALL_CONDITIONS:
            raise SystemExit(f"[FATAL] unknown condition: {c}")

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = MODEL_NAME.split("/")[-1].lower().replace(".", "-")
    run_name = args.run_name or f"{EXPERIMENT_NAME}_{model_tag}_{timestamp}"
    neutral_turn = C.resolve_neutral_turn(MODEL_NAME, override=args.neutral_turn or None)
    neutral_v2_turn = C.resolve_neutral_v2_turn(
        MODEL_NAME, override=args.neutral_v2_turn or None
    )
    short_ack_turn = C.resolve_short_ack_turn(
        MODEL_NAME, override=args.short_ack_turn or None
    )

    dataset = load_dataset(DATA_PATH)
    if args.limit:
        dataset = dataset[: args.limit]
    dataset_md5 = _md5(DATA_PATH)
    conditions_md5 = _md5(os.path.join(CURRENT_DIR, "conditions.py"))

    reference_jsonl = os.path.abspath(args.reference_jsonl) if args.reference_jsonl else None
    initial_references = None
    reference_jsonl_md5 = None
    if reference_jsonl:
        if not os.path.isfile(reference_jsonl):
            raise SystemExit(f"[FATAL] reference JSONL not found: {reference_jsonl}")
        initial_references = _load_initial_answer_references(reference_jsonl)
        _validate_initial_answer_references(initial_references, dataset)
        reference_jsonl_md5 = _md5(reference_jsonl)

    tokenizer = None
    tokenizer_path = args.tokenizer_path or MODEL_NAME
    if args.compute_tokens:
        try:
            tokenizer = _load_local_tokenizer(tokenizer_path)
        except Exception as exc:  # noqa: BLE001 - fail with an actionable audit message
            raise SystemExit(
                "[FATAL] token auditing was requested but the local tokenizer "
                f"could not be loaded from {tokenizer_path!r}: {type(exc).__name__}: {exc}"
            ) from exc

    neutral_turn_token_count = (
        _text_token_count(neutral_turn, tokenizer) if tokenizer is not None else None
    )
    neutral_v2_turn_token_count = (
        _text_token_count(neutral_v2_turn, tokenizer) if tokenizer is not None else None
    )
    short_ack_turn_token_count = (
        _text_token_count(short_ack_turn, tokenizer) if tokenizer is not None else None
    )
    if "sequential_final_neutral_v2" in conditions:
        if tokenizer is None:
            raise SystemExit(
                "[FATAL] sequential_final_neutral_v2 requires --compute-tokens "
                "and the exact local --tokenizer-path"
            )
        if neutral_v2_turn_token_count != neutral_turn_token_count:
            raise SystemExit(
                "[FATAL] Neutral-V2 is not token-length matched to Neutral-V1: "
                f"{neutral_v2_turn_token_count} vs {neutral_turn_token_count}"
            )

    # --- output dir + hard guards against clobbering the paper artifact -------
    out_dir = os.path.join(RESULT_DIR, "runs", run_name, OUTPUT_SUBDIR)
    if LEGACY_SUBDIR in out_dir:
        raise SystemExit("[FATAL] refusing to write under the legacy experiment tree")
    os.makedirs(out_dir, exist_ok=True)
    if not args.resume:
        for cond in conditions:
            p = os.path.join(out_dir, f"{cond}.jsonl")
            if os.path.exists(p) and not args.force:
                raise SystemExit(f"[FATAL] {p} exists; use --force to overwrite (won't by default)")

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "output_subdir": OUTPUT_SUBDIR,
        "run_name": run_name,
        "timestamp": timestamp,
        "model_name": MODEL_NAME,
        "decoding": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "top_logprobs": 20},
        "conditions": conditions,
        "neutral_turn": neutral_turn,
        "neutral_v2_turn": neutral_v2_turn,
        "short_ack_turn": short_ack_turn,
        "dataset_path": DATA_PATH,
        "dataset_md5": dataset_md5,
        "n_samples": len(dataset),
        "n_wrong_guides": C.N_WRONG_GUIDES,
        "git_commit": _git_commit(),
        "conditions_module_md5": conditions_md5,
        "compute_tokens": bool(args.compute_tokens),
        "tokenizer_path": tokenizer_path if args.compute_tokens else None,
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "reference_jsonl": reference_jsonl,
        "reference_jsonl_md5": reference_jsonl_md5,
        "initial_answer_source": "reference_jsonl" if reference_jsonl else "model_call",
        "neutral_turn_token_count": neutral_turn_token_count,
        "neutral_v2_turn_token_count": neutral_v2_turn_token_count,
        "neutral_v2_token_match_verified": (
            neutral_v2_turn_token_count == neutral_turn_token_count
            if tokenizer is not None
            else None
        ),
        "short_ack_turn_token_count": short_ack_turn_token_count,
        "note": "Archived Exp3 kept as as-published reference; 'Full-History' term retired.",
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    completed_ids = set()
    if args.resume:
        if not os.path.exists(manifest_path):
            raise SystemExit(f"[FATAL] cannot resume: missing manifest {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            existing_manifest = json.load(f)
        # Preserve the original run timestamp and provenance while validating
        # every setting that can change model behavior or output interpretation.
        timestamp = existing_manifest.get("timestamp")
        manifest["timestamp"] = timestamp
        manifest["git_commit"] = existing_manifest.get("git_commit")
        _validate_resume_manifest(existing_manifest, manifest)
        completed_ids = _resume_completed_ids(out_dir, conditions)
        dataset_ids = {item.get("id") for item in dataset}
        unexpected = completed_ids - dataset_ids
        if unexpected:
            raise SystemExit(
                f"[FATAL] cannot resume: {len(unexpected)} completed IDs are absent from the dataset"
            )
        print(f"[RESUME] {len(completed_ids)}/{len(dataset)} aligned items already complete")
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    answer_fn = make_real_answer_fn()
    file_mode = "a" if args.resume else "w"
    handles = {
        c: open(os.path.join(out_dir, f"{c}.jsonl"), file_mode, encoding="utf-8")
        for c in conditions
    }
    n_leak_fail = 0
    n_new = 0
    try:
        for i, item in enumerate(dataset, start=1):
            if item.get("id") in completed_ids:
                continue
            rows = run_item(
                item,
                answer_fn,
                conditions,
                MODEL_NAME,
                run_name,
                timestamp,
                neutral_turn,
                neutral_v2_turn=neutral_v2_turn,
                short_ack_turn=short_ack_turn,
                initial_reference=(initial_references or {}).get(item.get("id")),
                reference_jsonl=reference_jsonl,
            )
            for cond, row in rows.items():
                if cond == "stepwise_no_history" and row.get("no_history_step5_equals_sequential_final") is False:
                    n_leak_fail += 1
                    print(f"[WARN] item {row['id']}: no_history step-5 != sequential_final !!", file=sys.stderr)
                if tokenizer is not None:
                    _add_token_audit(row, tokenizer, MODEL_NAME)
                handles[cond].write(json.dumps(row, ensure_ascii=False) + "\n")
                handles[cond].flush()
            n_new += 1
            total_complete = len(completed_ids) + n_new
            if total_complete % 25 == 0 or total_complete == len(dataset):
                print(f"[OK] {total_complete}/{len(dataset)} items", flush=True)
    finally:
        for h in handles.values():
            h.close()

    print(
        f"[DONE] wrote {n_new} new item(s) across {len(conditions)} condition file(s) "
        f"to {out_dir}"
    )
    if "stepwise_no_history" in conditions:
        print(
            "[AUDIT] no_history step-5 == sequential_final identity failures "
            f"among newly processed items: {n_leak_fail}/{n_new}"
        )


if __name__ == "__main__":
    main()
