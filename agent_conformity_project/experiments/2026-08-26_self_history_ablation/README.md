# Self-History Ablation (journal Major Revision)

Isolates the confound a reviewer flagged between the published **Sequential-Final
(Exp2)** and **Sequential-Stepwise (Exp3)** conditions. Everything is held fixed
(question, options, correct answer, target distractor, five peer opinions, peer
order, system prompt, answer instruction, decoding, parser, model checkpoint)
except one controlled dimension per condition.

This experiment is **fully isolated** from the paper artifact
(`experiments/2026-04-29_five_wrong_guidance/`); it reuses `src/` utilities but
never modifies or overwrites the archived code, results, JSONL, tarballs, or
dataset.

## Audit findings that shaped the design (verified against archived data)

1. **Exp3's intermediate output is `ANSWER: X\nCONFIDENCE: NN` — no reasoning.**
   All 2500 archived step outputs (Qwen CQA main-500) are 24–25 chars; 0 exceptions.
   `USE_CONFIDENCE=1` and `max_tokens=128` ⇒ the protocol never emits rationales.
   → We do **not** use the term "Full-History". The raw-replay condition is named
   **Stepwise-Answer+Confidence-History**. Any Answer-vs-Answer+Confidence effect
   is a *self-reported-confidence* effect, **not** a reasoning effect.

2. **Decoding is deterministic (greedy).** `temperature=0`, `max_tokens=128`,
   `top_logprobs=20`; no `seed`/`top_p`/`do_sample` in the payload. Caveat: vLLM
   greedy can vary at the FP level across server/batch — so we run the
   determinism check *in the same session* (see below).

3. **Main-500 dataset.** CQA main-500 = `data/dataset.json` == 
   `data/datasets/commonsenseqa_five_wrong_guidance_refined_500.json`
   (md5 `cc5699972e9da35dd0fd2581b8bd2598`). `new500` is a *different* sample
   (md5 `f66a37…`) and is **not** used here. Exp2/Exp3 are paired on identical
   500 IDs, same order.

4. **The Exp2↔Exp3 confound is threefold**, not just self-history: Exp2's final
   request has a trailing `FINAL_DECISION_INSTRUCTION` user turn that Exp3 lacks;
   Exp3 carries 4 self-answer turns; and turn count/structure differ (for
   `gemma` the 6 consecutive user turns are also collapsed by
   `normalize_messages_for_model`). The new family removes these confounds.

## Conditions

All stepwise conditions are anchored on the **Exp2 final request** so that
No-History step-5 is byte-identical to Sequential-Final. History turns are
inserted *after* each `Agent i` for i < step; the request always ends with
`FINAL_DECISION_INSTRUCTION`.

| condition | calls/item | what enters context | vs neighbor |
|---|---|---|---|
| `sequential_final` | 1 | peers only (== archived Exp2) | baseline |
| `stepwise_no_history` | 5 | peers only; **self-answers discarded** | step-5 ≡ sequential_final |
| `stepwise_answer_history` | 5 | + prev **`ANSWER: X`** (confidence stripped) | +normalized answers only |
| `stepwise_answer_confidence_history` | 5 | + prev raw **`ANSWER: X\nCONFIDENCE: N`** | +confidence line only |
| `sequential_final_length_matched` | 1 | + 4 fixed **neutral** turns after Agent 1–4 | +turns only; Agent 5 → FINAL_INSTR |
| `sequential_final_neutral_v2` | 1 | + 4 alternative fixed neutral turns | wording-robustness replication |
| `sequential_final_short_ack` | 1 | + 4 `Acknowledged.` turns | assistant-role markers without length matching |

`stepwise_no_history` still calls the model at steps 1–4 (their outputs are saved
for analysis) but never feeds them back — so, for a stateless greedy model,
`No-History-final ≡ Sequential-Final` is a **mathematical identity / harness
check**, not evidence that the discarded calls themselves have a persistent
effect. Because Answer-History vs No-History changes both assistant-role turns
and answer content, the key content test is Answer+Confidence-History vs the
turn- and length-matched neutral condition.

**Neutral turn (Length-Matched).** Qwen2.5-3B (confirmed):
`Acknowledged. Message received. Proceeding to the next message.` — an exact
13-token match to that model's self-history turn, semantically empty (no
stance / no answer / no confidence / no decision steer). Phi / Gemma variants
must be length-checked on their own tokenizers first (`compute_neutral_token_match.py`).

**Neutral-V2** is `Acknowledged. Message received. Continuing to the next
message.`. It changes the main lexical choice while preserving the same control
structure; the supplied Qwen job refuses to proceed unless Neutral-V1 and V2
have equal token counts on the exact local tokenizer. **Short-Ack** is
`Acknowledged.` and is intentionally shorter: it tests whether assistant-role
turn markers alone reproduce the structural effect.

## Comparisons (paired, common-valid subset)

* **A** No-History vs Sequential-Final → harness/determinism check (expect ≡).
* **B** Answer-History vs No-History → combined assistant-turn + answer-content effect.
* **C** Answer+Confidence vs Answer-History → extra effect of self-reported confidence (NOT reasoning).
* **D** Length-Matched vs Sequential-Final → effect of turn/context structure alone.
* **E** Answer+Confidence vs Length-Matched → history content beyond matched turn/context structure (**key mechanism test**).
* **F** Answer-History vs Length-Matched → answer-only diagnostic; interpret cautiously because token length is not matched.
* **G** Neutral-V2 vs Sequential-Final → wording-robust structural effect.
* **H** Short-Ack vs Sequential-Final → short assistant-role-marker effect.
* **I** Neutral-V2 vs Neutral-V1 → sensitivity to neutral wording.
* **J** Answer+Confidence vs Neutral-V2 → key content test under the alternative neutral wording.
* **K** Neutral-V1 vs Short-Ack → added neutral text/length beyond a short role marker.

Report Accuracy, Harmful-Conformity Rate, target-distractor adoption, flip rate,
valid/invalid N, pp differences, exact McNemar, and question-level paired
bootstrap (10,000 resamples, both conditions of a question resampled together).
Holm correction is applied separately within each outcome across the inferential
family B--E and G--K. A is a deterministic harness check, while F and the exact-
token sensitivity analyses are diagnostics; those are excluded from the Holm
family.
Archived Exp2/Exp3 are read directly for the paired comparison; archived Exp3 is
kept only as the "as-published" reference.

## Files

| file | purpose |
|---|---|
| `conditions.py` | single source of truth: all message builders (pure) |
| `run_ablation.py` | runner engine (injectable model call); writes JSONL + manifest |
| `test_prompt_equivalence.py` | static Tests 1–5 + legacy-identity + no-leakage (no model) |
| `smoke_offline.py` | deterministic-stub smoke; prints full auditable traces (no GPU) |
| `show_condition_diffs.py` | exact line-level message diffs between conditions |
| `compute_neutral_token_match.py` | AutoDL: exact tokenizer counts per model family |

## Run

```bash
# static tests + offline smoke (no GPU, run anywhere)
python3 test_prompt_equivalence.py
python3 smoke_offline.py
python3 show_condition_diffs.py

# real run (on a host with the vLLM OpenAI server up) — see AUTODL section
MODEL_NAME=Qwen/Qwen2.5-3B-Instruct \
API_URL=http://127.0.0.1:8000/v1/chat/completions \
DATA_PATH=data/dataset.json \
python3 experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --compute-tokens \
  --tokenizer-path /local/path/to/the/exact/model-tokenizer
```

Output → `results/runs/<run_name>/2026-08-26_self_history_ablation/{condition}.jsonl`
plus `manifest.json` (model, decoding, dataset md5, git commit, conditions,
neutral turn, tokenizer path/class, and neutral-turn token count). When token
auditing is requested, tokenizer loading or tokenization failure is fatal rather
than silently producing null counts. The runner **refuses** to write under the
legacy tree or to overwrite existing files (use `--force` only intentionally).

Resume an interrupted aligned run with:

```bash
python3 experiments/2026-08-26_self_history_ablation/run_ablation.py \
  --run-name <existing-run-name> \
  --resume \
  --compute-tokens \
  --tokenizer-path /same/local/model-tokenizer
```

Resume fails closed if condition files contain duplicate IDs, have unequal ID
sets, or disagree with the existing manifest. This prevents mixing partial
conditions or incompatible decoding/tokenizer settings.

### Qwen 500 add-on controls (the next GPU job)

Do **not** rerun the original five conditions. The add-on job runs only
Neutral-V2 and Short-Ack and replays each item's exact initial answer from the
completed Qwen run. `--reference-jsonl` validates IDs, question, options,
correct answer, distractor, and raw/parsed initial-answer consistency before the
first model request; this avoids a fresh greedy baseline call becoming a hidden
source of drift.

The ready-to-submit job is at repository root:

```bash
sbatch hpc_self_history_qwen3b_neutral_controls.slurm
```

It expects the completed base run at
`results/runs/qwen25_3b_cqa_main500_2542766/2026-08-26_self_history_ablation/`,
checks the dataset MD5 and 500-line reference file, performs the tokenizer
preflight, runs exactly two conditions, then combines the old and new JSONLs in
one analysis. If the cluster checkout or result folder differs, edit only the
path variables at the top of the Slurm file.

Analyze a completed run and write a machine-readable result summary:

```bash
python3 experiments/2026-08-26_self_history_ablation/analyze_ablation.py \
  --run-dir results/runs/<run-name>/2026-08-26_self_history_ablation \
  --condition sequential_final_neutral_v2=results/runs/<control-run>/2026-08-26_self_history_ablation/sequential_final_neutral_v2.jsonl \
  --condition sequential_final_short_ack=results/runs/<control-run>/2026-08-26_self_history_ablation/sequential_final_short_ack.jsonl \
  --bootstrap-reps 10000 \
  --seed 12345 \
  --output-json results/runs/<run-name>/self_history_analysis.json
```

The analyzer rejects duplicate IDs and cross-condition metadata drift, reports
comparisons A--K, audits prompt-token coverage, reruns Comparisons E and J on
the exact final-prompt-token-match subset, and prints stepwise behavioral
trajectories plus option-logprob coverage.

## Reproducibility invariants (asserted by tests)

* `sequential_final` builder == archived Exp2 builder (byte-identical).
* `stepwise_no_history` step-5 messages == `sequential_final` messages.
* `answer_history` − `no_history` = previous `ANSWER: X` turns only.
* `answer_confidence` − `answer_history` = the `CONFIDENCE:` line only.
* `length_matched` − `sequential_final` = 4 neutral turns only.
* Neutral-V2 and Short-Ack change only the text of those 4 assistant turns.
* reference replay skips the initial model call and fails closed on metadata drift.
* decoding locked to `temperature=0, max_tokens=128`.
