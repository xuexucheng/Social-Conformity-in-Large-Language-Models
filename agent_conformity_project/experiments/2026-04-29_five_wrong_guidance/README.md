# Four Protocol Experiments with Five Wrong Peer Signals

This folder contains a runner for the four protocol experiments in which each question is paired with five controlled wrong peer opinions.

## Design

- The number of wrong peer opinions is fixed at `5`.
- Every wrong peer opinion points to the same target distractor: `item["distractor"]`.
- The model text output is only one option letter; no textual confidence is requested.
- Per-option confidence evidence is recorded from model token logprobs.
- Each answer request asks the model to output exactly one option letter.
- Per-option logprobs are saved as `A/B/C/D/E`; labels missing from `top_logprobs` are stored as `null`.

## Experiments

- `exp1_all_at_once_final`: Exp1, all-at-once majority; all five wrong peer opinions are shown together, followed by one answer.
- `exp2_sequential_context_final_only`: Exp2, sequential final-only; five wrong peer opinions are shown as separate user messages in one API call, with no model call before the final answer.
- `exp3_sequential_answer_each_step`: Exp3, sequential with stepwise output; wrong peer opinions are shown one by one, the model answers after each step, and the final prediction is the step-five answer.
- `exp4_all_at_once_self_iter5`: Exp4, all-at-once with self-iteration; all five wrong peer opinions are shown first, then the model self-iterates for five total answers, with the fifth answer used as the final prediction.

## Run

From `agent_conformity_project`:

```powershell
python experiments\2026-04-29_five_wrong_guidance\run_experiments.py
```

The runner writes four independent `.jsonl` files and `metadata.json` to:

```text
results/2026-04-29_five_wrong_guidance/
```

It also archives the same result files under the current `RUN_DIR`:

```text
results/runs/<RUN_NAME>/2026-04-29_five_wrong_guidance/
```
