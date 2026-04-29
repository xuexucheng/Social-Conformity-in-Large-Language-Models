# 2026-04-29 Five Wrong Guidance Experiments

This folder contains a runner for four attack-input experiments using five wrong guidance messages.

## Design

- The number of wrong guides is fixed at `5`.
- Every wrong guide points to the same target wrong option: `item["distractor"]`.
- The model text output is only one option letter; no textual confidence is requested.
- Per-option confidence evidence is recorded from model token logprobs.
- Each answer request asks the model to output exactly one option letter.
- Per-option logprobs are saved as `A/B/C/D/E`; labels missing from `top_logprobs` are stored as `null`.

## Experiments

- `exp1_all_at_once_final`: all five wrong guides in one message; answer once.
- `exp2_sequential_context_final_only`: five wrong guides are separate user messages in one API call; no model call is made before the final answer.
- `exp3_sequential_answer_each_step`: five wrong guides are provided one by one; model answers each time; final answer is step five.
- `exp4_all_at_once_self_iter5`: all five wrong guides in one message; model answers, then self-iterates in the same conversation for five total answers; final answer is iteration five.

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
