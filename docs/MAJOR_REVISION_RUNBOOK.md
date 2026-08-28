# Major-Revision Experiment Runbook

This runbook separates inference-free analysis from new GPU add-ons.  It is
designed to preserve the original main500 results: do not overwrite, force-add,
or rerun the archived four-protocol files unless the authors explicitly decide
that a missing raw artifact cannot be recovered.

## 1. Recover the archived per-item main outputs

For each of the six model-dataset cells, locate these four files:

```text
exp1_all_at_once_final.jsonl
exp2_sequential_context_final_only.jsonl
exp3_sequential_answer_each_step.jsonl
exp4_all_at_once_self_iter5.jsonl
```

Required cells:

```text
Qwen/Qwen2.5-3B-Instruct          CommonsenseQA
Qwen/Qwen2.5-3B-Instruct          MMLU
microsoft/Phi-3.5-mini-instruct   CommonsenseQA
microsoft/Phi-3.5-mini-instruct   MMLU
google/gemma-2-2b-it              CommonsenseQA
google/gemma-2-2b-it              MMLU
```

Do not infer paired counts from aggregate paper tables.  Copy the raw files to
an analysis host, edit a copy of
`agent_conformity_project/analysis/main_paired_manifest.example.json`, and use
absolute paths or paths relative to that manifest.  Raw JSONL is intentionally
ignored by Git; distribute it through the project artifact archive rather than
normal Git history.

Run the common-valid paired analysis:

```bash
python agent_conformity_project/analysis/paired_protocol_analysis.py \
  --manifest /path/to/main_paired_manifest.json \
  --output-dir /path/to/main_paired_analysis \
  --bootstrap-reps 10000 \
  --seed 12345
```

Expected outputs:

```text
main_paired_summary.json
main_paired_comparisons.csv
main_paired_summary.md
```

The command fails on duplicate IDs, missing files, or question/options/gold/
distractor/private-baseline drift.  Deltas are always condition 2 minus
condition 1.  Holm correction is performed separately within each outcome over
all prespecified model-dataset cells and both protocol contrasts.

## 2. Repeated 500-item robustness from cached predictions

This step does not call a model.  The manifest must point to a common attempted
item pool of at least 500 IDs per dataset; 1,000 or more is preferred.  Use the
same manifest as the paired analysis:

```bash
python agent_conformity_project/analysis/repeated_subset_analysis.py \
  --manifest /path/to/main_paired_manifest.json \
  --output-dir /path/to/repeated_subset_analysis \
  --sample-size 500 \
  --repetitions 30 \
  --seed-start 1000
```

Each repetition samples without replacement.  Repetitions may overlap.  One
dataset-level sample is reused across every model and protocol.  The JSON output
records all sampled IDs and seeds.

If only the original first500 and one non-overlapping new500 summary are
available, keep the manuscript wording as a two-subset replication; do not call
it a 30-repetition robustness distribution.

## 3. Cross-setting self-history replication

### Qwen2.5-3B on MMLU first500

```bash
sbatch hpc_self_history_qwen3b_mmlu500.slurm
```

The job checks the tracked full-MMLU MD5, runs only rows 0-499, verifies the
Qwen neutral-control token length, executes all seven conditions, and writes a
10,000-bootstrap analysis JSON.  It creates a new run directory and does not
touch the original CommonsenseQA run.

### Gemma2-2B on CommonsenseQA main500

Confirm the local model path.  If it differs from the default:

```bash
sbatch --export=ALL,MODEL_PATH=/exact/local/gemma-2-2b-it \
  hpc_self_history_gemma2b_cqa500.slurm
```

This job is deliberately staged:

1. run Sequential-Final, No-History, Answer-History, and Answer+Confidence;
2. measure the actual four retained self-history outputs under the Gemma tokenizer;
3. select two distinct neutral phrasings at the observed modal token length;
4. fail before controls if two exact matches cannot be found;
5. replay the exact stage-1 private answers for Neutral-V1, Neutral-V2, and Short-Ack;
6. combine all seven conditions in one paired analysis.

Never remove the neutral-token assertions to make the job pass.

## 4. Clean social-label control

Run once for Qwen and once for Gemma.  The required reference may be the
completed self-history `sequential_final.jsonl` or an archived main protocol
file containing the same 500 item IDs and non-empty `raw_initial_output`.

Qwen example:

```bash
sbatch --export=ALL,\
MODEL_PATH=/exact/local/qwen25_3b_instruct,\
MODEL_NAME_LOCAL=Qwen/Qwen2.5-3B-Instruct,\
REFERENCE_JSONL=/exact/run/sequential_final.jsonl,\
RUN_TAG=qwen3b \
  hpc_clean_social_label.slurm
```

Gemma example:

```bash
sbatch --export=ALL,\
MODEL_PATH=/exact/local/gemma-2-2b-it,\
MODEL_NAME_LOCAL=google/gemma-2-2b-it,\
REFERENCE_JSONL=/exact/gemma/reference.jsonl,\
RUN_TAG=gemma2b \
  hpc_clean_social_label.slurm
```

The primary comparison is `neutral_source_label` versus `model_source_label`.
The preflight selects a generic label whose complete chat prompt is token-count
matched to `Model` on all 500 items.  `unlabelled_numbered` is descriptive only.
The mixed-support legacy condition remains exploratory and must not be used as
label-only causal evidence.

After each model finishes, create a paired-analysis manifest containing:

```json
{
  "comparisons": [["neutral_source_label", "model_source_label"]],
  "cells": [
    {
      "dataset": "CommonsenseQA",
      "model": "MODEL_ID",
      "conditions": {
        "neutral_source_label": "/run/neutral_source_label.jsonl",
        "model_source_label": "/run/model_source_label.jsonl",
        "unlabelled_numbered": "/run/unlabelled_numbered.jsonl"
      }
    }
  ]
}
```

Analyze with `paired_protocol_analysis.py` and the same 10,000-bootstrap/12345
settings.

## 5. Files to return to the manuscript maintainer

For every new run, return:

- Slurm stdout and stderr;
- vLLM server log;
- manifest JSON;
- condition JSONL files or a stable artifact-archive link;
- analysis JSON, CSV, and Markdown;
- neutral-control or neutral-label selection JSON;
- the Git commit hash and exact submitted job ID.

Before handoff, verify:

- expected row count and unique ID count;
- no duplicate IDs;
- dataset MD5;
- exact model ID and local tokenizer path;
- prompt-token preflight success;
- No-History final-prompt identity;
- common-valid paired denominators;
- no writes under the archived `2026-04-29_five_wrong_guidance` result tree.
