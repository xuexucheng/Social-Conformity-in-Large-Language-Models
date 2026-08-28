# Reproducibility Notes

This document describes what can be reproduced from the public repository and what requires external raw result artifacts.

## Reproduction Scope

The repository supports three levels of checking:

1. **Code inspection**: prompts, parsers, metrics, dataset conversion scripts, and experiment runners are available in `agent_conformity_project/`.
2. **Dataset audit**: the processed CommonsenseQA and MMLU dataset files used by the paper can be inspected directly under `agent_conformity_project/data/datasets/`.
3. **Analysis-layer validation**: analysis scripts can be used to regenerate validation summaries locally.

Full model inference is not rerun by default. Reproducing the entire paper from scratch requires local access to the listed models and an OpenAI-compatible chat-completions server with logprob support.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The project was organized around Python 3.11-compatible scripts. The lightweight audit scripts use standard Python plus the dependencies listed in `requirements.txt`.

## Datasets

Tracked dataset files:

- `agent_conformity_project/data/dataset.json`
- `agent_conformity_project/data/datasets/commonsenseqa_five_wrong_guidance_refined_500.json`
- `agent_conformity_project/data/datasets/commonsenseqa_new500_validation_rows500_999_seed42.json`
- `agent_conformity_project/data/datasets/mmlu_all_validation.json`
- `agent_conformity_project/data/datasets/mmlu_new500_all_validation_rows500_999.json`

## Main Protocols

The paper-facing protocol names are:

- Exp1: all-at-once majority
- Exp2: sequential, final-only
- Exp3: sequential + intermediate commitments
- Exp4: all-at-once + self-iteration

The local result filenames use matching implementation names:

- `exp1_all_at_once_final.jsonl`
- `exp2_sequential_context_final_only.jsonl`
- `exp3_sequential_answer_each_step.jsonl`
- `exp4_all_at_once_self_iter5.jsonl`

Raw JSONL files are not tracked in Git. If external result artifacts are available, unpack or copy them under:

```text
agent_conformity_project/results/
```

## Safe Checks

These commands do not run models, start vLLM, or download data:

```powershell
python -m py_compile agent_conformity_project\analysis\analyze_exp5_label_attention.py
python -m py_compile agent_conformity_project\analysis\check_jsonl_integrity.py
python -m py_compile agent_conformity_project\analysis\exact_mcnemar.py
```

If the Exp5 JSONL is present locally, regenerate its summary with:

```powershell
python agent_conformity_project\analysis\analyze_exp5_label_attention.py
```

Generated Markdown validation reports are not committed to keep the repository lightweight. They can be regenerated locally from the analysis scripts when the corresponding raw result files are available.

To compute an exact two-sided McNemar test from paired discordant counts:

```powershell
python agent_conformity_project\analysis\exact_mcnemar.py --b 10 --c 35
```

Here `b` and `c` are the two off-diagonal discordant counts for a paired 2x2 comparison.

## Paper-Facing Paired and Repeated-Subset Analysis

Once archived per-item main outputs are restored locally, use:

```powershell
python agent_conformity_project\analysis\paired_protocol_analysis.py `
  --manifest C:\path\to\main_paired_manifest.json `
  --output-dir C:\path\to\main_paired_analysis `
  --bootstrap-reps 10000 `
  --seed 12345

python agent_conformity_project\analysis\repeated_subset_analysis.py `
  --manifest C:\path\to\main_paired_manifest.json `
  --output-dir C:\path\to\repeated_subset_analysis `
  --sample-size 500 `
  --repetitions 30 `
  --seed-start 1000
```

The detailed GPU and analysis handoff procedure is in
`docs/MAJOR_REVISION_RUNBOOK.md`; exact prompt templates and parsing order are
in `docs/PROMPT_APPENDIX.md`.

## Verified Qwen Mechanism-Ablation Environment

The retained stdout and vLLM logs for the completed Qwen2.5-3B
CommonsenseQA500 self-history run record:

- NVIDIA GeForce RTX 4090, 24,564 MiB;
- NVIDIA driver 565.77 and reported CUDA 12.7;
- Python 3.10.21;
- PyTorch 2.5.1+cu124;
- vLLM 0.6.6.post1;
- Transformers 4.57.6;
- vLLM model dtype resolved to `torch.bfloat16`;
- maximum model length 4096;
- vLLM server seed 0;
- GPU memory utilization setting 0.85;
- default model chat template (`chat_template=None`, auto-detected string format).

The request payload specifies `temperature=0`, `max_tokens=128`,
`logprobs=true`, and `top_logprobs=20`.  It does not explicitly send `top_p`,
`do_sample`, or a request-level seed.  This distinction must be preserved in the
paper rather than reporting settings that were not present in the request.

## External Artifacts

Large files are intentionally excluded from Git:

- raw model output JSONL files
- extracted `results/runs/` directories
- compressed result packages
- logs and caches
- model weights

For artifact review, distribute those files through an external archive service and keep this repository focused on code, dataset definitions, small reports, and reproducibility instructions.
