# Protocol-Dependent Conformity in Large Language Models

This repository is the public artifact for the paper **Protocol-Dependent Conformity in Large Language Models** (also referred to as **Social Conformity in LLMs**).

Its purpose is to make the paper's experiment code, dataset construction files, analysis utilities, and reproducibility documentation inspectable. Large raw model outputs, extracted runs, logs, model weights, generated reports, and compressed result packages are intentionally not committed to Git.

## Paper Experiments

The main paper studies whether multiple-choice LLM answers change after exposure to wrong peer signals, and whether that effect depends on the interaction protocol.

The four main protocols are:

- **Exp1 - All-at-once majority**: five wrong peer signals are shown together, followed by one final answer.
- **Exp2 - Sequential, final-only**: wrong peer signals are shown one by one, but only the final answer is observed.
- **Exp3 - Sequential + intermediate commitments**: the model answers after each peer signal, so prior answers can become anchors.
- **Exp4 - All-at-once + self-iteration**: all wrong peer signals are shown first, then the model self-iterates before the final answer.

The paper also includes:

- **Log-probability shift analysis** for correct and target-wrong options.
- **Robustness checks** using non-overlapping 500-example subsets.
- **Exact McNemar tests** for paired protocol comparisons.
- **Exp5 social-label framing** on Gemma-2-2B-it, testing whether explicit peer labels add influence beyond repeated recommendation content.

## Datasets And Models

Datasets:

- CommonsenseQA
- MMLU `all/validation`

Models:

- `Qwen/Qwen2.5-3B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `google/gemma-2-2b-it`

Reusable dataset files are under:

```text
agent_conformity_project/data/datasets/
```

The current repository includes the main 500-example datasets and the non-overlapping new500 dataset files used for robustness checks.

## Metrics

The paper reports:

- **Initial accuracy**: accuracy before peer exposure.
- **Final accuracy**: accuracy after the protocol.
- **Conformity rate (CR)**: among examples that did not initially choose the target distractor, the proportion whose final answer changes to that target distractor, i.e., `count(y0 != d and ya = d) / count(y0 != d)`.
- **Harmful conformity rate (HCR)**: initially correct examples that end at the wrong peer target.
- **Beneficial revision rate (BRR)**: initially wrong examples that end correct.
- **Answer-change rate (ChangeRate)**: fraction of valid examples where final answer differs from initial answer.

Some older code uses related names such as `wrong_conformity_rate`, `distractor_rate`, or `change_rate`; paper-facing summaries can be regenerated locally from the analysis scripts.

## Repository Layout

```text
agent_conformity_project/
  src/                 Core prompt, parsing, model API, and metric helpers
  scripts/             Dataset and experiment runner scripts
  analysis/            Lightweight analysis and integrity-check scripts
  data/datasets/       Reusable processed dataset files
docs/
  REPRODUCIBILITY.md   Artifact-review reproduction notes
  ARTIFACT_MANIFEST.md Public file manifest
```

Large local outputs are ignored by Git:

```text
agent_conformity_project/results/
results/
logs/
runs/
*.jsonl
*.tar.gz
```

## Generated Validation Summaries

Generated Markdown validation reports are not committed to keep the repository lightweight. They can be regenerated locally from the analysis scripts when the corresponding raw result files are available.

## Setup

Create an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The experiment code expects an OpenAI-compatible chat completions endpoint when running model inference. For artifact review, the lightweight checks below do not start vLLM, call remote APIs, download models, or rerun large experiments.

## Safe Sanity Checks

Compile Python files:

```powershell
python -m py_compile agent_conformity_project\analysis\analyze_exp5_label_attention.py
python -m py_compile agent_conformity_project\analysis\check_jsonl_integrity.py
python -m py_compile agent_conformity_project\analysis\exact_mcnemar.py
```

Regenerate the Exp5 summary if the corresponding local result JSONL is available:

```powershell
python agent_conformity_project\analysis\analyze_exp5_label_attention.py
```

Run an exact McNemar test from paired discordant counts:

```powershell
python agent_conformity_project\analysis\exact_mcnemar.py --b 10 --c 35
```

Run the paper-ready item-paired protocol analysis on a standard result directory:

```powershell
python agent_conformity_project\analysis\paired_protocol_analysis.py `
  --input-dir agent_conformity_project\results\2026-04-29_five_wrong_guidance `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --bootstrap-reps 10000 `
  --seed 42 `
  --output-dir agent_conformity_project\analysis_outputs\paired_primary
```

The paired analysis uses common-valid items, requires matching initial answers by
default for initial-state-dependent metrics, and writes condition validity counts,
effect sizes with 95% confidence intervals, exact McNemar tests, Holm-adjusted
p-values, and a per-item inclusion audit.

Analyze whether the protocol effect persists after controlling for the target
distractor's baseline plausibility:

```powershell
python agent_conformity_project\analysis\distractor_plausibility_analysis.py `
  --input-dir agent_conformity_project\results\2026-04-29_five_wrong_guidance `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --reference-condition Batch-Single `
  --bootstrap-reps 10000 `
  --seed 42 `
  --output-dir agent_conformity_project\analysis_outputs\distractor_plausibility
```

This analysis defines plausibility as `initial log p(target) - initial log p(gold)`,
reports logprob coverage without imputing missing options, estimates protocol effects
within model-dataset-specific plausibility quartiles, and fits adjusted logistic models
with item-clustered robust standard errors.

Audit invalid outputs and test sensitivity to the current parser and missing outcomes:

```powershell
python agent_conformity_project\analysis\invalid_parser_sensitivity.py `
  --input-dir agent_conformity_project\results\2026-04-29_five_wrong_guidance `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --bootstrap-reps 10000 `
  --seed 42 `
  --output-dir agent_conformity_project\analysis_outputs\invalid_parser_sensitivity
```

This audit keeps stored common-valid predictions as the primary analysis. It separately
reports current-parser results, deterministic manual-review samples, invalid-as-non-event
results, and worst-case bounds for missing binary outcomes.

Analyze the non-overlapping main500/new500 runs separately and then perform repeated
subsampling from their pooled common-valid items:

```powershell
python agent_conformity_project\analysis\repeated_subset_robustness.py `
  --batch qwen_csqa::main500=C:\path\to\qwen_csqa_main500 `
  --batch qwen_csqa::new500=C:\path\to\qwen_csqa_new500 `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --repetitions 30 `
  --sample-size 500 `
  --bootstrap-reps 10000 `
  --seed 42 `
  --output-dir agent_conformity_project\analysis_outputs\repeated_subset
```

The script rejects duplicate item IDs across batches, records every repetition's seed
and membership, and quantifies overlap between repetitions. Empirical ranges across
overlapping subsets are reported as descriptive robustness ranges, not as confidence
intervals from independent experiments.

Run its regression tests:

```powershell
python -m unittest discover -s agent_conformity_project\tests -p "test_*.py" -v
```

## What Is Not Included

The Git repository intentionally excludes:

- raw model output JSONL files
- extracted run directories
- logs and caches
- model weights
- vLLM caches
- compressed result packages
- paper drafts, slides, private feedback, and internal audit notes

If a reviewer needs raw result artifacts, distribute them through a release archive such as GitHub Releases, Zenodo, OSF, or Hugging Face Datasets, then place them locally under `agent_conformity_project/results/` before rerunning analysis scripts.

## Artifact Scope

This repository is organized as a paper-facing artifact for the protocol-dependent conformity experiments. It keeps runnable code, lightweight processed data, analysis utilities, and reproducibility documentation while excluding bulky local outputs and unrelated legacy artifacts.
