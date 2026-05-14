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

## Paper-Ready Paired Protocol Analysis

The primary comparison script aligns protocol results by item ID and uses the
intersection of valid final answers. Outcomes that depend on the initial state use
a shared baseline: by default, an item is eligible only when both protocol files
contain valid and identical `initial_prediction` values.

For the two primary protocol comparisons in one model-dataset result directory:

```powershell
python agent_conformity_project\analysis\paired_protocol_analysis.py `
  --input-dir agent_conformity_project\results\2026-04-29_five_wrong_guidance `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --bootstrap-reps 10000 `
  --seed 42 `
  --output-dir agent_conformity_project\analysis_outputs\paired_primary
```

For several model-dataset strata in one multiplicity family, repeat `--stratum`:

```powershell
python agent_conformity_project\analysis\paired_protocol_analysis.py `
  --stratum qwen_csqa=C:\path\to\qwen_csqa_results `
  --stratum qwen_mmlu=C:\path\to\qwen_mmlu_results `
  --stratum gemma_csqa=C:\path\to\gemma_csqa_results `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --output-dir agent_conformity_project\analysis_outputs\paired_primary_all
```

For custom ablation filenames, use repeated condition mappings:

```powershell
python agent_conformity_project\analysis\paired_protocol_analysis.py `
  --condition csqa::Sequential-Final=C:\path\to\sequential_final.jsonl `
  --condition csqa::No-History=C:\path\to\no_history.jsonl `
  --condition csqa::Answer-History=C:\path\to\answer_history.jsonl `
  --pair Sequential-Final:No-History `
  --pair No-History:Answer-History `
  --output-dir agent_conformity_project\analysis_outputs\paired_ablation
```

Generated files:

- `condition_quality.csv`: total, valid, invalid, error, and metadata counts by condition.
- `pairwise_results.csv`: rates, Wilson 95% CIs, paired difference bootstrap CIs,
  discordant counts, exact McNemar p-values, and Holm-adjusted p-values.
- `paired_item_audit.csv`: item-level inclusion, baseline, and binary outcome decisions.
- `paired_protocol_report.md`: paper-readable Markdown tables.
- `analysis_metadata.json`: analysis settings, source paths, outcome definitions, and pair order.

The reported difference is always `right condition - left condition`. Holm correction
is applied across all selected strata and comparisons separately within each outcome
by default. Use only pre-specified primary comparisons in the primary analysis; place
additional pairwise comparisons in an explicitly exploratory family.

## Distractor Plausibility Analysis

The main four-protocol runner stores `initial_option_logprobs` for each item. The
plausibility analysis uses these existing values and therefore does not require new
model inference when both the gold and target option logprobs were captured.

The pre-social plausibility margin is:

```text
initial log p(target distractor) - initial log p(gold answer)
```

Run the two primary comparisons with a frozen `Batch-Single` baseline:

```powershell
python agent_conformity_project\analysis\distractor_plausibility_analysis.py `
  --input-dir agent_conformity_project\results\2026-04-29_five_wrong_guidance `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --reference-condition Batch-Single `
  --baseline-condition Batch-Single `
  --bootstrap-reps 10000 `
  --seed 42 `
  --output-dir agent_conformity_project\analysis_outputs\distractor_plausibility
```

As with the paired primary analysis, several model-dataset strata can be supplied in
one invocation with repeated `--stratum NAME=DIR` arguments. Quartiles and margin
z-scores are computed separately within each stratum so that raw logprob scales are
not treated as directly comparable across models and datasets.

Generated files:

- `logprob_coverage.csv`: gold/target logprob availability and analyzable counts.
- `baseline_margin_audit.csv`: one frozen baseline margin and quartile per item.
- `plausibility_item_audit.csv`: condition-level consistency and exclusion reasons.
- `quartile_condition_rates.csv`: condition rates and Wilson CIs within each quartile.
- `paired_quartile_effects.csv`: paired protocol differences, bootstrap CIs, McNemar
  tests, and Holm correction within quartile.
- `plausibility_adjusted_regression.csv`: odds ratios from main-effect and
  protocol-by-margin logistic models.
- `regression_diagnostics.csv`: convergence, cluster counts, parameter counts,
  condition numbers, and separation warnings.
- `distractor_plausibility_report.md`: paper-readable coverage, quartile, and
  regression tables.

The regression uses a balanced condition panel, stratum fixed effects, margin
z-standardization within stratum, and item-clustered sandwich standard errors. Missing
gold or target logprobs are excluded and explicitly reported; they are never replaced
with zero or an arbitrary probability floor. Inspect the regression diagnostics before
using any coefficient with a convergence or separation warning.

## Invalid Output and Parser Sensitivity

Stored predictions remain the primary paper data. The invalid-output audit compares
them against the current `src/parser.py` only as an explicitly labelled sensitivity
analysis; it never silently overwrites the source JSONL files.

```powershell
python agent_conformity_project\analysis\invalid_parser_sensitivity.py `
  --input-dir agent_conformity_project\results\2026-04-29_five_wrong_guidance `
  --pair Batch-Single:Sequential-Final `
  --pair Sequential-Final:Sequential-Stepwise `
  --bootstrap-reps 10000 `
  --seed 42 `
  --sample-per-category 50 `
  --output-dir agent_conformity_project\analysis_outputs\invalid_parser_sensitivity
```

The parser audit distinguishes:

- stored and current parser agree;
- stored prediction is invalid but current parser recovers an answer;
- stored and current parser return different valid answers;
- stored prediction is valid but the current parser rejects the raw text;
- both are invalid;
- raw text is unavailable;
- the inference call is recorded as an execution error;
- several different explicit answer labels occur in the same output.

Generated files:

- `parser_condition_summary.csv`: valid/invalid rates and disagreement categories by
  stratum, condition, and initial/final phase.
- `parser_method_counts.csv`: current parser method frequencies.
- `parser_item_audit.csv`: item-level stored/reparsed decisions and raw-text previews.
- `parser_review_sample.csv`: deterministic samples from every non-agreement category
  for manual adjudication.
- `stored_vs_reparsed_pairwise_results.csv`: full paired statistics under the stored
  policy and the current-parser-when-raw-available sensitivity policy.
- `invalid_missing_data_bounds.csv`: available-case rates, complete-case differences,
  invalid-as-non-event differences, and worst-case paired difference bounds.
- `invalid_parser_sensitivity_report.md`: paper-readable audit and sensitivity tables.

Execution-error final rows remain invalid even if they contain partial raw text. When
raw text is unavailable, the parser sensitivity retains the stored prediction rather
than inventing a replacement. Worst-case bounds allow each missing binary outcome to
be either zero or one and therefore show whether invalid outputs could theoretically
reverse a reported protocol effect.

## Independent-Batch and Repeated-Subset Robustness

Use the main500 and new500 result directories as explicitly named non-overlapping
batches. The analysis first reports each batch separately, then pools common-valid
paired items and repeatedly samples questions without replacement.

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
  --output-dir agent_conformity_project\analysis_outputs\repeated_subset_qwen_csqa
```

Repeat `--batch STRATUM::BATCH=DIR` for every model-dataset stratum. Each batch
directory must contain the standard protocol JSONL filenames. Within a repetition,
the same sampled question IDs are used for both sides of a protocol comparison and
sampling is without replacement. Repetitions may overlap with one another.

Important MMLU detail: `mmlu_new500_all_validation_rows500_999.json` is already
contained within the full `mmlu_all_validation.json`. The main result batch must be
the model outputs for rows 0–499 only. Do not combine outputs for the full 1,531-row
validation file with the rows 500–999 new500 run, because that would duplicate item
IDs. The script rejects any such cross-batch duplicates.

Generated files:

- `batch_condition_quality.csv`: validity counts for every batch and protocol.
- `independent_batch_pairwise_results.csv`: common-valid paired effects, bootstrap
  CIs, exact McNemar tests, and Holm correction separately for main500 and new500.
- `pooled_pair_quality.csv`: eligible pooled item counts, batch composition, and
  exclusion counts.
- `pooled_item_audit.csv`: item-level reasons for inclusion or exclusion from the pool.
- `full_pool_results.csv`: paired metrics on the complete merged pool.
- `repeated_subset_runs.csv`: every outcome estimate for every repetition and seed.
- `repetition_item_membership.csv`: exact question IDs sampled in each repetition.
- `repetition_batch_composition.csv`: main500/new500 composition of each sample.
- `repeated_subset_summary.csv`: mean, SD, minimum, maximum, and empirical 2.5–97.5%
  ranges across repetitions.
- `repetition_overlap_summary.csv`: shared-item counts and Jaccard overlap between
  every pair of repetitions.
- `repeated_subset_report.md`: paper-readable independent-batch and repeated-subset
  tables.

The repeated-subset empirical interval is a descriptive range across overlapping
samples. It must not be called a 95% confidence interval or treated as evidence from
30 independent experiments. The main500/new500 estimates in
`independent_batch_pairwise_results.csv` are the genuine non-overlapping replication
comparison.

## External Artifacts

Large files are intentionally excluded from Git:

- raw model output JSONL files
- extracted `results/runs/` directories
- compressed result packages
- logs and caches
- model weights

For artifact review, distribute those files through an external archive service and keep this repository focused on code, dataset definitions, small reports, and reproducibility instructions.
