# Data Files

Reusable processed datasets live under `agent_conformity_project/data/`. The files use a common multiple-choice schema for the protocol experiments.

## Sources

- **CommonsenseQA**: validation examples converted to the project schema.
- **MMLU**: `all/validation` converted to the project schema.

The full raw benchmark releases are not vendored in this repository. The tracked files are lightweight processed datasets and metadata needed for code inspection and analysis-level reproduction.

## Files

| Path | Dataset | Use |
|---|---|---|
| `dataset.json` | CommonsenseQA | Default active dataset used when `DATA_PATH` is not set. |
| `datasets/commonsenseqa_five_wrong_guidance_refined_500.json` | CommonsenseQA | Main CommonsenseQA first500 evaluation subset. |
| `datasets/commonsenseqa_new500_validation_rows500_999_seed42.json` | CommonsenseQA | Non-overlapping new500 robustness subset. |
| `datasets/mmlu_all_validation.json` | MMLU | Converted MMLU `all/validation`; rows 0-499 correspond to the main first500 MMLU subset. |
| `datasets/mmlu_new500_all_validation_rows500_999.json` | MMLU | Non-overlapping new500 robustness subset. |

## Subsets

- **first500**: the primary 500-example evaluation subset for the main protocol comparisons.
- **new500**: a second 500-example subset from later validation rows, used as a non-overlapping robustness check.

## Schema

Each row is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `id` | Stable sample identifier. |
| `source` | Dataset/source tag used by the project. |
| `subject` | MMLU subject name, when applicable. |
| `question` | Multiple-choice question text. |
| `options` | Mapping from answer labels to option text, usually `A`-`D` for MMLU and `A`-`E` for CommonsenseQA. |
| `correct_answer` | Gold answer label. |
| `distractor` | Target wrong option used for controlled peer signals. |

The `distractor` field is the target option that wrong peer agents endorse in the conformity protocols. It is controlled to be different from `correct_answer`.

## Using A Different Dataset

To run scripts on MMLU without changing the default dataset:

```powershell
$env:DATA_PATH="data/datasets/mmlu_all_validation.json"
```

To download or refresh MMLU from the source benchmark:

```powershell
python scripts\download_mmlu.py --subject all --split validation
```

Add `--activate` only if you want to overwrite `data/dataset.json` with the downloaded MMLU split.
