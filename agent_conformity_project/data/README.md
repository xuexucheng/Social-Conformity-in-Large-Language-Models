# Data Files

This project keeps reusable datasets separate so one experiment dataset does not overwrite another.

- `dataset.json`: active default dataset used by scripts when `DATA_PATH` is not set.
- `datasets/commonsenseqa_five_wrong_guidance_refined_500.json`: reconstructed dataset from the latest five-wrong-guidance run.
- `datasets/mmlu_all_validation.json`: MMLU `all/validation` converted to the project format.

To run experiments on MMLU without changing the default dataset:

```powershell
$env:DATA_PATH="data/datasets/mmlu_all_validation.json"
```

To download or refresh MMLU:

```powershell
python scripts\download_mmlu.py --subject all --split validation
```

Add `--activate` only if you want to overwrite `data/dataset.json` with the downloaded MMLU split.
