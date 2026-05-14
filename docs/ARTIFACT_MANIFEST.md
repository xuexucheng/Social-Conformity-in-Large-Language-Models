# Artifact Manifest

This manifest separates public Git-tracked files from local-only artifacts.

## Public Paper Artifact

| Path | Purpose | Include in Git |
|---|---|---|
| `README.md` | Paper-facing repository overview and setup notes | yes |
| `requirements.txt` | Lightweight Python dependencies for scripts and analysis | yes |
| `.gitignore` | Prevents accidental commits of runs, logs, archives, model files, and caches | yes |
| `docs/REPRODUCIBILITY.md` | Artifact-review reproduction scope and safe commands | yes |
| `docs/ARTIFACT_MANIFEST.md` | Public manifest of included and excluded artifact files | yes |
| `agent_conformity_project/src/` | Core code: prompts, parsing, API wrapper, metrics, and helper logic | yes |
| `agent_conformity_project/scripts/` | Dataset and experiment runner scripts | yes |
| `agent_conformity_project/analysis/` | Lightweight integrity and Exp5 analysis scripts | yes |
| `agent_conformity_project/analysis/exact_mcnemar.py` | Dependency-free exact McNemar test helper for paired counts | yes |
| `agent_conformity_project/analysis/paired_protocol_analysis.py` | Common-valid paired protocol analysis, CIs, McNemar tests, Holm correction, and audit outputs | yes |
| `agent_conformity_project/analysis/distractor_plausibility_analysis.py` | Baseline target-plausibility coverage, quartile effects, and item-clustered adjusted regression | yes |
| `agent_conformity_project/analysis/invalid_parser_sensitivity.py` | Invalid-output audit, current-parser sensitivity, manual-review samples, and worst-case missing-data bounds | yes |
| `agent_conformity_project/analysis/repeated_subset_robustness.py` | Independent-batch replication, duplicate-ID audit, repeated subsampling, and overlap analysis | yes |
| `agent_conformity_project/tests/test_paired_protocol_analysis.py` | Synthetic regression tests for the paired statistical pipeline | yes |
| `agent_conformity_project/tests/test_distractor_plausibility_analysis.py` | Synthetic tests for margin extraction, missing logprobs, quartiles, and adjusted regression | yes |
| `agent_conformity_project/tests/test_invalid_parser_sensitivity.py` | Synthetic tests for parser recovery/disagreement, ambiguity, reanalysis, and missing-output bounds | yes |
| `agent_conformity_project/tests/test_repeated_subset_robustness.py` | Synthetic tests for independent batches, pooled sampling, membership tracking, and repetition overlap | yes |
| `agent_conformity_project/data/datasets/` | Processed reusable dataset files for CommonsenseQA and MMLU | yes |

## Generated Validation Summaries

Generated Markdown validation reports are not committed to keep the repository lightweight. They can be regenerated locally from the analysis scripts when the corresponding raw result files are available.

## Local-Only Artifacts

These files should remain outside normal Git commits:

| Pattern | Reason |
|---|---|
| `agent_conformity_project/results/` | Extracted result runs and packages can be very large |
| `results/` | Local result mirror |
| `runs/` | Extracted or temporary run directories |
| `logs/` | Runtime logs are environment-specific |
| `*.jsonl` | Raw model outputs can be large and may contain prompt transcripts |
| `*.tar`, `*.tar.gz`, `*.zip`, `*.7z` | Result packages should be released externally |
| `*.pt`, `*.pth`, `*.safetensors`, `*.bin` | Model weights are not part of this source artifact |
| `*.docx`, `*.pptx`, `*.pdf` | Manuscript drafts and slides should not be committed |
| generated Markdown validation reports | Intentionally excluded from Git and regenerated locally when needed |

## Current Cleanup Status

Legacy local files and bulky result archives have been removed from the current tracked file tree. Large raw outputs and generated reports are intentionally excluded and should be stored externally if needed.
