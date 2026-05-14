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
| `agent_conformity_project/data/datasets/` | Processed reusable dataset files for CommonsenseQA and MMLU | yes |
| `agent_conformity_project/analysis_outputs/*.md` | Small validation reports for datasets and imported result checks | yes |

## Paper Validation Reports

| Report | Role |
|---|---|
| `agent_conformity_project/analysis_outputs/imported_results_validation_report.md` | Recomputed first500 table checks against locally imported result packages |
| `agent_conformity_project/analysis_outputs/new500_dataset_integrity_report.md` | Confirms new500 dataset structure and non-overlap with first500 |
| `agent_conformity_project/analysis_outputs/exp5_label_attention_gemma2_2b_commonsenseqa500_fixed_summary.md` | Summarizes the Gemma Exp5 social-label framing run |
| `agent_conformity_project/analysis_outputs/mmlu_gemma_first4_integrity_report.md` | Structural integrity diagnosis for the fixed Gemma MMLU first4 package |

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

## Known Tracked Cleanup Candidates

The current history already contains some old or bulky files. They were not removed automatically in this cleanup branch because removing tracked files is a destructive repository decision.

Candidate categories for a later, explicit cleanup PR:

- tracked `.tmp_results1/results/...` JSONL and checkpoint outputs
- tracked `agent_conformity_project/*.tar.gz` result packages
- tracked IDE metadata under `.idea/`

Before removing them, verify whether any paper table still depends on those exact files or whether they have been archived externally.
