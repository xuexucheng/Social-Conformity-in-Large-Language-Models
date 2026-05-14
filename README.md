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

- Qwen2.5-3B-Instruct
- Phi-3.5-mini-instruct
- Gemma-2-2B-it

Reusable dataset files are under:

```text
agent_conformity_project/data/datasets/
```

The current repository includes the main 500-example datasets and the non-overlapping new500 dataset files used for robustness checks.

## Metrics

The paper reports:

- **Initial accuracy**: accuracy before peer exposure.
- **Final accuracy**: accuracy after the protocol.
- **Conformity rate (CR)**: fraction of valid examples where the final answer changes toward the peer target.
- **Harmful conformity rate (HCR)**: initially correct examples that end at the wrong peer target.
- **Beneficial revision rate (BRR)**: initially wrong examples that end correct.
- **Answer-change rate**: fraction of valid examples where final answer differs from initial answer.

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

## Notes On Legacy Files

The repository also contains earlier CROWN-Ace and social-influence utilities. They are retained for provenance and related experiments, but the paper-facing artifact is centered on Exp1-Exp4, Exp5, dataset files, analysis scripts, and reproducibility documentation.
