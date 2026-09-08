# Protocol-Dependent Conformity in Open-Weight Large Language Models under Fixed Wrong-Peer Signals

This repository is the public artifact for the paper **Protocol-Dependent Conformity in Open-Weight Large Language Models under Fixed Wrong-Peer Signals**.

Its purpose is to make the paper's experiment code, dataset construction files, analysis utilities, and reproducibility documentation inspectable. Large raw model outputs, extracted runs, logs, model weights, generated reports, and compressed result packages are intentionally not committed to Git.

## Paper Experiments

The main paper studies whether multiple-choice LLM answers change after exposure to wrong peer signals, and whether that effect depends on the interaction protocol.

The four main protocols are:

- **Exp1 - Batch-Single**: five fixed wrong-peer signals are shown in one user turn, followed by one post-exposure response.
- **Exp2 - Sequential-Final**: the same five wrong-peer signals arrive in successive user turns, with one response only after the final peer.
- **Exp3 - Sequential-Stepwise**: the model responds after each peer signal and prior assistant responses remain in conversational history.
- **Exp4 - Batch-Self-Iterative**: all five wrong-peer signals arrive first, followed by five self-revision iterations; iteration 5 is the final analyzed response.

The revision additionally includes:

- **Retained-history decomposition** with no-history, answer-history, answer-plus-confidence, token-matched neutral-history, and short-acknowledgement controls.
- **Source-versus-Model lexical control** for Qwen2.5-3B and Gemma2-2B on both CommonsenseQA and MMLU.
- **Paired inference** for Exp1-Exp2 and Exp2-Exp3 using bootstrap confidence intervals, exact McNemar tests, and Holm correction.
- **Repeated aligned 500-item subset analysis** across 30 draws.
- **Target-distractor plausibility analysis** based on pre-exposure option scores.
- **Targeted exploratory generalization probes** involving Qwen2.5-3B versus Qwen2.5-7B checkpoints, 1/3/5 wrong-peer settings, and fixed generic-justification augmentation.
- **Option-level log-probability analysis** for comparable Qwen and Gemma cells; Phi is excluded from the cross-protocol option-level comparison because archived top-20 truncation yields protocol-dependent missingness.

## Datasets And Models

Datasets:

- CommonsenseQA
- MMLU `all/validation`

Models:

- `Qwen/Qwen2.5-3B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`
- `google/gemma-2-2b-it`
- `Qwen/Qwen2.5-7B-Instruct` (targeted exploratory checkpoint comparison)

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
  experiments/         Main protocols and reviewer-requested controlled add-ons
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
python -m py_compile agent_conformity_project\analysis\check_jsonl_integrity.py
python -m py_compile agent_conformity_project\analysis\exact_mcnemar.py
```

Run an exact McNemar test from paired discordant counts:

```powershell
python agent_conformity_project\analysis\exact_mcnemar.py --b 10 --c 35
```

Prepare paper-facing paired and repeated-subset statistics after restoring the
archived per-item result JSONL files:

```powershell
python agent_conformity_project\analysis\paired_protocol_analysis.py --help
python agent_conformity_project\analysis\repeated_subset_analysis.py --help
```

The full colleague handoff sequence is documented in
`docs/MAJOR_REVISION_RUNBOOK.md`.

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
