# Social Conformity in Large Language Models

This repository studies how social signals change an LLM's answer and implements the `CROWN-Ace (Epistemic Vigilance Edition)` defense pipeline.

The current codebase supports:

- private baseline answering
- wrong-signal social attack settings from Zhu-style group structures
- CROWN-Ace with blind evidence extraction, synthesized re-reasoning, and cognitive audit gating
- prompt-defense and no-defense baselines
- per-experiment result saving for rows and summaries

## Repository Layout

- `agent_conformity_project/src/`
  - core model calling, parsing, metrics, and CROWN-Ace logic
- `agent_conformity_project/scripts/`
  - dataset building and experiment runners
- `agent_conformity_project/results/`
  - latest experiment outputs

## Main CROWN-Ace Files

- Core defense logic: [agent_conformity_project/src/crown_ace.py](C:\Users\21572\IdeaProjects\Social-Conformity-in-Large-Language-Models\agent_conformity_project\src\crown_ace.py)
- Experiment runner: [agent_conformity_project/scripts/run_crown_ace_experiments.py](C:\Users\21572\IdeaProjects\Social-Conformity-in-Large-Language-Models\agent_conformity_project\scripts\run_crown_ace_experiments.py)
- JSON parsing helpers: [agent_conformity_project/src/structured_parser.py](C:\Users\21572\IdeaProjects\Social-Conformity-in-Large-Language-Models\agent_conformity_project\src\structured_parser.py)
- API calls and logprobs: [agent_conformity_project/src/llm_api.py](C:\Users\21572\IdeaProjects\Social-Conformity-in-Large-Language-Models\agent_conformity_project\src\llm_api.py)

## CROWN-Ace Flow

`CROWN-Ace` follows this sequence:

1. `Step 0: Private Baseline`
   - answer independently
   - record `y_init`, reasoning, answer probability `p_init`, and entropy `H_init`
2. `Step 1: Sequential Protocol`
   - social opinions can be injected one-by-one instead of as a monolithic block
3. `Step 2: Blind Evidence Extraction`
   - peer answer labels are hidden
   - only hidden reasoning is exposed for evidence extraction
4. `Step 3: Synthesized Re-Reasoning`
   - recompute a candidate answer from baseline reasoning plus extracted evidence
5. `Step 4: Cognitive Audit Gate`
   - trigger only if candidate answer differs from the baseline
   - compute forward audit, logic echo, reasoning score, counterevidence score, and logic-echo strength
6. `Step 5: Final Decision`
   - accept only when the audit passes
   - otherwise roll back to the private baseline

## Implemented Experiments

The runner executes four experiment families.

### Experiment 1: Sequential Pressure Test

Purpose:
- compare `Single` vs `Sequential` exposure under wrong social pressure

Subconditions:
- `unanimous_wrong`
- `diverse_wrong`
- `devils_advocate_wrong`

Methods:
- `CROWN_ACE_SINGLE`
- `CROWN_ACE_SEQUENTIAL`

Eligibility:
- only samples where the private baseline is initially correct

### Experiment 2: Sequential Correction Test

Purpose:
- compare `Single` vs `Sequential` exposure when peers are unanimously correct

Subconditions:
- `unanimous_right`

Methods:
- `CROWN_ACE_SINGLE`
- `CROWN_ACE_SEQUENTIAL`

Eligibility:
- only samples where the private baseline is initially wrong

### Experiment 3: Algorithm-Only Correction Test

Purpose:
- remove the sequential protocol and compare audit logic under single-input exposure

Subconditions:
- `unanimous_right`

Methods:
- `PD`
- `CROWN_ACE_SINGLE`
- `ROLLBACK`

Eligibility:
- only samples where the private baseline is initially wrong

### Experiment 4: Full-System Comparison

Purpose:
- compare full `ND`, `PD`, and `CROWN-Ace` under Zhu-style wrong-signal settings

Subconditions:
- `unanimous_wrong`
- `diverse_wrong`
- `devils_advocate_wrong`

Methods:
- `ND`
- `PD`
- `CROWN_ACE`

## Zhu-Style Group Definitions

These match the structure shown in the user-provided Zhu summary image.

- `unanimous_wrong`
  - all peers choose the same wrong distractor
- `diverse_wrong`
  - peer answers are distributed across wrong options
- `devils_advocate_wrong`
  - `N-1` peers choose one wrong distractor and `1` peer chooses a different wrong distractor

## Metrics

The runner saves or derives:

- `WCR`
- `CR`
- `SAR`
- `CR_coll`
- `CR_res`
- `DNR`
- `ES`
- `delta_CR`
- `delta_CR_vs_rollback`
- `avg_delta_p`
- `avg_delta_h`
- `avg_reasoning_score`
- `avg_logic_echo_strength`

Notes:

- `p_init` and `p_final` are computed from real `logprobs` returned by the model API.
- `delta_h` is computed from the answer-option distribution, not from text confidence alone.
- the audit gate uses probability gain `delta_p`, not parsed confidence text.

## Environment Requirements

You need:

1. Python 3.11 or compatible
2. a local OpenAI-compatible chat completion API
3. a dataset file at `agent_conformity_project/data/dataset.json`

The default API endpoint is:

```text
http://127.0.0.1:8000/v1/chat/completions
```

The code expects `logprobs` support for answer-option scoring.

## Dataset Format

Create `agent_conformity_project/data/dataset.json` with entries like:

```json
[
  {
    "id": 1,
    "question": "Which planet is known as the Red Planet?",
    "options": {
      "A": "Earth",
      "B": "Mars",
      "C": "Jupiter",
      "D": "Venus",
      "E": "Mercury"
    },
    "correct_answer": "B",
    "distractor": "D"
  }
]
```

If you only have raw data, build the dataset first with:

```powershell
cd agent_conformity_project
python scripts/build_dataset.py
```

The dataset builder reads:

- `data/raw/validation.json`

and writes:

- `data/dataset.json`

## Configuration

Key environment variables are defined in [config.py](C:\Users\21572\IdeaProjects\Social-Conformity-in-Large-Language-Models\agent_conformity_project\src\config.py).

Common ones:

```powershell
$env:API_URL="http://127.0.0.1:8000/v1/chat/completions"
$env:MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
$env:DATA_PATH="data/dataset.json"
$env:RESULT_DIR="results"
$env:N_ATTACK_AGENTS="5"
$env:SEED="42"
```

Audit thresholds:

```powershell
$env:CROWN_ACE_REASONING_THRESHOLD="0.65"
$env:CROWN_ACE_COUNTER_THRESHOLD="0.35"
$env:CROWN_ACE_CONFIDENCE_GAIN_THRESHOLD="0.0"
```

## Running the CROWN-Ace Experiments

From the project directory:

```powershell
cd agent_conformity_project
python scripts/run_crown_ace_experiments.py
```

This runner will:

- load `data/dataset.json`
- compute private baselines
- generate social settings
- run all four experiment families
- save per-experiment rows and summaries
- archive a copy under the run directory

## Output Structure

Latest outputs are written to:

```text
agent_conformity_project/results/crown_ace/
```

Archived outputs are written to:

```text
agent_conformity_project/results/runs/<run_name>/crown_ace/
```

Each experiment directory contains:

- `rows.jsonl`
- `rows.csv`
- `summary.json`
- `summary.csv`
- `metadata.json`

The root experiment directory also contains:

- `overall_summary.json`
- `overall_summary.csv`
- `run_manifest.json`

## Supporting Scripts

Other available scripts in `agent_conformity_project/scripts/`:

- `run_clean.py`
- `run_attack.py`
- `run_defense.py`
- `run_social_influence_mechanisms.py`
- `run_directional_probability_analysis.py`
- `run_semantic_exclusion_test.py`

These are older or parallel experiment utilities. The main script for the current defense study is:

- `run_crown_ace_experiments.py`

## Troubleshooting

- If `run_manifest.json` says `blocked`, check whether `data/dataset.json` exists.
- If API calls fail, verify your local model server is running and supports `logprobs`.
- If you want deterministic `diverse_wrong` sampling, keep `SEED` fixed.

## Current Status

The code is prepared for execution, but no formal experiment results are bundled in the repository by default. You need to provide data and a running model API, then launch the experiment runner.


