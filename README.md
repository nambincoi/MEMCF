# MEMCF

This repository restructures the original experiment script
[`data_process/agent_memcf_v2.py`](/home/hoangnam/Memrec/data_process/agent_memcf_v2.py)
into a cleaner project layout with minimal changes to the underlying logic.

The goal is readability and reproducibility, not a research rewrite. The code
is now split by responsibility instead of leaving the entire implementation in
one file.

## Overview

MEMCF is a memory-augmented recommendation pipeline built around three ideas:

- simulate user choice with an AgentCF-style pairwise interaction between a true item and a sampled false item
- update user memory and item memory when the simulated choice is wrong
- convert failed interactions into a shared cross-user behavior memory pool that can later be retrieved during ranking

In this codebase, evaluation supports two settings:

- `v1`: recent user history + item information + retrieved behavior memories
- `v2`: updated user memory + updated item memory + retrieved behavior memories

## Pipeline

The repository follows the MEMCF pipeline below.

![MEMCF pipeline](assets/pipeline.png)

At a high level, the run is split into two phases:

1. Training from fail interactions
   - initialize a user state from recent history
   - let the user agent choose between a positive item and a negative item
   - if the choice is wrong, reflect and update:
     - user memory
     - item memory
     - global behavior memory
   - optionally link and evolve related behavior memories

2. Inference with retrieved memories
   - build the evaluation context for one user
   - retrieve top-k relevant behavior memories
   - rerank candidates with the LLM
   - compute Recall and NDCG

## How this repository maps to the pipeline

- [`memory_system.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/memory_system.py):
  chat model calls, behavior memory creation, linking, evolution, retrieval, and LLM ranking
- [`models.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/models.py):
  `BehaviorMemory`, `UserInteraction`, `AgentCFUserState`, and `AgentCFItemState`
- [`training.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/training.py):
  AgentCF-style wrong-choice simulation, reflection, and memory creation
- [`evaluation.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/evaluation.py):
  per-user ranking and metrics for `v1` / `v2`
- [`io_utils.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/io_utils.py):
  dataset loading, metric helpers, result export, and agent-state persistence
- [`experiment.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/experiment.py):
  top-level argument parsing, path resolution, train-or-load flow, and evaluation orchestration

## Repository layout

```text
agent_memcf_v2_repo/
├── README.md
├── pyproject.toml
├── requirements.txt
├── run.py
├── assets/
│   ├── pipeline.png
│   └── results.png
├── configs/
│   └── env.example
├── scripts/
│   └── run_video_game.sh
├── src/
│   └── agent_memcf/
│       ├── __init__.py
│       ├── __main__.py
│       ├── evaluation.py
│       ├── experiment.py
│       ├── io_utils.py
│       ├── memory_system.py
│       ├── models.py
│       └── training.py
├── data/
│   └── README.md
├── agent_memory/
│   └── README.md
└── evaluation_results/
    └── README.md
```

## Installation

```bash
cd agent_memcf_v2_repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

By default, the script expects:

```text
data/<DATASET_NAME>/
├── items.json
├── user_sequences_10.json
└── user_negatives_10.json
```

The default paths can be overridden with environment variables:

- `AGENTICREC_REPO_ROOT`
- `AGENTICREC_DATA_ROOT`
- `AGENTICREC_MEMORY_ROOT`
- `AGENTICREC_EVAL_ROOT`

Example:

```bash
source configs/env.example
```

## Run

Main entrypoint:

```bash
python run.py \
  --data_name Video_Game \
  --number_of_users 100 \
  --max_iterations 1 \
  --k_memories 1 \
  --eval_variants both
```

Equivalent direct run of the orchestration layer:

```bash
python src/agent_memcf/experiment.py \
  --data_name Video_Game \
  --number_of_users 100 \
  --max_iterations 1 \
  --k_memories 1 \
  --eval_variants both
```

Example helper script:

```bash
bash scripts/run_video_game.sh
```

## Main options

- `--data_name`: dataset folder under `data/`
- `--number_of_users`: number of users to run
- `--max_iterations`: max AgentCF reflection iterations per pair
- `--k_memories`: number of retrieved behavior memories during ranking
- `--eval_variants`: `v1`, `v2`, or `both`
- `--LOAD_SAVED_MEMORY`: reuse saved global memory and agent states
- `--wo_evolving`: disable memory evolution
- `--wo_link`: disable memory linking
- `--fewshot_ranking`: enable few-shot ranking prompts

## Outputs

Runtime outputs are written to:

- `agent_memory/<DATASET>/`
- `evaluation_results/<DATASET>/`

The main summary file is:

```text
evaluation_results/<DATASET>/nuser<...>_fail_interactions_no_evolving_k<...>_iter<...>_memory.summary.json
```

That summary contains:

- baseline metrics
- `variant_metrics.v1`
- `variant_metrics.v2`

## Results snapshot

The figure below shows an example comparison table across datasets and baselines.

![MEMCF results](assets/results.png)

In the reported examples shown here:

- MEMCF outperforms AgentCF on `Video_Game`
- MEMCF also improves over the listed baselines on `Digital_Music`
- the strongest gains appear in NDCG@10 and NDCG@20 for the shown comparisons

## Notes on code fidelity

- This repository is a structural cleanup of the original script.
- Prompt logic, memory flow, and evaluation behavior are intentionally preserved.
- The main code-level adjustment is path resolution so the project can run from this repo root instead of the old ad hoc layout.
- The split into `models.py`, `memory_system.py`, `training.py`, `evaluation.py`, and `io_utils.py` is organizational only.
