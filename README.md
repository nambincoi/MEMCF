# MEMCF

**MEMCF** is a memory-augmented recommendation framework that learns from
recommendation failures. Instead of treating wrong recommendations as pure
noise, MEMCF converts them into reusable behavioral knowledge, stores that
knowledge in a shared memory pool, and retrieves it later to improve ranking.

This repository presents MEMCF as a clean, readable research codebase while
preserving the original prompts, memory flow, and evaluation logic from
[`data_process/agent_memcf_v2.py`](/home/hoangnam/Memrec/data_process/agent_memcf_v2.py).

## Abstract

Large language model recommenders can produce plausible but incorrect ranking
choices. MEMCF addresses this by explicitly learning from those failures.
During training, an LLM-based user agent is asked to choose between a ground
truth positive item and a sampled negative item. When the wrong item is chosen,
MEMCF performs reflection, updates user memory and item memory, and distills the
mistake into a cross-user behavioral memory. During inference, the system
retrieves the most relevant behavior memories and injects them into the ranking
prompt. The result is a recommender that does not only imitate user history,
but also accumulates collaborative corrective signals from past failures.

## Method Overview

MEMCF has three core components.

- **User memory**: a mutable natural-language representation of user taste,
  preferences, and recurring behavioral patterns.
- **Item memory**: a mutable natural-language representation of what makes an
  item attractive or unattractive under different user contexts.
- **Behavior memory pool**: a shared collection of fail-interaction memories
  distilled from incorrect recommendations across users.

The method operates in two phases.

### Phase 1: Learn from Fail Interactions

For each user, MEMCF simulates an AgentCF-style pairwise decision.

- Build the current user state from recent history.
- Present one positive item and one negative item to the LLM user agent.
- If the positive item is selected, no corrective memory is created.
- If the negative item is selected, MEMCF:
  - updates the user memory,
  - updates the relevant item memories,
  - creates a structured fail-interaction memory,
  - optionally links that memory to related past failures,
  - optionally evolves linked memories using the new failure.

### Phase 2: Rank with Retrieved Memories

At evaluation time, MEMCF builds a ranking context for each user and candidate
set.

- Retrieve the top-k most relevant behavior memories.
- Construct the ranking prompt with user context, candidate item information,
  and retrieved memories.
- Score the output with Recall@K and NDCG@K.

This repository supports two evaluation variants.

- **`v1`**: recent user history + candidate item information + retrieved
  behavior memories.
- **`v2`**: updated user memory + updated item memory + retrieved behavior
  memories.

## Pipeline Figure

![MEMCF pipeline](assets/pipeline.png)

The figure above captures the same flow implemented in code: simulate failure,
reflect, create memory, retrieve memory, rerank.

## Results Snapshot

![MEMCF results](assets/results.png)

The result summary in this repository is included as a presentation artifact for
MEMCF. Exact numbers depend on dataset preparation, local model behavior,
evaluation candidate construction, and whether saved memories are reused or
retrained.

## Repository as an Implementation Map

The repository is intentionally organized so each file corresponds to one part
of the MEMCF pipeline.

### Top Level

- [`README.md`](/home/hoangnam/Memrec/agent_memcf_v2_repo/README.md)
  - research-style overview, method description, and repository map.
- [`run.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/run.py)
  - convenience entrypoint for running MEMCF from the repo root.
- [`requirements.txt`](/home/hoangnam/Memrec/agent_memcf_v2_repo/requirements.txt)
  - Python dependencies used by the original experiment.
- [`pyproject.toml`](/home/hoangnam/Memrec/agent_memcf_v2_repo/pyproject.toml)
  - packaging metadata for the cleaned repository layout.

### Configuration and Assets

- [`configs/env.example`](/home/hoangnam/Memrec/agent_memcf_v2_repo/configs/env.example)
  - example environment variables for data, memory, and evaluation paths.
- [`assets/pipeline.png`](/home/hoangnam/Memrec/agent_memcf_v2_repo/assets/pipeline.png)
  - MEMCF pipeline illustration.
- [`assets/results.png`](/home/hoangnam/Memrec/agent_memcf_v2_repo/assets/results.png)
  - example results figure used in the README.
- [`scripts/run_video_game.sh`](/home/hoangnam/Memrec/agent_memcf_v2_repo/scripts/run_video_game.sh)
  - example shell wrapper for launching a MEMCF run.

### Source Code

- [`src/agent_memcf/models.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/models.py)
  - core data structures:
  - `UserInteraction`
  - `BehaviorMemory`
  - `AgentCFUserState`
  - `AgentCFItemState`

- [`src/agent_memcf/memory_system.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/memory_system.py)
  - the memory engine of MEMCF:
  - chat model calls,
  - embedding creation,
  - behavior-memory construction,
  - memory linking,
  - memory evolution,
  - memory retrieval,
  - LLM ranking.

- [`src/agent_memcf/training.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/training.py)
  - the fail-interaction learning phase:
  - pairwise autonomous interaction,
  - reflection after wrong choices,
  - conversion of failed interactions into reusable memories.

- [`src/agent_memcf/evaluation.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/evaluation.py)
  - per-user evaluation logic:
  - prompt construction for ranking,
  - `v1` and `v2` evaluation behavior,
  - candidate reranking,
  - metric-ready outputs.

- [`src/agent_memcf/io_utils.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/io_utils.py)
  - input/output and reproducibility helpers:
  - dataset loading,
  - state persistence,
  - result export,
  - Recall@K and NDCG@K computation.

- [`src/agent_memcf/experiment.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/experiment.py)
  - orchestration layer:
  - argument parsing,
  - path resolution,
  - train-or-load control flow,
  - evaluation loop,
  - summary generation.

- [`src/agent_memcf/__main__.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/__main__.py)
  - module entrypoint for `python -m agent_memcf`.

- [`src/agent_memcf/__init__.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/__init__.py)
  - package marker.

### Runtime Directories

- [`data/README.md`](/home/hoangnam/Memrec/agent_memcf_v2_repo/data/README.md)
  - expected dataset layout.
- [`agent_memory/README.md`](/home/hoangnam/Memrec/agent_memcf_v2_repo/agent_memory/README.md)
  - saved global memories and agent states.
- [`evaluation_results/README.md`](/home/hoangnam/Memrec/agent_memcf_v2_repo/evaluation_results/README.md)
  - saved ranking outputs and summary metrics.

## Repository Layout

```text
agent_memcf_v2_repo/
├── README.md                         # research-style presentation of MEMCF
├── pyproject.toml                    # packaging metadata
├── requirements.txt                  # dependencies
├── run.py                            # root entrypoint
├── assets/
│   ├── pipeline.png                  # MEMCF pipeline figure
│   └── results.png                   # MEMCF results figure
├── configs/
│   └── env.example                   # environment variable template
├── scripts/
│   └── run_video_game.sh             # example run script
├── src/
│   └── agent_memcf/
│       ├── __init__.py               # package marker
│       ├── __main__.py               # module entrypoint
│       ├── models.py                 # dataclasses and state objects
│       ├── memory_system.py          # memory creation, retrieval, evolution
│       ├── training.py               # fail-interaction learning
│       ├── evaluation.py             # ranking and metrics logic
│       ├── io_utils.py               # loading, saving, metric helpers
│       └── experiment.py             # top-level experiment orchestration
├── data/
│   └── README.md                     # expected dataset structure
├── agent_memory/
│   └── README.md                     # saved memory/state outputs
└── evaluation_results/
    └── README.md                     # saved ranking/metric outputs
```

## Code Fidelity to the Original Script

This repository is a structural refactor, not an algorithmic rewrite.

Preserved intentionally.

- Prompt texts.
- Role prompts.
- Memory creation logic.
- Memory linking and evolution logic.
- Ranking behavior for `v1` and `v2`.
- State saving/loading semantics.
- Output naming conventions.

Changed intentionally.

- The single large script is split into focused modules.
- Path resolution defaults to this repository root.
- The README and layout are rewritten for readability and presentation.

## Installation

```bash
cd agent_memcf_v2_repo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

By default, MEMCF expects each dataset directory to contain:

```text
data/<DATASET_NAME>/
├── items.json
├── user_sequences_10.json
└── user_negatives_10.json
```

These defaults can be overridden with environment variables.

- `AGENTICREC_REPO_ROOT`
- `AGENTICREC_DATA_ROOT`
- `AGENTICREC_MEMORY_ROOT`
- `AGENTICREC_EVAL_ROOT`

Example:

```bash
source configs/env.example
```

## Running MEMCF

Run from the repo root.

```bash
python run.py \
  --data_name Video_Game \
  --number_of_users 100 \
  --max_iterations 1 \
  --k_memories 1 \
  --eval_variants both
```

Equivalent module-level invocation:

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

## Main Arguments

- `--data_name`: dataset folder under `data/`
- `--number_of_users`: number of users to run
- `--max_iterations`: maximum reflection iterations per fail pair
- `--k_memories`: number of retrieved behavior memories during ranking
- `--eval_variants`: `v1`, `v2`, or `both`
- `--LOAD_SAVED_MEMORY`: load saved memory and agent states instead of retraining
- `--wo_evolving`: disable memory evolution
- `--wo_link`: disable memory linking
- `--fewshot_ranking`: enable few-shot ranking prompts
- `--k_shot`: number of few-shot examples when few-shot ranking is enabled

## Outputs

Runtime artifacts are written to:

- `agent_memory/<DATASET>/`
- `evaluation_results/<DATASET>/`

The main summary file follows the original naming pattern:

```text
evaluation_results/<DATASET>/nuser<...>_fail_interactions_no_evolving_k<...>_iter<...>_memory.summary.json
```

That summary contains:

- baseline metrics
- `variant_metrics.v1`
- `variant_metrics.v2`

## Recommended Reading Order

If you want to understand the code quickly, read in this order.

1. [`README.md`](/home/hoangnam/Memrec/agent_memcf_v2_repo/README.md)
2. [`src/agent_memcf/experiment.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/experiment.py)
3. [`src/agent_memcf/training.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/training.py)
4. [`src/agent_memcf/memory_system.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/memory_system.py)
5. [`src/agent_memcf/evaluation.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/evaluation.py)
6. [`src/agent_memcf/io_utils.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/io_utils.py)
7. [`src/agent_memcf/models.py`](/home/hoangnam/Memrec/agent_memcf_v2_repo/src/agent_memcf/models.py)
