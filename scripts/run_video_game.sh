#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AGENTICREC_REPO_ROOT="$ROOT"
export AGENTICREC_DATA_ROOT="${AGENTICREC_DATA_ROOT:-$ROOT/data}"
export AGENTICREC_MEMORY_ROOT="${AGENTICREC_MEMORY_ROOT:-$ROOT/agent_memory}"
export AGENTICREC_EVAL_ROOT="${AGENTICREC_EVAL_ROOT:-$ROOT/evaluation_results}"

python "$ROOT/run.py" \
  --data_name Video_Game \
  --number_of_users 100 \
  --max_iterations 1 \
  --k_memories 1 \
  --eval_variants both
