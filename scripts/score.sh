#!/usr/bin/env bash
# Score a run directory with the OFFICIAL evaluator, natively.
#
# The challenge ships evaluate.py inside a Docker image bundling an Ollama judge.
# This host has no Docker (see docs/plan.md), so we invoke the same evaluate.py
# directly and point it at our own ground truth and run directory. The scoring
# code is byte-identical to the container's -- only the judge transport differs.
#
#   scripts/score.sh work/run/constant             # deterministic only
#   USE_RATIONALE_JUDGE=1 scripts/score.sh <run>   # + local Ollama judge
#
# Rule reminder: only the official pipeline may be used to report challenge
# performance. Never substitute the fast in-process scorer for these numbers.
set -euo pipefail

RUN_DIR="${1:?usage: scripts/score.sh <run-dir> [task ...]}"
shift || true
TASKS=("$@")
[[ ${#TASKS[@]} -eq 0 ]] && TASKS=(task1 task2 task3)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="${REPO_ROOT}/refs/challenge/evaluation/evaluate.py"

# Defaults to the official ground truth. Override with CHIMERA_GT_ROOT to score
# against the synthetic cohort (work/synth/ground_truth) when smoke-testing the
# metric surface -- those numbers are harness diagnostics, never results.
GT_ROOT="${CHIMERA_GT_ROOT:-${REPO_ROOT}/refs/challenge/evaluation/ground_truth}"

[[ -f "${EVAL_PY}" ]] || { echo "missing ${EVAL_PY}; clone refs first" >&2; exit 1; }

RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

for TASK in "${TASKS[@]}"; do
  OUT="${RUN_DIR}/_scores/${TASK}"
  mkdir -p "${OUT}"
  echo "=== ${TASK} ==="
  TASK_ID="${TASK}" \
  INPUT_DIRECTORY="${RUN_DIR}" \
  PREDICTIONS_FILE="${RUN_DIR}/predictions.json" \
  GROUND_TRUTH_DIR="${GT_ROOT}/${TASK}" \
  SECTION_MAPPING_FILE="${GT_ROOT}/section_variable_mapping.json" \
  EVAL_OUTPUT_DIR="${OUT}" \
  USE_RATIONALE_JUDGE="${USE_RATIONALE_JUDGE:-0}" \
    python "${EVAL_PY}"
done
