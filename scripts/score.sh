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
# Since upstream b0ae4eb the evaluator scores *every* task present in one dump
# in a single pass: TASK_ID is gone, GROUND_TRUTH_DIR is the root holding
# task<N>/, and metrics.json carries a per-task aggregate plus an
# overall_ranking_score. So this no longer loops per task.
#
# Rule reminder: only the official pipeline may be used to report challenge
# performance. Never substitute the fast in-process scorer for these numbers.
set -euo pipefail

RUN_DIR="${1:?usage: scripts/score.sh <run-dir>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="${REPO_ROOT}/refs/challenge/evaluation/evaluate.py"

# Defaults to the official ground truth. Override with CHIMERA_GT_ROOT to score
# against the synthetic cohort (work/synth/ground_truth) when smoke-testing the
# metric surface -- those numbers are harness diagnostics, never results.
GT_ROOT="${CHIMERA_GT_ROOT:-${REPO_ROOT}/refs/challenge/evaluation/ground_truth}"

[[ -f "${EVAL_PY}" ]] || { echo "missing ${EVAL_PY}; clone refs first" >&2; exit 1; }

RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
OUT="${RUN_DIR}/_scores"
mkdir -p "${OUT}"

TMP_DIR="$(mktemp -d -t chimera_score_XXXXXX)"
trap 'rm -rf "${TMP_DIR}"' EXIT

# The evaluator resolves case_id for *file-backed* inputs by joining
# ComponentInterfaceValue PKs against a CSV Grand Challenge exports after
# archive creation, and it exits if that file is absent. Our dump inlines
# case_id in the structured-prompt value, which _case_id_for_job still tries
# first, so an empty map is correct -- but it has to exist. A real one shipped
# with the ground truth wins.
CASE_MAP="${CASE_MAP_FILE:-}"
if [[ -z "${CASE_MAP}" ]]; then
  CASE_MAP="${GT_ROOT}/debug_archive_pks.csv"
  if [[ ! -f "${CASE_MAP}" ]]; then
    CASE_MAP="${TMP_DIR}/debug_archive_pks.csv"
    echo "case_id,structured-prompt_pk" > "${CASE_MAP}"
  fi
fi

# run() now takes the set of predicted cases as the *phase* and refuses the run
# if any of them lacks ground truth. Our cohorts are deliberately larger than
# their label sets -- 195 Task 1 cases against 91 labels -- so the unlabeled
# jobs have to come out of the dump before the evaluator sees it. On Grand
# Challenge every archive case is labeled and this filter removes nothing.
PREDICTIONS="${TMP_DIR}/predictions.json"
python - "${RUN_DIR}/predictions.json" "${GT_ROOT}" "${PREDICTIONS}" <<'PY'
import json, sys
from pathlib import Path

src, gt_root, dst = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
jobs = json.loads(src.read_text())

labeled = {
    case.name
    for task_dir in sorted(gt_root.glob("task*"))
    if task_dir.is_dir()
    for case in task_dir.iterdir()
    if case.is_dir()
}


def case_id(job):
    for sv in job.get("inputs") or []:
        if sv.get("socket", {}).get("slug") == "structured-prompt":
            value = sv.get("value")
            if isinstance(value, dict):
                return value.get("case_id")
    return None


kept = [j for j in jobs if case_id(j) in labeled]
dst.write_text(json.dumps(kept))
dropped = len(jobs) - len(kept)
if dropped:
    print(f"[score.sh] {dropped}/{len(jobs)} jobs have no ground truth; not scored")
PY

CASE_MAP_FILE="${CASE_MAP}" \
INPUT_DIRECTORY="${RUN_DIR}" \
PREDICTIONS_FILE="${PREDICTIONS}" \
GROUND_TRUTH_DIR="${GT_ROOT}" \
SECTION_MAPPING_FILE="${GT_ROOT}/section_variable_mapping.json" \
EVAL_OUTPUT_DIR="${OUT}" \
USE_RATIONALE_JUDGE="${USE_RATIONALE_JUDGE:-0}" \
  python "${EVAL_PY}"

# With the judge off, evaluate.py takes its `rs is None` branch, which prices
# section grounding at 0.175 against the live 0.05 -- a factor of 3.5 on exactly the
# term that penalises weighting an unrevealed variable. The numbers printed above are
# therefore not leaderboard numbers, and a policy change can lose here and win there.
# So always print both. Components stay the evaluator's; only the sum is recomputed.
if [[ "${USE_RATIONALE_JUDGE:-0}" == "0" ]]; then
  python -m chimera.scoring.reprice "${RUN_DIR}"
fi
