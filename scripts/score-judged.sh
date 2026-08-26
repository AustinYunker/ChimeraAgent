#!/usr/bin/env bash
# scripts/score.sh, but with the LLM rationale judge switched on.
#
# The challenge scores `free_text` with a DeepEval GEval rubric driven by a local
# Ollama model. `scripts/score.sh` defaults that off, because the deterministic
# components are the reproducible ones and the judge needs a 9.6 GB model and a
# server. This wrapper supplies both, so the rationale component can actually be
# measured instead of assumed.
#
#   scripts/score-judged.sh work/run/guideline-v3
#
# Three things this arranges that `score.sh` cannot:
#
# * **A judge interpreter.** `evaluate.py` imports `deepeval`, which the chimera
#   env deliberately does not carry -- it pulls a large dependency tree that has
#   no business near the predictors. A separate venv at $JUDGE_VENV holds the
#   evaluator's own `docker/requirements.txt` and nothing else, and goes on PATH
#   only for the duration of this call.
# * **A running Ollama.** Started on demand and left running (models take ~40 s
#   to load; `OLLAMA_KEEP_ALIVE` keeps the weights resident between runs).
# * **The right base URL.** The evaluator defaults to `http://ollama:11434`, the
#   container's service name, which does not resolve here.
#
# Nothing about the scoring maths differs from `score.sh`; this only populates
# `mean_rationale_score` and the 0.20 case-score component that depends on it.
# Measured reproducible on this host -- two runs whose Task 2 and Task 3 text was
# identical scored identically to four decimals over 147 cases -- so an A/B of two
# rationale variants is a clean read. Still never a *parity* signal: that property
# is not guaranteed across a model or Ollama upgrade, and `tests/test_scorer_parity.py`
# and `chimera.cli.score_fast` remain judge-free by design. See docs/judge-setup.md.
set -euo pipefail

RUN_DIR="${1:?usage: scripts/score-judged.sh <run-dir>}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Installed outside the repo: the model store alone is ~9.6 GB and the ollama
# tree ~2.2 GB, and neither should ever be a `git clean` away from deletion.
OLLAMA_ROOT="${OLLAMA_ROOT:-/home/beams0/AYUNKER/opt}"
OLLAMA_BIN="${OLLAMA_BIN:-${OLLAMA_ROOT}/ollama/bin/ollama}"
JUDGE_VENV="${JUDGE_VENV:-${OLLAMA_ROOT}/judgeenv}"

export OLLAMA_MODELS="${OLLAMA_MODELS:-${OLLAMA_ROOT}/ollama-models}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://${OLLAMA_HOST}}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-2h}"
export JUDGE_MODEL="${JUDGE_MODEL:-gemma4:e4b}"

[[ -x "${OLLAMA_BIN}" ]] || {
  echo "no ollama at ${OLLAMA_BIN}; see docs/judge-setup.md" >&2; exit 1; }
[[ -x "${JUDGE_VENV}/bin/python" ]] || {
  echo "no judge venv at ${JUDGE_VENV}; see docs/judge-setup.md" >&2; exit 1; }

if ! curl -sf --max-time 3 "${OLLAMA_BASE_URL}/api/version" >/dev/null; then
  echo "[score-judged] starting ollama at ${OLLAMA_BASE_URL}"
  mkdir -p "${OLLAMA_ROOT}/ollama-logs"
  # setsid so the server outlives this script and the next run reuses the
  # already-resident weights.
  setsid nohup "${OLLAMA_BIN}" serve \
    > "${OLLAMA_ROOT}/ollama-logs/serve.log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do
    curl -sf --max-time 2 "${OLLAMA_BASE_URL}/api/version" >/dev/null && break
    sleep 1
  done
  curl -sf --max-time 2 "${OLLAMA_BASE_URL}/api/version" >/dev/null || {
    echo "ollama did not come up; see ${OLLAMA_ROOT}/ollama-logs/serve.log" >&2
    exit 1
  }
fi

# `evaluate.py` pulls the model itself if it is absent, but it does so with no
# progress output and a 9.6 GB silent wait looks like a hang. Do it here.
if ! "${OLLAMA_BIN}" list | awk 'NR>1 {print $1}' | grep -qx "${JUDGE_MODEL}"; then
  echo "[score-judged] pulling ${JUDGE_MODEL}"
  "${OLLAMA_BIN}" pull "${JUDGE_MODEL}"
fi

PATH="${JUDGE_VENV}/bin:${PATH}" \
USE_RATIONALE_JUDGE=1 \
  "${REPO_ROOT}/scripts/score.sh" "${RUN_DIR}"
