#!/usr/bin/env bash
# Run the built submission image over a cohort, the way Grand Challenge does,
# then assert the evaluator would score every case.
#
# This is the only place the image is ever executed. The build host has no
# container runtime -- no Docker, and rootless podman is unavailable because
# /etc/subuid has no entry for the build user -- so the GitHub Actions runner
# does double duty as build machine and test machine. Everything the platform
# imposes is reproduced here, because CI is our only chance to catch it:
#
#   --network none                  no internet, as on the platform
#   --volume <case>:/input:ro       one case per invocation, read-only
#   --volume <out>:/output          the two result sockets
#   --volume <vol>:/tmp             /tmp is a scratch volume, not image space
#   --volume ./model:/opt/ml/model  the separate model-tarball mount
#
#   scripts/smoke_test_image.sh <image-tag> [cases-root] [output-root]
set -euo pipefail

IMAGE="${1:?usage: scripts/smoke_test_image.sh <image-tag> [cases-root] [output-root]}"
CASES="${2:-work/fixtures}"
OUTPUT="${3:-work/smoke}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASES="$(cd "${CASES}" && pwd)"
NOOP_VOLUME="chimera-smoke-tmp"
MODEL_DIR="${REPO_ROOT}/model"

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}"
OUTPUT="$(cd "${OUTPUT}" && pwd)"

# Per-case container logs, kept so the MCP assertion below can read them. They
# are the only channel the platform gives us either, so what is asserted here is
# exactly what we will be able to check in a debug submission's logs.
LOG_DIR="${OUTPUT}/_logs"
mkdir -p "${LOG_DIR}"

# The model mount is empty for C1b -- no weights ship yet -- but the directory
# must exist so the bind mount resolves, and its presence is what the entrypoint
# reports back in the logs.
mkdir -p "${MODEL_DIR}"

docker volume create "${NOOP_VOLUME}" >/dev/null
cleanup() {
  # The container runs as a non-root user, so files it wrote may not be owned by
  # the host user. Fix that from inside before anything on the host reads them.
  docker run --rm --platform=linux/amd64 --quiet \
    --volume "${OUTPUT}":/output --entrypoint /bin/sh "${IMAGE}" \
    -c "chmod -R -f o+rwX /output || true" >/dev/null 2>&1 || true
  docker volume rm "${NOOP_VOLUME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

n=0
while IFS= read -r inputs_file; do
  case_dir="$(dirname "${inputs_file}")"
  rel="${case_dir#"${CASES}"/}"          # task<N>/<case_id>
  out_dir="${OUTPUT}/${rel}"
  mkdir -p -m o+rwX "${out_dir}"

  echo "=+= ${rel}"
  docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --volume "${case_dir}":/input:ro \
    --volume "${out_dir}":/output \
    --volume "${NOOP_VOLUME}":/tmp \
    --volume "${MODEL_DIR}":/opt/ml/model:ro \
    "${IMAGE}" 2>&1 | tee "${LOG_DIR}/$(echo "${rel}" | tr / _).log"
  n=$((n + 1))
done < <(find "${CASES}" -name inputs.json | sort)

if [ "${n}" -eq 0 ]; then
  echo "=+= no cases (inputs.json) found under ${CASES}" >&2
  exit 1
fi

# Hand ownership back before the checker reads the tree.
cleanup
trap - EXIT

# Every case must have reached its evidence over MCP. `inference.py` degrades to
# an in-process read when the stdio transport fails, which keeps a case worth
# scoring but makes the entry's central claim -- that tool access goes through
# the official MCP interface -- false in the shipped artefact. That degradation
# is invisible in the outputs: `check_outputs` passes either way, because a
# DirectStore prediction is a perfectly well-formed one. The logs are the only
# place it shows, so the build fails on them rather than on the sockets.
echo "=+= ran ${n} case(s); asserting every case went through MCP"
degraded=0
for log_file in "${LOG_DIR}"/*.log; do
  case_name="$(basename "${log_file}" .log)"
  if grep -qE "falling back|retrying with a direct read" "${log_file}"; then
    echo "=+= ${case_name}: DEGRADED -- took a non-MCP path" >&2
    grep -nE "falling back|retrying with a direct read" "${log_file}" >&2
    degraded=1
  elif ! grep -q "mcp server" "${log_file}"; then
    echo "=+= ${case_name}: no 'mcp server' line -- the handshake never happened" >&2
    degraded=1
  fi
done
if [ "${degraded}" -ne 0 ]; then
  echo "=+= at least one case did not use MCP; see the logs above" >&2
  exit 1
fi
echo "=+= all ${n} case(s) handshook with the MCP server"

echo "=+= validating result sockets"
python -m chimera.cli.check_outputs --cases "${CASES}" --outputs "${OUTPUT}"
