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
    "${IMAGE}"
  n=$((n + 1))
done < <(find "${CASES}" -name inputs.json | sort)

if [ "${n}" -eq 0 ]; then
  echo "=+= no cases (inputs.json) found under ${CASES}" >&2
  exit 1
fi

# Hand ownership back before the checker reads the tree.
cleanup
trap - EXIT

echo "=+= ran ${n} case(s); validating result sockets"
python -m chimera.cli.check_outputs --cases "${CASES}" --outputs "${OUTPUT}"
