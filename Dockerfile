# Grand Challenge submission image for the CHIMERA-agent entry.
#
# Deliberately tiny. The payload is `GuidelinePredictor` (see inference.py), with
# `ConstantPredictor` as the fallback when a case defeats the normal route. Its
# entire runtime import closure is the Python standard library -- no numpy, no
# scikit-learn, no pydantic, no torch -- so there is nothing to gain from the
# CUDA base image the eventual LLM pipeline will need. It builds in about a
# minute, which keeps the one path we exercise least often (see below) as short
# and as reliable as possible.
#
# The size is measured, never asserted here: the `build-image` workflow prints
# it, and README.md records the figure for the last tagged image. Do not restate
# a number in this file -- the build host cannot run a container (see below), so
# anything written here would be a guess that outlives the build it described.
#
# The build host has no Docker. Rootless podman is installed and does start, so
# "no container runtime" overstated it -- but it maps a single UID, because
# /etc/subuid has no entry for the build user, and that is not enough to unpack
# the base image, let alone run this file. Measured, not assumed:
#
#   ApplyLayer ... potentially insufficient UIDs or GIDs available in user
#   namespace (requested 0:42 for /etc/gshadow) ... lchown: invalid argument
#
# It fails on `FROM`, before any instruction below is reached. The authoritative
# build is therefore in GitHub Actions, which also smoke-tests the image against
# the platform contract: see .github/workflows/build-image.yml. That workflow
# triggers on a `v*` tag or by workflow_dispatch.
#
# Platform contract, mirrored by .github/workflows/build-image.yml:
#   /input          read-only, one case, flat sockets described by inputs.json
#   /output         the two result sockets, written flat
#   /opt/ml/model   read-only mount for the separately-uploaded model tarball
#   no network, /tmp is a scratch volume, process runs as a non-root user
FROM --platform=linux/amd64 python:3.11-slim

# Unbuffered so the platform's log capture sees output as it happens rather than
# only on exit -- the container logs are the only diagnostic channel we get.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

COPY --chown=user:user pyproject.toml LICENSE NOTICE /opt/app/
COPY --chown=user:user src/ /opt/app/src/
COPY --chown=user:user inference.py /opt/app/

# --no-deps is deliberate, not an oversight: pyproject.toml declares numpy,
# scikit-learn and pydantic because training and the fast scorer need them, and
# none of them is imported by the entrypoint. tests/test_entrypoint.py pins that
# invariant, so if it ever stops holding the suite fails before the image does.
RUN python3 -m pip install --no-deps .

USER user

ENTRYPOINT ["python3", "inference.py"]
