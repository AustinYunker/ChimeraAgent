# Grand Challenge submission image for the CHIMERA-agent entry.
#
# Deliberately tiny. The C1b payload is the fitted constant prior, whose entire
# runtime import closure is the Python standard library -- no numpy, no
# scikit-learn, no pydantic, no torch -- so there is nothing to gain from the
# CUDA base image the eventual LLM pipeline will need. ~150 MB builds in about a
# minute, which keeps the one path we cannot exercise locally (see below) as
# short and as reliable as possible.
#
# The build host has no container runtime at all: no Docker, and rootless podman
# is unavailable because /etc/subuid has no entry for the build user. This image
# is therefore built *and smoke-tested* in GitHub Actions -- see
# .github/workflows/build-image.yml -- and never on a developer machine.
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
