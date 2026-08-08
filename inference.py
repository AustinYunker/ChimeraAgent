"""Grand Challenge algorithm entrypoint.

The platform runs this container once per case: input sockets are mounted flat
and read-only at ``/input`` (described by ``/input/inputs.json``), and the two
result sockets are expected as flat JSON files under ``/output``. There is no
network, ``/tmp`` is a scratch volume, and any tarball uploaded alongside the
algorithm appears read-only at ``/opt/ml/model``.

All three paths are environment-overridable so the entrypoint can be exercised
natively on a development host that has no container runtime at all -- which is
the situation on our build machine, where the image can only ever be run inside
CI.

This is written against our own :mod:`chimera.contract` implementation. The
reference baseline's ``inference.py`` is unlicensed and therefore not
redistributable, so nothing here derives from it; the socket layout it also
targets is a published interface, not borrowed code.

**This must not raise.** A crashed case is not skipped by the evaluator -- it is
scored against a sentinel label, which costs the true class its recall. Every
failure path below degrades to a valid, conservative prediction instead.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from chimera.contract.io import read_case, socket_paths, write_case_outputs
from chimera.contract.io import detect_task
from chimera.contract.types import Prediction
from chimera.predictors import ConstantPredictor, GuidelinePredictor

INPUT_PATH = Path(os.environ.get("CHIMERA_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("CHIMERA_OUTPUT", "/output"))
MODEL_PATH = Path(os.environ.get("CHIMERA_MODEL", "/opt/ml/model"))

log = logging.getLogger("chimera.inference")


def log_environment() -> None:
    """Report what the platform actually gave us.

    The first validation submission exists to learn about the runtime, and the
    container logs are the only channel through which it can tell us anything.
    """
    log.info("python           : %s", sys.version.split()[0])
    log.info("input            : %s (exists=%s)", INPUT_PATH, INPUT_PATH.is_dir())
    log.info("output           : %s (exists=%s)", OUTPUT_PATH, OUTPUT_PATH.is_dir())
    log.info("model mount      : %s (exists=%s)", MODEL_PATH, MODEL_PATH.is_dir())
    if INPUT_PATH.is_dir():
        log.info("input contents   : %s", sorted(p.name for p in INPUT_PATH.iterdir()))


def fallback_prediction(task: int) -> Prediction:
    """A schema-valid prediction for ``task`` when the real path has failed.

    Deliberately the plain :class:`ConstantPredictor` rather than the fitted
    prior: if we are here, something about this case defeated the normal route,
    and the fallback should depend on as little as possible -- in particular not
    on the parameter file or on anything read from the case.
    """
    from chimera.contract.io import CaseInputs

    empty = CaseInputs(
        task=task,
        case_id="gc-case",
        structured_prompt={},
        clinical_data={},
        neural_representations={},
    )
    return ConstantPredictor().predict(empty)


def detect_task_safely() -> int | None:
    """Identify the interface without reading any payload, or ``None``."""
    try:
        return detect_task(socket_paths(INPUT_PATH))
    except Exception:
        log.exception("could not determine the task from %s", INPUT_PATH / "inputs.json")
        return None


def run() -> int:
    log_environment()

    task = detect_task_safely()
    if task is None:
        # Without the interface we cannot even choose which sockets to write.
        # Nothing valid can be emitted, so fail loudly rather than silently.
        log.error("no recognisable input interface; writing nothing")
        return 1
    log.info("task             : %d", task)

    try:
        case = read_case(INPUT_PATH)
        prediction = GuidelinePredictor().predict(case)
        log.info("case_id          : %s", case.case_id)
    except Exception:
        log.exception("prediction failed; falling back to the constant predictor")
        prediction = fallback_prediction(task)

    try:
        written = write_case_outputs(OUTPUT_PATH, prediction)
    except Exception:
        # The prediction itself was rejected by contract validation. Fall back
        # once more; if that also fails the container has a real defect.
        log.exception("writing the prediction failed; retrying with the fallback")
        written = write_case_outputs(OUTPUT_PATH, fallback_prediction(task))

    for role, path in written.items():
        log.info("wrote %-9s : %s", role, path)
    return 0


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
