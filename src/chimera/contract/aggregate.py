"""Rebuild the Grand Challenge ``predictions.json`` job dump for local scoring.

The official evaluator does not read a directory of per-case predictions. It
reads a job dump plus one directory per job pk::

    <root>/predictions.json
    <root>/<job_pk>/<relative_path>.json

Two details are load-bearing and cost a silent zero if missed:

* ``case_id`` is read from the job's **inline** ``structured-prompt`` input
  value. Without it the job is skipped with a warning and the ground-truth case
  is reported as a missing candidate with ``case_score = 0``.
* The rationale judge reads the clinical context from the job's **inline**
  clinical-data input value, because the ground truth no longer repeats it. An
  absent value does not fail the run -- it just degrades the judged component.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable

from chimera.contract import spec
from chimera.contract.io import CaseInputs, write_json
from chimera.contract.types import Prediction

# Stable namespace so a case always maps to the same pk across runs, which keeps
# result directories diffable between iterations.
_PK_NAMESPACE = uuid.UUID("6f2a1c7e-9d3b-4a55-8f21-0b7c9e4d1a30")


def job_pk(task: int, case_id: str) -> str:
    return str(uuid.uuid5(_PK_NAMESPACE, f"task{task}/{case_id}"))


def _input_socket(slug: str, relative_path: str, value: Any) -> dict[str, Any]:
    return {
        "socket": {
            "slug": slug,
            "relative_path": relative_path,
            "is_json_kind": True,
            "is_file_kind": False,
        },
        "file": None,
        "image": None,
        "value": value,
    }


def _output_socket(slug: str, relative_path: str) -> dict[str, Any]:
    return {
        "socket": {
            "slug": slug,
            "relative_path": relative_path,
            "is_json_kind": True,
            "is_file_kind": False,
        },
        "file": None,
        "image": None,
        "value": None,
    }


def build_job(case: CaseInputs, pred: Prediction) -> dict[str, Any]:
    """One entry of the job dump, describing a single scored case."""
    task = case.task
    clinical_slug = spec.CLINICAL_SLUG_BY_TASK[task]

    # The prompt must carry case_id -- that is how the evaluator finds ground
    # truth. Copy rather than mutate the caller's dict.
    prompt = dict(case.structured_prompt)
    prompt["case_id"] = case.case_id

    inputs = [
        _input_socket(spec.STRUCTURED_PROMPT_SLUG, "structured-prompt.json", prompt),
        _input_socket(
            spec.NEURAL_REP_SLUG,
            "prostate-modality-level-neural-representations.json",
            # Deliberately not inlined: the evaluator never reads it, and the
            # vectors would bloat the dump by megabytes per case.
            None,
        ),
        _input_socket(clinical_slug, f"{clinical_slug}.json", case.clinical_data),
    ]

    sockets = spec.OUTPUT_SOCKETS[task]
    outputs = [
        _output_socket(*sockets["decision"]),
        _output_socket(*sockets["reasoning"]),
    ]

    return {"pk": job_pk(task, case.case_id), "inputs": inputs, "outputs": outputs}


def write_predictions_dump(
    root: Path, entries: Iterable[tuple[CaseInputs, Prediction]]
) -> Path:
    """Write ``predictions.json`` describing every case already written under ``root``.

    Assumes each case's output files are at ``root/<job_pk>/<relative_path>``,
    which is what :func:`chimera.contract.io.write_case_outputs` produces when
    pointed at ``root / job_pk``.
    """
    jobs = [build_job(case, pred) for case, pred in entries]
    path = root / "predictions.json"
    write_json(path, jobs)
    return path
