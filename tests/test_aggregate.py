"""The ``predictions.json`` job dump.

Two fields in here are load-bearing and fail *silently* when wrong: the inline
``case_id`` (without it the evaluator skips the job and scores the ground-truth
case as a missing candidate worth zero) and the output ``relative_path`` (which
is how the evaluator locates the files we wrote). Both are asserted by driving
the official evaluator's own resolution helpers over our output.
"""

from __future__ import annotations

import json

import pytest

from chimera.contract import spec
from chimera.contract.aggregate import build_job, job_pk, write_predictions_dump
from chimera.contract.io import CaseInputs, write_case_outputs
from chimera.contract.types import DecisionPrediction, Reasoning, RecurrencePrediction


def _case(task: int, case_id: str) -> CaseInputs:
    return CaseInputs(
        task=task,
        case_id=case_id,
        structured_prompt={"age": 66, "psa": 8.1},
        clinical_data={"radiology_report": "PI-RADS 4 lesion, left peripheral zone."},
        neural_representations={"MRI image": [[0.0] * 1024]},
    )


def _pred(task: int):
    if task == 3:
        return RecurrencePrediction(months_to_recurrence=30.0, event=1, free_text="x" * 60)
    decision = "yes" if task == 1 else "active_treatment"
    return DecisionPrediction(
        task=task,
        decision=decision,
        reasoning=Reasoning(
            confidence="clear",
            variable_weights={v: "noted" for v in spec.VARIABLES_BY_TASK[task]},
            reveal_sequence=["radiology_report"],
            free_text="x" * 60,
        ),
    )


def test_case_id_is_inlined_in_the_structured_prompt():
    """The evaluator reads case_id from this inline value; absent means skipped."""
    job = build_job(_case(1, "BX_42"), _pred(1))
    prompt = next(
        i["value"] for i in job["inputs"] if i["socket"]["slug"] == spec.STRUCTURED_PROMPT_SLUG
    )
    assert prompt["case_id"] == "BX_42"


def test_build_job_does_not_mutate_the_caller_prompt():
    case = _case(1, "BX_42")
    build_job(case, _pred(1))
    assert "case_id" not in case.structured_prompt


def test_clinical_data_is_inlined_for_the_rationale_judge():
    """Ground truth no longer carries clinical context; the judge reads it here."""
    job = build_job(_case(2, "T2_07"), _pred(2))
    slug = spec.CLINICAL_SLUG_BY_TASK[2]
    clinical = next(i["value"] for i in job["inputs"] if i["socket"]["slug"] == slug)
    assert clinical["radiology_report"].startswith("PI-RADS 4")


def test_embeddings_are_not_inlined():
    """The evaluator never reads them, and they would bloat the dump."""
    job = build_job(_case(1, "BX_42"), _pred(1))
    reps = next(i for i in job["inputs"] if i["socket"]["slug"] == spec.NEURAL_REP_SLUG)
    assert reps["value"] is None


def test_job_pk_is_stable_and_task_scoped():
    assert job_pk(1, "X") == job_pk(1, "X")
    assert job_pk(1, "X") != job_pk(2, "X")


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "gc-nested"])
def test_evaluator_resolves_every_output_file_we_write(
    tmp_path, official_evaluator, monkeypatch, nested
):
    """End-to-end path resolution, using the evaluator's own helper.

    Grand Challenge nests a job's outputs under ``<job_pk>/output/`` while our
    local runner writes them flat under ``<job_pk>/``. The evaluator accepts
    either, so both layouts are exercised here -- the flat one is what we test
    against locally, the nested one is what actually happens on the platform.
    """
    ev = official_evaluator
    monkeypatch.setattr(ev, "INPUT_DIRECTORY", tmp_path)

    entries = []
    for task, case_id in ((1, "BX_01"), (2, "T2_01"), (3, "T3-006")):
        case, pred = _case(task, case_id), _pred(task)
        case_dir = tmp_path / job_pk(task, case_id)
        write_case_outputs(case_dir / "output" if nested else case_dir, pred)
        entries.append((case, pred))

    dump_path = write_predictions_dump(tmp_path, entries)
    jobs = json.loads(dump_path.read_text())
    assert len(jobs) == 3

    for job in jobs:
        # The interface key is how the evaluator picks the task handler.
        assert ev.get_interface_key(job) in ev.INTERFACE_TASK_ID
        for socket in job["outputs"]:
            location = ev.get_file_location(
                job_pk=job["pk"], values=job["outputs"], slug=socket["socket"]["slug"]
            )
            assert location.is_file(), f"evaluator cannot find {location}"
            # And it must parse as JSON, not merely exist.
            ev.load_json_file(location=location)
