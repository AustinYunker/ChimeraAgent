"""Contract conformance.

These are the cheapest tests in the repo and they guard the most expensive
failure mode: a submission that runs perfectly and scores zero because a
filename, a JSON shape, or an enum token was wrong. Everything here is asserted
against the official ``evaluate.py`` where possible rather than against our own
reading of it -- see :mod:`test_spec_matches_evaluator`.
"""

from __future__ import annotations

import json

import pytest

from chimera.contract import spec
from chimera.contract.io import detect_task, read_case, write_case_outputs
from chimera.contract.types import (
    ContractError,
    DecisionPrediction,
    Reasoning,
    RecurrencePrediction,
    validate,
)


def _reasoning(task: int, **overrides) -> Reasoning:
    kwargs = {
        "confidence": "clear",
        "variable_weights": {v: "noted" for v in spec.VARIABLES_BY_TASK[task]},
        "reveal_sequence": ["radiology_report"],
        "free_text": "A sufficiently specific rationale referencing the case.",
    }
    kwargs.update(overrides)
    return Reasoning(**kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Output shapes on disk
# --------------------------------------------------------------------------- #

def test_task1_writes_both_spellings_of_the_biopsy_socket(tmp_path):
    """Task 1's socket was misspelled 'biospy' upstream and has been corrected.

    The debug phase rejected the misspelling on 2026-08-24, so the corrected
    name is canonical. The old one is still written because sockets are
    configured per phase and the test phase -- one submission, no retry -- has
    never been observed. See spec.LEGACY_OUTPUT_FILENAMES.
    """
    pred = DecisionPrediction(task=1, decision="yes", reasoning=_reasoning(1))
    write_case_outputs(tmp_path, pred)

    for name in (
        "prostate-biopsy-decision.json",
        "prostate-biopsy-decision-reasoning.json",
        "prostate-biospy-decision.json",
        "prostate-biospy-decision-reasoning.json",
    ):
        assert (tmp_path / name).is_file(), name

    # Identical content, so whichever socket a phase resolves scores the same.
    for correct, legacy in (
        ("prostate-biopsy-decision.json", "prostate-biospy-decision.json"),
        (
            "prostate-biopsy-decision-reasoning.json",
            "prostate-biospy-decision-reasoning.json",
        ),
    ):
        assert json.loads((tmp_path / correct).read_text()) == json.loads(
            (tmp_path / legacy).read_text()
        )


def test_only_task1_has_a_legacy_alias(tmp_path):
    """Tasks 2 and 3 write exactly two files. The alias is not a general habit."""
    write_case_outputs(
        tmp_path / "t2",
        DecisionPrediction(task=2, decision="active_treatment", reasoning=_reasoning(2)),
    )
    assert len(list((tmp_path / "t2").iterdir())) == 2


def test_decision_file_is_a_bare_json_value(tmp_path):
    """Tasks 1 and 2 write a bare string, not an object wrapping it."""
    write_case_outputs(tmp_path, DecisionPrediction(task=1, decision="no", reasoning=_reasoning(1)))
    assert json.loads((tmp_path / "prostate-biopsy-decision.json").read_text()) == "no"

    write_case_outputs(
        tmp_path,
        DecisionPrediction(task=2, decision="watchful_waiting", reasoning=_reasoning(2)),
    )
    payload = json.loads((tmp_path / "prostate-treatment-decision.json").read_text())
    assert payload == "watchful_waiting"


def test_reasoning_file_has_exactly_four_keys(tmp_path):
    """The baseline's Pydantic models carry more; the evaluator wants these four."""
    write_case_outputs(tmp_path, DecisionPrediction(task=1, decision="yes", reasoning=_reasoning(1)))
    payload = json.loads((tmp_path / "prostate-biopsy-decision-reasoning.json").read_text())
    assert set(payload) == {"confidence", "variable_weights", "reveal_sequence", "free_text"}


def test_task3_outcome_and_bare_string_reasoning(tmp_path):
    pred = RecurrencePrediction(months_to_recurrence=42.5, event=1, free_text="Because of pT3b and positive margins.")
    write_case_outputs(tmp_path, pred)

    outcome = json.loads((tmp_path / "prostate-time-to-recurrence-or-last-follow-up.json").read_text())
    assert outcome == {"months_to_recurrence": 42.5, "event": 1}

    reasoning_path = tmp_path / "prostate-time-to-recurrence-or-last-follow-up-reasoning.json"
    # Task 3's reasoning socket is a bare string, unlike Tasks 1 and 2.
    assert isinstance(json.loads(reasoning_path.read_text()), str)


def test_invalid_prediction_writes_nothing(tmp_path):
    """Validation runs before the first write, so a bad case leaves no partial output."""
    bad = DecisionPrediction(task=1, decision="maybe", reasoning=_reasoning(1))
    with pytest.raises(ContractError):
        write_case_outputs(tmp_path, bad)
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------- #
# Validation vocabularies
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "task, decision",
    [(1, "Yes"), (1, "true"), (2, "surveillance"), (2, "active-treatment")],
)
def test_rejects_out_of_vocabulary_decisions(task, decision):
    with pytest.raises(ContractError):
        validate(DecisionPrediction(task=task, decision=decision, reasoning=_reasoning(task)))


def test_rejects_bad_confidence_and_weight_tokens():
    with pytest.raises(ContractError, match="confidence"):
        validate(DecisionPrediction(task=1, decision="yes", reasoning=_reasoning(1, confidence="high")))

    weights = {v: "noted" for v in spec.TASK1_VARIABLES} | {"pirads": "critical"}
    with pytest.raises(ContractError, match="weight"):
        validate(DecisionPrediction(task=1, decision="yes", reasoning=_reasoning(1, variable_weights=weights)))


def test_rejects_wrong_variable_key_set():
    """Task 2's eleven variables are not Task 1's ten."""
    wrong = {v: "noted" for v in spec.TASK2_VARIABLES}
    with pytest.raises(ContractError):
        validate(DecisionPrediction(task=1, decision="yes", reasoning=_reasoning(1, variable_weights=wrong)))


def test_rejects_out_of_vocabulary_reveal():
    """An unrecognised section name is charged as an extra reveal by the evaluator."""
    with pytest.raises(ContractError, match="out-of-vocabulary"):
        validate(
            DecisionPrediction(
                task=1, decision="yes", reasoning=_reasoning(1, reveal_sequence=["mri_report"])
            )
        )


def test_rejects_duplicate_reveals():
    with pytest.raises(ContractError, match="duplicates"):
        validate(
            DecisionPrediction(
                task=1,
                decision="yes",
                reasoning=_reasoning(1, reveal_sequence=["psa_trend", "psa_trend"]),
            )
        )


def test_empty_reveal_sequence_is_valid():
    """Revealing nothing is legal and scores 1.0 on tool efficiency."""
    validate(DecisionPrediction(task=1, decision="yes", reasoning=_reasoning(1, reveal_sequence=[])))


@pytest.mark.parametrize("event", [-1, 2, None, "1"])
def test_rejects_bad_recurrence_event(event):
    with pytest.raises(ContractError):
        validate(RecurrencePrediction(months_to_recurrence=1.0, event=event, free_text="x" * 50))


@pytest.mark.parametrize("months", [-1.0, float("inf"), float("nan")])
def test_rejects_months_that_are_negative_or_non_finite(months):
    """`inf` and `nan` serialise as bare `Infinity`/`NaN`, which is not JSON."""
    with pytest.raises(ContractError):
        validate(RecurrencePrediction(months_to_recurrence=months, event=0, free_text="x" * 50))


def test_task3_months_is_always_serialised_as_a_json_number(tmp_path):
    """A string here would take down the whole Task 3 evaluation job.

    See test_official_evaluator_crashes_on_a_non_numeric_months_in_a_failed_case
    for the failure it prevents. ``decision_json`` coerces, so even a numeric
    string handed to the dataclass reaches disk as a number.
    """
    pred = RecurrencePrediction(months_to_recurrence="42.5", event=1, free_text="x" * 50)
    write_case_outputs(tmp_path, pred)
    outcome = json.loads(
        (tmp_path / "prostate-time-to-recurrence-or-last-follow-up.json").read_text()
    )
    assert isinstance(outcome["months_to_recurrence"], float)
    assert isinstance(outcome["event"], int) and not isinstance(outcome["event"], bool)


def test_official_evaluator_crashes_on_a_non_numeric_months_in_a_failed_case(official_evaluator):
    """Pin a crash in the official evaluator so we never trip it.

    ``evaluate_recurrence_case`` stores the *raw* ``months_to_recurrence`` when
    a case fails schema validation, but normalises it when the case passes.
    ``aggregate_recurrence_metrics`` then does arithmetic on that raw value, so
    a single schema-failed case carrying a string blows up the aggregation for
    the *entire task* with a TypeError -- not just that one case.

    Our writers coerce to float and validate before writing, so we cannot emit
    this. If a future evaluator fixes it, this test fails and can be deleted.
    """
    ev = official_evaluator
    gt = {"case_id": "T3-x", "months_to_recurrence": 40.0, "event": 1}
    # Valid months, but an invalid event -- so the schema gate fails and the
    # unnormalised "36.5" survives into the aggregator.
    pred = {"case_id": "T3-x", "months_to_recurrence": "36.5", "event": 2}

    row = ev.evaluate_case(gt, pred, None, None)
    ev.attach_kappa_fields(row, gt, pred)
    assert row["gate"] == "schema_failed"
    assert row["pred_months"] == "36.5", "raw value no longer leaks; the crash may be fixed"

    with pytest.raises(TypeError):
        ev.compute_aggregate_metrics([row])


def test_official_evaluator_crashes_on_a_non_string_treatment_decision(official_evaluator):
    """The Task 2 twin of the crash above, from the same root cause.

    A schema-failed treatment case copies its raw ``treatment_recommendation.
    primary`` into ``pred_decision``, which then goes into the dataset F1
    unnormalised. A truthy non-string value survives the ``or`` sentinel and
    reaches sklearn, which refuses to mix string and numeric labels.

    Which exception escapes depends on the cohort: ``accuracy_score`` raises
    ValueError on the mixed dtypes, while a cohort that gets past it dies in
    ``classification_report``'s ``sorted(set(y_true) | set(y_pred))`` with a
    TypeError. Either way the task's aggregation is lost, which is the point.

    ``""`` and ``None`` are safe -- they fall through to the missing sentinel.
    Only a truthy non-string trips it, and our validation cannot emit one.
    """
    ev = official_evaluator
    gt = {"case_id": "T2-x", "treatment_recommendation": {"primary": "active_treatment"},
          "confidence": "clear", "variable_weights": {}, "reveal_sequence": []}
    pred = {"case_id": "T2-x", "treatment_recommendation": {"primary": 7}}

    row = ev.evaluate_case(gt, pred, None, None)
    ev.attach_kappa_fields(row, gt, pred)
    assert row["gate"] == "schema_failed"
    assert row["pred_decision"] == 7, "raw value no longer leaks; the crash may be fixed"

    with pytest.raises((TypeError, ValueError)):
        ev.compute_aggregate_metrics([row])


# --------------------------------------------------------------------------- #
# Interface detection
# --------------------------------------------------------------------------- #

def test_detects_task_from_truncated_task3_slug():
    """GC clips slugs at 50 chars, so the Task 3 clinical slug arrives truncated."""
    slug = spec.CLINICAL_SLUG_BY_TASK[3]
    assert len(slug) == 50
    assert detect_task({slug, spec.STRUCTURED_PROMPT_SLUG, spec.NEURAL_REP_SLUG}) == 3


def test_detect_task_rejects_unknown_interface():
    with pytest.raises(ValueError, match="no known clinical-data socket"):
        detect_task({spec.STRUCTURED_PROMPT_SLUG})


def test_read_case_roundtrip(tmp_fixture_case):
    case = read_case(tmp_fixture_case)
    assert case.task in (1, 2, 3)
    assert case.case_id
    assert isinstance(case.clinical_data, dict) and case.clinical_data
    # MRI is present for every task and is a single 1024-d vector.
    mri = case.embeddings("MRI image")
    assert len(mri) == 1 and len(mri[0]) == 1024
    # An absent modality is an empty list, never a missing key or None.
    if case.task in (1, 2):
        assert case.embeddings("Prostatectomy slide") == []
    if case.task == 1:
        assert case.embeddings("Biopsy slide") == []
