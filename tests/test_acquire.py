"""The acquisition loop, and the compliance properties it exists to satisfy.

The organizers ruled on 8 Sep that retrieval must be chosen "dynamically by the
agent based on the case and the information available at that point", and that a
strategy deciding the sequence in advance does not qualify. Three of the tests
below are that requirement written as assertions -- that the sequence varies
with retrieved *content*, that no case retrieves nothing, and that the sequence
is not a function of the task alone. The rest guard the invariant those must not
be bought at the expense of: the declaration still names exactly what was read.
"""

from __future__ import annotations

import pytest

from chimera.contract import spec
from chimera.contract.io import CaseInputs
from chimera.contract.types import validate
from chimera.mcp.client import DirectStore
from chimera.predictors import acquire
from chimera.predictors.guideline import GuidelinePredictor

#: An MRI report at the biopsy threshold, stating a prior negative biopsy.
RAD_HIGH_WITH_HISTORY = (
    "Exam: mpMRI Prostate. Clinical indication: PSA 9.1 ng/mL, previously "
    "negative biopsy in 2022.\nIMPRESSION: PI-RADS 5. PSA density 0.31 "
    "ng/mL/mL. Clinical T stage cT2b."
)
#: The same score, but silent about any earlier biopsy.
RAD_HIGH_NO_HISTORY = (
    "Exam: mpMRI Prostate.\nIMPRESSION: PI-RADS 5. PSA density 0.31 "
    "ng/mL/mL. Clinical T stage cT2b."
)
#: Benign imaging: the band where PSA behaviour is what decides.
RAD_LOW_WITH_HISTORY = (
    "Exam: mpMRI Prostate. Clinical indication: previously negative biopsy.\n"
    "IMPRESSION: PI-RADS 2. PSA density 0.09 ng/mL/mL. Clinical T stage cT1c."
)
PATH_LOW_GRADE = (
    "Timepoint 1 (Jan 2024): Prostate tissue biopsy reveals Gleason 3+3, ISUP "
    "grade group 1.\nAI-predicted ISUP grade group (digital pathology): grade group 4."
)
PATH_HIGH_GRADE = (
    "Timepoint 1 (Mar 2024): Prostate tissue biopsy reveals Gleason 4+4, ISUP "
    "grade group 4. Cribriform pattern identified."
)
PSA_SERIES = [{"date": "Jan 2024", "val": 4.1}, {"date": "Jan 2025", "val": 6.8}]
NOTES = [{"date": "01 Feb 2024", "author": "Dr. X", "text": "Referred to urology."}]


def _case(task: int, clinical: dict) -> CaseInputs:
    return CaseInputs(
        task=task,
        case_id=f"T{task}-x",
        structured_prompt={"psa": 7.4, "age": 64, "pirads": "4"},
        clinical_data=clinical,
        neural_representations={},
    )


def _acquired(task: int, clinical: dict) -> list[str]:
    return acquire.acquire(task, DirectStore(_case(task, clinical)))


def _declared(task: int, clinical: dict) -> list[str]:
    case = _case(task, clinical)
    pred = GuidelinePredictor().predict(case, DirectStore(case))
    return list(pred.reasoning.reveal_sequence)


# -- the compliance properties ---------------------------------------------- #


def test_sequence_depends_on_retrieved_content_not_just_the_case():
    """The organizers' requirement, reduced to its smallest witness.

    Two Task 1 cases identical in every structured field and differing only in
    whether the MRI report happens to state a prior biopsy must retrieve
    different sections -- because the decision to call ``previous_notes`` is
    made *after* reading the report, from what the report said.
    """
    both = {"psa_trend": PSA_SERIES, "previous_notes": NOTES}
    stated = _acquired(1, {"radiology_report": RAD_HIGH_WITH_HISTORY, **both})
    silent = _acquired(1, {"radiology_report": RAD_HIGH_NO_HISTORY, **both})

    assert stated == ["radiology_report"]
    assert silent == ["radiology_report", "previous_notes"]


def test_imaging_band_reopens_the_kinetics_question():
    """PI-RADS 4-5 is an indication on its own; at 2 the PSA trend is what decides.

    Both cases are served the identical section set, so the difference is
    entirely in what the first retrieval returned.
    """
    both = {"psa_trend": PSA_SERIES, "previous_notes": NOTES}
    high = _acquired(1, {"radiology_report": RAD_HIGH_WITH_HISTORY, **both})
    low = _acquired(1, {"radiology_report": RAD_LOW_WITH_HISTORY, **both})

    assert "psa_trend" not in high
    assert "psa_trend" in low


def test_grade_reopens_the_surveillance_questions_on_task_2():
    """A treatment workup asks about PSA over time only where surveillance is live."""
    rest = {
        "radiology_report": RAD_HIGH_NO_HISTORY,
        "psa_trend": PSA_SERIES,
        "previous_notes": NOTES,
    }
    low = _acquired(2, {"pathology_report": PATH_LOW_GRADE, **rest})
    high = _acquired(2, {"pathology_report": PATH_HIGH_GRADE, **rest})

    assert low[0] == "pathology_report", "grade is asked first"
    assert "psa_trend" in low
    assert "psa_trend" not in high


@pytest.mark.parametrize("task", [1, 2])
def test_every_case_retrieves_something(task):
    """No case may reach a decision without consulting the MCP server.

    Until 8 Sep every Task 2 case declared the empty set, which is weakly
    dominant under ``cost_aware_tool_score`` and makes no tool call at all.
    """
    sections = {s: "PI-RADS 3. ISUP grade group 2." for s in spec.REVEAL_SECTIONS}
    assert _acquired(task, sections)


def test_the_policy_is_not_a_function_of_the_task_alone():
    """The whole objection was a per-task constant. Assert it cannot come back."""
    served = {
        "radiology_report": RAD_HIGH_WITH_HISTORY,
        "pathology_report": PATH_LOW_GRADE,
        "psa_trend": PSA_SERIES,
        "previous_notes": NOTES,
    }
    variants = {
        tuple(_acquired(1, {**served, "radiology_report": rad}))
        for rad in (RAD_HIGH_WITH_HISTORY, RAD_HIGH_NO_HISTORY, RAD_LOW_WITH_HISTORY)
    }
    assert len(variants) > 1


# -- reveal honesty, which the above must not cost -------------------------- #


@pytest.mark.parametrize("task", [1, 2])
def test_declaration_is_exactly_what_was_retrieved(task):
    """Declared set == ledger set, in both directions.

    Compared as sets rather than as lists on purpose: the declaration is in
    agenda order and the ledger is in call order, and they legitimately differ
    because ``extract_structured`` may reach for a section the agenda has not
    got to yet. What may never differ is *membership* -- declaring a section no
    tool returned would be dishonest, and reading one we do not declare would
    under-report the evidence we used.
    """
    clinical = {
        "radiology_report": RAD_LOW_WITH_HISTORY,
        "pathology_report": PATH_LOW_GRADE,
        "psa_trend": PSA_SERIES,
        "previous_notes": NOTES,
    }
    case = _case(task, clinical)
    store = DirectStore(case)
    declared = list(GuidelinePredictor().predict(case, store).reasoning.reveal_sequence)
    assert set(declared) == set(store.retrieved)
    assert len(set(declared)) == len(declared)


@pytest.mark.parametrize("task", [1, 2])
def test_a_section_the_case_does_not_carry_is_never_declared(task):
    """An absent section is not a reveal, however loudly the agenda asks for it."""
    declared = _declared(task, {"radiology_report": RAD_LOW_WITH_HISTORY})
    assert "previous_notes" not in declared
    assert "psa_trend" not in declared


@pytest.mark.parametrize("task", [1, 2])
def test_a_blank_section_is_not_evidence(task):
    assert _acquired(task, {s: "   " for s in spec.REVEAL_SECTIONS}) == []


@pytest.mark.parametrize("task", [1, 2])
def test_declaration_stays_inside_the_vocabulary_and_has_no_duplicates(task):
    sections = {s: "PI-RADS 2. ISUP grade group 1." for s in spec.REVEAL_SECTIONS}
    sections["surgical_pathology_report"] = "not a reveal name"
    declared = _declared(task, sections)
    assert set(declared) <= set(spec.REVEAL_SECTIONS)
    assert len(set(declared)) == len(declared)


@pytest.mark.parametrize("task", [1, 2, 3])
def test_output_still_passes_contract_validation(task):
    sections = {s: "PI-RADS 3. ISUP grade group 2." for s in spec.REVEAL_SECTIONS}
    case = _case(task, sections)
    validate(GuidelinePredictor().predict(case, DirectStore(case)))


def test_loop_terminates_when_every_section_is_absent():
    """The agenda must retire an unanswerable question rather than re-ask it."""
    assert _acquired(1, {}) == []


# -- the readers ------------------------------------------------------------ #


def test_ai_predicted_grade_is_not_read_as_the_histopathology_grade():
    """``PATH_LOW_GRADE`` carries a real ISUP 1 and an AI-predicted grade 4.

    Branching the workup on a model's output rather than on the specimen would
    be wrong regardless of score, and the two disagree on 49 released cases.
    """
    findings = acquire.Findings()
    findings.absorb_pathology(PATH_LOW_GRADE)
    assert findings.isup == 1


def test_a_prior_pirads_is_not_read_as_the_current_one():
    findings = acquire.Findings()
    findings.absorb_radiology(
        "Comparison with earlier PI-RADS 2 study.\nIMPRESSION: PI-RADS 5."
    )
    assert findings.pirads == 5


def test_readers_never_raise_on_junk():
    findings = acquire.Findings()
    for absorb in (findings.absorb_radiology, findings.absorb_pathology,
                   findings.absorb_notes, findings.absorb_labs):
        absorb("")
        absorb("PI-RADS 9. ISUP grade group 0. Gleason nonsense.")
    findings.absorb_psa_trend([{"val": "not-a-number"}, "junk", None])
    assert findings.pirads is None
    assert findings.isup is None
