"""What the rationale may and may not say.

The rules under test are not stylistic. ``free_text`` is worth 0.20 of the case
score on Tasks 1 and 2, and the judge that scores it sees only the clinical-data
socket -- never the patient card the values come from. So a card value the reports
rarely restate reads to it as a fabrication, and the module's job is to keep those
out while still sounding like the terse clinical one-liners it is compared
against. See :mod:`chimera.predictors.rationale` for the measured corroboration
rates behind :data:`~chimera.predictors.rationale.CITABLE`.
"""

from __future__ import annotations

import pytest

from chimera.evidence.reports import NOT_ASSESSED, SurgicalPathology
from chimera.evidence.structured import StructuredFeatures
from chimera.models.guidelines import capra_s_points
from chimera.predictors.rationale import (
    CITABLE,
    biopsy_rationale,
    recurrence_rationale,
    treatment_rationale,
)

# --------------------------------------------------------------------------- #
# The uncitable fields
# --------------------------------------------------------------------------- #

def test_age_and_cspca_are_never_quoted():
    """The two fields the reports almost never restate (22%/7% and 3%/3%).

    Both are real, both are on the card, and ``age`` is ``important`` in every
    weight vector we ship -- none of which helps a judge that cannot see them.
    """
    features = StructuredFeatures(age=64.0, cspca=0.37, psa=7.4, pirads=4, psad=0.19)
    for text in (
        biopsy_rationale(features, "yes", "clear"),
        treatment_rationale(features, "active_treatment", "clear", "intermediate"),
    ):
        assert "64" not in text
        assert "0.37" not in text

    assert "age" not in CITABLE and "cspca" not in CITABLE


@pytest.mark.parametrize("dre_abnormal", [0, 1])
def test_the_rectal_examination_is_never_mentioned(dre_abnormal):
    """A category is a factual claim too -- this one asserts the exam happened.

    Measured: the reports mention a rectal examination at all in 12% of Task 1
    cases and 42% of Task 2 cases, less often than they state the age. Stating
    it wiped out the whole of the first rewrite's Task 1 gain, the judge
    replacing 60 ``age`` complaints with 51 DRE ones for a net 0.5662 -> 0.5646.
    """
    features = StructuredFeatures(pirads=4, psa=7.4, dre_abnormal=dre_abnormal)
    for text in (
        biopsy_rationale(features, "yes", "clear"),
        treatment_rationale(features, "active_treatment", "clear", "high"),
    ):
        assert "rectal" not in text and "DRE" not in text

    assert "dre" not in CITABLE and "dre_abnormal" not in CITABLE


def test_no_procedural_boilerplate():
    """The judge named these two strings as the defect in the measured baseline:
    the first "procedural meta-data ... not present in the Input", the second an
    unsupported claim. Neither may come back."""
    features = StructuredFeatures(psa=7.4, pirads=4)
    text = biopsy_rationale(features, "yes", "clear")
    assert "Weighted most heavily" not in text
    assert "section retrieval" not in text
    assert "Guideline basis" not in text


# --------------------------------------------------------------------------- #
# Tasks 1 and 2
# --------------------------------------------------------------------------- #

def test_biopsy_rationale_states_the_decisive_findings_then_the_decision():
    features = StructuredFeatures(
        pirads=2, psa=4.7, psad=0.14, dre_abnormal=0, prior_biopsy="negative"
    )
    text = biopsy_rationale(features, "no", "clear")
    assert "PI-RADS 2" in text
    assert "unlikely to harbour clinically significant disease" in text
    assert "PSA 4.7 ng/mL (density 0.14)" in text
    assert "previous negative biopsy" in text
    assert text.rstrip().endswith("Prostate biopsy is not indicated at present.")


def test_treatment_rationale_leads_with_the_eau_stratum():
    features = StructuredFeatures(
        bx_isup=3, bx_gl_prim=4, bx_gl_sec=3, psa=9.3, psad=0.516, ct_ordinal=3, pirads=5
    )
    text = treatment_rationale(features, "active_treatment", "clear", "intermediate")
    assert text.startswith("Localised prostate cancer, EAU intermediate risk:")
    assert "ISUP grade group 3 (Gleason 4+3)" in text
    assert "clinical stage cT1c" in text
    assert "Active treatment should be offered." in text


def test_a_very_high_psa_is_not_described_as_localised():
    """EAU high risk is a stratification *of localised disease*, so the stratum
    name cannot carry a PSA of 187 ng/mL. Two released cases (190 and 187) were
    scored 0.4 and 0.2 for exactly this, the judge asking for the staging the
    reference rationale calls for; the claim is wrong on its own terms too."""
    features = StructuredFeatures(psa=187.0, psad=3.26, pirads=5, bx_isup=4, ct_ordinal=3)
    text = treatment_rationale(features, "active_treatment", "clear", "high")
    assert "ocalised prostate cancer" not in text
    assert "staging for metastatic disease" in text
    # The value belongs in the findings, once.
    assert text.count("187") == 1


def test_a_high_but_not_extreme_psa_stays_localised():
    """The cut is ~100 ng/mL, not EAU high risk at 20 -- a PSA of 25 is routine
    high-risk localised disease and must not be talked up into a staging case."""
    features = StructuredFeatures(psa=25.0, psad=0.5, pirads=5, bx_isup=4, ct_ordinal=3)
    text = treatment_rationale(features, "active_treatment", "clear", "high")
    assert text.startswith("Localised prostate cancer, EAU high risk:")
    assert "metastatic" not in text


def test_an_unknown_eau_stratum_does_not_invent_one():
    features = StructuredFeatures(psa=7.4)
    text = treatment_rationale(features, "active_surveillance", "borderline", None)
    assert "incompletely characterised risk" in text
    assert "EAU" not in text


@pytest.mark.parametrize(
    "confidence,marker",
    [("clear", None), ("borderline", "borderline"), ("uncertain", "genuinely uncertain")],
)
def test_the_prose_carries_the_stated_confidence(confidence, marker):
    """Rubric item 5 scores whether the expressed uncertainty matches the
    ``confidence`` field, and it is reading the prose, not the field."""
    features = StructuredFeatures(pirads=3, psa=7.4)
    text = biopsy_rationale(features, "yes", confidence)
    if marker is None:
        assert "borderline" not in text and "uncertain" not in text
    else:
        assert marker in text


def test_an_empty_case_still_yields_a_decision_sentence():
    """Nothing readable is not a licence to say nothing, or to guess."""
    text = biopsy_rationale(StructuredFeatures(), "no", "uncertain")
    assert text.startswith("Prostate biopsy is not indicated at present.")
    assert "genuinely uncertain" in text


# --------------------------------------------------------------------------- #
# Task 3
# --------------------------------------------------------------------------- #

_FULL = SurgicalPathology(
    gleason_primary=4, gleason_secondary=3, isup=3, pt_stage="pT3a",
    epe=True, positive_margins=True, svi=False, lymph_nodes=False,
)


def test_recurrence_rationale_names_the_specimen_findings():
    text = recurrence_rationale(_FULL, 12.0, 48.0, capra_s_points(_FULL, 12.0))
    assert "Gleason 4+3 (ISUP 3)" in text
    assert "pathological stage pT3a" in text
    assert "positive surgical margins" in text
    assert "no seminal-vesicle invasion" in text
    assert "extraprostatic extension" in text
    assert "negative lymph nodes" in text
    assert "preoperative PSA 12 ng/mL" in text
    assert "48 months" in text


def test_unsampled_nodes_are_not_reported_as_negative():
    """pNx is not pN0. CAPRA-S scores them the same; the prose must not."""
    pathology = SurgicalPathology(lymph_nodes=NOT_ASSESSED)
    text = recurrence_rationale(pathology, None, 60.0, capra_s_points(pathology, None))
    assert "no lymph nodes sampled" in text
    assert "negative lymph nodes" not in text


def test_a_complete_specimen_quotes_the_score_out_of_twelve():
    earned, assessable = capra_s_points(_FULL, 12.0)
    assert assessable == 12
    text = recurrence_rationale(_FULL, 12.0, 48.0, (earned, assessable))
    assert f"CAPRA-S score of {earned} of 12" in text


def test_a_rescaled_score_is_never_quoted_as_a_raw_one():
    """The failure this guards: a specimen where only the PSA parsed rescales to
    a flat 12 of 12, which would claim a fully-staged high-risk case we never
    read. Only the points actually assessable may be stated."""
    bare = SurgicalPathology()
    points = capra_s_points(bare, 86.0)
    assert points == (3, 3)
    text = recurrence_rationale(bare, 86.0, 24.0, points)
    assert "of 12" not in text
    assert "3 of the 3 CAPRA-S points" in text


def test_an_unreadable_specimen_says_so_rather_than_scoring_it():
    bare = SurgicalPathology()
    assert capra_s_points(bare, None) is None
    text = recurrence_rationale(bare, None, 60.0, None)
    assert "no interpretable findings" in text
    assert "CAPRA-S" not in text
    assert "60 months" in text
