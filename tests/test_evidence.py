"""Feature extraction, and what it does when the input is not what we saw in training.

The robustness tests matter more than the happy-path ones. 100 of the 250 test cases
come from Karolinska, whose report templates we have never seen, and Task 3's entire
feature set is parsed out of free text. The requirement is not that extraction keeps
working on unfamiliar prose -- it cannot -- but that it degrades to ``None`` and lets
the model fall back, rather than raising and losing the case to a sentinel label.
"""

from __future__ import annotations

import pytest

from chimera.contract.io import CaseInputs
from chimera.evidence import extract_reports, extract_structured
from chimera.evidence.reports import NOT_ASSESSED
from chimera.models.guidelines import capra_s, eau_risk, stratum

SURGICAL = (
    "The robot-assisted radical prostatectomy specimen showed Gleason 4+3 "
    "(ISUP grade group 3), pathological stage pT4b. Extraprostatic extension was "
    "present; surgical margins were positive; the seminal vesicles were invaded; "
    "lymphovascular invasion was absent; lymph node metastasis was present."
)
RADIOLOGY = (
    "Prostate volume: 37.29 cc. PSA density: 2.306 ng/mL/cc. PI-RADS: 5. "
    "AI model-predicted probability of clinically significant prostate cancer "
    "(0–1): 0.9527162."
)


def _case(task: int, prompt: dict | None = None, clinical: dict | None = None) -> CaseInputs:
    return CaseInputs(
        task=task,
        case_id="X-1",
        structured_prompt=prompt or {},
        clinical_data=clinical or {},
        neural_representations={},
    )


# --------------------------------------------------------------------------- #
# Structured prompt
# --------------------------------------------------------------------------- #

def test_structured_reads_the_numeric_panel():
    f = extract_structured(_case(1, {"psa": 4.7, "age": 67, "psad": 0.14, "pirads": "2"}))
    assert (f.psa, f.age, f.psad, f.pirads) == (4.7, 67.0, 0.14, 2)


@pytest.mark.parametrize("value", ["NA", "", None, "unknown", [], {}])
def test_pirads_out_of_range_becomes_none(value):
    """One released case literally has `pirads: "NA"`."""
    assert extract_structured(_case(1, {"pirads": value})).pirads is None


def test_ctx_is_not_a_stage():
    """`cTx` means not assessable. Treating it as early stage would under-risk
    exactly the patients we know least about."""
    assert extract_structured(_case(2, {"ct": "cTx"})).ct_ordinal is None
    assert extract_structured(_case(2, {"ct": "cT2a"})).ct_ordinal is not None


def test_ct_ordinal_orders_stages():
    def o(v):
        return extract_structured(_case(2, {"ct": v})).ct_ordinal

    assert o("cT1c") < o("cT2a") < o("cT2c") < o("cT3a") < o("cT4")


def test_dre_not_done_is_not_normal():
    assert extract_structured(_case(1, {"dre": "Not done"})).dre_abnormal is None
    assert extract_structured(_case(1, {"dre": "Normal"})).dre_abnormal == 0
    assert extract_structured(_case(1, {"dre": "Nodus"})).dre_abnormal == 1


def test_prior_biopsy_none_means_never_biopsied():
    """The string "None" is a real category, not a missing value, and it is the
    strongest single feature in Task 2."""
    assert extract_structured(_case(1, {"bx": "None"})).prior_biopsy == "none"
    assert extract_structured(_case(1, {})).prior_biopsy is None


def test_ipss_score_is_dug_out_of_its_sentence():
    f = extract_structured(_case(1, {"ipss": "IPSS score: 18/35 (moderate LUTS)"}))
    assert f.ipss == 18


def test_structured_survives_garbage():
    weird = {"psa": {"nested": 1}, "age": ["list"], "pirads": object(), "ct": 42, "ipss": None}
    f = extract_structured(_case(1, weird))
    assert f.psa is None and f.age is None and f.pirads is None and f.ct_ordinal is None


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def test_surgical_pathology_is_fully_parsed():
    p = extract_reports(_case(3, clinical={"surgical_pathology_report": SURGICAL}))
    assert (p.gleason_primary, p.gleason_secondary, p.isup) == (4, 3, 3)
    assert p.pt_stage == "pT4b"
    assert p.epe is True and p.positive_margins is True and p.svi is True
    assert p.lvi is False and p.lymph_nodes is True
    assert p.gleason_sum == 7


def test_negative_phrasing_is_not_read_as_positive():
    """"There was no extraprostatic extension" contains "extraprostatic extension",
    so a positive-first match would invert the answer."""
    text = "There was no extraprostatic extension; surgical margins were negative."
    p = extract_reports(_case(3, clinical={"surgical_pathology_report": text}))
    assert p.epe is False
    assert p.positive_margins is False


def test_unsampled_nodes_are_not_negative_nodes():
    """pNx is clinically distinct from pN0, and 43 of 75 released cases are pNx."""
    unsampled = extract_reports(
        _case(3, clinical={"surgical_pathology_report": "no lymph nodes were removed."})
    )
    negative = extract_reports(
        _case(3, clinical={"surgical_pathology_report": "there was no lymph node metastasis."})
    )
    assert unsampled.lymph_nodes == NOT_ASSESSED
    assert negative.lymph_nodes is False


def test_a_value_ending_a_sentence_still_parses():
    """`[\\d.]+` would swallow the full stop and fail the float cast."""
    p = extract_reports(_case(3, clinical={"radiology_report": RADIOLOGY}))
    assert p.cspca == pytest.approx(0.9527162)
    assert p.prostate_volume == pytest.approx(37.29)
    assert p.pirads == 5


def test_missing_field_is_unknown_not_negative():
    """Silence about EPE must not be read as its absence."""
    p = extract_reports(_case(3, clinical={"surgical_pathology_report": "Gleason 3+3."}))
    assert p.epe is None and p.svi is None and p.positive_margins is None


@pytest.mark.parametrize(
    "clinical",
    [
        {},
        {"surgical_pathology_report": None},
        {"surgical_pathology_report": 42},
        {"surgical_pathology_report": ["a", {"t": "b"}]},
        {"surgical_pathology_report": "Rapport i helt annan mall utan kända fraser."},
        {"surgical_pathology_report": SURGICAL[:40]},
    ],
    ids=["empty", "none", "int", "list", "foreign-template", "truncated"],
)
def test_reports_degrade_rather_than_raise(clinical):
    """The Karolinska proxy: unfamiliar input must yield None, never an exception."""
    p = extract_reports(_case(3, clinical=clinical))
    assert p.epe in (None, True, False)


# --------------------------------------------------------------------------- #
# Guidelines
# --------------------------------------------------------------------------- #

def test_capra_s_matches_a_hand_computed_case():
    """PSA 12 (2) + Gleason 4+3 (2) + margins (2) + SVI (2) + EPE (1) + LNI (1) = 10."""
    p = extract_reports(_case(3, clinical={"surgical_pathology_report": SURGICAL}))
    assert capra_s(p, 12.0) == pytest.approx(10.0)


def test_capra_s_rescales_when_a_component_is_missing():
    """A partly-readable report must not rank as low-risk purely for being partial."""
    full = extract_reports(_case(3, clinical={"surgical_pathology_report": SURGICAL}))
    partial = extract_reports(
        _case(3, clinical={"surgical_pathology_report":
                           "Gleason 4+3 (ISUP grade group 3). Surgical margins were positive."})
    )
    assert capra_s(partial, 12.0) > capra_s(full, 12.0) / 2
    assert capra_s(extract_reports(_case(3)), None) is None


def test_capra_s_orders_risk_monotonically():
    benign = extract_reports(_case(3, clinical={"surgical_pathology_report":
        "Gleason 3+3 (ISUP grade group 1). There was no extraprostatic extension; "
        "surgical margins were negative; the seminal vesicles were not invaded; "
        "there was no lymph node metastasis."}))
    severe = extract_reports(_case(3, clinical={"surgical_pathology_report": SURGICAL}))
    assert capra_s(benign, 4.0) < capra_s(severe, 30.0)


def test_pnx_scores_no_nodal_points_but_still_counts():
    """CAPRA-S awards its nodal point only for confirmed pN1."""
    text = SURGICAL.replace("lymph node metastasis was present", "no lymph nodes were removed")
    p = extract_reports(_case(3, clinical={"surgical_pathology_report": text}))
    assert p.lymph_nodes == NOT_ASSESSED
    assert capra_s(p, 12.0) == pytest.approx(9.0)


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ({"bx_isup": 1, "psa": 5.0, "ct": "cT1c"}, "low"),
        ({"bx_isup": 2, "psa": 5.0, "ct": "cT1c"}, "intermediate"),
        ({"bx_isup": 5, "psa": 5.0, "ct": "cT1c"}, "high"),
        ({"bx_isup": 1, "psa": 25.0, "ct": "cT1c"}, "high"),
        ({"bx_isup": 1, "psa": 5.0, "ct": "cT3a"}, "high"),
    ],
)
def test_eau_risk_groups(prompt, expected):
    assert eau_risk(extract_structured(_case(2, prompt))) == expected


def test_eau_unknown_field_cannot_produce_low_risk():
    """Low requires every criterion; missing data must not buy a low-risk label."""
    assert eau_risk(extract_structured(_case(2, {"bx_isup": 1, "psa": 5.0}))) != "low"


def test_stratum_is_total():
    """Every case lands in a leaf, including one with no usable features at all."""
    for task in (1, 2):
        leaf = stratum(task, extract_structured(_case(task, {})))
        assert isinstance(leaf, str) and leaf
