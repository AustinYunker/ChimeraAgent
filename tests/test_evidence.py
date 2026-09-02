"""Feature extraction, and what it does when the input is not what we saw in training.

The robustness tests matter more than the happy-path ones. 100 of the 250 test cases
come from Karolinska, whose report templates we have never seen, and Task 3's entire
feature set is parsed out of free text. The requirement is not that extraction keeps
working on unfamiliar prose -- it cannot -- but that it degrades to ``None`` and lets
the model fall back, rather than raising and losing the case to a sentinel label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chimera.contract import spec
from chimera.contract.io import CaseInputs
from chimera.evidence import (
    classify_prior_biopsy,
    extract_prior_context,
    extract_reports,
    extract_structured,
)
from chimera.evidence.reports import NOT_ASSESSED, PriorContext
from chimera.mcp.client import DirectStore
from chimera.models.guidelines import TASK2_LEAVES, capra_s, eau_risk, stratum
from chimera.models.stratified import (
    FALLBACK_MONTHS,
    MONTHS_AT_ZERO_RISK,
    MONTHS_PER_CAPRA_POINT,
    predict_months,
)

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


# These cases exist only in memory, so there is no directory for an MCP server
# to serve and the in-process store is the honest stand-in. It enforces the same
# per-task tool registry and keeps the same ledger, so what is exercised here is
# the extraction; `tests/test_mcp.py` exercises the wire.
def _features(task: int, prompt: dict | None = None, clinical: dict | None = None):
    case = _case(task, prompt, clinical)
    return extract_structured(case, DirectStore(case))


def _reports(task: int, prompt: dict | None = None, clinical: dict | None = None):
    return extract_reports(DirectStore(_case(task, prompt, clinical)))


# --------------------------------------------------------------------------- #
# Structured prompt
# --------------------------------------------------------------------------- #

def test_structured_reads_the_numeric_panel():
    f = _features(1, {"psa": 4.7, "age": 67, "psad": 0.14, "pirads": "2"})
    assert (f.psa, f.age, f.psad, f.pirads) == (4.7, 67.0, 0.14, 2)


@pytest.mark.parametrize("value", ["NA", "", None, "unknown", [], {}])
def test_pirads_out_of_range_becomes_none(value):
    """One released case literally has `pirads: "NA"`."""
    assert _features(1, {"pirads": value}).pirads is None


def test_ctx_is_not_a_stage():
    """`cTx` means not assessable. Treating it as early stage would under-risk
    exactly the patients we know least about."""
    assert _features(2, {"ct": "cTx"}).ct_ordinal is None
    assert _features(2, {"ct": "cT2a"}).ct_ordinal is not None


def test_ct_ordinal_orders_stages():
    def o(v):
        return _features(2, {"ct": v}).ct_ordinal

    assert o("cT1c") < o("cT2a") < o("cT2c") < o("cT3a") < o("cT4")


def test_dre_not_done_is_not_normal():
    assert _features(1, {"dre": "Not done"}).dre_abnormal is None
    assert _features(1, {"dre": "Normal"}).dre_abnormal == 0
    assert _features(1, {"dre": "Nodus"}).dre_abnormal == 1


def test_prior_biopsy_none_means_never_biopsied():
    """The string "None" is a real category, not a missing value, and it is the
    strongest single feature in Task 2."""
    assert _features(1, {"bx": "None"}).prior_biopsy == "none"
    assert _features(1, {}).prior_biopsy is None


def test_ipss_score_is_dug_out_of_its_sentence():
    f = _features(1, {"ipss": "IPSS score: 18/35 (moderate LUTS)"})
    assert f.ipss == 18


def test_structured_survives_garbage():
    weird = {"psa": {"nested": 1}, "age": ["list"], "pirads": object(), "ct": 42, "ipss": None}
    f = _features(1, weird)
    assert f.psa is None and f.age is None and f.pirads is None and f.ct_ordinal is None


# --------------------------------------------------------------------------- #
# Prior biopsy recovered from the notes
#
# Release Version 3 deleted `bx` from all 195 Task 1 prompts, and it is the first
# thing the Task 1 stratifier branches on. These pin the rules that get it back;
# the cohort-level number (193/195 against Version 2) is in
# `chimera.evidence.notes`.
# --------------------------------------------------------------------------- #

def _notes(*texts: str) -> dict:
    return {"previous_notes": [{"date": "1 Jan 2025", "author": "Dr. X", "text": t}
                               for t in texts]}


@pytest.mark.parametrize(
    "text, expected",
    [
        # A grade is assigned to tissue, so it cannot exist without a specimen.
        ("Biopsy in March showed Gleason 3+4 disease in two cores.", "positive"),
        ("Histology reported ISUP grade group 2.", "positive"),
        ("Systematic biopsy was positive in the left base.", "positive"),
        # `risk` must not read as a hedge -- this is a diagnosis, not a question.
        ("Subsequent histopathology confirmed low-risk prostate cancer.", "positive"),
        ("He underwent radical prostatectomy in 2019.", "positive"),
        ("Previous TRUS biopsy was negative.", "negative"),
        ("Biopsy cores were benign.", "negative"),
        # The hedge guard: a prior *negative* biopsy plus a cancer question in one
        # clause. Reading the disease word as a diagnosis inverts the answer.
        ("Prior negative biopsy; assess for clinically significant prostate cancer.",
         "negative"),
        ("Previous negative TRUS biopsy, prompting re-evaluation for occult malignancy.",
         "negative"),
        # Never biopsied -- a finding in its own right, and the Version 2 wording.
        ("PSA 4.9 ng/mL on annual screening; referred to urology for imaging.", "none"),
        # Biopsied, outcome unstated. Abstain: guessing here routes the case to the
        # wrong leaf, and on Task 1 a wrong decision scores zero however good the
        # reasoning.
        ("PSA rose to 12.0 following a previous biopsy event.", None),
        ("Biopsy was performed in 2021.", None),
    ],
)
def test_prior_biopsy_is_read_from_prose(text, expected):
    assert classify_prior_biopsy(text) == expected


def test_contradictory_evidence_abstains():
    """Both polarities in one record means we do not know which visit won."""
    assert classify_prior_biopsy(
        "First biopsy was negative. Repeat biopsy showed Gleason 3+3."
    ) is None


def test_the_patient_card_wins_where_it_speaks():
    """Task 2 still carries `bx`; its coding is the organizers' and beats our regex."""
    clinical = _notes("Biopsy showed Gleason 4+3.")
    assert _features(2, {"bx": "Negative"}, clinical).prior_biopsy == "negative"


def test_the_notes_keep_task_2_alive_if_bx_leaves_its_card_too():
    """The whole reason this extractor exists.

    Version 3 deleted `bx` from every Task 1 prompt. If the test release does the
    same to Task 2, every case falls into the `unknown` leaf and the task degrades
    to a constant -- unless the status can be read back out of the prose.
    """
    trimmed = _case(2, {"bx_isup": 4, "psa": 8.0, "ct": "cT2a"},
                    _notes("Biopsy in March showed Gleason 4+4 disease."))
    f = extract_structured(trimmed, DirectStore(trimmed))
    assert f.prior_biopsy == "positive"
    assert stratum(2, f) == "positive_high"


def test_task_1_does_not_pay_for_a_feature_it_no_longer_uses():
    """Task 1 stratifies on PI-RADS alone, so it must not go reading the notes.

    The tool score is precision over declared reveals, so a section read for a
    feature no leaf consults is a straight subtraction.
    """
    f = _features(1, {}, _notes("Biopsy showed Gleason 4+3 disease."))
    assert f.prior_biopsy is None
    assert f.evidence_sections == ()


def test_only_sections_actually_read_are_reported():
    """`evidence_sections` is the reveal-honesty half: read it, declare it.

    Empty when the card answered -- nothing was retrieved -- and naming exactly
    the sections that carried text, not the ones we would have liked.
    """
    from chimera.contract import spec

    card = _features(2, {"bx": "Positive"}, _notes("Biopsy positive."))
    assert card.evidence_sections == ()

    both = _features(2, {}, {
        "radiology_report": "Indication: rising PSA.",
        "previous_notes": [{"text": "Biopsy showed Gleason 3+4."}],
    })
    assert both.evidence_sections == ("radiology_report", "previous_notes")

    one = _features(2, {}, {
        "radiology_report": "Biopsy showed Gleason 3+4.", "previous_notes": [],
    })
    assert one.evidence_sections == ("radiology_report",)

    assert set(both.evidence_sections) <= set(spec.REVEAL_SECTIONS)


@pytest.mark.parametrize("clinical", [
    {},
    {"radiology_report": None, "previous_notes": None},
    {"previous_notes": "Free-text block with no note records."},   # Task 3's shape
    {"previous_notes": [{"date": "1 Jan"}, "bare string", 42, None]},
    {"radiology_report": ["fragment", {"text": "and a dict"}]},
    {"radiology_report": 3.14, "previous_notes": {"text": "Biopsy: benign."}},
])
def test_the_notes_extractor_degrades_rather_than_raises(clinical):
    """The Karolinska proxy: unfamiliar shapes cost the feature, never the case."""
    f = _features(1, {}, clinical)
    assert f.prior_biopsy in (None, "none", "negative", "positive")
    assert isinstance(f.evidence_sections, tuple)


def test_a_truncated_report_does_not_invent_a_diagnosis():
    """Prose cut mid-sentence must abstain or under-call, never over-call.

    Truncation is the cheap stand-in for a template we have never seen: the
    evidence for the real answer disappears a clause at a time. Losing it has to
    cost the feature, not flip its polarity -- ``positive`` is the expensive
    direction, because it routes a Task 1 case to ``prior_positive`` and the gate
    turns a wrong leaf into a zero.
    """
    for full in (
        "Previous transrectal biopsy of the prostate was negative for malignancy.",
        "PSA 4.9 ng/mL on annual screening; referred to urology for imaging.",
    ):
        for cut in range(4, len(full) + 1):
            assert classify_prior_biopsy(full[:cut]) != "positive", full[:cut]


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #

def test_surgical_pathology_is_fully_parsed():
    p = _reports(3, clinical={"surgical_pathology_report": SURGICAL})
    assert (p.gleason_primary, p.gleason_secondary, p.isup) == (4, 3, 3)
    assert p.pt_stage == "pT4b"
    assert p.epe is True and p.positive_margins is True and p.svi is True
    assert p.lvi is False and p.lymph_nodes is True
    assert p.gleason_sum == 7


def test_negative_phrasing_is_not_read_as_positive():
    """"There was no extraprostatic extension" contains "extraprostatic extension",
    so a positive-first match would invert the answer."""
    text = "There was no extraprostatic extension; surgical margins were negative."
    p = _reports(3, clinical={"surgical_pathology_report": text})
    assert p.epe is False
    assert p.positive_margins is False


def test_unsampled_nodes_are_not_negative_nodes():
    """pNx is clinically distinct from pN0, and 43 of 75 released cases are pNx."""
    unsampled = _reports(
        3, clinical={"surgical_pathology_report": "no lymph nodes were removed."}
    )
    negative = _reports(
        3, clinical={"surgical_pathology_report": "there was no lymph node metastasis."}
    )
    assert unsampled.lymph_nodes == NOT_ASSESSED
    assert negative.lymph_nodes is False


def test_a_value_ending_a_sentence_still_parses():
    """`[\\d.]+` would swallow the full stop and fail the float cast."""
    p = _reports(3, clinical={"radiology_report": RADIOLOGY})
    assert p.cspca == pytest.approx(0.9527162)
    assert p.prostate_volume == pytest.approx(37.29)
    assert p.pirads == 5


def test_missing_field_is_unknown_not_negative():
    """Silence about EPE must not be read as its absence."""
    p = _reports(3, clinical={"surgical_pathology_report": "Gleason 3+3."})
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
    p = _reports(3, clinical=clinical)
    assert p.epe in (None, True, False)


# --------------------------------------------------------------------------- #
# Guidelines
# --------------------------------------------------------------------------- #

def test_capra_s_matches_a_hand_computed_case():
    """PSA 12 (2) + Gleason 4+3 (2) + margins (2) + SVI (2) + EPE (1) + LNI (1) = 10."""
    p = _reports(3, clinical={"surgical_pathology_report": SURGICAL})
    assert capra_s(p, 12.0) == pytest.approx(10.0)


def test_capra_s_rescales_when_a_component_is_missing():
    """A partly-readable report must not rank as low-risk purely for being partial."""
    full = _reports(3, clinical={"surgical_pathology_report": SURGICAL})
    partial = _reports(
        3, clinical={"surgical_pathology_report":
                     "Gleason 4+3 (ISUP grade group 3). Surgical margins were positive."}
    )
    assert capra_s(partial, 12.0) > capra_s(full, 12.0) / 2
    assert capra_s(_reports(3), None) is None


def test_capra_s_orders_risk_monotonically():
    benign = _reports(3, clinical={"surgical_pathology_report":
        "Gleason 3+3 (ISUP grade group 1). There was no extraprostatic extension; "
        "surgical margins were negative; the seminal vesicles were not invaded; "
        "there was no lymph node metastasis."})
    severe = _reports(3, clinical={"surgical_pathology_report": SURGICAL})
    assert capra_s(benign, 4.0) < capra_s(severe, 30.0)


def test_pnx_scores_no_nodal_points_but_still_counts():
    """CAPRA-S awards its nodal point only for confirmed pN1."""
    text = SURGICAL.replace("lymph node metastasis was present", "no lymph nodes were removed")
    p = _reports(3, clinical={"surgical_pathology_report": text})
    assert p.lymph_nodes == NOT_ASSESSED
    assert capra_s(p, 12.0) == pytest.approx(9.0)


# --------------------------------------------------------------------------- #
# Task 1: the history the MRI report carries
#
# Task 1's decision turns on a history the payload mostly does not contain --
# release Version 3 removed `bx` and the grade fields from all 195 prompts, and
# the task is served no pathology report -- so the MRI report's own indication
# line is the only place any of it can appear. Everything here reads
# `radiology_report`, which the Task 1 policy already reveals.
# --------------------------------------------------------------------------- #

def _prior(text: str) -> PriorContext:
    return extract_prior_context(DirectStore(_case(1, clinical={"radiology_report": text})))


@pytest.mark.parametrize("text,expected", [
    ("Indication: prior biopsy positive for adenocarcinoma.", "positive"),
    ("Indication: prior biopsy was negative.", "negative"),
    ("Previously negative biopsy; rising PSA.", "negative"),
    ("Positive previous biopsy of the left peripheral zone.", "positive"),
])
def test_a_stated_prior_biopsy_result_is_read(text, expected):
    """Only templates that name the polarity outright. Measured against the
    notes-derived status on every released case that states one: 28 of 28
    agree, and it never fires on a never-biopsied case."""
    assert _prior(text).biopsy_result == expected


def test_a_quoted_grade_settles_the_polarity():
    """A biopsy that produced a grade was a positive biopsy -- stated rather
    than inferred, since the report is quoting a histopathology result. Three of
    the 91 released reports do this, which is why the grade is parsed instead of
    being declared absent."""
    p = _prior("Re-evaluation of prior biopsy, ISUP grade group 2, left apex.")
    assert (p.prior_grade, p.biopsy_result) == (2, "positive")


def test_a_recommendation_to_biopsy_is_not_a_history():
    """The failure the `prior_care` pattern was tightened for: a bare `biops`
    matched the reports *recommending* one, which is every high-PI-RADS first
    presentation."""
    p = _prior("PI-RADS 5 lesion. Targeted biopsy is recommended.")
    assert not p.has_history and not p.prior_care


def test_this_studys_pirads_is_not_read_as_an_earlier_one():
    p = _prior("PI-RADS: 4. Prostate volume: 40 cc.")
    assert p.prior_pirads is None and not p.has_history


def test_interval_language_is_recognised_even_though_no_released_case_has_it():
    """0 of 91 released Task 1 reports state a comparison. Parsed anyway: 100 of
    the 250 test cases come from Karolinska, and a rationale saying "no
    comparison is reported" when one *is* reported would be a fabrication."""
    assert _prior("Compared with the prior MRI, interval growth of the lesion.").states_comparison
    assert _prior("The lesion is unchanged since the previous study.").states_comparison
    assert not _prior("PI-RADS 5 lesion in the left apex.").states_comparison


def test_an_unreadable_report_yields_an_empty_context():
    """Never raises, and an absent section is not evidence of anything."""
    assert extract_prior_context(DirectStore(_case(1))) == PriorContext()
    assert not _prior("").has_history


@pytest.mark.parametrize(
    "text",
    [
        "Histopathology from 2019 showed Gleason 3+4 adenocarcinoma.",
        "Known prostate cancer, referred for staging.",
        "Status post prostatectomy; rising PSA.",
    ],
)
def test_the_looser_reader_supplies_what_the_strict_patterns_miss(text):
    """None of these name a biopsy *and* its outcome in one clause.

    The strict patterns therefore abstain on all three, and the notes classifier
    settles them -- a quoted Gleason, an established diagnosis and a completed
    prostatectomy each require a positive specimen. This is the swap that takes
    the stated-history count from 28 of 91 to 63.
    """
    assert _prior(text).biopsy_result == "positive"


def test_a_never_biopsied_report_is_left_unstated_rather_than_asserted():
    """``classify_prior_biopsy`` answers ``none`` here; we refuse to repeat it.

    ``none`` is the classifier's *finding* that the prose contains no biopsy, and
    it is wrong on 2 of the 91 labelled cases -- both men who had in fact been
    biopsied. An abstention costs a sentence; asserting "no previous biopsy"
    about them would be a fabrication, and would also arm the history-gap clause
    via ``has_history``, since ``bool("none")`` is true.
    """
    prior = _prior("MRI of the prostate for elevated PSA. PI-RADS 3 lesion.")
    assert prior.biopsy_result is None
    assert not prior.has_history


def test_contradictory_polarities_still_abstain():
    """The classifier refuses to pick a side, and nothing downstream picks one."""
    assert _prior("Previous negative biopsy. Gleason 4+3 on the 2020 cores.").biopsy_result is None


# --------------------------------------------------------------------------- #
# Task 3: the csPCa tie-break
#
# The whole safety argument for consulting the MRI is that its weight is a *bound*
# rather than a tuned coefficient -- under one CAPRA-S point, so it can only order
# cases the nomogram scores equally. These pin that bound, because it is what makes
# the term safe on a cohort whose csPCa model may be calibrated differently.
# --------------------------------------------------------------------------- #

def _months(clinical: dict, psa: float | None) -> float:
    case = _case(3, {"psa": psa} if psa is not None else {}, clinical)
    return predict_months(case, DirectStore(case))


def test_cspca_never_crosses_a_full_capra_point():
    """The guarantee: a case cannot outrank one scoring a full point higher.

    Swept over the whole probability range, including the extremes, since the
    bound has to hold for any value Karolinska's model might emit.
    """
    lower = SURGICAL.replace("lymph node metastasis was present",
                             "there was no lymph node metastasis")   # one point less
    for p in (0.0, 0.01, 0.5, 0.99, 1.0):
        radiology = RADIOLOGY.replace("0.9527162", str(p))
        # The lower-risk specimen gets the most favourable csPCa, the higher-risk
        # one the least -- the hardest case for the bound.
        low = _months({"surgical_pathology_report": lower, "radiology_report": radiology}, 12.0)
        high = _months({"surgical_pathology_report": SURGICAL,
                        "radiology_report": RADIOLOGY.replace("0.9527162", "0.0")}, 12.0)
        assert high < low, f"csPCa {p} reordered across a CAPRA-S point"


def test_cspca_does_break_a_tie():
    """Bounded is not inert: equal CAPRA-S must still be ordered by the MRI."""
    same = {"surgical_pathology_report": SURGICAL}
    hot = _months({**same, "radiology_report": RADIOLOGY.replace("0.9527162", "0.95")}, 12.0)
    cold = _months({**same, "radiology_report": RADIOLOGY.replace("0.9527162", "0.05")}, 12.0)
    assert hot < cold


def test_absent_cspca_leaves_the_capra_ordering_untouched():
    """A report that omits the line must reproduce the pre-tie-break months exactly."""
    clinical = {"surgical_pathology_report": SURGICAL}
    p = _reports(3, clinical=clinical)
    assert p.cspca is None
    assert _months(clinical, 12.0) == pytest.approx(
        MONTHS_AT_ZERO_RISK - MONTHS_PER_CAPRA_POINT * capra_s(p, 12.0)
    )


def test_unreadable_specimen_still_falls_back_rather_than_ranking_on_the_mri_alone():
    """No CAPRA-S means no ordering claim at all -- the MRI must not stand in for it."""
    assert _months({"radiology_report": RADIOLOGY}, None) == pytest.approx(FALLBACK_MONTHS)


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
    assert eau_risk(_features(2, prompt)) == expected


def test_eau_unknown_field_cannot_produce_low_risk():
    """Low requires every criterion; missing data must not buy a low-risk label."""
    assert eau_risk(_features(2, {"bx_isup": 1, "psa": 5.0})) != "low"


def test_stratum_is_total():
    """Every case lands in a leaf, including one with no usable features at all."""
    for task in (1, 2):
        leaf = stratum(task, _features(task, {}))
        assert isinstance(leaf, str) and leaf


def test_low_grade_intermediate_splits_out_of_the_residual_band():
    """ISUP 1 at PSA 10-20 is intermediate only by PSA -- the split's whole point.

    That subgroup held 4 of the 7 training active-surveillance cases we were
    calling active_treatment, and none of the leaf's active_treatment cases.
    """
    f = _features(2, {"bx": "Positive", "bx_isup": 1, "psa": 13.5, "ct": "cT1c"})
    assert eau_risk(f) == "intermediate"
    assert stratum(2, f) == "positive_intermediate_isup1"


@pytest.mark.parametrize("isup", [2, 3])
def test_the_intermediate_residue_is_left_alone(isup):
    """3/12/2 is a coin flip; only ISUP 1 leaves the residual band."""
    f = _features(2, {"bx": "Positive", "bx_isup": isup, "psa": 13.5, "ct": "cT1c"})
    assert stratum(2, f) == "positive_intermediate"


def test_an_unknown_grade_cannot_buy_the_low_grade_leaf():
    """The mirror of `test_eau_unknown_field_cannot_produce_low_risk`.

    Missing data must not route a case toward the less aggressive label. An
    absent `bx_isup` stays in the residual band, which is the safe direction.
    """
    f = _features(2, {"bx": "Positive", "psa": 13.5, "ct": "cT1c"})
    assert eau_risk(f) == "intermediate"
    assert stratum(2, f) == "positive_intermediate"


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ({"bx_isup": 1, "psa": 5.0, "ct": "cT1c"}, "positive_low"),
        ({"bx_isup": 1, "psa": 25.0, "ct": "cT1c"}, "positive_high"),
        ({"bx_isup": 1, "psa": 5.0, "ct": "cT3a"}, "positive_high"),
    ],
)
def test_the_split_only_touches_the_intermediate_band(prompt, expected):
    """Low and high are decided before the split is consulted, so ISUP 1 alone
    must not pull a case out of either."""
    assert stratum(2, _features(2, {"bx": "Positive", **prompt})) == expected


def test_every_task_2_leaf_carries_a_fitted_label():
    """A leaf with no entry in `leaf_labels` would fall through to a default at
    inference time -- silently, and only on the cohort that reaches it."""
    params = json.loads(
        (Path("src/chimera/predictors/guideline_params.json")).read_text()
    )
    labels = params["task2"]["leaf_labels"]
    for leaf in TASK2_LEAVES:
        assert leaf in labels, f"{leaf} has no fitted label"
        assert labels[leaf] in spec.TREATMENT_DECISIONS
