"""Features from the clinical reports -- primarily Task 3's surgical pathology.

Task 3's structured prompt carries almost nothing: age, PSA, and a free-text DRE
sentence in 69% of cases. Everything that predicts biochemical recurrence -- grade,
stage, margins, extraprostatic extension, seminal-vesicle invasion, nodal status --
lives in the ``surgical_pathology_report`` string. Fortunately that report is
templated machine-generated prose, and over the 75 released cases every one of those
fields parses at 100%.

The templates admit a small number of phrasings for the same fact, and both
polarities have to be recognised explicitly::

    "Extraprostatic extension was present"   -> True
    "There was no extraprostatic extension"  -> False

Absence of a match therefore means *unknown*, not *negative*, which matters: a
missing field must not be silently scored as a favourable finding.

One distinction is clinical rather than textual. ``"no lymph nodes were removed"``
is pNx -- nodes were never assessed -- and is genuinely different from ``"there was
no lymph node metastasis"`` (pN0). Collapsing the two would tell a survival model
that 43 unstaged patients were node-negative.

**Robustness is the point of this module, not completeness.** 100 of the 250 test
cases come from Karolinska, whose report templates we have never seen. Every
function returns ``None`` when it cannot find what it is looking for, and
:class:`SurgicalPathology` is designed to be usefully incomplete.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any

from chimera.evidence.notes import classify_prior_biopsy, mentions_biopsy
from chimera.mcp.client import ClinicalStore

#: Nodal status when the pelvic nodes were never sampled (pNx).
NOT_ASSESSED = "not_assessed"

#: A decimal number that stops before a sentence-ending period.
_NUM = r"(\d+(?:\.\d+)?)"


def _text(store: ClinicalStore, section: str) -> str:
    """A section's text, or ``""`` -- sections are sometimes ``None`` or a list."""
    value = store.section(section)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # `previous_notes` is a list of note records in Tasks 1/2 and a plain string
        # in Task 3; flatten so either shape is searchable.
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.extend(str(v) for v in item.values())
        return "\n".join(parts)
    return ""


def _search(text: str, pattern: str) -> re.Match[str] | None:
    return re.search(pattern, text, re.I)


def _polar(text: str, positive: str, negative: str) -> bool | None:
    """``True`` / ``False`` / ``None`` for a fact stated in either polarity.

    Negative is tested first: several templates phrase the negative as a superset of
    the positive ("There was no extraprostatic extension" contains "extraprostatic
    extension"), so checking the positive first would invert the answer.
    """
    if _search(text, negative):
        return False
    if _search(text, positive):
        return True
    return None


@dataclass(slots=True)
class SurgicalPathology:
    """Post-prostatectomy findings. Any field may be ``None``."""

    gleason_primary: int | None = None
    gleason_secondary: int | None = None
    isup: int | None = None
    pt_stage: str | None = None
    epe: bool | None = None
    positive_margins: bool | None = None
    svi: bool | None = None
    lvi: bool | None = None
    #: ``True`` / ``False`` / :data:`NOT_ASSESSED` / ``None``.
    lymph_nodes: bool | str | None = None
    #: From the radiology report rather than the pathology one.
    pirads: int | None = None
    prostate_volume: float | None = None
    psa_density: float | None = None
    cspca: float | None = None
    #: Charlson Comorbidity Index; stated in only ~25% of cases.
    cci: int | None = None

    @property
    def gleason_sum(self) -> int | None:
        if self.gleason_primary is None or self.gleason_secondary is None:
            return None
        return self.gleason_primary + self.gleason_secondary

    def as_dict(self) -> dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["gleason_sum"] = self.gleason_sum
        return out


def extract_reports(store: ClinicalStore) -> SurgicalPathology:
    """Parse whatever the case's reports state. Never raises.

    Takes a store rather than the case: every one of these four documents is a
    masked section, so reaching it is a tool call, and the calls this makes are
    recorded in the store's ledger for the caller to declare.
    """
    surgical = _text(store, "surgical_pathology_report")
    pathology = _text(store, "pathology_report")
    radiology = _text(store, "radiology_report")
    notes = _text(store, "previous_notes")

    # Grade comes from the prostatectomy specimen when available; the biopsy report
    # is the fallback, since post-surgical grade is the one CAPRA-S wants.
    grade_text = surgical if _search(surgical, r"Gleason") else pathology

    gleason = _search(grade_text, r"Gleason\s*(\d)\s*\+\s*(\d)")
    isup = _search(grade_text, r"ISUP\s+grade\s+group\s+(\d)")
    stage = _search(surgical, r"pathological stage\s+(pT\w+)")

    nodes: bool | str | None
    if _search(surgical, r"no lymph nodes were (?:removed|sampled|examined)"):
        nodes = NOT_ASSESSED
    else:
        nodes = _polar(
            surgical,
            r"lymph node metastas[ei]s was present|positive lymph nodes?",
            r"(?:was|were)?\s*no lymph node metastas[ei]s|lymph node metastas[ei]s was absent",
        )

    # `_NUM` rather than `[\d.]+`: the latter is greedy and swallows a
    # sentence-ending period, so a value at the end of a sentence parses as
    # "0.9527162." and fails the float cast. Values followed by a unit happen to
    # escape that, which is exactly the kind of bug that hides in a coverage check.
    volume = _search(radiology, rf"[Pp]rostate volume:\s*{_NUM}")
    density = _search(radiology, rf"PSA density:\s*{_NUM}")
    pirads = _search(radiology, r"PI-?RADS:?\s*(\d)")
    cspca = _search(radiology, rf"clinically significant prostate cancer[^:]*:\s*{_NUM}")
    cci = _search(notes, r"(?:Charlson Comorbidity Index|CCI)\D{0,12}(\d+)")

    def _num(match: re.Match[str] | None, cast):
        if match is None:
            return None
        try:
            return cast(match.group(1))
        except (TypeError, ValueError):
            return None

    return SurgicalPathology(
        gleason_primary=_num(gleason, int),
        gleason_secondary=int(gleason.group(2)) if gleason else None,
        isup=_num(isup, int),
        pt_stage=stage.group(1) if stage else None,
        epe=_polar(
            surgical,
            r"[Ee]xtraprostatic extension was present",
            r"no extraprostatic extension|extraprostatic extension was absent",
        ),
        positive_margins=_polar(
            surgical,
            r"surgical margins were positive",
            r"surgical margins were negative|no positive surgical margins",
        ),
        svi=_polar(
            surgical,
            r"seminal vesicles were invaded|seminal vesicle invasion was present",
            r"seminal vesicles were not invaded|no seminal vesicle invasion"
            r"|seminal vesicle invasion was absent",
        ),
        lvi=_polar(
            surgical,
            r"lymphovascular invasion was present",
            r"lymphovascular invasion was absent|no lymphovascular invasion",
        ),
        lymph_nodes=nodes,
        pirads=_num(pirads, int),
        prostate_volume=_num(volume, float),
        psa_density=_num(density, float),
        cspca=_num(cspca, float),
        cci=_num(cci, int),
    )


# --------------------------------------------------------------------------- #
# Task 1: the history the MRI report carries, and the gaps it leaves
# --------------------------------------------------------------------------- #

#: An explicitly stated prior biopsy result. Only the templates that name the
#: polarity outright -- an inferred result would be a claim we cannot corroborate.
_PRIOR_BIOPSY_RESULT = (
    r"prior (?:biopsy|bx)[^.;|]{0,30}?\b(positive|negative)\b"
    r"|previously (positive|negative) biopsy"
    r"|(positive|negative) (?:prior|previous|earlier) biops"
)

#: The only two answers :func:`extract_prior_context` will take from the notes
#: classifier. It also returns ``none`` -- a positive finding that the prose
#: describes a work-up containing no biopsy -- which is refused there.
_STATED_POLARITIES = frozenset({"positive", "negative"})

#: An ISUP grade group from a *previous* histopathology, as quoted by the MRI
#: report. A graded biopsy is a positive one, so this also settles the polarity.
_PRIOR_GRADE = r"ISUP\s*(?:grade\s*group\s*|GG\s*)?(\d)\b"

#: A PI-RADS score attributed to an earlier study rather than to this one.
_PRIOR_PIRADS = r"(?:earlier|previous|prior)\s+PI-?RADS\s*\"?(\d)"

#: Evidence that this patient is already under prostate-cancer care.
_PRIOR_CARE = (
    r"active surveillance|previously diagnosed|prior prostate cancer"
    r"|known (?:prostate )?(?:cancer|malignancy)|re-evaluation of prior"
    r"|prior (?:biopsy|bx)|previous biopsy|previous histopathology|earlier biops"
)

#: An interval comparison against a previous study. **No released Task 1 report
#: contains one** -- 0 of 91 -- which is precisely the gap a third of the reference
#: rationales complain about. It is parsed anyway rather than assumed absent,
#: because a rationale that says "no comparison is reported" when one *is* reported
#: would be the same fabrication we are trying to avoid, and the Karolinska
#: templates are unseen.
_COMPARISON = (
    r"compared (?:with|to) (?:the )?(?:prior|previous|earlier)"
    r"|interval (?:growth|increase|change|progression)"
    r"|\b(?:stable|stabile|unchanged)\b"
    r"|no change since|since the (?:prior|previous|earlier)"
)


@dataclass(slots=True)
class PriorContext:
    """What the MRI report says about this patient's history -- and what it omits.

    Task 1's decision hangs on a history the payload mostly does not contain. The
    reference rationales say so out loud: 24% of the prior-biopsy-positive cases
    are the urologist asking for the earlier ISUP grade or for whether the lesion
    has grown, against 4% of the never-biopsied ones. Neither fact is on the
    patient card -- release Version 3 removed ``bx`` and the grade fields from all
    195 Task 1 prompts -- and Task 1 is served no pathology report, so the MRI
    report's own indication line is the only place either can appear.

    Every field is read from ``radiology_report``, which the Task 1 policy already
    reveals, so this costs no tool call and does not touch ``reveal_sequence``.
    """

    #: ``positive`` / ``negative``, where the report states it outright. Filled
    #: first by this module's strict patterns and then, where those abstain, by
    #: :func:`chimera.evidence.notes.classify_prior_biopsy` reading the same
    #: section -- see :func:`extract_prior_context` for why the looser reader is
    #: the safe one here. Never ``none``: this field's job is to name a history,
    #: and the absence of one is not something to assert.
    biopsy_result: str | None = None
    #: Does the report name a biopsy, or only a cancer history? The looser
    #: reader answers ``positive`` from "re-evaluation of prior prostate cancer
    #: diagnosis" as readily as from "previously positive biopsy", and the
    #: polarity is right both times -- but only the second licenses the word
    #: *biopsy* in the rationale. The Aug 30 debug run is the evidence: the
    #: platform judge accepted "a previous biopsy positive for cancer" on the two
    #: cases whose report says "previously positive biopsy" and called it an
    #: unsupported claim on the one whose report says only "prior prostate cancer
    #: diagnosis". Consumed by the rationale, never by the decision.
    states_biopsy: bool = False
    #: ISUP grade group of the earlier biopsy. Stated in 3 of 91 cases.
    prior_grade: int | None = None
    #: PI-RADS attributed to an earlier study. Present on 5 never-biopsied cases,
    #: which are men with a prior MRI and no biopsy -- prior context all the same.
    prior_pirads: int | None = None
    #: The patient is already under prostate-cancer care.
    prior_care: bool = False
    #: The report compares this study against a previous one.
    states_comparison: bool = False

    @property
    def has_history(self) -> bool:
        """Is there any earlier episode for the record to be silent about?"""
        return bool(
            self.biopsy_result
            or self.prior_grade is not None
            or self.prior_pirads is not None
            or self.prior_care
        )


def extract_prior_context(store: ClinicalStore) -> PriorContext:
    """Read the Task 1 history out of the MRI report. Never raises.

    The prior-biopsy polarity has two readers, tried strictest first. This
    module's own patterns want the report to name the biopsy and its outcome in
    one clause, which is unambiguous but rare -- 28 of the 91 labelled cases.
    :mod:`chimera.evidence.notes` accepts the many other ways a report says the
    same thing, chiefly a quoted Gleason score or ISUP grade, neither of which
    exists without a positive specimen.

    Handing this field to the looser reader is a measurement, not a relaxation.
    Scored over all 91 labelled Task 1 cases against the Version 2 patient card,
    which still carried ``bx``, ``classify_prior_biopsy`` on ``radiology_report``
    alone is right 87 times and **never states the wrong polarity**; its four
    misses all under-call. Restricted to the cases where it answers ``positive``
    or ``negative`` -- the only two this function accepts from it -- it is 63 for
    63. So the swap converts 35 abstentions into stated history and introduces no
    claim the report does not support.

    ``none`` is refused on purpose. Two of the four misses are under-calls *to*
    ``none``, and unlike an abstention that is an assertion: it would tell the
    judge a man with a positive prior biopsy had never been biopsied. It would
    also arm :attr:`PriorContext.has_history`, since ``bool("none")`` is true,
    and so put "no comparison with prior imaging is reported" on never-biopsied
    men who have no prior imaging to compare against.

    Both readers see only ``radiology_report``, which the Task 1 policy already
    declares, so nothing here costs a tool call or changes ``reveal_sequence``.
    Adding ``previous_notes`` would carry the classifier to 91 of 91, but it buys
    4 further cases for a measured -0.0045 on Task 1 through tool precision, so
    it is deliberately not read. See ``docs/plan.md``.
    """
    text = _text(store, "radiology_report")
    if not text:
        return PriorContext()

    match = _search(text, _PRIOR_BIOPSY_RESULT)
    result = next((g.lower() for g in match.groups() if g), None) if match else None
    if result is None:
        classified = classify_prior_biopsy(text)
        result = classified if classified in _STATED_POLARITIES else None

    grade_match = _search(text, _PRIOR_GRADE)
    grade = int(grade_match.group(1)) if grade_match else None
    if grade is not None and not 1 <= grade <= 5:
        grade = None
    # A biopsy that produced a grade was a positive biopsy. Stated rather than
    # inferred: the report is quoting a histopathology result.
    if grade is not None and result is None:
        result = "positive"

    pirads_match = _search(text, _PRIOR_PIRADS)
    prior_pirads = int(pirads_match.group(1)) if pirads_match else None
    if prior_pirads is not None and not 1 <= prior_pirads <= 5:
        prior_pirads = None

    return PriorContext(
        biopsy_result=result,
        # A quoted grade is a quoted histopathology result, so it names a biopsy
        # as surely as the word does.
        states_biopsy=mentions_biopsy(text) or grade is not None,
        prior_grade=grade,
        prior_pirads=prior_pirads,
        prior_care=bool(_search(text, _PRIOR_CARE)),
        states_comparison=bool(_search(text, _COMPARISON)),
    )
