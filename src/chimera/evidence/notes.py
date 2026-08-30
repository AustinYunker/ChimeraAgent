"""Prior-biopsy status recovered from the narrative sections.

Release Version 3 deleted the ``bx`` field from every Task 1 structured prompt --
195 of 195 cases now have no key at all, where Version 2 gave 116 Positive, 30
Negative and 49 None. The fact did not leave the data, only the patient card: it
is still stated in the radiology report and the referral notes, in prose. This
module recovers it from there.

**Nothing consumes it today, and that is the point.** Task 1 turned out to
stratify better on PI-RADS alone (see
:data:`chimera.models.guidelines.TASK1_LEAVES`), and Task 2 -- the one model that
does branch on prior-biopsy status -- still has ``bx`` on all 153 of its cards, so
:func:`chimera.evidence.structured.extract_structured` never reaches for this.
It is a guard against the trimming continuing. If the test release takes ``bx``
off Task 2's card the way Version 3 took it off Task 1's, every Task 2 case falls
into the ``unknown`` leaf and our strongest task -- 0.71 cross-validated against
0.27 for a constant -- degrades to a constant. Carrying this costs nothing while
the field is present.

**Measured, not assumed.** Version 2 is kept alongside Version 3 in
``data/train_release_v2`` purely as an answer key, and this classifier is scored
against it over all 195 Task 1 cases:

===========  =====  ========  ========  ========
true \\ pred   none  negative  positive  abstain
===========  =====  ========  ========  ========
none            49         0         0        0
negative         0        30         0        0
positive         1         0       114        1
===========  =====  ========  ========  ========

193/195 = 0.990, with **zero false positives** and both misses under-calling on
cases whose text genuinely never states the outcome. ``tests/test_evidence.py``
pins the rules against transcribed excerpts.

Two design rules follow from how this is used:

* **Abstain rather than guess.** ``None`` means "unknown", which the stratifier
  already handles as its own leaf; a wrong ``positive`` would route the case to
  the wrong leaf and, on Task 1, the gate turns that into a zero. So a text that
  mentions a biopsy without stating its outcome returns ``None``, and evidence
  pointing both ways returns ``None`` rather than picking a side.
* **Report what was read.** Reading these sections obliges declaring them in
  ``reveal_sequence``, so the extractor returns the sections it consumed
  alongside the answer. See :mod:`chimera.predictors.prior` for the rule.

Pure standard library, and never raises: 100 of the 250 test cases come from
Karolinska, whose referral prose we have never seen, and an unrecognised phrasing
must cost one feature rather than one case.
"""

from __future__ import annotations

import re
from typing import Any

from chimera.mcp.client import ClinicalStore

#: Clinical-data sections this module reads, in reveal-vocabulary order. Both are
#: in :data:`chimera.contract.spec.REVEAL_SECTIONS`, so both can be declared.
NOTE_SECTIONS: tuple[str, ...] = ("radiology_report", "previous_notes")

# Words that turn a mention of cancer into a *question* rather than a diagnosis.
# "prior negative biopsy; assess for clinically significant prostate cancer" names
# a prior biopsy and a cancer in one clause and means the opposite of a positive
# one, so the disease patterns below refuse to span any of these.
#
# `risk` deliberately is *not* here: "histopathology confirmed low-risk prostate
# cancer" is a positive, and hedging on `risk` cost exactly that case.
_HEDGE = (
    r"(?:negativ|suspic|suspect|assess|rule|occult|screen|evaluat|concern|query"
    r"|exclud|unreveal)"
)

#: Up to ``n`` characters within one sentence, none of which starts a hedge word.
def _span(n: int) -> str:
    return r"(?:(?!" + _HEDGE + r")[^.\n]){0," + str(n) + r"}"


#: Any one of these means cancer was found on a previous biopsy. A reported
#: Gleason score or ISUP grade is by itself conclusive -- both are assigned to
#: tissue, so neither exists without a positive specimen -- as is a prostatectomy.
_POSITIVE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\bpositive\b" + _span(60) + r"\b(?:biops\w*|sampling)\b",
        r"\b(?:biops\w*|sampling)\b" + _span(60) + r"\bpositive\b",
        r"\bgleason\b",
        r"\bISUP\b",
        r"\b(?:prior|previous(?:ly)?|earlier|established|known|documented|confirmed|"
        r"pre-?existing)\b" + _span(60) + r"\b(?:prostat\w+\s+)?"
        r"(?:cancer|malignan\w+|carcinoma|adenocarcinoma|disease|PCa)\b",
        r"\b(?:cancer|malignan\w+|carcinoma|adenocarcinoma|PCa)\b" + _span(40)
        + r"\b(?:diagnos\w+|confirm\w+|identified)\b",
        r"\b(?:diagnos\w+|confirm\w+)\b" + _span(40)
        + r"\b(?:cancer|malignan\w+|carcinoma|adenocarcinoma)\b",
        r"\b(?:biops\w*|sampling)\b[^.\n]{0,60}\b(?:adenocarcinoma|carcinoma)\b",
        r"\bprostatectom\w*",
    )
)

#: A biopsy that found nothing. No hedge guard here: these patterns already
#: require the negative word and the biopsy word in the same clause.
_NEGATIVE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.I)
    for p in (
        r"\b(?:negative|unrevealing|benign)\b[^.\n]{0,60}\b(?:biops\w*|sampling)\b",
        r"\b(?:biops\w*|sampling)\b[^.\n]{0,80}\b(?:negative|unrevealing|benign)\b",
    )
)

#: A biopsy is mentioned at all. Separates "never biopsied" from "biopsied, and
#: the note does not say what it showed" -- the first is the informative ``none``
#: category, the second has to abstain.
_MENTIONED = re.compile(
    r"\bbiops\w*|\b(?:needle|systematic|initial|prior|previous)\s+sampling\b", re.I
)


def mentions_biopsy(text: str) -> bool:
    """Does ``text`` name a biopsy at all?

    :func:`classify_prior_biopsy` answers ``positive`` from cancer-history
    phrasings that never say "biopsy" -- "known prostate cancer", "status post
    prostatectomy". The polarity it returns is right either way, but the *claim
    shape* is not: a caller that renders every ``positive`` as "a previous
    biopsy positive for cancer" asserts a procedure the source never mentioned.
    This is the test that separates the two, so the prose can track its source.
    """
    return bool(text) and bool(_MENTIONED.search(text))


def classify_prior_biopsy(text: str) -> str | None:
    """``positive`` / ``negative`` / ``none``, or ``None`` when the text is unclear.

    ``none`` is a *finding* -- the notes describe a work-up with no biopsy in it --
    and matches the string the Version 2 patient card used. ``None`` is an
    abstention. Contradictory evidence abstains rather than preferring a polarity;
    on this cohort that costs one case and buys zero false positives.
    """
    if not text:
        return None
    positive = any(p.search(text) for p in _POSITIVE)
    negative = any(p.search(text) for p in _NEGATIVE)
    if positive and negative:
        return None
    if positive:
        return "positive"
    if negative:
        return "negative"
    if _MENTIONED.search(text):
        return None  # a biopsy happened; the outcome is not stated
    return "none"


def _section_text(value: Any) -> str:
    """Readable prose from a clinical-data section, whatever shape it arrives in.

    ``previous_notes`` is a list of note records in Tasks 1 and 2 and a plain
    string in Task 3. Within a record only ``text`` is taken when it exists --
    the metadata fields around it (dates, specialties) add words this module
    matches on without adding facts.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        body = value.get("text")
        if isinstance(body, str):
            return body
        return "\n".join(v for v in value.values() if isinstance(v, str))
    if isinstance(value, list):
        return "\n".join(part for part in (_section_text(v) for v in value) if part)
    return ""


def prior_biopsy_from_notes(store: ClinicalStore) -> tuple[str | None, tuple[str, ...]]:
    """Read :data:`NOTE_SECTIONS` and classify. Returns ``(status, sections_read)``.

    ``sections_read`` names the sections that actually carried text, which is
    what may be declared in ``reveal_sequence``. It is returned even when the
    classification abstains: we read them either way, and the reveal has to
    describe what was retrieved rather than what was useful.

    Both sections are fetched through the store, so each is a tool call and the
    caller's declaration is backed by the store's ledger rather than by this
    return value alone.
    """
    read: list[str] = []
    parts: list[str] = []
    for section in NOTE_SECTIONS:
        text = _section_text(store.section(section))
        if text.strip():
            read.append(section)
            parts.append(text)

    if not parts:
        return None, ()
    return classify_prior_biopsy("\n".join(parts)), tuple(read)
