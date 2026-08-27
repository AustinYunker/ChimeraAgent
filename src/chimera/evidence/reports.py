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
