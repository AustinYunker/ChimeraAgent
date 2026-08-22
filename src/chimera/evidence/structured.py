"""Features from the structured prompt (the "patient card").

For Tasks 1 and 2 this is the whole feature source: the numeric panel is populated
in 100% of released cases and the biopsy grades in 74% / 100%, so there is no need
to parse the narrative reports -- which, unlike Task 3's, are genuinely varied prose
with per-hospital headers and phrasings. Task 3 is the reverse case and is handled
in :mod:`chimera.evidence.reports`.

Measured fill rates over ``data/train_release`` (195 / 153 / 75 cases) drove what is
here. Worth recording what is *not* here, because the field names invite the mistake:
``ct``, ``cores_positive``, ``cores_total``, ``max_core_pct``, ``pni``, ``lvi``,
``growth_pattern``, ``high_risk_patterns``, ``tumor_location`` and ``bx_isup_pred``
appear as keys on every Task 1 case and are empty on all 195. ``ct`` *is* populated
for Task 2. ``active_treatment_flag`` is 0 for every case in both tasks and carries
no information.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any

from chimera.contract.io import CaseInputs
from chimera.evidence.notes import prior_biopsy_from_notes

#: Clinical T stage, ordered. `cTx` means "not assessable" and is not a stage, so it
#: maps to ``None`` rather than to a low value -- treating unknown as early-stage
#: would systematically under-risk exactly the patients least well characterised.
_CT_ORDER: tuple[str, ...] = (
    "cT1", "cT1a", "cT1b", "cT1c",
    "cT2", "cT2a", "cT2b", "cT2c",
    "cT3", "cT3a", "cT3b",
    "cT4",
)
_CT_ORDINAL: dict[str, int] = {stage: i for i, stage in enumerate(_CT_ORDER)}

#: Observed DRE vocabulary. `Not done` is distinct from `Normal`: absence of a
#: finding is not a normal finding.
_DRE_VALUES = frozenset({"normal", "nodus", "abnormal", "suspicious", "not done"})
_DRE_ABNORMAL = frozenset({"nodus", "abnormal", "suspicious"})

#: Observed prior-biopsy vocabulary. `None` here is the string "None" meaning
#: "never biopsied", not a missing value -- and it carries real signal.
_BX_VALUES = frozenset({"positive", "negative", "none"})

#: Tasks whose decision model actually branches on prior-biopsy status, and so may
#: pay to go looking for it when the patient card omits it.
#:
#: Only Task 2 does. Task 3 never used it, and Task 1 dropped it at C2 because
#: PI-RADS alone stratifies better out of fold (see
#: :data:`chimera.models.guidelines.TASK1_LEAVES`). The gate is not tidiness: the
#: evaluator's tool score is *precision* over the declared reveals, so reading a
#: section costs real points, and a feature no leaf consults is pure cost. One
#: line to widen if a future release takes ``bx`` off another task's card too.
_TASKS_USING_PRIOR_BIOPSY = frozenset({2})


def as_float(value: Any) -> float | None:
    """Best-effort numeric read. Returns ``None`` rather than raising."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def as_int(value: Any) -> int | None:
    f = as_float(value)
    return int(f) if f is not None else None


def pirads(prompt: dict[str, Any]) -> int | None:
    """PI-RADS 1-5. Stored as a string, and literally ``"NA"`` in one case."""
    value = prompt.get("pirads")
    if not isinstance(value, (str, int, float)):
        return None
    score = as_int(value)
    return score if score is not None and 1 <= score <= 5 else None


def ct_stage(prompt: dict[str, Any]) -> str | None:
    """Normalised clinical T stage, e.g. ``cT2a``. ``cTx`` -> ``None``."""
    value = prompt.get("ct")
    if not isinstance(value, str):
        return None
    match = re.search(r"c?T([1-4][a-c]?)", value, re.I)
    if not match:
        return None
    stage = f"cT{match.group(1).lower()}"
    return stage if stage in _CT_ORDINAL else None


def ct_ordinal(prompt: dict[str, Any]) -> int | None:
    stage = ct_stage(prompt)
    return _CT_ORDINAL.get(stage) if stage else None


def dre(prompt: dict[str, Any]) -> str | None:
    """Normalised DRE category, or ``None`` when absent or free text.

    Task 3 stores a whole sentence here ("Digital rectal examination was abnormal
    bilaterally.") rather than a category, so this returns ``None`` there by design
    and Task 3 takes its findings from the reports instead.
    """
    value = prompt.get("dre")
    if not isinstance(value, str):
        return None
    normalised = value.strip().lower()
    return normalised if normalised in _DRE_VALUES else None


def dre_abnormal(prompt: dict[str, Any]) -> int | None:
    """1 if the DRE found something, 0 if explicitly normal, ``None`` if not done."""
    category = dre(prompt)
    if category is None or category == "not done":
        return None
    return 1 if category in _DRE_ABNORMAL else 0


def prior_biopsy(prompt: dict[str, Any]) -> str | None:
    """``positive`` / ``negative`` / ``none``, where ``none`` means never biopsied.

    Release Version 3 removed this key from every Task 1 prompt (Task 2 kept it),
    so a ``None`` here is now the common case for Task 1 and
    :func:`chimera.evidence.notes.prior_biopsy_from_notes` takes over --
    see :func:`extract_structured`.
    """
    value = prompt.get("bx")
    if not isinstance(value, str):
        return None
    normalised = value.strip().lower()
    return normalised if normalised in _BX_VALUES else None


def ipss(prompt: dict[str, Any]) -> int | None:
    """The IPSS symptom score out of 35, dug out of ``"IPSS score: 12/35 (...)"``."""
    value = prompt.get("ipss")
    if not isinstance(value, str):
        return None
    match = re.search(r"(\d+)\s*/\s*35", value)
    return int(match.group(1)) if match else None


def comorbidity_count(prompt: dict[str, Any]) -> int | None:
    """Number of past-medical-history entries; ``None`` when the field is absent.

    An empty list is a real zero -- the field was recorded and held nothing -- so it
    is distinguished from the key being missing entirely.
    """
    value = prompt.get("pmhx")
    return len(value) if isinstance(value, list) else None


@dataclass(slots=True)
class StructuredFeatures:
    """The Task 1/2 panel. Every field is optional; none of them is ever invented."""

    age: float | None = None
    psa: float | None = None
    psad: float | None = None
    psav: float | None = None
    psap: float | None = None
    vol: float | None = None
    cspca: float | None = None
    months: float | None = None
    pirads: int | None = None
    ct_ordinal: int | None = None
    bx_isup: int | None = None
    bx_gl_prim: int | None = None
    bx_gl_sec: int | None = None
    dre_abnormal: int | None = None
    ipss: int | None = None
    comorbidity_count: int | None = None
    #: Kept as categories rather than numbers; the model one-hots them.
    dre: str | None = None
    prior_biopsy: str | None = None
    #: Clinical-data sections that had to be *read* to produce these features,
    #: as opposed to those taken from the patient card. Empty whenever the card
    #: answered everything. Carried here rather than recomputed downstream so
    #: the declared ``reveal_sequence`` is derived from the same call that did
    #: the reading -- see :mod:`chimera.evidence.notes`.
    evidence_sections: tuple[str, ...] = ()

    #: Not features: excluded from :meth:`numeric` and from the model's inputs.
    _NON_NUMERIC = ("dre", "prior_biopsy", "evidence_sections")

    def numeric(self) -> dict[str, float | None]:
        """Just the orderable fields, for a linear model."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in self._NON_NUMERIC
        }

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def extract_structured(case: CaseInputs) -> StructuredFeatures:
    """Read the patient card, falling back to the notes for what it no longer says.

    Never raises; unreadable fields stay ``None``.

    The only fallback is prior-biopsy status, and it fires only when both are true:
    the card omits ``bx`` entirely, and this task's model actually branches on it
    (:data:`_TASKS_USING_PRIOR_BIOPSY`). The card wins wherever it speaks -- it is
    the organizers' own coding.

    On release Version 3 that means the fallback never fires: Task 2 is the only
    consumer and Task 2's card still carries ``bx`` on all 153 cases. It is there
    for the test cohort. Version 3 already deleted ``bx`` from all 195 Task 1
    prompts, and if the same trimming reaches Task 2 then every case collapses
    into the ``unknown`` leaf and Task 2 -- our strongest task by a wide margin --
    degrades to a constant. The cost of carrying this guard is zero while the
    field is present; the cost of not carrying it is the whole task.
    """
    prompt = case.structured_prompt if isinstance(case.structured_prompt, dict) else {}

    bx = prior_biopsy(prompt)
    evidence_sections: tuple[str, ...] = ()
    if bx is None and case.task in _TASKS_USING_PRIOR_BIOPSY:
        bx, evidence_sections = prior_biopsy_from_notes(case.clinical_data)

    return StructuredFeatures(
        age=as_float(prompt.get("age")),
        psa=as_float(prompt.get("psa")),
        psad=as_float(prompt.get("psad")),
        psav=as_float(prompt.get("psav")),
        psap=as_float(prompt.get("psap")),
        vol=as_float(prompt.get("vol")),
        cspca=as_float(prompt.get("cspca")),
        months=as_float(prompt.get("months")),
        pirads=pirads(prompt),
        ct_ordinal=ct_ordinal(prompt),
        bx_isup=as_int(prompt.get("bx_isup")),
        bx_gl_prim=as_int(prompt.get("bx_gl_prim")),
        bx_gl_sec=as_int(prompt.get("bx_gl_sec")),
        dre_abnormal=dre_abnormal(prompt),
        ipss=ipss(prompt),
        comorbidity_count=comorbidity_count(prompt),
        dre=dre(prompt),
        prior_biopsy=bx,
        evidence_sections=evidence_sections,
    )
