"""The clinical-document tool registry, one tool per masked section.

Each tool returns exactly one section of the case's clinical-data socket. That
one-to-one shape is the point: the urologist forms hid each "Extended EHR view"
document behind its own click, so revealing one is a separate act, and the
retrieval a submission declares is a list of sections rather than a single
"read the record" call.

Six of the seven sections are in :data:`chimera.contract.spec.REVEAL_SECTIONS`
and may therefore be declared in ``reveal_sequence``.
``surgical_pathology_report`` is the exception: Task 3 needs it for CAPRA-S, but
it is outside the reveal vocabulary, so declaring it would register as an extra
reveal and cost tool-efficiency precision. It is retrievable and never declared,
which is consistent rather than contradictory -- the vocabulary bounds what can
be *said*, not what may be *read*.

Registries are per task because the tasks mask different documents: Task 1 has
no biopsy report to offer (its cohort is pre-biopsy), and Task 3's
post-treatment record carries no PSA series or laboratory panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chimera.contract import spec


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool name bound to the clinical-data section it returns."""

    name: str
    section: str
    description: str

    @property
    def declarable(self) -> bool:
        """Whether retrieving this may be named in ``reveal_sequence``."""
        return self.section in spec.REVEAL_SECTIONS

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the call arguments, as ``tools/list`` must publish."""
        return {
            "type": "object",
            "properties": {
                "case_id": {
                    "type": "string",
                    "description": "Identifier of the case to retrieve this section for.",
                }
            },
            "required": ["case_id"],
        }


PSA_TREND = ToolSpec(
    name="get_psa_trend",
    section="psa_trend",
    description=(
        "Retrieve the patient's serial PSA measurements as dated values. The "
        "single current PSA is already on the patient card; this is the "
        "trajectory behind it."
    ),
)

LAB_RESULTS = ToolSpec(
    name="get_lab_results",
    section="laboratory_results",
    description=(
        "Retrieve the full laboratory panel as dated, flagged results -- PSA "
        "and free PSA, haematology, renal function, testosterone and the "
        "remaining chemistry."
    ),
)

MRI_REPORT = ToolSpec(
    name="get_mri_report",
    section="radiology_report",
    description=(
        "Retrieve the radiologist's mpMRI report as free text. The headline "
        "imaging numbers (PI-RADS, prostate volume, PSA density, predicted "
        "csPCa probability) are already on the card; this is the prose behind "
        "them, including lesion location and extraprostatic assessment."
    ),
)

PATHOLOGY_REPORT = ToolSpec(
    name="get_pathology_report",
    section="pathology_report",
    description=(
        "Retrieve the needle-biopsy pathology report as free text: Gleason "
        "patterns, ISUP grade group, core counts and involvement. Returns no "
        "section for a patient who has not been biopsied."
    ),
)

SURGICAL_PATHOLOGY_REPORT = ToolSpec(
    name="get_surgical_pathology_report",
    section="surgical_pathology_report",
    description=(
        "Retrieve the radical-prostatectomy specimen report as free text: "
        "post-operative Gleason and ISUP grade, pathological stage, surgical "
        "margins, extraprostatic extension, seminal-vesicle invasion and nodal "
        "status. Returns no section for a patient who has not had surgery."
    ),
)

PREVIOUS_NOTES = ToolSpec(
    name="get_previous_notes",
    section="previous_notes",
    description=(
        "Retrieve prior general-practice and urology consultation notes, "
        "including the referral, as dated authored entries."
    ),
)

FAMILY_HISTORY = ToolSpec(
    name="get_family_history",
    section="family_history",
    description=(
        "Ask the patient about first-degree family history of prostate cancer. "
        "This is anamnesis taken during the consultation rather than a document "
        "already in the record, so it exists only if it is requested."
    ),
)


#: Which documents each task's form masked. Task 1 is a pre-biopsy cohort and
#: has no biopsy report; Task 3 is post-treatment and carries no PSA series or
#: laboratory panel.
TOOLS_BY_TASK: dict[int, tuple[ToolSpec, ...]] = {
    1: (PSA_TREND, LAB_RESULTS, MRI_REPORT, PREVIOUS_NOTES, FAMILY_HISTORY),
    2: (
        PSA_TREND,
        LAB_RESULTS,
        MRI_REPORT,
        PATHOLOGY_REPORT,
        PREVIOUS_NOTES,
        FAMILY_HISTORY,
    ),
    3: (
        MRI_REPORT,
        PATHOLOGY_REPORT,
        SURGICAL_PATHOLOGY_REPORT,
        PREVIOUS_NOTES,
        FAMILY_HISTORY,
    ),
}

ALL_TOOLS: tuple[ToolSpec, ...] = (
    PSA_TREND,
    LAB_RESULTS,
    MRI_REPORT,
    PATHOLOGY_REPORT,
    SURGICAL_PATHOLOGY_REPORT,
    PREVIOUS_NOTES,
    FAMILY_HISTORY,
)

TOOL_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in ALL_TOOLS}


def tools_for_task(task: int) -> tuple[ToolSpec, ...]:
    """The registry for ``task``; empty for an unknown task rather than raising."""
    return TOOLS_BY_TASK.get(task, ())


def section_is_present(value: Any) -> bool:
    """Whether a retrieved section carries usable content.

    Sections are heterogeneous -- ``radiology_report`` is a string, ``psa_trend``
    a list of records -- so this tests emptiness rather than type. Both ends use
    it: the server to decide whether to include a key at all, and the client to
    decide whether the call counts as a retrieval. Keeping one definition is
    what stops "the server sent it" and "we retrieved it" from disagreeing, and
    a disagreement there would show up as a dishonest ``reveal_sequence``.
    """
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


def tool_for_section(task: int, section: str) -> ToolSpec | None:
    """The tool serving ``section`` on ``task``, or ``None`` if it is not masked there.

    ``None`` is the honest answer for a section this task does not expose, and
    callers treat it as "no data" -- the same outcome as a case that simply has
    nothing under that key.
    """
    for tool in tools_for_task(task):
        if tool.section == section:
            return tool
    return None
