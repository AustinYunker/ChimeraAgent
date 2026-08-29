"""Turning a case into features.

Everything here takes a :class:`~chimera.contract.io.CaseInputs` -- the object the
container builds at inference time -- rather than a file path or a dataframe row.
That is deliberate: training and serving then compute features through the same
code, so the two cannot silently diverge.

Two constraints apply to every extractor in this package:

* **Pure standard library.** These modules run inside the submission image, which
  installs with ``--no-deps``. ``tests/test_entrypoint.py`` enforces it.
* **Never raise, always degrade.** An unparseable field returns ``None``. 100 of the
  250 test cases come from Karolinska with different report templates and scanners,
  and a crashed case is scored against a sentinel label rather than skipped -- so a
  missing feature must cost a little accuracy, never a whole case.

A third constraint applies to :mod:`chimera.evidence.notes`, which reads the
narrative sections rather than the patient card: whatever it reads has to be
declared in ``reveal_sequence``, so it reports the sections it consumed and
:class:`~chimera.evidence.structured.StructuredFeatures` carries them through.
"""

from chimera.evidence.notes import (
    NOTE_SECTIONS,
    classify_prior_biopsy,
    prior_biopsy_from_notes,
)
from chimera.evidence.reports import (
    PriorContext,
    SurgicalPathology,
    extract_prior_context,
    extract_reports,
)
from chimera.evidence.structured import StructuredFeatures, extract_structured

__all__ = [
    "NOTE_SECTIONS",
    "PriorContext",
    "StructuredFeatures",
    "SurgicalPathology",
    "classify_prior_biopsy",
    "extract_prior_context",
    "extract_reports",
    "extract_structured",
    "prior_biopsy_from_notes",
]
