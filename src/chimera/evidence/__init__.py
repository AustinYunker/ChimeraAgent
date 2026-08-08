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
"""

from chimera.evidence.reports import SurgicalPathology, extract_reports
from chimera.evidence.structured import StructuredFeatures, extract_structured

__all__ = [
    "StructuredFeatures",
    "SurgicalPathology",
    "extract_reports",
    "extract_structured",
]
