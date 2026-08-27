"""The predictor interface every model plugs into.

One method, one case plus one document store in, one prediction out. Keeping
this narrow is what lets C0's constant predictor and the eventual hybrid agent
share the same runner, the same contract validation, and the same scoring
harness.

The store is a separate argument rather than a field on
:class:`~chimera.contract.io.CaseInputs` on purpose. ``CaseInputs`` is what the
platform hands us; the store is how the masked "Extended EHR view" documents are
*reached*, and every retrieval through it is a recorded MCP tool call. Splitting
them keeps the distinction visible at every call site: a predictor that never
touches ``store`` provably never read a masked document.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chimera.contract.io import CaseInputs
from chimera.contract.types import Prediction
from chimera.mcp.client import ClinicalStore


@runtime_checkable
class Predictor(Protocol):
    """Produces one submission-shaped prediction for one case."""

    name: str

    def predict(self, case: CaseInputs, store: ClinicalStore) -> Prediction:
        """Return a prediction for ``case``, reaching documents through ``store``.

        Implementations must not raise for a well-formed case: a crash loses the
        case entirely, and the evaluator scores a missing case as a decision
        error rather than dropping it. Degrade to a conservative prediction
        instead.
        """
        ...
