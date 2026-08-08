"""The predictor interface every model plugs into.

One method, one case in, one prediction out. Keeping this narrow is what lets
C0's constant predictor and the eventual hybrid agent share the same runner,
the same contract validation, and the same scoring harness.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chimera.contract.io import CaseInputs
from chimera.contract.types import Prediction


@runtime_checkable
class Predictor(Protocol):
    """Produces one submission-shaped prediction for one case."""

    name: str

    def predict(self, case: CaseInputs) -> Prediction:
        """Return a prediction for ``case``.

        Implementations must not raise for a well-formed case: a crash loses the
        case entirely, and the evaluator scores a missing case as a decision
        error rather than dropping it. Degrade to a conservative prediction
        instead.
        """
        ...
