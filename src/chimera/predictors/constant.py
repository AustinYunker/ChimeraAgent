"""A constant predictor -- the C0 plumbing check.

This exists to prove the I/O contract end to end before any modelling, and to
be the payload of the first validation submission (C1b): if the container runs
on Grand Challenge at all, this tells us so a month before the one-shot test
deadline.

It also fixes the floor. Every later model is measured against these numbers,
so a change that fails to beat a constant is a change that did nothing.
"""

from __future__ import annotations

from chimera.contract import spec
from chimera.contract.io import CaseInputs
from chimera.contract.types import (
    DecisionPrediction,
    Prediction,
    Reasoning,
    RecurrencePrediction,
)
from chimera.mcp.client import ClinicalStore

# Placeholder prose. The rationale judge scores case-agnostic text LOW by
# design, so this is a deliberate zero on that component, not an attempt at one.
_PLACEHOLDER_TEXT = (
    "Baseline constant prediction emitted by the contract-conformance harness. "
    "No case-specific evidence was consulted."
)


class ConstantPredictor:
    """Emits a fixed decision and a fixed reasoning payload for every case."""

    def __init__(
        self,
        *,
        biopsy_decision: str = "yes",
        treatment_decision: str = "active_treatment",
        confidence: str = "borderline",
        weight: str = "noted",
        reveal_sequence: tuple[str, ...] = (),
        months_to_recurrence: float = 60.0,
        event: int = 0,
        name: str = "constant",
    ) -> None:
        if biopsy_decision not in spec.BIOPSY_DECISIONS:
            raise ValueError(f"bad biopsy_decision {biopsy_decision!r}")
        if treatment_decision not in spec.TREATMENT_DECISIONS:
            raise ValueError(f"bad treatment_decision {treatment_decision!r}")
        if confidence not in spec.CONFIDENCE_LEVELS:
            raise ValueError(f"bad confidence {confidence!r}")
        if weight not in spec.WEIGHT_LEVELS:
            raise ValueError(f"bad weight {weight!r}")

        self.biopsy_decision = biopsy_decision
        self.treatment_decision = treatment_decision
        self.confidence = confidence
        self.weight = weight
        self.reveal_sequence = list(reveal_sequence)
        self.months_to_recurrence = months_to_recurrence
        self.event = event
        self.name = name

    def predict(self, case: CaseInputs, store: ClinicalStore | None = None) -> Prediction:
        """``store`` is accepted and never used -- that is the point of this class.

        It consults no evidence, so it makes no tool calls, and its declared
        ``reveal_sequence`` is whatever it was constructed with. Optional here
        alone, so :func:`inference.fallback_prediction` can still produce a
        valid payload on a path where the transport itself is what failed.
        """
        if case.task == 3:
            return RecurrencePrediction(
                months_to_recurrence=self.months_to_recurrence,
                event=self.event,
                free_text=_PLACEHOLDER_TEXT,
            )

        decision = self.biopsy_decision if case.task == 1 else self.treatment_decision
        return DecisionPrediction(
            task=case.task,
            decision=decision,
            reasoning=Reasoning(
                confidence=self.confidence,
                variable_weights={v: self.weight for v in spec.VARIABLES_BY_TASK[case.task]},
                reveal_sequence=list(self.reveal_sequence),
                free_text=_PLACEHOLDER_TEXT,
            ),
        )
