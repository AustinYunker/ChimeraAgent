"""Typed prediction records and their serialisation to the submission shapes.

The submission shapes are deliberately *not* the baseline's Pydantic models
(``Task1Output`` and friends in ``chimera-agent-baseline``). Those carry
``schema_version``, ``patient``, and a ``reveal_sequence`` of rich objects; the
evaluator wants a bare JSON value for the decision and a flat four-key object
for the reasoning. Serialising the baseline models verbatim would fail the gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from chimera.contract import spec


class ContractError(ValueError):
    """Raised when a prediction cannot be serialised to a valid submission."""


@dataclass(slots=True)
class Reasoning:
    """The reasoning payload for Tasks 1 and 2 -- exactly four keys on the wire."""

    confidence: str
    variable_weights: dict[str, str]
    reveal_sequence: list[str]
    free_text: str

    def to_json(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "variable_weights": dict(self.variable_weights),
            "reveal_sequence": list(self.reveal_sequence),
            "free_text": self.free_text,
        }


@dataclass(slots=True)
class DecisionPrediction:
    """A Task 1 or Task 2 prediction: a categorical decision plus reasoning."""

    task: int
    decision: str
    reasoning: Reasoning

    def decision_json(self) -> Any:
        # Bare JSON value -- "yes" / "no" or a treatment token, not an object.
        return self.decision

    def reasoning_json(self) -> Any:
        return self.reasoning.to_json()


@dataclass(slots=True)
class RecurrencePrediction:
    """A Task 3 prediction.

    Only the ordering of ``months_to_recurrence`` across the cohort affects the
    leaderboard (Harrell's C-index, shorter predicted time = higher risk).
    ``event`` and ``free_text`` must be present and valid but do not rank.
    """

    months_to_recurrence: float
    event: int
    free_text: str
    task: int = 3

    def decision_json(self) -> Any:
        return {
            "months_to_recurrence": float(self.months_to_recurrence),
            "event": int(self.event),
        }

    def reasoning_json(self) -> Any:
        # Task 3's reasoning socket is a bare string, not an object.
        return self.free_text


Prediction = DecisionPrediction | RecurrencePrediction


# --------------------------------------------------------------------------- #
# Validation. Run this before writing -- a schema failure scores the case zero.
# --------------------------------------------------------------------------- #

def validate(pred: Prediction) -> None:
    """Raise :class:`ContractError` if ``pred`` would not survive the evaluator.

    Mirrors ``validate_record`` in ``evaluation/evaluate.py`` and additionally
    enforces the vocabularies the evaluator merely scores badly rather than
    rejecting, since an out-of-vocabulary token is never the intent.
    """
    if isinstance(pred, RecurrencePrediction):
        _validate_recurrence(pred)
        return
    _validate_decision(pred)


def _validate_recurrence(pred: RecurrencePrediction) -> None:
    try:
        months = float(pred.months_to_recurrence)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"months_to_recurrence not a float: {pred.months_to_recurrence!r}") from exc
    # Non-finite values pass every downstream check and then serialise as the
    # bare tokens `Infinity` / `NaN`, which are a Python extension rather than
    # legal JSON. They also poison the cohort MAE.
    if not math.isfinite(months) or months < 0:
        raise ContractError(f"months_to_recurrence must be finite and >= 0, got {months!r}")
    if pred.event not in (0, 1):
        raise ContractError(f"event must be 0 or 1, got {pred.event!r}")
    if not isinstance(pred.free_text, str) or not pred.free_text.strip():
        raise ContractError("free_text must be a non-empty string")


def _validate_decision(pred: DecisionPrediction) -> None:
    if pred.task not in (1, 2):
        raise ContractError(f"decision prediction must be task 1 or 2, got {pred.task!r}")

    allowed = spec.BIOPSY_DECISIONS if pred.task == 1 else spec.TREATMENT_DECISIONS
    if pred.decision not in allowed:
        raise ContractError(f"task {pred.task} decision {pred.decision!r} not in {allowed}")

    r = pred.reasoning
    if r.confidence not in spec.CONFIDENCE_LEVELS:
        raise ContractError(f"confidence {r.confidence!r} not in {spec.CONFIDENCE_LEVELS}")

    expected_vars = set(spec.VARIABLES_BY_TASK[pred.task])
    got_vars = set(r.variable_weights)
    if missing := expected_vars - got_vars:
        raise ContractError(f"task {pred.task} missing variable weights: {sorted(missing)}")
    if extra := got_vars - expected_vars:
        raise ContractError(f"task {pred.task} unknown variable weights: {sorted(extra)}")
    for var, weight in r.variable_weights.items():
        if weight not in spec.WEIGHT_LEVELS:
            raise ContractError(f"weight {weight!r} for {var!r} not in {spec.WEIGHT_LEVELS}")

    if len(set(r.reveal_sequence)) != len(r.reveal_sequence):
        raise ContractError(f"reveal_sequence has duplicates: {r.reveal_sequence}")
    if unknown := [s for s in r.reveal_sequence if s not in spec.REVEAL_SECTIONS]:
        # The evaluator keeps these verbatim and charges them as extra reveals,
        # which is never what we want.
        raise ContractError(f"reveal_sequence has out-of-vocabulary sections: {unknown}")

    if not isinstance(r.free_text, str) or not r.free_text.strip():
        raise ContractError("free_text must be a non-empty string")
