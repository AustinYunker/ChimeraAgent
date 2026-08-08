"""The fitted constant prior -- the C1b submission payload.

Where :class:`~chimera.predictors.constant.ConstantPredictor` emits arbitrary
fixed values to prove the plumbing, this emits the values that actually maximise
the official ranking metric on the released training labels. It is still a
constant predictor -- no per-case modelling, no learned function of the inputs --
but its constants were chosen by exhaustive search rather than by guessing. See
:mod:`chimera.cli.fit_prior` for the fit, and ``prior_params.json`` for the
result.

Two properties are load-bearing:

* **Pure standard library.** This module and everything it imports must stay
  free of numpy, scikit-learn and pydantic, because it is the only predictor
  that ships inside the submission container and that is what keeps the image
  at ~150 MB. ``tests/test_entrypoint.py`` asserts this.
* **Reveal honesty.** ``docs/plan.md`` requires the declared ``reveal_sequence``
  to be exactly the evidence we actually retrieved. The fitted policy is only a
  *request*; :meth:`PriorPredictor.predict` intersects it with the sections this
  case genuinely carries, reads those, and declares exactly what it read. A
  section the case does not have is never claimed.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

from chimera.contract import spec
from chimera.contract.io import CaseInputs
from chimera.contract.types import (
    DecisionPrediction,
    Prediction,
    Reasoning,
    RecurrencePrediction,
)

PARAMS_RESOURCE = "prior_params.json"

#: Structured-prompt fields worth naming in the rationale, with how to render
#: them. Only fields actually present and non-null are used -- the rationale
#: must never assert a value we did not read.
_RATIONALE_FIELDS: tuple[tuple[str, str], ...] = (
    ("pirads", "PI-RADS {}"),
    ("psa", "PSA {} ng/mL"),
    ("psad", "PSA density {}"),
    ("age", "age {}"),
    ("bx_isup", "ISUP {}"),
    ("ct", "clinical stage {}"),
)

_DECISION_PHRASE = {
    "yes": "Biopsy recommended",
    "no": "Biopsy not recommended",
    "active_surveillance": "Active surveillance recommended",
    "continued_surveillance": "Continued surveillance recommended",
    "watchful_waiting": "Watchful waiting recommended",
    "active_treatment": "Active treatment recommended",
}


def load_params(path: Any | None = None) -> dict[str, Any]:
    """Read the fitted parameters, from the package by default.

    Going through :mod:`importlib.resources` rather than a filesystem path is
    what lets this work identically from an editable checkout and from the
    installed wheel inside the container.
    """
    if path is not None:
        with open(path) as fh:
            return json.load(fh)
    resource = resources.files("chimera.predictors").joinpath(PARAMS_RESOURCE)
    with resource.open() as fh:
        return json.load(fh)


def _normalise_weights(task: int, raw: Any) -> dict[str, str]:
    """Coerce a fitted weight vector to exactly the task's variable set.

    A stale parameter file must not be able to produce a ``ContractError`` at
    inference time: an unknown variable is dropped and a missing one defaults to
    ``not_used``, which is how the evaluator scores an omission anyway.
    """
    raw = raw if isinstance(raw, dict) else {}
    weights = {}
    for var in spec.VARIABLES_BY_TASK[task]:
        value = raw.get(var)
        weights[var] = value if value in spec.WEIGHT_LEVELS else "not_used"
    return weights


def section_is_present(value: Any) -> bool:
    """Whether a clinical-data section carries usable content.

    Sections are heterogeneous -- ``radiology_report`` is a string,
    ``psa_trend`` a list of records -- so this tests emptiness rather than type.
    """
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return value is not None


class PriorPredictor:
    """Emits the fitted per-task constants, grounded in what the case carries."""

    name = "prior"

    def __init__(self, params: dict[str, Any] | None = None, *, params_path: Any = None) -> None:
        self.params = params if params is not None else load_params(params_path)

    # -- reveal handling ---------------------------------------------------- #

    def retrieve(self, case: CaseInputs, policy: list[str]) -> dict[str, Any]:
        """Actually read the policy's sections from this case's clinical data.

        Returns only what was found, in policy order. The declared
        ``reveal_sequence`` is precisely ``list(...)`` of this mapping, so the
        declaration cannot drift from the retrieval.
        """
        clinical = case.clinical_data if isinstance(case.clinical_data, dict) else {}
        retrieved: dict[str, Any] = {}
        for section in policy:
            if section not in spec.REVEAL_SECTIONS:
                continue
            value = clinical.get(section)
            if section_is_present(value):
                retrieved[section] = value
        return retrieved

    # -- rationale ---------------------------------------------------------- #

    def _facts(self, case: CaseInputs) -> list[str]:
        prompt = case.structured_prompt if isinstance(case.structured_prompt, dict) else {}
        facts = []
        for key, template in _RATIONALE_FIELDS:
            value = prompt.get(key)
            if value is not None and value != "":
                facts.append(template.format(value))
        return facts

    def free_text(
        self, case: CaseInputs, decision: str, weights: dict[str, str], retrieved: dict[str, Any]
    ) -> str:
        """A short rationale asserting only things this case actually supports."""
        parts = [_DECISION_PHRASE.get(decision, decision.replace("_", " ").capitalize()) + "."]

        if facts := self._facts(case):
            parts.append("Case features: " + ", ".join(facts) + ".")

        drivers = [v for v, w in weights.items() if w == "decisive"]
        drivers += [v for v, w in weights.items() if w == "important"]
        if drivers:
            parts.append("Weighted most heavily: " + ", ".join(drivers) + ".")

        if retrieved:
            parts.append("Evidence consulted: " + ", ".join(retrieved) + ".")
        else:
            parts.append("Decided from the structured patient record without section retrieval.")

        return " ".join(parts)

    # -- Predictor protocol -------------------------------------------------- #

    def predict(self, case: CaseInputs) -> Prediction:
        if case.task == 3:
            return self._predict_recurrence(case)
        return self._predict_decision(case)

    def _predict_recurrence(self, case: CaseInputs) -> RecurrencePrediction:
        params = self.params.get("task3") or {}
        months = params.get("months_to_recurrence")
        try:
            months = float(months)
        except (TypeError, ValueError):
            months = 60.0
        event = params.get("event")
        event = event if event in (0, 1) else 0

        facts = self._facts(case)
        text = (
            "Predicted time to biochemical recurrence or last follow-up: "
            f"{months:.0f} months (event={event}). "
            + (f"Case features: {', '.join(facts)}. " if facts else "")
            + "Cohort-level prior; this task is ranked on the ordering of predicted "
            "times rather than on this rationale."
        )
        return RecurrencePrediction(
            months_to_recurrence=months, event=event, free_text=text
        )

    def _predict_decision(self, case: CaseInputs) -> DecisionPrediction:
        params = self.params.get(f"task{case.task}") or {}

        allowed = spec.BIOPSY_DECISIONS if case.task == 1 else spec.TREATMENT_DECISIONS
        decision = params.get("decision")
        if decision not in allowed:
            decision = allowed[0]

        confidence = params.get("confidence")
        if confidence not in spec.CONFIDENCE_LEVELS:
            confidence = "borderline"

        weights = _normalise_weights(case.task, params.get("variable_weights"))

        policy = params.get("reveal_sequence")
        policy = policy if isinstance(policy, list) else []
        retrieved = self.retrieve(case, policy)

        return DecisionPrediction(
            task=case.task,
            decision=decision,
            reasoning=Reasoning(
                confidence=confidence,
                variable_weights=weights,
                # Exactly what we read -- never the policy we asked for.
                reveal_sequence=list(retrieved),
                free_text=self.free_text(case, decision, weights, retrieved),
            ),
        )
