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
  that ships inside the submission container, and that is what keeps the image
  small enough to build and push in about a minute.
  ``tests/test_entrypoint.py`` asserts the stdlib closure; the size itself is
  measured by the ``build-image`` workflow and recorded in ``README.md``.
* **Reveal honesty.** ``docs/plan.md`` requires the declared ``reveal_sequence``
  to be exactly the evidence we actually retrieved. The fitted policy is only a
  *request*; :meth:`PriorPredictor.predict` intersects it with the sections this
  case genuinely carries, reads those, and declares exactly what it read. A
  section the case does not have is never claimed. Since C4a "reads those"
  means a tool call through a :class:`~chimera.mcp.client.ClinicalStore`, so
  the declaration is backed by the store's ledger rather than by convention.
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
from chimera.evidence.reports import extract_reports
from chimera.evidence.structured import StructuredFeatures, extract_structured
from chimera.mcp.client import ClinicalStore
from chimera.models.guidelines import eau_risk
# Submodule form, not ``from chimera.predictors import rationale``: the package
# __init__ imports this module, so the package object is only half-built while
# this line runs and an attribute lookup on it would fail.
from chimera.predictors.rationale import (
    biopsy_rationale,
    recurrence_rationale,
    treatment_rationale,
)

PARAMS_RESOURCE = "prior_params.json"


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


class PriorPredictor:
    """Emits the fitted per-task constants, grounded in what the case carries."""

    name = "prior"

    def __init__(self, params: dict[str, Any] | None = None, *, params_path: Any = None) -> None:
        self.params = params if params is not None else load_params(params_path)

    # -- reveal handling ---------------------------------------------------- #

    def retrieve(self, store: ClinicalStore, policy: list[str]) -> dict[str, Any]:
        """Call the policy's tools, and return what they actually gave us.

        Returns only what was found, in policy order. The declared
        ``reveal_sequence`` is precisely ``list(...)`` of this mapping, so the
        declaration cannot drift from the retrieval.

        Two filters apply and they are not the same filter. ``REVEAL_SECTIONS``
        bounds what may be *declared*; the store's per-task registry bounds what
        can be *retrieved*. A section outside the vocabulary is skipped here
        because declaring it would register as an extra reveal, which is why
        Task 3's surgical pathology is fetched by the recurrence path directly
        rather than through a policy.

        Order is deliberately the policy's, not the ledger's. The ledger records
        call order, which extraction perturbs; the fitted policy is a stable
        ordering that the tool score was measured against.
        """
        retrieved: dict[str, Any] = {}
        for section in policy:
            if section not in spec.REVEAL_SECTIONS:
                continue
            value = store.section(section)
            if value is not None:
                retrieved[section] = value
        return retrieved

    # -- rationale ---------------------------------------------------------- #

    def free_text(
        self, task: int, features: StructuredFeatures, decision: str, confidence: str
    ) -> str:
        """A short clinical rationale, in the register the judge scores against.

        Delegates to :mod:`chimera.predictors.rationale`, which owns the reasons
        behind every choice in the wording -- chiefly that the judge's evidence
        context is the clinical-data socket and not the patient card, so a card
        value the reports never state reads to it as a hallucination.

        Takes the already-extracted features rather than the case, because
        extraction may read narrative sections and the caller is the one that
        has to declare them.
        """
        if task == 1:
            return biopsy_rationale(features, decision, confidence)
        return treatment_rationale(features, decision, confidence, eau_risk(features))

    # -- Predictor protocol -------------------------------------------------- #

    def predict(self, case: CaseInputs, store: ClinicalStore) -> Prediction:
        if case.task == 3:
            return self._predict_recurrence(case, store)
        return self._predict_decision(case, store)

    def _predict_recurrence(self, case: CaseInputs, store: ClinicalStore) -> RecurrencePrediction:
        params = self.params.get("task3") or {}
        months = params.get("months_to_recurrence")
        try:
            months = float(months)
        except (TypeError, ValueError):
            months = 60.0
        event = params.get("event")
        event = event if event in (0, 1) else 0

        # The months are a cohort constant here, but the rationale still has to
        # describe *this* specimen: the recurrence rubric scores clinical
        # specificity against the reports, which we are handed either way.
        #
        # No CAPRA-S, though. `GuidelinePredictor` passes the score because the
        # score is what orders its cases; this predictor's ordering is a
        # constant, and saying otherwise would be the one thing the rubric
        # actually punishes -- a rationale that does not match the prediction.
        pathology = extract_reports(store)
        psa = extract_structured(case, store).psa
        text = recurrence_rationale(pathology, psa, months, None)
        return RecurrencePrediction(
            months_to_recurrence=months, event=event, free_text=text
        )

    def _predict_decision(self, case: CaseInputs, store: ClinicalStore) -> DecisionPrediction:
        params = self.params.get(f"task{case.task}") or {}

        allowed = spec.BIOPSY_DECISIONS if case.task == 1 else spec.TREATMENT_DECISIONS
        decision = params.get("decision")
        if decision not in allowed:
            decision = allowed[0]

        confidence = params.get("confidence")
        if confidence not in spec.CONFIDENCE_LEVELS:
            confidence = "borderline"

        weights = _normalise_weights(case.task, params.get("variable_weights"))

        features = extract_structured(case, store)

        policy = params.get("reveal_sequence")
        policy = list(policy) if isinstance(policy, list) else []
        # Sections extraction had to read to build the features. Empty whenever
        # the patient card answered everything, which on the current release is
        # every Task 1 and Task 2 case -- but reveal honesty runs both ways, and
        # under-declaring what we read is as wrong as over-declaring it.
        for section in features.evidence_sections:
            if section not in policy:
                policy.append(section)
        retrieved = self.retrieve(store, policy)

        return DecisionPrediction(
            task=case.task,
            decision=decision,
            reasoning=Reasoning(
                confidence=confidence,
                variable_weights=weights,
                # Exactly what we read -- never the policy we asked for.
                reveal_sequence=list(retrieved),
                free_text=self.free_text(case.task, features, decision, confidence),
            ),
        )
