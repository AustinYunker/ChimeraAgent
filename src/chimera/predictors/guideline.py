"""The C2 predictor: guideline strata with metric-fitted labels.

Subclasses :class:`~chimera.predictors.prior.PriorPredictor` rather than duplicating
it, because everything around the decision is unchanged and worth keeping identical:
the reveal-honesty rule (declare only sections actually read from this case), the
grounded rationale, and the defensive coercion of a stale parameter file.

What changes is where the decision comes from. The prior emitted one constant per
task; this routes the case through a clinical stratification and emits that leaf's
fitted label, then attaches the reasoning constants fitted for that decision.

Still pure standard library, and still no per-case learned function -- the fitted
object is a handful of labels, one per guideline leaf. See
:mod:`chimera.models.stratified` for the fit.
"""

from __future__ import annotations

from typing import Any

from chimera.contract import spec
from chimera.contract.io import CaseInputs
from chimera.contract.types import DecisionPrediction, Reasoning, RecurrencePrediction
from chimera.evidence import extract_structured
from chimera.models import stratified
from chimera.models.guidelines import describe
from chimera.predictors.prior import PriorPredictor, _normalise_weights

PARAMS_RESOURCE = "guideline_params.json"


class GuidelinePredictor(PriorPredictor):
    """Stratify, look up the fitted label, then reason conditioned on it."""

    name = "guideline"

    def __init__(self, params: dict[str, Any] | None = None, *, params_path: Any = None) -> None:
        if params is None and params_path is None:
            from importlib import resources
            import json

            resource = resources.files("chimera.predictors").joinpath(PARAMS_RESOURCE)
            with resource.open() as fh:
                params = json.load(fh)
        super().__init__(params, params_path=params_path)

    # -- Task 1 / 2 ---------------------------------------------------------- #

    def _predict_decision(self, case: CaseInputs) -> DecisionPrediction:
        task = case.task
        entry = self.params.get(f"task{task}") or {}
        allowed = spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS

        model = {
            "task": task,
            "leaf_labels": entry.get("leaf_labels") or {},
            "reasoning": entry.get("reasoning") or {},
        }
        decision, reasoning = stratified.predict_decision(case, model)
        if decision not in allowed:
            decision = allowed[0]

        confidence = reasoning.get("confidence")
        if confidence not in spec.CONFIDENCE_LEVELS:
            confidence = "borderline"
        weights = _normalise_weights(task, reasoning.get("variable_weights"))

        policy = reasoning.get("reveal_sequence")
        policy = list(policy) if isinstance(policy, list) else []
        # Sections the extractor had to read to reach the decision -- for Task 1
        # since release Version 3 that is the radiology report and the referral
        # notes, which is where `bx` now lives. `stratified.fit` forces these into
        # the fitted policy too, so this is normally a no-op; it matters when the
        # parameter file predates the change, and reveal honesty runs the wrong
        # way round (under-declaring what we read) if it is skipped.
        for section in extract_structured(case).evidence_sections:
            if section not in policy:
                policy.append(section)
        retrieved = self.retrieve(case, policy)

        return DecisionPrediction(
            task=task,
            decision=decision,
            reasoning=Reasoning(
                confidence=confidence,
                variable_weights=weights,
                reveal_sequence=list(retrieved),
                free_text=self._guideline_text(case, decision, weights, retrieved),
            ),
        )

    def _guideline_text(
        self, case: CaseInputs, decision: str, weights: dict[str, str], retrieved: dict[str, Any]
    ) -> str:
        """The prior's rationale, plus the guideline path that produced the decision."""
        base = self.free_text(case, decision, weights, retrieved)
        try:
            trace = describe(case.task, extract_structured(case))
        except Exception:  # pragma: no cover - describe is total, this is belt and braces
            return base

        parts = []
        if trace.get("prior_biopsy"):
            parts.append(f"prior biopsy {trace['prior_biopsy']}")
        if trace.get("eau_risk"):
            parts.append(f"EAU {trace['eau_risk']} risk")
        if not parts:
            return base
        return f"{base} Guideline basis: {', '.join(parts)}."

    # -- Task 3 -------------------------------------------------------------- #

    def _predict_recurrence(self, case: CaseInputs) -> RecurrencePrediction:
        """CAPRA-S ordering. Nothing here is fitted, so no parameters are consulted.

        Only the ordering of predicted months reaches the leaderboard; the absolute
        scale is free, and ``event`` does not enter the C-index at all.
        """
        months = stratified.predict_months(case)
        facts = self._facts(case)
        text = (
            f"Predicted time to biochemical recurrence or last follow-up: {months:.0f} months. "
            "Risk ordered by the CAPRA-S score from the surgical pathology report "
            "(grade, stage, margins, extraprostatic extension, seminal-vesicle "
            "invasion, nodal status) together with preoperative PSA."
            + (f" Case features: {', '.join(facts)}." if facts else "")
        )
        return RecurrencePrediction(months_to_recurrence=months, event=0, free_text=text)
