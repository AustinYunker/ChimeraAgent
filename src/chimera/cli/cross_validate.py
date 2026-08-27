"""Compare decision models by repeated pooled out-of-fold cross-validation.

The C2 pass condition. Every model is refit inside each fold -- decision rule *and*
reasoning constants -- so the constant baseline here is not the shipped C1b prior
read off disk but the same procedure re-run on each training split. Comparing a
model against a baseline that was fitted on the full cohort would flatter the model.

Usage::

    python -m chimera.cli.cross_validate --repeats 5
    python -m chimera.cli.cross_validate --tasks 2 --repeats 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from chimera.cli.fit_prior import available_sections
from chimera.contract import spec
from chimera.eval.cv import cross_validate, load_rows, summarise
from chimera.mcp.client import McpSession
from chimera.models import stratified

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = REPO_ROOT / "work" / "train" / "cases"
DEFAULT_GT = REPO_ROOT / "work" / "train" / "ground_truth"
DEFAULT_DATA = REPO_ROOT / "data" / "train_release"


def _constant_fit(task: int, sections: Sequence[str]):
    """The C1b prior as a one-leaf model, so it shares the guideline model's code path.

    Expressing the baseline this way rather than calling ``fit_prior`` keeps the two
    arms genuinely comparable: same reasoning optimiser, same objective, same
    fold discipline. The only difference is that this one has a single stratum and
    therefore cannot condition on anything.
    """

    def fit(rows):
        reasoning = stratified.fit_reasoning_by_decision(task, rows, sections)
        decisions = spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS
        best, best_score = decisions[0], float("-inf")
        for decision in decisions:
            score = stratified._score_leaf_map(
                task, rows, {"all": decision}, reasoning["__default__"], ["all"] * len(rows)
            )
            if score > best_score:
                best, best_score = decision, score
        return {"task": task, "constant": best, "reasoning": reasoning}

    def predict(case, store, params):
        decision = params["constant"]
        return stratified.decision_record(
            params["task"], case.case_id, decision,
            stratified.reasoning_for(params, decision),
        )

    return fit, predict


def _guideline_fit(task: int, sections: Sequence[str]):
    def fit(rows):
        return stratified.fit(task, rows, list(sections))

    return fit, stratified.predict_record


def _evaluate(args, task: int, rows: Sequence[Any]) -> dict[str, Any]:
    """Every arm for one task, keyed by the name it is reported under."""
    results: dict[str, Any] = {}

    if task == 3:
        # Nothing is fitted, so folds are irrelevant: one pass over the cohort
        # is the estimate. Reported through the same path for consistency.
        results["capra_s"] = cross_validate(
            3, rows, lambda train: None, stratified.predict_recurrence_record,
            folds=args.folds, repeats=1,
        )
        results["constant"] = cross_validate(
            3, rows, lambda train: None,
            lambda case, store, p: {"case_id": case.case_id,
                                    "months_to_recurrence": 60.0, "event": 0},
            folds=args.folds, repeats=1,
        )
        return results

    sections = available_sections(args.data, task)
    for name, builder in (("constant (C1b prior)", _constant_fit),
                          ("guideline strata (C2)", _guideline_fit)):
        fit, predict = builder(task, sections)
        results[name] = cross_validate(
            task, rows, fit, predict, folds=args.folds, repeats=args.repeats
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", dest="as_json", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {}

    # One cohort-scoped MCP server for the whole sweep. Every clinical document
    # the fit or the scoring touches is fetched through it.
    with McpSession.for_cohort(args.cases) as session:
        for task in args.tasks:
            rows = load_rows(args.cases, args.gt, task, session)
            if not rows:
                print(f"task{task}: no labeled rows under {args.gt}")
                continue

            print(f"===== task{task} (n={len(rows)}) =====")
            results = _evaluate(args, task, rows)

            for name, result in results.items():
                print(summarise(name, result))
            print()
            report[f"task{task}"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "last_aggregate"}
                for k, v in results.items()
            }

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
