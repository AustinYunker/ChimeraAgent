"""Cross-validation, built around the official metric rather than accuracy.

This is the instrument C2's pass condition is measured on, so three design choices
are load-bearing:

**Pooled out-of-fold predictions, not averaged per-fold scores.** The ranking metric
is cohort-level -- Task 1 is ``(mean_case_score + F1_yes) / 2``, Task 3 is a C-index.
An F1 over an 18-case fold, or a C-index over a 15-case Task 3 fold containing about
four events, is mostly noise, and averaging such numbers does not estimate the
cohort-level quantity we are actually scored on. Instead every case is predicted by a
model that never saw it, and the pooled predictions are scored **once**.

**Everything fitted is refit inside the fold.** The decision rule and the reasoning
constants are both fitted from labels. Holding the reasoning constants fixed across
folds -- the tempting shortcut, since they were fitted once for C1b -- leaks the
held-out labels into the score through the confidence and variable-weight terms.

**Repeated, with a spread.** At n = 72 the difference between two split seeds is
routinely larger than the difference between two models. Any change smaller than the
reported standard deviation is not evidence.

Not part of the container: this module imports scikit-learn and is a development
tool only.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from chimera.contract.io import CaseInputs, read_case
from chimera.scoring.fast import score_cohort
from chimera.scoring.records import load_ground_truth

#: Fit a model from training rows, returning opaque parameters.
FitFn = Callable[[Sequence["Row"]], Any]
#: Turn one case plus fitted parameters into the flat record the scorer compares on.
PredictFn = Callable[[CaseInputs, Any], dict[str, Any]]


@dataclass(slots=True)
class Row:
    """One labeled case: the inputs a predictor sees, and the target it is scored on."""

    case: CaseInputs
    gt: dict[str, Any]

    @property
    def case_id(self) -> str:
        return str(self.gt.get("case_id") or self.case.case_id)


def load_rows(cases_root: Path, gt_root: Path, task: int) -> list[Row]:
    """Pair every labeled case under ``gt_root`` with its inputs under ``cases_root``.

    Cases without labels are dropped -- 104 of 195 for Task 1, 81 of 153 for Task 2 --
    since they cannot contribute to a supervised estimate. They remain useful for
    exercising the inference path at full cohort size.
    """
    targets = {rec["case_id"]: rec for rec in load_ground_truth(gt_root / f"task{task}", task)}
    rows: list[Row] = []
    task_dir = cases_root / f"task{task}"
    if not task_dir.is_dir():
        return rows
    for case_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
        gt = targets.get(case_dir.name)
        if gt is None:
            continue
        rows.append(Row(case=read_case(case_dir, fallback_case_id=case_dir.name), gt=gt))
    return rows


def stratification_key(task: int, row: Row) -> Any:
    """What to balance folds on: the decision for Tasks 1/2, the event for Task 3.

    Stratifying Task 3 on the event indicator is what stops a fold landing with zero
    events, which would make its contribution to the pooled C-index undefined.
    """
    if task == 3:
        return row.gt.get("event")
    if task == 1:
        return row.gt.get("biopsy_decision")
    rec = row.gt.get("treatment_recommendation") or {}
    return rec.get("primary") if isinstance(rec, dict) else None


def _splits(task: int, rows: Sequence[Row], folds: int, seed: int):
    """Stratified folds, falling back to unstratified when a class is too rare.

    ``watchful_waiting`` has 2 examples in 72, so a 5-fold stratified split is not
    always possible; scikit-learn raises rather than degrading, and losing the whole
    estimate over two cases would be worse than slightly unbalanced folds.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    labels = [stratification_key(task, r) for r in rows]
    counts: dict[Any, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    if all(c >= folds for c in counts.values()) and len(counts) > 1:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(range(len(rows)), labels))
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(range(len(rows))))


def out_of_fold_records(
    task: int,
    rows: Sequence[Row],
    fit: FitFn,
    predict: PredictFn,
    *,
    folds: int = 5,
    seed: int = 0,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``(gt, prediction)`` for every row, each predicted by a model that never saw it."""
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for train_idx, test_idx in _splits(task, rows, folds, seed):
        train = [rows[i] for i in train_idx]
        params = fit(train)
        for i in test_idx:
            record = predict(rows[i].case, params)
            record.setdefault("case_id", rows[i].case_id)
            paired.append((rows[i].gt, record))
    return paired


def cross_validate(
    task: int,
    rows: Sequence[Row],
    fit: FitFn,
    predict: PredictFn,
    *,
    folds: int = 5,
    repeats: int = 5,
    seed: int = 0,
    metric: str | None = None,
) -> dict[str, Any]:
    """Repeated pooled out-of-fold estimate of the official ranking metric."""
    if not rows:
        return {"n": 0, "mean": None, "sd": None, "runs": []}

    key = metric or ("concordance_index" if task == 3 else "ranking_score")
    runs: list[float] = []
    aggregates: list[dict[str, Any]] = []
    for repeat in range(repeats):
        paired = out_of_fold_records(
            task, rows, fit, predict, folds=folds, seed=seed + repeat
        )
        aggregate = score_cohort(paired)
        aggregates.append(aggregate)
        value = aggregate.get(key)
        if value is not None:
            runs.append(float(value))

    return {
        "n": len(rows),
        "metric": key,
        "mean": statistics.fmean(runs) if runs else None,
        "sd": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "runs": runs,
        "last_aggregate": aggregates[-1] if aggregates else None,
    }


def summarise(name: str, result: dict[str, Any]) -> str:
    if result.get("mean") is None:
        return f"  {name:<28} n/a"
    return (
        f"  {name:<28} {result['mean']:.4f} +/- {result['sd']:.4f}"
        f"   (n={result['n']}, {result['metric']})"
    )
