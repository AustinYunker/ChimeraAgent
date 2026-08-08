"""The cross-validation harness.

C2's pass condition is a number this module produces, so the harness itself needs
testing more than most code here does. A harness that silently leaked would not
crash or look wrong -- it would just report encouraging figures and send us into the
one-shot test submission with a model that never generalised.

The central test is :func:`test_a_memorising_model_scores_at_chance_out_of_fold`: a
lookup table from case id to true label is perfect in-sample and worthless on unseen
cases, so it separates a real hold-out from a fake one in a way that inspecting the
code cannot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chimera.contract.io import CaseInputs
from chimera.eval.cv import (
    Row,
    cross_validate,
    load_rows,
    out_of_fold_records,
    stratification_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "work" / "train" / "cases"
GT = REPO_ROOT / "work" / "train" / "ground_truth"


#: Reasoning that scores a perfect 1.0 when echoed back: confidence matches, the
#: weight vector matches exactly, no reveals are declared (tool precision 1.0), and
#: the only actively-weighted variables are `psa` and `age`, which are always
#: available and therefore grounded without retrieval. Without this a "perfect"
#: model tops out at 0.775 and the test's thresholds stop meaning anything.
_PERFECT_REASONING = {
    "confidence": "clear",
    "variable_weights": {"psa": "important", "age": "important"},
    "reveal_sequence": [],
}


def _synthetic_rows(n: int = 40) -> list[Row]:
    """Cases whose labels are pure noise -- alternating, unrelated to any feature."""
    rows = []
    for i in range(n):
        label = "yes" if i % 2 == 0 else "no"
        case = CaseInputs(
            task=1,
            case_id=f"C{i:03d}",
            structured_prompt={"psa": float(i)},
            clinical_data={},
            neural_representations={},
        )
        rows.append(Row(case=case, gt={"case_id": f"C{i:03d}", "biopsy_decision": label,
                                       **_PERFECT_REASONING}))
    return rows


def _record(case_id: str, decision: str) -> dict:
    return {"case_id": case_id, "biopsy_decision": decision, **_PERFECT_REASONING}


def test_a_memorising_model_scores_at_chance_out_of_fold():
    """The guard against a leaking harness.

    This model is a dictionary from case id to the right answer. If the harness ever
    fit on the evaluation fold it would score a perfect 1.0; held out properly it
    knows nothing about the cases it is asked about.
    """
    rows = _synthetic_rows()

    def fit(train):
        return {r.case_id: r.gt["biopsy_decision"] for r in train}

    def predict(case, table):
        # Unseen case ids fall back to a fixed guess.
        return _record(case.case_id, table.get(case.case_id, "yes"))

    honest = cross_validate(1, rows, fit, predict, folds=5, repeats=2)

    # Same model, but fitted on everything including what it is scored on.
    leaked = cross_validate(1, rows, lambda train: fit(rows), predict, folds=5, repeats=1)

    assert leaked["mean"] > 0.95, "the leaked control should be near-perfect"
    assert honest["mean"] < 0.70, (
        f"memorising model scored {honest['mean']:.3f} out of fold; the harness is leaking"
    )


def test_training_and_evaluation_rows_never_overlap():
    rows = _synthetic_rows(30)
    seen: list[tuple[set[str], set[str]]] = []

    def fit(train):
        return {"train_ids": {r.case_id for r in train}}

    def predict(case, params):
        seen.append(({case.case_id}, params["train_ids"]))
        return _record(case.case_id, "yes")

    out_of_fold_records(1, rows, fit, predict, folds=5, seed=0)
    for evaluated, trained_on in seen:
        assert not (evaluated & trained_on), "a case was scored by a model fitted on it"


def test_every_case_is_predicted_exactly_once_per_repeat():
    rows = _synthetic_rows(30)
    paired = out_of_fold_records(
        1, rows,
        lambda train: None,
        lambda case, p: _record(case.case_id, "yes"),
        folds=5, seed=0,
    )
    ids = [pred["case_id"] for _, pred in paired]
    assert len(ids) == len(rows)
    assert len(set(ids)) == len(rows)


def test_repeats_vary_the_split_and_report_a_spread():
    """A single split is not an estimate; the spread is what makes it one."""
    rows = _synthetic_rows(40)

    def fit(train):
        # Sensitive to the split, so different seeds must give different answers.
        return "yes" if sum(1 for r in train if r.gt["biopsy_decision"] == "yes") % 2 else "no"

    def predict(case, decision):
        return _record(case.case_id, decision)

    result = cross_validate(1, rows, fit, predict, folds=5, repeats=4)
    assert len(result["runs"]) == 4
    assert result["sd"] is not None


def test_stratification_key_reads_each_task_correctly():
    row1 = Row(CaseInputs(1, "a", {}, {}, {}), {"case_id": "a", "biopsy_decision": "no"})
    row2 = Row(CaseInputs(2, "b", {}, {}, {}),
               {"case_id": "b", "treatment_recommendation": {"primary": "active_treatment"}})
    row3 = Row(CaseInputs(3, "c", {}, {}, {}),
               {"case_id": "c", "event": 1, "months_to_recurrence": 12.0})
    assert stratification_key(1, row1) == "no"
    assert stratification_key(2, row2) == "active_treatment"
    assert stratification_key(3, row3) == 1


def test_splits_survive_a_class_too_rare_to_stratify():
    """`watchful_waiting` has 2 examples in 72; sklearn raises rather than degrading,
    and losing the whole estimate over two cases would be the wrong trade."""
    rows = _synthetic_rows(20)
    rows[0].gt["biopsy_decision"] = "maybe"  # a class with a single member
    result = cross_validate(
        1, rows,
        lambda train: None,
        lambda case, p: _record(case.case_id, "yes"),
        folds=5, repeats=1,
    )
    assert result["n"] == 20


@pytest.mark.parametrize("task", [1, 2, 3])
def test_load_rows_pairs_real_cases_with_their_labels(task):
    if not (CASES / f"task{task}").is_dir():
        pytest.skip("release cohort not built; run chimera.cli.make_release_cases")
    rows = load_rows(CASES, GT, task)
    expected = {1: 91, 2: 72, 3: 75}[task]
    assert len(rows) == expected
    assert all(r.gt["case_id"] == r.case.case_id for r in rows)
