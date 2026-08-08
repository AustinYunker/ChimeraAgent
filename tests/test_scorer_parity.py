"""C1: the fast scorer must agree with the official one, exactly.

``chimera.scoring.fast`` is a transcription of ``evaluation/evaluate.py``, and
transcriptions rot in ways that are invisible until they have silently steered
weeks of model selection. So rather than testing the fast scorer against
hand-computed expectations, every test here drives *both* scorers over the same
records and asserts the outputs are equal.

The randomised cases are deliberately hostile: invalid tokens, missing keys,
wrong types, out-of-vocabulary reveals, `not_revealed` aliases and absent
predictions. Agreeing on well-formed input is easy; the leniency rules are
where a replica actually drifts.

The rationale judge is out of scope by construction -- both sides run with it
disabled, which is the ``USE_RATIONALE_JUDGE=0`` renormalised composite.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from chimera.contract import spec
from chimera.scoring import fast

TOL = 1e-9

# Keys the official aggregate carries that the fast scorer deliberately omits:
# a human-readable sklearn report, and a flag that only appears when sklearn is
# missing (in which case parity is moot).
_OFFICIAL_ONLY_KEYS = {"decision_classification_report", "sklearn_unavailable"}


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #

def _assert_close(ours: Any, theirs: Any, path: str) -> None:
    if theirs is None or ours is None:
        assert ours is theirs, f"{path}: ours={ours!r} official={theirs!r}"
        return
    if isinstance(theirs, dict):
        assert isinstance(ours, dict), f"{path}: ours is {type(ours).__name__}, not a dict"
        assert set(ours) == set(theirs), (
            f"{path}: key mismatch, ours-only={sorted(set(ours) - set(theirs))} "
            f"official-only={sorted(set(theirs) - set(ours))}"
        )
        for k in theirs:
            _assert_close(ours[k], theirs[k], f"{path}.{k}")
        return
    if isinstance(theirs, bool) or isinstance(ours, bool):
        assert ours == theirs, f"{path}: ours={ours!r} official={theirs!r}"
        return
    if isinstance(theirs, (int, float)):
        assert isinstance(ours, (int, float)), f"{path}: ours={ours!r} official={theirs!r}"
        assert abs(ours - theirs) <= TOL, f"{path}: ours={ours!r} official={theirs!r}"
        return
    assert ours == theirs, f"{path}: ours={ours!r} official={theirs!r}"


def _official_aggregate(ev, cases: list[tuple[dict, dict | None]]) -> dict:
    """Run the official scorer over the same records, judge and tool metric off."""
    rows = []
    for gt, pred in cases:
        row = ev.evaluate_case(gt, pred, None, None)
        ev.attach_kappa_fields(row, gt, pred)
        rows.append(row)
    return ev.compute_aggregate_metrics(rows)


def _compare_cohort(ev, cases: list[tuple[dict, dict | None]]) -> None:
    theirs = _official_aggregate(ev, cases)
    ours = fast.score_cohort(cases)

    extra = set(ours) - set(theirs)
    assert not extra, f"fast scorer invents keys the official one lacks: {sorted(extra)}"
    for key, value in theirs.items():
        if key in _OFFICIAL_ONLY_KEYS:
            continue
        assert key in ours, f"fast scorer is missing aggregate key {key!r}"
        _assert_close(ours[key], value, key)


@pytest.fixture(scope="session")
def evaluator_with_mapping(official_evaluator):
    """The official evaluator, skipped unless its section mapping resolved.

    Without the mapping the official ``section_grounding_score`` short-circuits
    to ``None`` for every case, so a parity run would pass while proving nothing
    about the grounding component.
    """
    if not official_evaluator._get_section_var_mapping():
        pytest.skip("section_variable_mapping.json not resolvable from the evaluator")
    return official_evaluator


# --------------------------------------------------------------------------- #
# Record generators
# --------------------------------------------------------------------------- #

_BAD_WEIGHTS = ("critical", "NOT_USED ", "not_revealed", "", None, 3, "Decisive")
_BAD_CONF = ("high", "  CLEAR ", "", None, 2)
_BAD_REVEALS = ("mri_report", "", None, 42, "Radiology_Report")


def _gt_decision_record(rng: random.Random, task: int, case_id: str) -> dict:
    variables = spec.VARIABLES_BY_TASK[task]
    weights = {}
    for var in variables:
        # 'not_revealed' is the form's own alias for not_used and shows up in
        # real ground truth, so it has to appear on the reference side too.
        weights[var] = rng.choice(
            list(spec.WEIGHT_LEVELS) + ["not_revealed"] if rng.random() < 0.15 else list(spec.WEIGHT_LEVELS)
        )
    record: dict[str, Any] = {
        "case_id": case_id,
        "confidence": rng.choice(spec.CONFIDENCE_LEVELS),
        "variable_weights": weights,
        "reveal_sequence": rng.sample(
            list(spec.REVEAL_SECTIONS), k=rng.randint(0, len(spec.REVEAL_SECTIONS))
        ),
        "free_text": "reference rationale",
    }
    if task == 1:
        record["biopsy_decision"] = rng.choice(spec.BIOPSY_DECISIONS)
    else:
        record["treatment_recommendation"] = {"primary": rng.choice(spec.TREATMENT_DECISIONS)}
    return record


def _pred_decision_record(rng: random.Random, task: int, case_id: str) -> dict | None:
    if rng.random() < 0.08:
        return None  # no job produced this case at all

    variables = list(spec.VARIABLES_BY_TASK[task])

    # Weights: usually complete and valid, sometimes partial, extra, or junk.
    roll = rng.random()
    weights: Any
    if roll < 0.08:
        weights = None
    elif roll < 0.12:
        weights = ["not", "a", "dict"]
    elif roll < 0.20:
        weights = {}
    else:
        keys = list(variables)
        if rng.random() < 0.20:  # omit some -> scored as not_used
            keys = rng.sample(keys, k=rng.randint(0, len(keys)))
        weights = {k: rng.choice(spec.WEIGHT_LEVELS) for k in keys}
        if rng.random() < 0.15:  # invent a variable the ground truth never had
            weights["invented_variable"] = rng.choice(spec.WEIGHT_LEVELS)
        if rng.random() < 0.20:  # a token the evaluator has to normalise or drop
            weights[rng.choice(variables)] = rng.choice(_BAD_WEIGHTS)

    # Reveals: valid subsets, out-of-vocabulary names, duplicates, wrong types.
    roll = rng.random()
    reveals: Any
    if roll < 0.10:
        reveals = []
    elif roll < 0.15:
        reveals = "radiology_report"  # not a list
    elif roll < 0.20:
        reveals = None
    else:
        reveals = rng.sample(list(spec.REVEAL_SECTIONS), k=rng.randint(0, len(spec.REVEAL_SECTIONS)))
        if rng.random() < 0.20:
            reveals.append(rng.choice(_BAD_REVEALS))
        if rng.random() < 0.15 and reveals:
            reveals.append(reveals[0])  # duplicate, must be deduplicated

    confidence = (
        rng.choice(spec.CONFIDENCE_LEVELS) if rng.random() < 0.80 else rng.choice(_BAD_CONF)
    )

    record: dict[str, Any] = {
        "case_id": case_id,
        "confidence": confidence,
        "variable_weights": weights,
        "reveal_sequence": reveals,
        "free_text": "candidate rationale",
    }
    if rng.random() < 0.05:
        del record["confidence"]

    if task == 1:
        decision: Any = rng.choice(spec.BIOPSY_DECISIONS)
        if rng.random() < 0.12:
            decision = rng.choice(["Yes", "maybe", "", None, 1])
        record["biopsy_decision"] = decision
    else:
        primary: Any = rng.choice(spec.TREATMENT_DECISIONS)
        if rng.random() < 0.12:
            # Casing/hyphen/whitespace variants the evaluator normalises, plus
            # tokens it rejects outright. Non-string values are excluded: a
            # schema-failed case leaks its raw `primary` into the dataset F1,
            # and a truthy non-string crashes the official aggregator outright
            # -- pinned by
            # test_official_evaluator_crashes_on_a_non_string_treatment_decision.
            primary = rng.choice(
                ["Active-Treatment", " active  surveillance ", "surgery", "", None]
            )
        record["treatment_recommendation"] = {"primary": primary}
    return record


def _gt_recurrence_record(rng: random.Random, case_id: str) -> dict:
    return {
        "case_id": case_id,
        "months_to_recurrence": round(rng.uniform(0.5, 96.0), 2),
        "event": rng.choice([0, 1]),
    }


def _pred_recurrence_record(rng: random.Random, case_id: str) -> dict | None:
    if rng.random() < 0.08:
        return None
    months: Any = round(rng.uniform(0.0, 120.0), 2)
    event: Any = rng.choice([0, 1])
    if rng.random() < 0.12:
        months = rng.choice(["36.5", None, -5.0])
    if rng.random() < 0.12:
        event = rng.choice(["1", None, 2, -1, True, 1.7])

    # A schema-failed case keeps its *raw* months, and the official aggregator
    # then does arithmetic on it -- so a string here takes the whole evaluation
    # job down rather than just this case. Parity cannot be measured against a
    # crash, so that combination is excluded here and pinned separately by
    # test_official_evaluator_crashes_on_a_non_numeric_months_in_a_failed_case.
    if isinstance(months, str) and fast._norm_event(event) is None:
        event = 1

    return {
        "case_id": case_id,
        "months_to_recurrence": months,
        "event": event,
        "free_text": "candidate rationale",
    }


def _cohort(task: int, n: int, seed: int) -> list[tuple[dict, dict | None]]:
    rng = random.Random(f"{seed}:{task}:{n}")
    cases = []
    for i in range(n):
        case_id = f"{task}_{i:04d}"
        if task == 3:
            cases.append((_gt_recurrence_record(rng, case_id), _pred_recurrence_record(rng, case_id)))
        else:
            cases.append(
                (_gt_decision_record(rng, task, case_id), _pred_decision_record(rng, task, case_id))
            )
    return cases


# --------------------------------------------------------------------------- #
# Parity over randomised cohorts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("task", [1, 2, 3])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_aggregate_parity_on_randomised_cohorts(evaluator_with_mapping, task, seed):
    """The headline check: identical aggregate metrics on hostile input."""
    _compare_cohort(evaluator_with_mapping, _cohort(task, 60, seed))


@pytest.mark.parametrize("task", [1, 2, 3])
def test_per_case_parity_on_randomised_cases(evaluator_with_mapping, task):
    """Case-level parity, so a failure points at the component, not the mean."""
    ev = evaluator_with_mapping
    compared = ("case_score", "gate", "decision_score", "confidence_score",
                "variable_weight_score", "important_decisive_factor_score",
                "tool_score", "section_grounding_score", "event_score", "time_score",
                "gt_decision", "pred_decision", "gt_event", "pred_event",
                "gt_months", "pred_months", "_weight_pairs")

    for gt, pred in _cohort(task, 400, seed=99):
        theirs = ev.evaluate_case(gt, pred, None, None)
        ev.attach_kappa_fields(theirs, gt, pred)
        ours = fast.score_case(gt, pred)
        for key in compared:
            if key not in theirs:
                continue
            _assert_close(ours.get(key), theirs[key], f"{gt['case_id']}.{key}")


@pytest.mark.parametrize("task", [1, 2])
def test_parity_when_every_case_fails_the_gate(evaluator_with_mapping, task):
    """An all-wrong cohort: mean case score 0, F1 0, and no division by zero."""
    cases = []
    for gt, pred in _cohort(task, 30, seed=7):
        if pred is None:
            continue
        if task == 1:
            wrong = "no" if gt["biopsy_decision"] == "yes" else "yes"
            pred["biopsy_decision"] = wrong
        else:
            options = [d for d in spec.TREATMENT_DECISIONS
                       if d != gt["treatment_recommendation"]["primary"]]
            pred["treatment_recommendation"] = {"primary": options[0]}
        cases.append((gt, pred))
    _compare_cohort(evaluator_with_mapping, cases)


@pytest.mark.parametrize("task", [1, 2, 3])
def test_parity_when_every_prediction_is_missing(evaluator_with_mapping, task):
    """Nothing scored. The sentinel label must still cost the true classes."""
    cases = [(gt, None) for gt, _ in _cohort(task, 20, seed=11)]
    _compare_cohort(evaluator_with_mapping, cases)


def test_parity_when_all_task3_cases_are_censored(evaluator_with_mapping):
    """The reference Task 3 cohort's exact shape: C-index must be None, not 0."""
    cases = []
    for gt, pred in _cohort(3, 20, seed=13):
        gt["event"] = 0
        cases.append((gt, pred))
    theirs = _official_aggregate(evaluator_with_mapping, cases)
    assert theirs["concordance_index"] is None
    _compare_cohort(evaluator_with_mapping, cases)


# --------------------------------------------------------------------------- #
# End-to-end: the same run the official pipeline would do, via our own files
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("task", [1, 2, 3])
def test_parity_end_to_end_against_a_real_cohort(evaluator_with_mapping, tmp_path, task):
    """Score real fixture output twice: through the files, and in process.

    This is the one test that also exercises the adapters -- the ground-truth
    loader and the prediction flattener -- rather than just the scoring maths.
    """
    from pathlib import Path

    from chimera.contract.io import read_case
    from chimera.predictors.constant import ConstantPredictor
    from chimera.scoring.records import load_ground_truth, record_from_prediction

    repo_root = Path(__file__).resolve().parents[1]
    gt_root = repo_root / "refs" / "challenge" / "evaluation" / "ground_truth" / f"task{task}"
    fixture_root = repo_root / "work" / "fixtures" / f"task{task}"
    if not gt_root.is_dir() or not fixture_root.is_dir():
        pytest.skip("reference ground truth or fixtures not available")

    gt_records = load_ground_truth(gt_root, task)
    assert gt_records, f"no ground truth loaded for task{task}"

    predictor = ConstantPredictor()
    cases: list[tuple[dict, dict | None]] = []
    for gt in gt_records:
        case_dir = fixture_root / gt["case_id"]
        if not case_dir.is_dir():
            cases.append((gt, None))
            continue
        prediction = predictor.predict(read_case(case_dir))
        cases.append((gt, record_from_prediction(prediction, gt["case_id"])))

    _compare_cohort(evaluator_with_mapping, cases)


def test_our_ground_truth_loader_matches_the_official_one(official_evaluator, request):
    """Both loaders must produce identical records, or parity is comparing fictions."""
    from pathlib import Path

    from chimera.scoring.records import load_ground_truth

    repo_root = Path(__file__).resolve().parents[1]
    for task in (1, 2, 3):
        gt_root = repo_root / "refs" / "challenge" / "evaluation" / "ground_truth" / f"task{task}"
        if not gt_root.is_dir():
            pytest.skip(f"no reference ground truth for task{task}")
        theirs = official_evaluator.load_ground_truth_records(gt_root, f"task{task}")
        ours = load_ground_truth(gt_root, task)
        assert ours == theirs, f"task{task} ground-truth records differ"
