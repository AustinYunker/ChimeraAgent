"""Fit the constant-prior predictor against the released training labels.

The prior has no per-case behaviour: it emits one decision, one confidence, one
variable-weight vector and one reveal set per task. That makes it exhaustively
optimisable -- the candidate space is only ``|decisions| x 3 x |weight
strategies| x 2^|sections|`` -- so this module enumerates it rather than
guessing, and picks the argmax of the official ranking metric.

Three constraints shape the search:

* **Reveal honesty.** ``docs/plan.md`` requires that the declared
  ``reveal_sequence`` be exactly what we retrieved. Candidate reveal sets are
  therefore restricted to sections that actually exist in that task's
  clinical-data payload -- see :func:`available_sections`. Task 1 has no
  ``pathology_report`` section at all, so grounding ``bx`` through a declared
  reveal is not available to us however well it would score.
* **Selection is in-sample; the reported estimate is not.** The chosen
  parameters are fitted on everything, because a prior fitted on a subset is a
  different prior. What gets *reported* is a K-fold estimate of the whole
  fit-then-predict procedure, which is the number that generalises.
* **Task 3 does not rank on any of this.** Harrell's C-index is 0.5 for any
  constant, since every pair ties. Its constants are chosen to maximise the
  (unranked, analysis-only) case score so the output is sensible rather than
  arbitrary.

Writes ``prior_params.json`` next to the predictor that consumes it. Re-run it
whenever labels grow -- the challenge site says they arrive incrementally.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Iterator

from chimera.contract import spec
from chimera.scoring.fast import score_cohort
from chimera.scoring.records import load_ground_truth

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA = REPO_ROOT / "data" / "train_release"
DEFAULT_OUT = REPO_ROOT / "src" / "chimera" / "predictors" / "prior_params.json"

#: Clinical-data filename stem per task, used to discover which sections exist.
_CLINICAL_STEM = {
    1: "prostate-biopsy-decision-clinical-data",
    2: "prostate-treatment-decision-clinical-data",
    3: "prostate-time-to-recurrence-or-last-follow-up-clinical-data",
}

#: Task 3 candidate predictions, in months. Coarse on purpose: the C-index is
#: 0.5 for every constant, so this only shapes the analysis-only case score.
_TASK3_MONTHS = tuple(float(m) for m in range(6, 133, 6))


def available_sections(data_root: Path, task: int) -> list[str]:
    """Reveal-vocabulary sections that actually appear in ``task``'s clinical data.

    A section we cannot read is a section we cannot honestly declare. Scans the
    whole cohort rather than one case, because a section absent from case 1 may
    still exist elsewhere.
    """
    seen: set[str] = set()
    for path in sorted((data_root / f"task{task}").glob(f"*/{_CLINICAL_STEM[task]}.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            seen |= {k for k in payload if k in spec.REVEAL_SECTIONS}
    # Preserve the canonical vocabulary order so the fit is deterministic.
    return [s for s in spec.REVEAL_SECTIONS if s in seen]


def modal_weights(gts: list[dict], variables: tuple[str, ...]) -> dict[str, str]:
    """Per-variable most-common ground-truth weight.

    Still a constant predictor -- the same vector goes out for every case -- but
    a far better one than a single level repeated across variables, because the
    ordinal error term is a mean over per-variable distances.
    """
    out: dict[str, str] = {}
    for var in variables:
        counts: dict[str, int] = {}
        for gt in gts:
            weights = gt.get("variable_weights") or {}
            value = weights.get(var, "not_used")
            counts[value] = counts.get(value, 0) + 1
        # Break ties by the canonical ordinal order, so the fit is reproducible.
        out[var] = max(spec.WEIGHT_LEVELS, key=lambda w: (counts.get(w, 0), -spec.WEIGHT_ORDINAL[w]))
    return out


def _weight_strategies(gts: list[dict], variables: tuple[str, ...]) -> dict[str, dict[str, str]]:
    strategies = {f"all_{level}": {v: level for v in variables} for level in spec.WEIGHT_LEVELS}
    strategies["modal"] = modal_weights(gts, variables)
    return strategies


def _decision_record(task: int, params: dict[str, Any], case_id: str) -> dict[str, Any]:
    """The flat record the scorer compares on, for a Task 1/2 prior."""
    record: dict[str, Any] = {
        "case_id": case_id,
        "confidence": params["confidence"],
        "variable_weights": params["variable_weights"],
        "reveal_sequence": list(params["reveal_sequence"]),
    }
    if task == 1:
        record["biopsy_decision"] = params["decision"]
    else:
        record["treatment_recommendation"] = {"primary": params["decision"]}
    return record


def _recurrence_record(params: dict[str, Any], case_id: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "months_to_recurrence": params["months_to_recurrence"],
        "event": params["event"],
    }


def build_record(task: int, params: dict[str, Any], case_id: str) -> dict[str, Any]:
    if task == 3:
        return _recurrence_record(params, case_id)
    return _decision_record(task, params, case_id)


def objective(task: int, gts: list[dict], params: dict[str, Any]) -> float:
    """The number we maximise: ranking score for Tasks 1/2, case score for Task 3.

    Task 3 uses the case score deliberately -- its ranking score is 0.5 for every
    constant, so optimising it would pick arbitrarily among ties.
    """
    pairs = [(gt, build_record(task, params, gt["case_id"])) for gt in gts]
    metrics = score_cohort(pairs)
    key = "mean_case_score" if task == 3 else "ranking_score"
    return metrics.get(key) or 0.0


def candidates(task: int, gts: list[dict], sections: list[str]) -> Iterator[dict[str, Any]]:
    """Every parameter combination worth scoring for ``task``."""
    if task == 3:
        for months, event in itertools.product(_TASK3_MONTHS, (0, 1)):
            yield {"months_to_recurrence": months, "event": event}
        return

    decisions = spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS
    variables = spec.VARIABLES_BY_TASK[task]
    strategies = _weight_strategies(gts, variables)
    reveal_sets = [
        list(combo)
        for size in range(len(sections) + 1)
        for combo in itertools.combinations(sections, size)
    ]
    for decision, confidence, (name, weights), reveals in itertools.product(
        decisions, spec.CONFIDENCE_LEVELS, strategies.items(), reveal_sets
    ):
        yield {
            "decision": decision,
            "confidence": confidence,
            "variable_weights": weights,
            "weight_strategy": name,
            "reveal_sequence": reveals,
        }


#: Guard on the alternating polish below. It converges in two or three rounds in
#: practice; this only stops a pathological oscillation from looping forever.
_MAX_POLISH_ROUNDS = 8

#: Improvements smaller than this are floating-point noise, not signal.
_EPS = 1e-12


def _reveal_sets(sections: list[str]) -> list[list[str]]:
    return [
        list(combo)
        for size in range(len(sections) + 1)
        for combo in itertools.combinations(sections, size)
    ]


def polish(
    task: int, gts: list[dict], sections: list[str], seed: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    """Alternating coordinate ascent from a grid-search seed.

    The grid only offers whole-vector weight strategies -- one level everywhere,
    or the per-variable mode. Neither is optimal, and the mode is not even
    well-defined when two levels tie, which for Task 1's ``pirads`` is exactly
    what happens. Rather than break that tie by fiat, hill-climb: vary one
    variable's weight at a time, then re-optimise the reveal set (grounding
    depends on which variables are active), then confidence and the decision.
    Repeat until nothing improves.

    Ascent on a coupled objective only guarantees a local optimum, but it starts
    from the exhaustive grid's best point and can only improve on it.
    """
    best = dict(seed)
    best["variable_weights"] = dict(seed["variable_weights"])
    best_score = objective(task, gts, best)
    variables = spec.VARIABLES_BY_TASK[task]
    decisions = spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS
    reveal_sets = _reveal_sets(sections)

    def try_all(options, apply) -> bool:
        nonlocal best, best_score
        improved = False
        for option in options:
            candidate = apply(best, option)
            score = objective(task, gts, candidate)
            if score > best_score + _EPS:
                best, best_score, improved = candidate, score, True
        return improved

    def with_weight(base: dict, pair) -> dict:
        var, level = pair
        weights = dict(base["variable_weights"])
        weights[var] = level
        return {**base, "variable_weights": weights}

    for _ in range(_MAX_POLISH_ROUNDS):
        improved = False
        for var in variables:
            improved |= try_all(((var, level) for level in spec.WEIGHT_LEVELS), with_weight)
        improved |= try_all(reveal_sets, lambda b, r: {**b, "reveal_sequence": list(r)})
        improved |= try_all(spec.CONFIDENCE_LEVELS, lambda b, c: {**b, "confidence": c})
        improved |= try_all(decisions, lambda b, d: {**b, "decision": d})
        if not improved:
            break

    return best, best_score


def fit(task: int, gts: list[dict], sections: list[str]) -> tuple[dict[str, Any], float]:
    """Exhaustive grid over :func:`candidates`, then coordinate-ascent polish."""
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    for params in candidates(task, gts, sections):
        score = objective(task, gts, params)
        if score > best_score:
            best, best_score = params, score
    assert best is not None, f"no candidates generated for task {task}"
    if task == 3:
        # Task 3's grid is already exhaustive over its two scalars.
        return best, best_score
    return polish(task, gts, sections, best)


def cv_estimate(task: int, gts: list[dict], sections: list[str], folds: int = 5) -> float | None:
    """K-fold estimate of the *procedure*, not of one fixed parameter set.

    Each fold refits from scratch on the other folds, so this measures what the
    prior would score on data it has not seen -- which is the only number worth
    quoting outside this repo.
    """
    if len(gts) < folds * 2:
        return None
    scores: list[float] = []
    for k in range(folds):
        # Deterministic interleaved split; the release order is not meaningful.
        held_out = [gt for i, gt in enumerate(gts) if i % folds == k]
        train = [gt for i, gt in enumerate(gts) if i % folds != k]
        if not held_out or not train:
            continue
        params, _ = fit(task, train, sections)
        scores.append(objective(task, held_out, params))
    return (sum(scores) / len(scores)) if scores else None


def _public_params(task: int, params: dict[str, Any]) -> dict[str, Any]:
    """Drop bookkeeping fields the predictor does not need."""
    return {k: v for k, v in params.items() if k != "weight_strategy"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help=f"release root containing task<N>/ (default: {DEFAULT_DATA})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"parameter file to write (default: {DEFAULT_OUT})")
    parser.add_argument("--folds", type=int, default=5,
                        help="cross-validation folds for the reported estimate (default: 5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the fit without writing the parameter file")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"no release data at {args.data}")

    fitted: dict[str, Any] = {
        "_comment": (
            "Fitted by chimera.cli.fit_prior against the released training labels. "
            "Do not hand-edit; re-run the fit instead. in_sample_score is the "
            "selection objective and is optimistic; cv_score estimates the "
            "fit-then-predict procedure on unseen cases."
        ),
    }

    for task in (1, 2, 3):
        gts = load_ground_truth(args.data / f"task{task}", task)
        if not gts:
            print(f"task{task}: no labeled cases under {args.data}, skipping")
            continue
        sections = available_sections(args.data, task)
        params, score = fit(task, gts, sections)
        cv = cv_estimate(task, gts, sections, args.folds)

        entry = _public_params(task, params)
        entry["n_labeled"] = len(gts)
        entry["available_sections"] = sections
        entry["in_sample_score"] = round(score, 6)
        entry["cv_score"] = round(cv, 6) if cv is not None else None
        fitted[f"task{task}"] = entry

        metric = "mean_case_score" if task == 3 else "ranking_score"
        print(f"=== task{task} (n={len(gts)}) ===")
        print(f"  sections available : {sections}")
        for key, value in _public_params(task, params).items():
            print(f"  {key:<19}: {value}")
        print(f"  {metric} in-sample : {score:.4f}")
        print(f"  {metric} {args.folds}-fold CV : {cv:.4f}" if cv is not None else "  CV: n/a")
        print()

    if args.dry_run:
        print("--dry-run: not written")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fitted, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
