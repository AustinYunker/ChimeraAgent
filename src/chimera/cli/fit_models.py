"""Fit the C2 decision models and export them for the container.

Writes ``guideline_params.json`` next to the predictor that reads it: leaf labels,
reasoning constants per decision, and the cross-validated score each task was
accepted on. Models cross the boundary into the submission image as **plain JSON**,
never as pickles, so the container keeps its standard-library-only import closure and
stays at ~47 MB -- and so every number that influences a prediction is legible in the
diff.

Re-run whenever labels grow; the challenge site says they arrive incrementally.

Usage::

    python -m chimera.cli.fit_models
    python -m chimera.cli.fit_models --repeats 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chimera.cli.cross_validate import _constant_fit, _guideline_fit
from chimera.cli.fit_prior import available_sections
from chimera.eval.cv import cross_validate, load_rows
from chimera.models import stratified

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES = REPO_ROOT / "work" / "train" / "cases"
DEFAULT_GT = REPO_ROOT / "work" / "train" / "ground_truth"
DEFAULT_DATA = REPO_ROOT / "data" / "train_release"
DEFAULT_OUT = REPO_ROOT / "src" / "chimera" / "predictors" / "guideline_params.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repeats", type=int, default=5,
                        help="repeated-CV runs used for the recorded estimate")
    parser.add_argument("--skip-cv", action="store_true",
                        help="fit only; leave the recorded CV scores null")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fitted: dict[str, Any] = {
        "_comment": (
            "Fitted by chimera.cli.fit_models. Do not hand-edit; re-run the fit. "
            "leaf_labels map guideline strata to decisions and were chosen against "
            "the official ranking metric, not accuracy. cv_* are repeated pooled "
            "out-of-fold estimates; cv_sd is the spread across split seeds and a "
            "difference smaller than it is not evidence."
        )
    }

    for task in (1, 2):
        rows = load_rows(args.cases, args.gt, task)
        if not rows:
            print(f"task{task}: no labeled rows under {args.gt}")
            continue
        sections = available_sections(args.data, task)
        params = stratified.fit(task, rows, list(sections))

        entry: dict[str, Any] = {
            "leaf_labels": params["leaf_labels"],
            "reasoning": params["reasoning"],
            "n_labeled": len(rows),
            "available_sections": list(sections),
        }

        if not args.skip_cv:
            fit_fn, predict_fn = _guideline_fit(task, sections)
            guided = cross_validate(task, rows, fit_fn, predict_fn, repeats=args.repeats)
            const_fit, const_predict = _constant_fit(task, sections)
            baseline = cross_validate(task, rows, const_fit, const_predict, repeats=args.repeats)
            entry["cv_score"] = guided["mean"]
            entry["cv_sd"] = guided["sd"]
            entry["cv_baseline_constant"] = baseline["mean"]
            entry["cv_beats_baseline_beyond_noise"] = bool(
                guided["mean"] is not None
                and baseline["mean"] is not None
                and (guided["mean"] - baseline["mean"]) > max(guided["sd"], baseline["sd"])
            )

        fitted[f"task{task}"] = entry

        print(f"=== task{task} (n={len(rows)}) ===")
        for leaf, label in sorted(params["leaf_labels"].items()):
            print(f"  {leaf:<24} -> {label}")
        print(f"  reasoning fitted for  : {sorted(k for k in params['reasoning'])}")
        if not args.skip_cv:
            print(f"  cv {entry['cv_score']:.4f} +/- {entry['cv_sd']:.4f} "
                  f"(constant {entry['cv_baseline_constant']:.4f}) "
                  f"beats-noise={entry['cv_beats_baseline_beyond_noise']}")
        print()

    # Task 3 has nothing to fit; the record exists so the file documents all three.
    rows3 = load_rows(args.cases, args.gt, 3)
    entry3: dict[str, Any] = {
        "model": "capra_s",
        "fitted": False,
        "n_labeled": len(rows3),
        "months_at_zero_risk": stratified.MONTHS_AT_ZERO_RISK,
        "months_per_capra_point": stratified.MONTHS_PER_CAPRA_POINT,
    }
    if rows3 and not args.skip_cv:
        result = cross_validate(
            3, rows3, lambda train: None, stratified.predict_recurrence_record, repeats=1
        )
        entry3["cv_score"] = result["mean"]
        entry3["cv_sd"] = 0.0
        print(f"=== task3 (n={len(rows3)}) ===")
        print(f"  CAPRA-S, nothing fitted -> c-index {result['mean']:.4f}\n")
    fitted["task3"] = entry3

    if args.dry_run:
        print("--dry-run: not written")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fitted, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
