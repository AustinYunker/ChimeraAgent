"""Score a run directory with the fast in-process scorer, and diff it against
the official one.

Two jobs in one command, deliberately:

* **Score.** ``--run <dir>`` reads the run's ``predictions.json`` and output
  files exactly as ``evaluate.py`` would, then scores them in-process. This is
  the loop C2/C3 will run per fold, without a subprocess or a judge.
* **Verify.** ``--compare`` additionally loads the official
  ``<run>/_scores/metrics.json`` written by ``scripts/score.sh`` -- one file for
  the whole run, with the aggregates keyed by task id -- and asserts every
  shared number agrees to within ``--tol`` (default 1e-9).

The comparison is the point. ``tests/test_scorer_parity.py`` drives both scorers
over in-memory records, which pins the *maths*; this pins the whole path --
file layout, slug resolution, job routing, case-id recovery, missing-case
handling. Both have to hold before the fast scorer can be trusted for model
selection.

Rule reminder: reported challenge performance always comes from
``scripts/score.sh``. This command exists to earn the right to use the fast
scorer *between* those runs, never to replace them.

Usage::

    scripts/score.sh work/run/constant                     # official, first
    python -m chimera.cli.score_fast --run work/run/constant --compare

    CHIMERA_GT_ROOT=$PWD/work/synth/ground_truth \\
      python -m chimera.cli.score_fast --run work/run/synth-constant --compare
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

from chimera.scoring.fast import CASE_COMPONENT_WEIGHTS_JUDGE_OFF, score_cohort_rows
from chimera.scoring.records import TASK_KIND, pair_run_with_ground_truth

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GT_ROOT = REPO_ROOT / "refs" / "challenge" / "evaluation" / "ground_truth"

# Written by the evaluator but not by us: the judge is off, and the sklearn text
# report is a formatted string rather than a number. Neither is a parity signal.
_IGNORED_AGGREGATE_KEYS = frozenset({
    "mean_rationale_score",
    "decision_classification_report",
})

# Per-case fields the evaluator strips before writing metrics.json, plus the
# judge column. Compared only where both sides carry the key.
_IGNORED_ROW_KEYS = frozenset({"rationale_score", "reason"})

# Headline numbers, in the order worth reading them.
_SUMMARY_KEYS: dict[str, tuple[str, ...]] = {
    "decision": (
        "n_cases", "n_evaluated", "mean_case_score",
        "mean_case_score_among_gate_passed", "decision_gate_pass_rate",
        "decision_accuracy", "decision_f1_yes", "decision_weighted_f1",
        "confidence_weighted_kappa", "variable_weight_weighted_kappa",
        "mean_tool_score", "mean_section_grounding_score", "ranking_score",
    ),
    "recurrence": (
        "n_cases", "n_evaluated", "mean_case_score",
        "recurrence_event_accuracy", "mean_event_score", "mean_time_score",
        "event1_time_mae_months", "concordance_index",
        "mean_time_dependent_auc", "ranking_score",
    ),
}


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _diff(
    ours: Any, theirs: Any, tol: float, path: str, out: list[str]
) -> None:
    """Collect every place ``ours`` and ``theirs`` disagree beyond ``tol``."""
    if isinstance(ours, dict) and isinstance(theirs, dict):
        for key in sorted(set(ours) | set(theirs)):
            if key in _IGNORED_AGGREGATE_KEYS:
                continue
            if key not in ours or key not in theirs:
                # A key only one side reports is a real structural difference
                # for aggregates, but per-case rows legitimately differ (the
                # evaluator strips private fields), so those are filtered by
                # the caller before reaching here.
                out.append(f"{path}.{key}: {'ours' if key in ours else 'official'} only")
                continue
            _diff(ours[key], theirs[key], tol, f"{path}.{key}", out)
        return

    if isinstance(ours, list) and isinstance(theirs, list):
        if len(ours) != len(theirs):
            out.append(f"{path}: length {len(ours)} != {len(theirs)}")
            return
        for i, (a, b) in enumerate(zip(ours, theirs)):
            _diff(a, b, tol, f"{path}[{i}]", out)
        return

    if isinstance(ours, bool) or isinstance(theirs, bool):
        # bool is an int subclass; compare identity of truth value, not 1.0.
        if bool(ours) != bool(theirs):
            out.append(f"{path}: {ours!r} != {theirs!r}")
        return

    if isinstance(ours, (int, float)) and isinstance(theirs, (int, float)):
        if math.isnan(ours) and math.isnan(theirs):
            return
        if abs(float(ours) - float(theirs)) > tol:
            out.append(f"{path}: {ours!r} != {theirs!r} (delta {abs(ours - theirs):.3e})")
        return

    if ours != theirs:
        out.append(f"{path}: {ours!r} != {theirs!r}")


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Strip the fields the evaluator strips before writing ``metrics.json``."""
    return {
        k: v for k, v in row.items()
        if not k.startswith("_") and k not in {
            "gt_biopsy_decision_conf", "pred_biopsy_decision_conf",
        }
    }


def compare_rows(
    ours: list[dict[str, Any]], theirs: list[dict[str, Any]], tol: float
) -> list[str]:
    """Diff per-case rows by case id, over the keys both sides report."""
    problems: list[str] = []
    mine = {r["case_id"]: _public_row(r) for r in ours}
    yours = {r["case_id"]: r for r in theirs}

    for cid in sorted(set(mine) - set(yours)):
        problems.append(f"results[{cid}]: scored by us, absent from the official run")
    for cid in sorted(set(yours) - set(mine)):
        problems.append(f"results[{cid}]: in the official run, not scored by us")

    for cid in sorted(set(mine) & set(yours)):
        a, b = mine[cid], yours[cid]
        shared = (set(a) & set(b)) - _IGNORED_ROW_KEYS
        _diff({k: a[k] for k in shared}, {k: b[k] for k in shared}, tol,
              f"results[{cid}]", problems)
    return problems


def official_metrics_path(run_dir: Path) -> Path:
    """Where ``scripts/score.sh`` leaves the official ``metrics.json``.

    One file for the whole run since upstream b0ae4eb: the evaluator scores
    every task in the dump in a single pass and keys its aggregates by task id.
    """
    return run_dir / "_scores" / "metrics.json"


def score_task(run_dir: Path, gt_root: Path, task: int, tol: float, compare: bool
               ) -> tuple[dict[str, Any], list[str]]:
    """Score one task; return its aggregate and any parity problems found."""
    pairs = pair_run_with_ground_truth(run_dir, gt_root / f"task{task}", task)
    # This command exists to diff against ``scripts/score.sh``, which runs the
    # official evaluator with USE_RATIONALE_JUDGE=0. Parity therefore needs the
    # judge-off weighting, not the judge-on one the selection loops default to.
    rows, aggregate = score_cohort_rows(
        pairs, weights=CASE_COMPONENT_WEIGHTS_JUDGE_OFF
    )

    if not compare:
        return aggregate, []

    metrics_path = official_metrics_path(run_dir)
    if not metrics_path.is_file():
        return aggregate, [
            f"no official metrics at {metrics_path}; run scripts/score.sh first"
        ]

    official = json.loads(metrics_path.read_text())
    task_id = f"task{task}"
    aggregates = official.get("aggregates", {})
    if task_id not in aggregates:
        return aggregate, [
            f"official metrics at {metrics_path} carry no {task_id} aggregate "
            f"(saw {sorted(k for k in aggregates if k.startswith('task'))})"
        ]

    # `results` spans every task in the dump, so select this one's rows by the
    # evaluator's own `task` field rather than by position.
    kind = TASK_KIND[task]
    their_rows = [
        r for r in official.get("results", []) if r.get("task") == kind
    ]

    problems: list[str] = []
    _diff(aggregate, aggregates[task_id], tol, "aggregates", problems)
    problems.extend(compare_rows(rows, their_rows, tol))
    return aggregate, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=REPO_ROOT / "work" / "run" / "constant",
        help="run directory holding predictions.json (default: work/run/constant)",
    )
    parser.add_argument(
        "--gt", type=Path, default=None,
        help="ground-truth root containing task<N>/ "
             "(default: $CHIMERA_GT_ROOT, else the official one)",
    )
    parser.add_argument(
        "--tasks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3],
        help="tasks to score (default: all)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="diff against the official metrics.json under <run>/_scores/",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-9,
        help="absolute tolerance for the comparison (default: 1e-9)",
    )
    parser.add_argument(
        "--json", dest="as_json", action="store_true",
        help="print the aggregates as JSON instead of a summary table",
    )
    args = parser.parse_args()

    # Same override scripts/score.sh honours, so the two commands can be pointed
    # at the synthetic cohort with one exported variable.
    gt_root = args.gt or Path(os.environ.get("CHIMERA_GT_ROOT") or DEFAULT_GT_ROOT)

    all_problems: list[str] = []
    payload: dict[str, Any] = {}

    for task in args.tasks:
        aggregate, problems = score_task(args.run, gt_root, task, args.tol, args.compare)
        payload[f"task{task}"] = aggregate
        all_problems.extend(f"task{task}: {p}" for p in problems)

        if args.as_json:
            continue
        kind = "recurrence" if task == 3 else "decision"
        print(f"=== task{task} ===")
        for key in _SUMMARY_KEYS[kind]:
            if key in aggregate:
                print(f"  {key:<38} {_fmt(aggregate[key])}")
        if args.compare:
            print(f"  {'parity vs official':<38} "
                  f"{'OK' if not problems else f'{len(problems)} MISMATCH'}")
        print()

    if args.as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))

    if all_problems:
        print(f"!! {len(all_problems)} mismatch(es) at tol={args.tol:g}:")
        for problem in all_problems[:50]:
            print(f"   {problem}")
        if len(all_problems) > 50:
            print(f"   ... and {len(all_problems) - 50} more")
        return 1

    if args.compare:
        print(f"fast scorer matches the official metrics.json to {args.tol:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
