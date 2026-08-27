"""Read the official evaluator's output back at the prices the leaderboard uses.

``scripts/score.sh`` runs the official ``evaluate.py`` with ``USE_RATIONALE_JUDGE=0``,
because the judge needs an Ollama container this host cannot run. That is not a
smaller version of the live score -- it is a *differently priced* one. ``evaluate.py``
carries two component vectors and picks between them on whether the judge returned a
score (``evaluate.py`` ~L1559):

===================  ==========  ===========
component            judge on    judge off
===================  ==========  ===========
confidence               0.20        0.225
var_weight               0.25        0.275
factor_f1                0.15        0.175
tool                     0.15        0.150
section_grounding    **0.05**    **0.175**
rationale                0.20         --
===================  ==========  ===========

Section grounding is priced **3.5x higher** with the judge off. That is not a detail:
grounding is exactly the term that penalises weighting a variable whose section was
never revealed, so the judge-off run systematically prefers policies that weight few
variables, and the judge-on leaderboard does not. A policy change can therefore lose
under ``score.sh`` and win on the platform. One nearly did.

This module does not re-score anything. Every component value it reads is the
official evaluator's own, taken from ``_scores/per_case_results.csv``; only the linear
combination is replaced. That keeps the standing rule intact -- the numbers come from
the official pipeline -- while reporting them at the weights that actually rank us.

**What is missing, and when that matters.** The judge-on case score is these five
components plus ``0.20 * rationale``, and we cannot compute the rationale term without
the judge. So :func:`reprice_run` reports the 0.80-weight deterministic part, which is
a lower bound in absolute terms. It is nonetheless *exact for comparisons* whenever
the rationale term is unchanged between two runs -- and that is checkable rather than
assumed, because the judge's entire input is ``case_id``, ``task``, ``clinical_data``,
the expected decision and confidence, and the predicted decision, confidence and
``free_text`` (``evaluate.py`` ``build_rationale_judge``). ``variable_weights`` and
``reveal_sequence`` are never shown to it. Two runs that differ only in those two
fields therefore have an identical rationale term, and the difference this module
reports is the whole difference. :func:`rationale_term_is_shared` checks that
precondition against the run directories instead of trusting it.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from chimera.scoring.fast import (
    CASE_COMPONENT_WEIGHTS,
    CASE_COMPONENT_WEIGHTS_JUDGE_OFF,
)

#: Our component names -> the evaluator's ``per_case_results.csv`` columns.
COMPONENT_COLUMNS: dict[str, str] = {
    "confidence": "confidence_score",
    "var_weight": "variable_weight_score",
    "factor_f1": "important_decisive_factor_score",
    "tool": "tool_score",
    "section_grounding": "section_grounding_score",
}

#: The evaluator's ``task`` column values, by task id. Task 3 is deliberately absent:
#: its ranking score is the C-index alone, so no repricing can move it.
CSV_TASK_NAMES: dict[int, str] = {1: "biopsy", 2: "treatment"}

#: ``evaluate.py`` L1986. Only tasks actually present share the denominator.
TASK_RANKING_WEIGHTS: dict[int, float] = {1: 2.0, 2: 2.0, 3: 1.0}

#: Fields whose value the rationale judge is shown. Anything outside this set can
#: differ between two runs without moving the rationale term.
JUDGE_VISIBLE_FIELDS: tuple[str, ...] = (
    "case_id",
    "clinical_data",
    "confidence",
    "free_text",
    "biopsy_decision",
    "treatment_recommendation",
    "event",
    "months_to_recurrence",
)


def _rows(run_dir: Path, task: int) -> list[dict[str, str]]:
    path = Path(run_dir) / "_scores" / "per_case_results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} -- run scripts/score.sh on this run first")
    name = CSV_TASK_NAMES[task]
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["task"] == name]


def _mean_case_score(rows: Iterable[dict[str, str]], weights: dict[str, float]) -> float:
    """Mean over *all* cases, scoring a gated-out case as zero.

    The gate is the evaluator's: a wrong decision earns nothing regardless of how good
    the reasoning was, and it still occupies a slot in the denominator.
    """
    rows = list(rows)
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        if row["gate"] != "passed":
            continue
        total += sum(
            weight * float(row[COMPONENT_COLUMNS[name]] or 0.0)
            for name, weight in weights.items()
        )
    return total / len(rows)


def reprice_run(run_dir: Path) -> dict[str, Any]:
    """Both pricings of a scored run, per task and overall.

    Returns ``{"task<N>": {...}, "overall_ranking_score_judge_off": float,
    "overall_ranking_score_judge_on_partial": float}``. Task 3 is carried through
    unchanged, since its ranking score is the C-index.

    The F1 term is taken from the official aggregate and left alone: it depends only
    on the decisions, which no reasoning constant can move.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "_scores" / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"{metrics_path} -- run scripts/score.sh on this run first")
    aggregates = json.loads(metrics_path.read_text())["aggregates"]

    out: dict[str, Any] = {}
    present: list[int] = []
    for task in (1, 2, 3):
        key = f"task{task}"
        if key not in aggregates:
            continue
        present.append(task)
        official = aggregates[key]

        if task == 3:
            out[key] = {
                "mean_case_score_judge_off": official["mean_case_score"],
                "ranking_score_judge_off": official["ranking_score"],
                "ranking_score_judge_on_partial": official["ranking_score"],
                "note": "C-index only; component prices do not enter the ranking score",
            }
            continue

        rows = _rows(run_dir, task)
        off = _mean_case_score(rows, CASE_COMPONENT_WEIGHTS_JUDGE_OFF)
        on = _mean_case_score(rows, CASE_COMPONENT_WEIGHTS)

        # The evaluator's own number, recomputed from its own per-case components. If
        # this drifts, the column mapping or the gate rule is wrong and every figure
        # below is untrustworthy -- so it is checked rather than assumed.
        if abs(off - official["mean_case_score"]) > 1e-6:
            raise AssertionError(
                f"{key}: repriced judge-off mean_case_score {off:.6f} != official "
                f"{official['mean_case_score']:.6f}; the CSV mapping is wrong"
            )

        # ranking = (mean_case_score + F1) / 2, so the F1 term falls straight out of
        # the official pair and needs no per-task key name.
        f1_term = 2.0 * official["ranking_score"] - official["mean_case_score"]
        out[key] = {
            "mean_case_score_judge_off": off,
            "mean_case_score_judge_on_partial": on,
            "f1_term": f1_term,
            "ranking_score_judge_off": official["ranking_score"],
            "ranking_score_judge_on_partial": (on + f1_term) / 2.0,
        }

    denominator = sum(TASK_RANKING_WEIGHTS[t] for t in present) or 1.0
    for label in ("judge_off", "judge_on_partial"):
        out[f"overall_ranking_score_{label}"] = sum(
            out[f"task{t}"][f"ranking_score_{label}"] * TASK_RANKING_WEIGHTS[t]
            for t in present
        ) / denominator
    return out


def rationale_term_is_shared(a: Path, b: Path, task: int) -> tuple[bool, dict[str, int]]:
    """Whether two runs are identical in everything the rationale judge can see.

    When this is true, the ``judge_on_partial`` *difference* between the two runs is
    the exact judge-on difference, because the omitted ``0.20 * rationale`` term is
    the same on both sides. Returns the verdict and a count of differing fields, so a
    false answer says which field broke it.
    """
    from chimera.scoring.records import predictions_from_run

    left = {r["case_id"]: r for r in predictions_from_run(Path(a), task)}
    right = {r["case_id"]: r for r in predictions_from_run(Path(b), task)}
    differing: dict[str, int] = {}
    if set(left) != set(right):
        differing["case_id"] = len(set(left) ^ set(right))
        return False, differing
    for case_id, lrow in left.items():
        rrow = right[case_id]
        for field in JUDGE_VISIBLE_FIELDS:
            if lrow.get(field) != rrow.get(field):
                differing[field] = differing.get(field, 0) + 1
    return not differing, differing


def _format(report: dict[str, Any]) -> str:
    lines = [
        "",
        "Repriced at live judge-on component weights (section grounding 0.05, not",
        "0.175). Components are the official evaluator's own; only the weighted sum",
        "is recomputed. The 0.20 rationale term is omitted -- absolute values are",
        "therefore lower bounds, but differences between runs whose free_text,",
        "decision, confidence and clinical_data match are exact.",
        "",
        f"  {'':7} {'ranking judge-off':>18} {'ranking judge-on*':>18}",
    ]
    for task in (1, 2, 3):
        key = f"task{task}"
        if key not in report:
            continue
        entry = report[key]
        lines.append(
            f"  {key:7} {entry['ranking_score_judge_off']:18.4f} "
            f"{entry['ranking_score_judge_on_partial']:18.4f}"
        )
    lines.append(
        f"  {'overall':7} {report['overall_ranking_score_judge_off']:18.4f} "
        f"{report['overall_ranking_score_judge_on_partial']:18.4f}"
    )
    lines.append("")
    lines.append("  * partial: excludes the 0.20 rationale term, which needs the judge.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a run directory already scored")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    report = reprice_run(args.run_dir)
    print(json.dumps(report, indent=2) if args.json else _format(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
