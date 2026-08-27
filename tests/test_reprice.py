"""The judge-off score is not a smaller live score; it is a differently priced one.

``scripts/score.sh`` cannot run the rationale judge on this host, so the official
evaluator takes its ``rs is None`` branch and prices section grounding at 0.175 --
3.5x the 0.05 the live leaderboard uses. Grounding is precisely the term that
penalises weighting a variable whose section was never revealed, so the two pricings
disagree about exactly the policies a reasoning fit moves between. A refit that gains
+0.018 on Task 2's live ranking score loses 0.017 under ``score.sh``.

:mod:`chimera.scoring.reprice` reads the evaluator's own per-case components back and
recombines them at the live weights. These tests pin the two things that can silently
rot: the arithmetic (including the 2:2:1 task aggregation), and the CSV column names,
which are the evaluator's and not ours.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from chimera.scoring.fast import (
    CASE_COMPONENT_WEIGHTS,
    CASE_COMPONENT_WEIGHTS_JUDGE_OFF,
)
from chimera.scoring.reprice import (
    COMPONENT_COLUMNS,
    rationale_term_is_shared,
    reprice_run,
)

# The parity module already knows how to build a run directory and drive the official
# evaluator over it; reusing it keeps one definition of "a scored run".
from tests.test_run_dir_parity import (  # noqa: F401  (refs_available is a fixture)
    _build_run,
    _run_official,
    refs_available,
)


def _write_scored_run(
    root: Path, task: int, rows: list[dict], f1: float, name: str
) -> Path:
    """A minimal scored run directory: the two files ``reprice_run`` reads."""
    out = root / "_scores"
    out.mkdir(parents=True, exist_ok=True)

    columns = ["case_id", "task", "gate", *COMPONENT_COLUMNS.values()]
    with (out / "per_case_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(
                {
                    "case_id": f"C{i:03d}",
                    "task": name,
                    "gate": row["gate"],
                    **{
                        COMPONENT_COLUMNS[k]: row.get(k, 0.0) for k in COMPONENT_COLUMNS
                    },
                }
            )

    mean_off = sum(
        sum(w * r.get(k, 0.0) for k, w in CASE_COMPONENT_WEIGHTS_JUDGE_OFF.items())
        for r in rows
        if r["gate"] == "passed"
    ) / len(rows)
    (out / "metrics.json").write_text(
        json.dumps(
            {
                "aggregates": {
                    f"task{task}": {
                        "n_cases": len(rows),
                        "mean_case_score": mean_off,
                        "ranking_score": (mean_off + f1) / 2.0,
                    }
                },
                "results": [],
            }
        )
    )
    return root


#: One gate-passed case with perfect everything, and one with grounding on the floor
#: -- the shape a "weight more variables than you revealed" policy produces.
ROWS = [
    {
        "gate": "passed",
        "confidence": 1.0,
        "var_weight": 1.0,
        "factor_f1": 1.0,
        "tool": 1.0,
        "section_grounding": 1.0,
    },
    {
        "gate": "passed",
        "confidence": 1.0,
        "var_weight": 1.0,
        "factor_f1": 1.0,
        "tool": 1.0,
        "section_grounding": 0.0,
    },
    {
        "gate": "failed",
        "confidence": 1.0,
        "var_weight": 1.0,
        "factor_f1": 1.0,
        "tool": 1.0,
        "section_grounding": 1.0,
    },
]


def test_the_judge_off_number_is_reproduced_from_the_components(tmp_path):
    """The guard inside ``reprice_run``: recombining at judge-off prices is a no-op.

    If this drifts, the column mapping or the gate rule is wrong and every repriced
    figure is untrustworthy -- which is why it raises rather than warns.
    """
    run = _write_scored_run(tmp_path / "run", 1, ROWS, f1=0.80, name="biopsy")
    out = reprice_run(run)
    official = json.loads((run / "_scores" / "metrics.json").read_text())
    assert out["task1"]["mean_case_score_judge_off"] == pytest.approx(
        official["aggregates"]["task1"]["mean_case_score"]
    )
    assert out["task1"]["ranking_score_judge_off"] == pytest.approx(
        official["aggregates"]["task1"]["ranking_score"]
    )


def test_a_bad_column_mapping_is_an_error_not_a_wrong_number(tmp_path):
    """A silently wrong reprice is worse than a crash: it would decide a refit."""
    run = _write_scored_run(tmp_path / "run", 1, ROWS, f1=0.80, name="biopsy")
    metrics = run / "_scores" / "metrics.json"
    payload = json.loads(metrics.read_text())
    payload["aggregates"]["task1"]["mean_case_score"] += 0.05
    metrics.write_text(json.dumps(payload))
    with pytest.raises(AssertionError, match="CSV mapping is wrong"):
        reprice_run(run)


def test_grounding_is_the_term_the_two_pricings_disagree_about(tmp_path):
    """The whole reason this module exists, as a number.

    Both cases here are otherwise perfect, so the only difference between the two
    pricings is what a lost grounding score costs: 0.175 with the judge off against
    0.05 with it on.
    """
    run = _write_scored_run(tmp_path / "run", 1, ROWS, f1=0.80, name="biopsy")
    out = reprice_run(run)["task1"]
    lost = 1.0 / len(ROWS)  # one of three cases has grounding 0.0
    penalty_off = CASE_COMPONENT_WEIGHTS_JUDGE_OFF["section_grounding"] * lost
    penalty_on = CASE_COMPONENT_WEIGHTS["section_grounding"] * lost

    perfect_off = sum(CASE_COMPONENT_WEIGHTS_JUDGE_OFF.values()) * 2 / len(ROWS)
    perfect_on = sum(CASE_COMPONENT_WEIGHTS.values()) * 2 / len(ROWS)
    assert out["mean_case_score_judge_off"] == pytest.approx(perfect_off - penalty_off)
    assert out["mean_case_score_judge_on_partial"] == pytest.approx(perfect_on - penalty_on)
    assert penalty_off / penalty_on == pytest.approx(3.5)


def test_the_f1_term_survives_repricing_untouched(tmp_path):
    """Decisions are not repriced. Only the case-score half of the ranking moves."""
    run = _write_scored_run(tmp_path / "run", 2, ROWS, f1=0.60, name="treatment")
    out = reprice_run(run)["task2"]
    assert out["f1_term"] == pytest.approx(0.60)
    assert out["ranking_score_judge_on_partial"] == pytest.approx(
        (out["mean_case_score_judge_on_partial"] + 0.60) / 2.0
    )


def test_tasks_are_aggregated_two_two_one_over_the_tasks_present(tmp_path):
    """``evaluate.py`` L1986, including that absent tasks leave the denominator."""
    run = tmp_path / "run"
    _write_scored_run(run, 1, ROWS, f1=0.80, name="biopsy")

    # Splice a second task in, so both weigh 2.0 and the denominator is 4, not 5.
    metrics = run / "_scores" / "metrics.json"
    payload = json.loads(metrics.read_text())
    payload["aggregates"]["task2"] = dict(payload["aggregates"]["task1"])
    metrics.write_text(json.dumps(payload))
    with (run / "_scores" / "per_case_results.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "task", "gate", *COMPONENT_COLUMNS.values()]
        )
        for i, row in enumerate(ROWS):
            writer.writerow(
                {
                    "case_id": f"T{i:03d}",
                    "task": "treatment",
                    "gate": row["gate"],
                    **{COMPONENT_COLUMNS[k]: row.get(k, 0.0) for k in COMPONENT_COLUMNS},
                }
            )

    out = reprice_run(run)
    both = out["task1"]["ranking_score_judge_on_partial"]
    assert out["overall_ranking_score_judge_on_partial"] == pytest.approx(both)


def test_task3_is_carried_through_because_the_c_index_has_no_components(tmp_path):
    """Task 3 ranks on Harrell's C alone, so no pricing can move it."""
    run = tmp_path / "run"
    out_dir = run / "_scores"
    out_dir.mkdir(parents=True)
    (out_dir / "per_case_results.csv").write_text(
        ",".join(["case_id", "task", "gate", *COMPONENT_COLUMNS.values()]) + "\n"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "aggregates": {
                    "task3": {"mean_case_score": 0.7513, "ranking_score": 0.7372}
                },
                "results": [],
            }
        )
    )
    out = reprice_run(run)
    assert out["task3"]["ranking_score_judge_on_partial"] == pytest.approx(0.7372)
    assert out["overall_ranking_score_judge_on_partial"] == pytest.approx(0.7372)


def test_the_column_names_are_the_evaluators_own(refs_available, tmp_path):
    """Against a real evaluator run, so a renamed column cannot pass unnoticed.

    The unit tests above build the CSV from ``COMPONENT_COLUMNS`` and would agree with
    themselves after any rename. This one lets ``evaluate.py`` write the file.
    """
    run_dir = tmp_path / "run"
    _build_run(run_dir, 1)
    _run_official(run_dir)
    out = reprice_run(run_dir)["task1"]  # raises if a column is missing or misnamed
    assert 0.0 <= out["mean_case_score_judge_on_partial"] <= 1.0
    assert out["mean_case_score_judge_off"] > 0.0


def test_two_runs_differing_only_in_weights_share_the_rationale_term(
    refs_available, tmp_path
):
    """The precondition that makes a repriced *difference* exact rather than a bound.

    The judge is shown the case, the decision, the confidence and the free text -- not
    ``variable_weights`` and not ``reveal_sequence``. So two runs that differ only in
    those have an identical rationale term, and omitting it cancels.
    """
    a = tmp_path / "a"
    b = tmp_path / "b"
    _build_run(a, 1)
    _build_run(b, 1)
    shared, differing = rationale_term_is_shared(a, b, 1)
    assert shared, f"identical runs should share everything: {differing}"

    from chimera.scoring.records import predictions_from_run

    # Perturb something the judge *can* see, and the guarantee must be withdrawn.
    # Located by content rather than by path: a run directory is flat job dirs, and
    # Task 1 writes the reasoning socket under both the corrected and the legacy
    # spelling, so patching one file by name can leave the reader seeing the other.
    victim = next(r for r in predictions_from_run(b, 1) if r.get("free_text"))
    original = victim["free_text"]
    patched = 0
    for path in b.rglob("*.json"):
        text = path.read_text()
        if original in text:
            path.write_text(text.replace(original, "CHANGED " + original))
            patched += 1
    assert patched, "found no file carrying the free_text we read back"

    shared, differing = rationale_term_is_shared(a, b, 1)
    assert not shared and "free_text" in differing
