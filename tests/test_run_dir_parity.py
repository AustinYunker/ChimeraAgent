"""C1, through the files: the fast scorer must agree with a real evaluator run.

``test_scorer_parity`` drives both scorers over the same in-memory records, so
it pins the maths and nothing else. Everything *around* the maths is where a
submission actually dies: a misspelled slug, a job routed to the wrong task, a
case id that never made it into the structured prompt, output files nested one
directory deeper than expected.

So this module builds a run directory the way ``run_local`` does, invokes the
official ``evaluate.py`` on it as a subprocess -- the same invocation
``scripts/score.sh`` makes -- and diffs its ``metrics.json`` against the fast
scorer reading the *same directory* back off disk. Nothing is shared between
the two sides but the files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chimera.cli.run_local import discover_cases, run
from chimera.cli.score_fast import score_task
from chimera.contract.aggregate import write_predictions_dump
from chimera.predictors.constant import ConstantPredictor
from chimera.scoring.records import predictions_from_run, record_from_prediction

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "refs" / "challenge" / "evaluation" / "evaluate.py"
GT_ROOT = REPO_ROOT / "refs" / "challenge" / "evaluation" / "ground_truth"
FIXTURES = REPO_ROOT / "work" / "fixtures"

TOL = 1e-9


def _build_run(out_root: Path, task: int) -> list:
    """Predict every fixture case for ``task`` into an evaluator-ready run dir."""
    case_dirs = discover_cases(FIXTURES, (task,))
    if not case_dirs:
        pytest.skip(f"no fixtures for task{task} under {FIXTURES}")
    done, failed = run(ConstantPredictor(), case_dirs, out_root)
    assert not failed, f"predictor failed on {[str(d) for d, _ in failed]}"
    write_predictions_dump(out_root, done)
    return done


def _run_official(run_dir: Path, task: int) -> Path:
    """Invoke the official evaluator exactly as ``scripts/score.sh`` does."""
    out = run_dir / "_scores" / f"task{task}"
    out.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "TASK_ID": f"task{task}",
        "INPUT_DIRECTORY": str(run_dir),
        "PREDICTIONS_FILE": str(run_dir / "predictions.json"),
        "GROUND_TRUTH_DIR": str(GT_ROOT / f"task{task}"),
        "SECTION_MAPPING_FILE": str(GT_ROOT / "section_variable_mapping.json"),
        "EVAL_OUTPUT_DIR": str(out),
        "USE_RATIONALE_JUDGE": "0",
    }
    proc = subprocess.run(
        [sys.executable, str(EVALUATOR)], env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, f"official evaluator failed:\n{proc.stdout}\n{proc.stderr}"
    return out / "metrics.json"


@pytest.fixture(scope="module")
def refs_available() -> None:
    if not EVALUATOR.is_file():
        pytest.skip(f"reference evaluator not cloned at {EVALUATOR}")
    if not GT_ROOT.is_dir():
        pytest.skip(f"reference ground truth not cloned at {GT_ROOT}")


@pytest.mark.parametrize("task", [1, 2, 3])
def test_fast_scorer_matches_a_real_evaluator_run(refs_available, tmp_path, task):
    """The C1 pass condition, end to end and through the filesystem."""
    run_dir = tmp_path / "run"
    _build_run(run_dir, task)
    metrics = _run_official(run_dir, task)
    assert metrics.is_file()

    aggregate, problems = score_task(run_dir, GT_ROOT, task, TOL, compare=True)
    assert not problems, "\n".join(problems)

    # A comparison that compared nothing would also report no problems.
    official = json.loads(metrics.read_text())
    assert official["aggregates"], "official run produced an empty aggregate"
    assert set(aggregate), "fast scorer produced an empty aggregate"
    assert official["results"], "official run scored no cases"


@pytest.mark.parametrize("task", [1, 2, 3])
def test_reading_a_run_directory_back_reproduces_what_was_written(
    refs_available, tmp_path, task
):
    """The reader is the writer's exact inverse.

    If these drift, the parity test above still passes -- both sides would read
    the same wrong thing -- while cross-validation quietly scores records that
    are not what the container will emit.
    """
    run_dir = tmp_path / "run"
    done = _build_run(run_dir, task)

    written = {
        case.case_id: record_from_prediction(pred, case.case_id) for case, pred in done
    }
    read_back = {rec["case_id"]: rec for rec in predictions_from_run(run_dir, task)}

    assert set(read_back) == set(written)
    for case_id, expected in written.items():
        got = dict(read_back[case_id])
        # `clinical_data` is added by the reader from the job's inline input
        # socket; the writer's record has no equivalent, and it is judge-only.
        got.pop("clinical_data", None)
        assert got == expected, f"task{task} {case_id} did not round-trip"


def test_the_reader_ignores_jobs_belonging_to_another_task(refs_available, tmp_path):
    """One run directory holds all three tasks; each read must see only its own."""
    run_dir = tmp_path / "run"
    case_dirs = discover_cases(FIXTURES, (1, 2, 3))
    if not case_dirs:
        pytest.skip(f"no fixtures under {FIXTURES}")
    done, failed = run(ConstantPredictor(), case_dirs, run_dir)
    assert not failed
    write_predictions_dump(run_dir, done)

    seen: dict[int, set[str]] = {}
    for task in (1, 2, 3):
        records = predictions_from_run(run_dir, task)
        seen[task] = {r["case_id"] for r in records}
        expected_kind = "months_to_recurrence" if task == 3 else (
            "treatment_recommendation" if task == 2 else "biopsy_decision"
        )
        assert records, f"task{task} read no records back"
        assert all(expected_kind in r for r in records), f"task{task} read a foreign record"

    assert not (seen[1] & seen[2]) and not (seen[2] & seen[3]) and not (seen[1] & seen[3])
    assert sum(len(v) for v in seen.values()) == len(done)


def test_a_ground_truth_case_with_no_prediction_is_scored_as_a_hard_zero(
    refs_available, tmp_path
):
    """Dropping a case must cost the full case score, not silently shrink n.

    This is the failure mode a crashed case produces in production, so the fast
    scorer has to charge for it the way the evaluator does.
    """
    from chimera.scoring.records import pair_run_with_ground_truth

    task = 1
    run_dir = tmp_path / "run"
    _build_run(run_dir, task)

    dump = run_dir / "predictions.json"
    jobs = json.loads(dump.read_text())
    assert len(jobs) >= 2, "need at least two cases to drop one"
    dropped = jobs.pop(0)
    dump.write_text(json.dumps(jobs))

    pairs = pair_run_with_ground_truth(run_dir, GT_ROOT / f"task{task}", task)
    missing = [(gt, pred) for gt, pred in pairs if pred is None]
    assert len(missing) == 1, "the dropped case should survive as an unmatched target"
    assert len(pairs) == len(jobs) + 1, "the target count must not shrink"

    metrics = _run_official(run_dir, task)
    aggregate, problems = score_task(run_dir, GT_ROOT, task, TOL, compare=True)
    assert not problems, "\n".join(problems)

    official = json.loads(metrics.read_text())
    row = next(r for r in official["results"] if r["gate"] == "missing_candidate")
    assert row["case_score"] == 0.0
    assert aggregate["n_cases"] == len(official["results"])
    assert dropped["pk"] not in dump.read_text()
