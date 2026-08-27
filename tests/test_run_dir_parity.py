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
from chimera.mcp.client import McpSession
from chimera.predictors.constant import ConstantPredictor
from chimera.scoring.records import predictions_from_run, record_from_prediction

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "refs" / "challenge" / "evaluation" / "evaluate.py"
GT_ROOT = REPO_ROOT / "refs" / "challenge" / "evaluation" / "ground_truth"
FIXTURES = REPO_ROOT / "work" / "fixtures"

# The reference ground truth ships two Task 3 cases and both are censored, so it
# has no comparable survival pair and the evaluator now refuses the run outright
# ("Ranking score is undefined for ['task3']"). The synthetic cohort is built
# with ~40% events for exactly this reason, so Task 3 parity runs there. Tasks 1
# and 2 stay on the real reference cases, where the labels are the organizers'.
SYNTH = REPO_ROOT / "work" / "synth"

#: ``(task, cases_root, gt_root)`` for each parity run.
COHORTS = [
    (1, FIXTURES, GT_ROOT),
    (2, FIXTURES, GT_ROOT),
    (3, SYNTH / "cases", SYNTH / "ground_truth"),
]

TOL = 1e-9


def _build_run(out_root: Path, task: int, cases_root: Path = FIXTURES) -> list:
    """Predict every case for ``task`` under ``cases_root`` into a run dir.

    One cohort-scoped MCP server for the whole sweep, exactly as
    ``run_local`` does it -- these tests are about the run directory, but they
    should reach the documents by the same route the harness does.
    """
    case_dirs = discover_cases(cases_root, (task,))
    if not case_dirs:
        pytest.skip(f"no cases for task{task} under {cases_root}")
    with McpSession.for_cohort(cases_root) as session:
        done, failed = run(ConstantPredictor(), case_dirs, out_root, session)
    assert not failed, f"predictor failed on {[str(d) for d, _ in failed]}"
    write_predictions_dump(out_root, done)
    return done


def _stub_case_map(run_dir: Path) -> Path:
    """An empty archive case map, which the evaluator requires but need not use.

    Since 2026-08-16 the evaluator resolves ``case_id`` for *file-backed* inputs
    by joining ComponentInterfaceValue PKs against a CSV Grand Challenge exports
    after archive creation, and it exits outright if the file is missing. Our run
    directories inline ``case_id`` in the structured-prompt value, which
    ``_case_id_for_job`` still tries first, so the map is never consulted -- but
    it has to exist. Writing it here rather than into the cloned ``refs/`` tree
    keeps the reference checkout pristine.
    """
    path = run_dir / "debug_archive_pks.csv"
    path.write_text("case_id,structured-prompt_pk\n")
    return path


def _run_official(run_dir: Path, gt_root: Path = GT_ROOT) -> Path:
    """Invoke the official evaluator exactly as ``scripts/score.sh`` does.

    One invocation for the whole run: the evaluator derives its task set from
    the dump, takes ``GROUND_TRUTH_DIR`` as the root holding ``task<N>/``, and
    writes a single ``metrics.json`` whose aggregates are keyed by task id.
    """
    out = run_dir / "_scores"
    out.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "CASE_MAP_FILE": str(_stub_case_map(run_dir)),
        "INPUT_DIRECTORY": str(run_dir),
        "PREDICTIONS_FILE": str(run_dir / "predictions.json"),
        "GROUND_TRUTH_DIR": str(gt_root),
        "SECTION_MAPPING_FILE": str(gt_root / "section_variable_mapping.json"),
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


@pytest.mark.parametrize("task,cases_root,gt_root", COHORTS, ids=["1", "2", "3"])
def test_fast_scorer_matches_a_real_evaluator_run(
    refs_available, tmp_path, task, cases_root, gt_root
):
    """The C1 pass condition, end to end and through the filesystem."""
    run_dir = tmp_path / "run"
    _build_run(run_dir, task, cases_root)
    metrics = _run_official(run_dir, gt_root)
    assert metrics.is_file()

    aggregate, problems = score_task(run_dir, gt_root, task, TOL, compare=True)
    assert not problems, "\n".join(problems)

    # A comparison that compared nothing would also report no problems.
    official = json.loads(metrics.read_text())
    assert official["aggregates"].get(f"task{task}"), "official run produced no aggregate"
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
    with McpSession.for_cohort(FIXTURES) as session:
        done, failed = run(ConstantPredictor(), case_dirs, run_dir, session)
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


def test_a_case_missing_from_the_dump_leaves_the_denominator(refs_available, tmp_path):
    """A dropped job now shrinks n rather than scoring a hard zero.

    This reversed at upstream b0ae4eb. ``run()`` derives the phase from the
    prediction dump -- ``phase_case_ids`` -- and filters the ground truth to it,
    so a target the dump never mentions is not scored at all. Before, it was
    kept and charged the full case score as a ``missing_candidate``.

    That is worth pinning rather than merely following: it means a case our
    container fails to emit costs nothing *here*, so the deterministic
    all-cases-or-fail behaviour of the container is what protects the score, not
    the evaluator. The fast scorer has to agree either way, or cross-validation
    is measuring a different cohort than the leaderboard.
    """
    from chimera.scoring.records import pair_run_with_ground_truth

    task = 1
    run_dir = tmp_path / "run"
    _build_run(run_dir, task)

    before = pair_run_with_ground_truth(run_dir, GT_ROOT / f"task{task}", task)

    dump = run_dir / "predictions.json"
    jobs = json.loads(dump.read_text())
    assert len(jobs) >= 2, "need at least two cases to drop one"
    dropped = jobs.pop(0)
    dump.write_text(json.dumps(jobs))

    pairs = pair_run_with_ground_truth(run_dir, GT_ROOT / f"task{task}", task)
    assert not [p for _, p in pairs if p is None], "no target should be left unmatched"
    assert len(pairs) == len(before) - 1, "the dropped case must leave the denominator"

    metrics = _run_official(run_dir)
    aggregate, problems = score_task(run_dir, GT_ROOT, task, TOL, compare=True)
    assert not problems, "\n".join(problems)

    official = json.loads(metrics.read_text())
    task_rows = [r for r in official["results"] if r["task"] == "biopsy"]
    assert not [r for r in task_rows if r["gate"] == "missing_candidate"]
    assert aggregate["n_cases"] == len(task_rows) == len(jobs)
    assert dropped["pk"] not in dump.read_text()
