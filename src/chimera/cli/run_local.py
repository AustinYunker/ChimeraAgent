"""Run a predictor over a local cohort and emit an evaluator-ready run directory.

Grand Challenge invokes the container once per case; locally we sweep a whole
cohort and then rebuild the ``predictions.json`` job dump the official evaluator
expects. Output layout::

    <run>/predictions.json
    <run>/<job_pk>/<relative_path>.json

Usage::

    python -m chimera.cli.run_local --cases work/fixtures --out work/run/constant
    python -m chimera.cli.run_local --cases work/fixtures --tasks 1 3
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from chimera.contract.aggregate import job_pk, write_predictions_dump
from chimera.contract.io import CaseInputs, read_case, write_case_outputs
from chimera.contract.types import Prediction
from chimera.mcp.client import McpSession
from chimera.predictors import (
    ConstantPredictor,
    GuidelinePredictor,
    Predictor,
    PriorPredictor,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

PREDICTORS: dict[str, type[Predictor]] = {
    "constant": ConstantPredictor,
    "prior": PriorPredictor,
    "guideline": GuidelinePredictor,
}


def discover_cases(cases_root: Path, tasks: tuple[int, ...]) -> list[Path]:
    """Every case directory under ``cases_root/task<N>/<case_id>/``."""
    found: list[Path] = []
    for task in tasks:
        task_dir = cases_root / f"task{task}"
        if not task_dir.is_dir():
            continue
        found.extend(sorted(d for d in task_dir.iterdir() if (d / "inputs.json").is_file()))
    return found


def run(
    predictor: Predictor,
    case_dirs: list[Path],
    out_root: Path,
    session: McpSession,
) -> tuple[list[tuple[CaseInputs, Prediction]], list[tuple[Path, str]]]:
    """Predict every case, writing outputs under ``out_root/<job_pk>/``.

    A case that raises is recorded and skipped rather than aborting the sweep --
    one bad case should not cost the other 249. The evaluator scores a missing
    case as a decision error, so failures stay visible in the metrics.

    ``session`` serves the whole cohort, so the sweep pays one server start
    rather than one per case, and each case gets its own store -- and therefore
    its own retrieval ledger.
    """
    done: list[tuple[CaseInputs, Prediction]] = []
    failed: list[tuple[Path, str]] = []

    for case_dir in case_dirs:
        try:
            case = read_case(case_dir, fallback_case_id=case_dir.name)
            pred = predictor.predict(case, session.store_for(case))
            write_case_outputs(out_root / job_pk(case.task, case.case_id), pred)
            done.append((case, pred))
        except Exception:  # noqa: BLE001 -- deliberately broad, see docstring
            failed.append((case_dir, traceback.format_exc()))

    return done, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=Path, default=REPO_ROOT / "work" / "fixtures",
        help="cohort root containing task<N>/<case_id>/ (default: work/fixtures)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="run directory to write (default: work/run/<predictor>)",
    )
    parser.add_argument(
        "--predictor", choices=sorted(PREDICTORS), default="constant",
        help="which predictor to run (default: constant)",
    )
    parser.add_argument(
        "--tasks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3],
        help="tasks to run (default: all)",
    )
    args = parser.parse_args()

    out_root = args.out or REPO_ROOT / "work" / "run" / args.predictor
    predictor = PREDICTORS[args.predictor]()

    case_dirs = discover_cases(args.cases, tuple(args.tasks))
    if not case_dirs:
        raise SystemExit(
            f"no cases found under {args.cases} for tasks {args.tasks}\n"
            "generate fixtures first: python -m chimera.cli.make_fixtures"
        )

    with McpSession.for_cohort(args.cases) as session:
        done, failed = run(predictor, case_dirs, out_root, session)
    write_predictions_dump(out_root, done)

    print(f"predictor : {predictor.name}")
    print(f"cases     : {len(done)} written, {len(failed)} failed")
    print(f"run dir   : {out_root}")

    for case_dir, tb in failed:
        print(f"\n!! {case_dir}\n{tb}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
