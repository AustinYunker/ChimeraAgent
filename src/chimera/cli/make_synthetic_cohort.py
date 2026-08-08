"""Generate a synthetic cohort *with ground truth*, for exercising the scorer.

The eight fixture cases carried by the reference repos are too few and too
degenerate to test the scoring surface: both Task 3 cases are censored, so there
are no comparable pairs and Harrell's C-index is undefined, and Task 2 never
sees all four treatment classes. That leaves the Task 3 leaderboard path -- the
*only* thing Task 3 is ranked on -- completely untested.

This module emits a larger synthetic cohort in the same layout as the real data,
paired with ground truth in the evaluator's own directory format, so the full
metric surface can be driven end to end.

    work/synth/cases/task<N>/<case_id>/...      inputs, GC flat layout
    work/synth/ground_truth/task<N>/<case_id>/  decision + reasoning / outcome
    work/synth/ground_truth/section_variable_mapping.json

The labels are drawn from a crude generative story, not from clinical reality.
They are adequate for testing that metrics compute, plumbing holds, and code
paths execute. They are **not** a validation signal: never report a number
measured on this cohort, and never tune a model against it.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

from chimera.cli.make_fixtures import (
    BASELINE_INPUTS,
    GT_ROOT,
    REPO_ROOT,
    _INTERFACE_DIR,
    _inputs_manifest,
    _neural_representations,
    _structured_prompt,
)
from chimera.contract import spec
from chimera.contract.io import read_json, write_json


def _synth_reasoning(task: int, rng: random.Random) -> dict[str, Any]:
    """A ground-truth-shaped reasoning record: exactly four keys."""
    variables = spec.VARIABLES_BY_TASK[task]
    # Skew toward the middle of the ordinal scale, as real annotations do.
    weights = {
        v: rng.choices(spec.WEIGHT_LEVELS, weights=(3, 4, 3, 1))[0] for v in variables
    }
    n_reveals = rng.randint(0, len(spec.REVEAL_SECTIONS))
    reveals = rng.sample(list(spec.REVEAL_SECTIONS), k=n_reveals)
    return {
        "confidence": rng.choices(spec.CONFIDENCE_LEVELS, weights=(1, 3, 4))[0],
        "variable_weights": weights,
        "reveal_sequence": reveals,
        "free_text": (
            "Synthetic reference rationale for harness testing. Decision driven by "
            "the variables marked important or decisive above."
        ),
    }


def _synth_outcome(rng: random.Random) -> dict[str, Any]:
    """A Task 3 outcome with a realistic mix of events and censoring.

    Roughly 40% events, which is enough for the C-index to have comparable pairs
    and for the IPCW time-dependent AUC to be defined at most horizons.
    """
    event = 1 if rng.random() < 0.4 else 0
    months = round(rng.uniform(3.0, 96.0), 1)
    return {"months_to_recurrence": months, "event": event}


def build(
    out_root: Path,
    *,
    n_per_task: dict[int, int],
    seed: int = 20260806,
) -> dict[int, list[str]]:
    cases_root = out_root / "cases"
    gt_root = out_root / "ground_truth"

    # The evaluator resolves grounding against this file; copy the official one
    # so synthetic runs score against real grounding semantics.
    src_mapping = GT_ROOT / "section_variable_mapping.json"
    gt_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_mapping, gt_root / "section_variable_mapping.json")

    built: dict[int, list[str]] = {}
    for task in (1, 2, 3):
        n = n_per_task.get(task, 0)
        if n <= 0:
            built[task] = []
            continue

        src_dir = BASELINE_INPUTS / _INTERFACE_DIR[task]
        clinical_src = next(src_dir.glob("*clinical-data.json"))
        clinical_payload = read_json(clinical_src)
        manifest = _inputs_manifest(task)

        case_ids: list[str] = []
        for i in range(n):
            case_id = f"SYN-T{task}-{i:03d}"
            case_ids.append(case_id)
            rng = random.Random(f"{seed}:synth:{task}:{case_id}")

            case_dir = cases_root / f"task{task}" / case_id
            write_json(case_dir / "inputs.json", manifest)
            write_json(
                case_dir / "structured-prompt.json", _structured_prompt(task, case_id, rng)
            )
            write_json(case_dir / clinical_src.name, clinical_payload)
            write_json(
                case_dir / "prostate-modality-level-neural-representations.json",
                _neural_representations(task, rng),
            )

            gt_dir = gt_root / f"task{task}" / case_id
            if task == 3:
                write_json(
                    gt_dir / "prostate-time-to-recurrence-or-last-follow-up.json",
                    _synth_outcome(rng),
                )
            else:
                stem = (
                    "prostate-biopsy-decision"
                    if task == 1
                    else "prostate-treatment-decision"
                )
                labels = (
                    spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS
                )
                write_json(gt_dir / f"{stem}.json", rng.choice(list(labels)))
                write_json(gt_dir / f"{stem}-reasoning.json", _synth_reasoning(task, rng))

        built[task] = case_ids
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "work" / "synth",
        help="output root (default: work/synth)",
    )
    parser.add_argument("--n-task1", type=int, default=60)
    parser.add_argument("--n-task2", type=int, default=60)
    parser.add_argument("--n-task3", type=int, default=60)
    args = parser.parse_args()

    built = build(
        args.out,
        n_per_task={1: args.n_task1, 2: args.n_task2, 3: args.n_task3},
    )
    for task, ids in sorted(built.items()):
        print(f"task{task}: {len(ids)} synthetic cases")
    print(f"\ncases        : {args.out / 'cases'}")
    print(f"ground truth : {args.out / 'ground_truth'}")
    print("\nSYNTHETIC LABELS -- for harness testing only, never for reporting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
