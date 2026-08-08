"""Turn the released training data into a runnable cohort plus a scorable ground truth.

``data/train_release`` is not in Grand Challenge layout: each case directory
holds its input sockets *and* its ground-truth files side by side, with no
``inputs.json`` manifest. :func:`chimera.contract.io.read_case` resolves sockets
through that manifest, so the release cannot be run as-is.

This splits the release into the two trees the rest of the harness expects,
without touching ``data/`` itself::

    work/train/cases/task<N>/<case_id>/         inputs only, GC flat layout
    work/train/ground_truth/task<N>/<case_id>/  labels only
    work/train/ground_truth/section_variable_mapping.json

The manifest is built by :func:`chimera.cli.make_fixtures._inputs_manifest`, the
same helper the synthetic cohorts use, so all three cohorts present an identical
interface to ``run_local``.

Only labeled cases get a ground-truth directory -- 91 of 195 for Task 1, 72 of
153 for Task 2, all 75 for Task 3. Unlabeled cases are still emitted as runnable
inputs, because the evaluator simply drops predictions with no matching target
and they are useful for exercising the inference path at full cohort size.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from chimera.cli.make_fixtures import GT_ROOT as REFERENCE_GT_ROOT
from chimera.cli.make_fixtures import REPO_ROOT, _inputs_manifest
from chimera.contract.io import write_json

DEFAULT_DATA = REPO_ROOT / "data" / "train_release"
DEFAULT_OUT = REPO_ROOT / "work" / "train"

#: Socket files that make up a case's *input*, per task. Everything else in a
#: release case directory is a label and must not leak into the cases tree.
#:
#: Not every case carries all three. Four Task 1 cases -- two of them labeled --
#: have no ``prostate-modality-level-neural-representations.json`` at all. Those
#: are still perfectly good cases, so the manifest is built per case from the
#: files that exist rather than from a fixed per-task list; dropping them would
#: throw away labeled training data and hide a missing-socket path we have to
#: survive anyway.
_INPUT_FILES: dict[int, tuple[str, ...]] = {
    1: (
        "structured-prompt.json",
        "prostate-modality-level-neural-representations.json",
        "prostate-biopsy-decision-clinical-data.json",
    ),
    2: (
        "structured-prompt.json",
        "prostate-modality-level-neural-representations.json",
        "prostate-treatment-decision-clinical-data.json",
    ),
    3: (
        "structured-prompt.json",
        "prostate-modality-level-neural-representations.json",
        "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
    ),
}

#: The one socket a case cannot do without: the evaluator and our own
#: ``detect_task`` both identify the interface from the clinical-data slug.
_REQUIRED_SUFFIX = "-clinical-data.json"

#: Label files per task, mirroring ``GT_FILENAMES`` in the official evaluator.
#: A case is "labeled" only when every one of its task's files is present.
_LABEL_FILES: dict[int, tuple[str, ...]] = {
    1: ("prostate-biopsy-decision.json", "prostate-biopsy-decision-reasoning.json"),
    2: ("prostate-treatment-decision.json", "prostate-treatment-decision-reasoning.json"),
    3: ("prostate-time-to-recurrence-or-last-follow-up.json",),
}


def build(data_root: Path, out_root: Path) -> dict[int, tuple[int, int]]:
    """Split ``data_root`` into cases and ground truth under ``out_root``.

    Returns ``{task: (n_cases, n_labeled)}``.
    """
    cases_root = out_root / "cases"
    gt_root = out_root / "ground_truth"
    gt_root.mkdir(parents=True, exist_ok=True)

    # The evaluator resolves grounding against this file and it is not part of
    # the release, so take the official one -- exactly as the synthetic cohort does.
    mapping = REFERENCE_GT_ROOT / "section_variable_mapping.json"
    if not mapping.is_file():
        raise SystemExit(
            f"section mapping not found at {mapping}\n"
            "clone the reference repos first: see README.md"
        )
    shutil.copyfile(mapping, gt_root / "section_variable_mapping.json")

    summary: dict[int, tuple[int, int]] = {}
    for task in (1, 2, 3):
        task_dir = data_root / f"task{task}"
        if not task_dir.is_dir():
            summary[task] = (0, 0)
            continue

        full_manifest = _inputs_manifest(task)
        n_cases = n_labeled = 0

        for case_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
            present = [case_dir / name for name in _INPUT_FILES[task] if (case_dir / name).is_file()]
            if not any(p.name.endswith(_REQUIRED_SUFFIX) for p in present):
                # No clinical-data socket means the interface is unidentifiable.
                continue

            dest = cases_root / f"task{task}" / case_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for src in present:
                shutil.copyfile(src, dest / src.name)

            # Describe only what we actually wrote, the way Grand Challenge's
            # own inputs.json describes only what it actually mounts.
            names = {p.name for p in present}
            write_json(
                dest / "inputs.json",
                [e for e in full_manifest if e["socket"]["relative_path"] in names],
            )
            n_cases += 1

            labels = [case_dir / name for name in _LABEL_FILES[task]]
            if all(p.is_file() for p in labels):
                gt_dir = gt_root / f"task{task}" / case_dir.name
                gt_dir.mkdir(parents=True, exist_ok=True)
                for src in labels:
                    shutil.copyfile(src, gt_dir / src.name)
                n_labeled += 1

        summary[task] = (n_cases, n_labeled)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help=f"release root containing task<N>/ (default: {DEFAULT_DATA})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output root for cases/ and ground_truth/ (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    if not args.data.is_dir():
        raise SystemExit(f"no release data at {args.data}")

    summary = build(args.data, args.out)
    for task, (n_cases, n_labeled) in sorted(summary.items()):
        print(f"task{task}: {n_cases} cases, {n_labeled} labeled")
    print(f"\ncases        : {args.out / 'cases'}")
    print(f"ground truth : {args.out / 'ground_truth'}")
    return 0 if any(n for n, _ in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
