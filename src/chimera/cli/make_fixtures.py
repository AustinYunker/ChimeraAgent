"""Build a runnable local cohort from the reference repos.

The real training data is not in hand yet, so C0 and C1 run against fixtures
synthesised to match the eight ground-truth cases shipped in
``refs/challenge/evaluation/ground_truth``. Clinical-data payloads are the real
ones from ``refs/baseline/test/input`` (the baseline's own structured prompts
are ``{"key": "value"}`` placeholders, so those are generated here instead).

The output layout mirrors a Grand Challenge ``/input`` directory per case, so
the same reader serves fixtures and the platform:

    work/fixtures/task<N>/<case_id>/
        inputs.json
        structured-prompt.json
        <clinical-data>.json
        prostate-modality-level-neural-representations.json

Regenerate any time with ``python -m chimera.cli.make_fixtures``. Delete and
replace this whole tree once the real data lands.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from chimera.contract import spec
from chimera.contract.io import write_json

REPO_ROOT = Path(__file__).resolve().parents[3]
GT_ROOT = REPO_ROOT / "refs" / "challenge" / "evaluation" / "ground_truth"
BASELINE_INPUTS = REPO_ROOT / "refs" / "baseline" / "test" / "input"

# Which baseline fixture directory carries each task's clinical-data payload.
_INTERFACE_DIR = {1: "interf0", 2: "interf1", 3: "interf2"}

# Embedding widths, from the challenge's Data Sources page.
_MRI_DIM = 1024
_SLIDE_DIM = 960


def _discover_case_ids(task: int) -> list[str]:
    task_dir = GT_ROOT / f"task{task}"
    if not task_dir.is_dir():
        return []
    return sorted(d.name for d in task_dir.iterdir() if d.is_dir())


def _structured_prompt(task: int, case_id: str, rng: random.Random) -> dict[str, Any]:
    """A plausible structured prompt using the documented variable names."""
    psa = round(rng.uniform(3.0, 22.0), 1)
    vol = round(rng.uniform(25.0, 90.0), 1)
    base: dict[str, Any] = {
        "case_id": case_id,
        "task": task,
        "age": rng.randint(52, 78),
        "psa": psa,
        "dre": rng.choice(["Normal", "Suspicious", "Not performed"]),
    }
    if task == 3:
        # Task 3's prompt is deliberately sparse; most signal is in the reports.
        base["active_treatment_prior_to_surgery"] = rng.choice([None, None, "Neoadjuvant ADT"])
        return base

    prev = round(psa - rng.uniform(0.2, 3.0), 1)
    base.update(
        {
            "psap": max(prev, 0.1),
            "psav": round(rng.uniform(0.1, 3.0), 2),
            "psad": round(psa / vol, 3),
            "vol": vol,
            "months": rng.randint(3, 30),
            "pirads": str(rng.randint(1, 5)),
            "ct": rng.choice(["cT1c", "cT2a", "cT2b", "cT3a"]),
            "cspca": round(rng.uniform(0.02, 0.98), 3),
            "bx": rng.choice(["Positive", "Negative", "None"]),
            "pmhx": rng.sample(
                ["Hypertension", "Type 2 diabetes", "BPH", "Atrial fibrillation"],
                k=rng.randint(0, 2),
            ),
        }
    )
    if task == 2:
        isup = rng.randint(1, 5)
        prim = min(3 + isup // 3, 5)
        base.update(
            {
                "bx_isup": isup,
                "bx_gl_prim": prim,
                "bx_gl_sec": min(prim + rng.randint(0, 1), 5),
                "bx_gl_tert": None,
            }
        )
    return base


def _neural_representations(task: int, rng: random.Random) -> dict[str, list[list[float]]]:
    """Frozen embeddings, with missing modalities as empty lists.

    Roughly a fifth of slide modalities are dropped so the missing-modality path
    is exercised from day one rather than discovered on the test set.
    """

    def vec(dim: int) -> list[float]:
        return [round(rng.gauss(0.0, 1.0), 6) for _ in range(dim)]

    reps: dict[str, list[list[float]]] = {
        "MRI image": [vec(_MRI_DIM)],
        "Biopsy slide": [],
        "Prostatectomy slide": [],
    }
    if task in (2, 3) and rng.random() > 0.2:
        reps["Biopsy slide"] = [vec(_SLIDE_DIM) for _ in range(rng.randint(1, 3))]
    if task == 3 and rng.random() > 0.2:
        reps["Prostatectomy slide"] = [vec(_SLIDE_DIM) for _ in range(rng.randint(1, 3))]
    return reps


def _inputs_manifest(task: int) -> list[dict[str, Any]]:
    clinical_slug = spec.CLINICAL_SLUG_BY_TASK[task]
    # The on-disk filename is the untruncated name; only the slug is clipped.
    clinical_file = (
        "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json"
        if task == 3
        else f"{clinical_slug}.json"
    )
    return [
        {
            "socket": {
                "slug": slug,
                "relative_path": rel,
                "is_json_kind": True,
                "is_file_kind": False,
            },
            "file": None,
            "image": None,
            "value": None,
        }
        for slug, rel in (
            (spec.STRUCTURED_PROMPT_SLUG, "structured-prompt.json"),
            (
                spec.NEURAL_REP_SLUG,
                "prostate-modality-level-neural-representations.json",
            ),
            (clinical_slug, clinical_file),
        )
    ]


def build(out_root: Path, *, seed: int = 20260806) -> dict[int, list[str]]:
    built: dict[int, list[str]] = {}
    for task in (1, 2, 3):
        case_ids = _discover_case_ids(task)
        if not case_ids:
            built[task] = []
            continue

        src_dir = BASELINE_INPUTS / _INTERFACE_DIR[task]
        clinical_src = next(src_dir.glob("*clinical-data.json"))
        clinical_payload = json.loads(clinical_src.read_text())
        manifest = _inputs_manifest(task)

        for case_id in case_ids:
            # Seed per case so regenerating is stable and cases differ.
            rng = random.Random(f"{seed}:{task}:{case_id}")
            case_dir = out_root / f"task{task}" / case_id
            write_json(case_dir / "inputs.json", manifest)
            write_json(
                case_dir / "structured-prompt.json",
                _structured_prompt(task, case_id, rng),
            )
            write_json(case_dir / clinical_src.name, clinical_payload)
            write_json(
                case_dir / "prostate-modality-level-neural-representations.json",
                _neural_representations(task, rng),
            )
        built[task] = case_ids
    return built


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "work" / "fixtures",
        help="fixture root to write (default: work/fixtures)",
    )
    args = parser.parse_args()

    if not GT_ROOT.is_dir():
        raise SystemExit(
            f"ground truth not found at {GT_ROOT}\n"
            "clone the reference repos first: see README.md"
        )

    built = build(args.out)
    total = sum(len(v) for v in built.values())
    for task, ids in sorted(built.items()):
        print(f"task{task}: {len(ids)} cases -> {', '.join(ids) if ids else '(none)'}")
    print(f"\n{total} fixture cases written to {args.out}")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
