"""Conversion to the flat record shape both scorers compare on.

The official evaluator never scores our files directly. Its per-interface
handlers first flatten each job into a single dict -- reasoning keys at the top
level, the decision folded in under a task-specific key -- and the ground-truth
loader flattens the reference files into *the same shape*. Every scorer then
sees identical keys on both sides.

Reproducing that flattening here, rather than inventing our own record, is what
makes parity checkable: the fast scorer and ``evaluate.py`` consume byte-equal
inputs, so any disagreement is a real disagreement about the maths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chimera.contract import spec
from chimera.contract.types import DecisionPrediction, Prediction, RecurrencePrediction

# Ground-truth filenames per task, mirroring ``GT_FILENAMES`` in evaluate.py.
# Note these use the *corrected* "biopsy" spelling -- unlike our output sockets,
# which must use the misspelled "biospy" slug.
_GT_FILENAMES: dict[int, dict[str, str]] = {
    1: {
        "decision": "prostate-biopsy-decision.json",
        "reasoning": "prostate-biopsy-decision-reasoning.json",
    },
    2: {
        "decision": "prostate-treatment-decision.json",
        "reasoning": "prostate-treatment-decision-reasoning.json",
    },
    3: {"outcome": "prostate-time-to-recurrence-or-last-follow-up.json"},
}

# Clinical data now travels with the *ground truth* rather than being read off the
# prediction's inline input, reversing the previous release. It is the context the
# rationale judge is shown, so it has to be on the record even though no
# judge-disabled component reads it -- otherwise our records and the evaluator's
# differ and parity is only apparently intact. Mirrors ``CLINICAL_FILENAMES``.
_GT_CLINICAL_FILENAMES: dict[int, str] = {
    1: "prostate-biopsy-decision-clinical-data.json",
    2: "prostate-treatment-decision-clinical-data.json",
    3: "prostate-time-to-recurrence-or-last-follow-up-clinical-data.json",
}

# The evaluator's own task names, used as the ``task`` field on every row.
TASK_KIND: dict[int, str] = {1: "biopsy", 2: "treatment", 3: "recurrence"}


def record_from_prediction(pred: Prediction, case_id: str) -> dict[str, Any]:
    """Flatten one of our predictions the way ``process_interf*`` would.

    This is deliberately the *serialised* payload rather than the dataclass
    fields, so anything the writers would put on disk is what gets scored.
    """
    if isinstance(pred, RecurrencePrediction):
        outcome = pred.decision_json()
        return {
            "case_id": case_id,
            "months_to_recurrence": outcome.get("months_to_recurrence"),
            "event": outcome.get("event"),
            "free_text": pred.reasoning_json(),
        }

    if not isinstance(pred, DecisionPrediction):  # pragma: no cover - guard
        raise TypeError(f"unsupported prediction type: {type(pred).__name__}")

    reasoning = pred.reasoning_json()
    record: dict[str, Any] = dict(reasoning) if isinstance(reasoning, dict) else {}
    if pred.task == 1:
        record["biopsy_decision"] = pred.decision_json()
    else:
        record["treatment_recommendation"] = {"primary": pred.decision_json()}
    record["case_id"] = case_id
    return record


def gt_record_from_dir(case_dir: Path, task: int) -> dict[str, Any] | None:
    """Load one ground-truth case, or ``None`` if a required file is absent.

    The directory name is authoritative for ``case_id``; the reference files
    carry no case id of their own.
    """
    names = _GT_FILENAMES[task]
    if any(not (case_dir / n).exists() for n in names.values()):
        return None

    if task == 3:
        outcome = json.loads((case_dir / names["outcome"]).read_text())
        record = dict(outcome) if isinstance(outcome, dict) else {}
    else:
        reasoning = json.loads((case_dir / names["reasoning"]).read_text())
        record = dict(reasoning) if isinstance(reasoning, dict) else {}
        decision = json.loads((case_dir / names["decision"]).read_text())
        if task == 1:
            record["biopsy_decision"] = decision
        else:
            record["treatment_recommendation"] = {"primary": decision}

    clinical_path = case_dir / _GT_CLINICAL_FILENAMES[task]
    clinical: Any = {}
    if clinical_path.exists():
        try:
            loaded = json.loads(clinical_path.read_text())
        except (OSError, json.JSONDecodeError):
            loaded = None
        clinical = loaded if isinstance(loaded, dict) else {}
    record["clinical_data"] = clinical

    record["case_id"] = case_dir.name
    return record


def load_ground_truth(root: Path, task: int) -> list[dict[str, Any]]:
    """Load every ground-truth case under ``root``, in sorted case-id order."""
    if not root.is_dir():
        raise FileNotFoundError(f"ground-truth directory not found: {root}")
    records = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        record = gt_record_from_dir(case_dir, task)
        if record is not None:
            records.append(record)
    return records


# --------------------------------------------------------------------------- #
# Reading a run directory back, the way the evaluator does
# --------------------------------------------------------------------------- #

# The evaluator tolerates more than one spelling per output socket, so a run
# directory produced by someone else -- or by a future version of our writer --
# can legitimately use either. Resolve the same set it does.
_ACCEPTED_OUTPUT_SLUGS: dict[int, dict[str, tuple[str, ...]]] = {
    1: {
        "decision": ("prostate-biospy-decision", "prostate-biopsy-decision"),
        "reasoning": (
            "prostate-biospy-decision-reasoning",
            "prostate-biopsy-decision-reasoning",
        ),
    },
    2: {
        "decision": ("prostate-treatment-decision",),
        "reasoning": ("prostate-treatment-decision-reasoning",),
    },
    3: {
        "decision": ("prostate-time-to-recurrence-or-last-follow-up",),
        "reasoning": (
            "prostate-time-to-recurrence-or-last-follow-up-reas",
            "prostate-time-to-recurrence-or-last-follow-up-reasoning",
        ),
    },
}


def _socket_value(values: Any, slugs: tuple[str, ...]) -> Any:
    """Inline ``value`` of the first matching *input* socket, or ``None``."""
    for slug in slugs:
        for sv in values or []:
            if isinstance(sv, dict) and sv.get("socket", {}).get("slug") == slug:
                return sv.get("value")
    return None


def _output_path(run_dir: Path, job: dict, slugs: tuple[str, ...]) -> Path:
    """Locate one output file of ``job``, by relative_path rather than slug.

    Grand Challenge nests job outputs under ``<pk>/output/``; our local runner
    keeps them flat under ``<pk>/``. The evaluator accepts either, preferring
    flat, so this does too.
    """
    for slug in slugs:
        for sv in job.get("outputs") or []:
            if isinstance(sv, dict) and sv.get("socket", {}).get("slug") == slug:
                relative = sv["socket"]["relative_path"]
                flat = run_dir / job["pk"] / relative
                return flat if flat.exists() else run_dir / job["pk"] / "output" / relative
    raise RuntimeError(f"job {job.get('pk')} has no output socket in {slugs}")


def predictions_from_run(run_dir: Path, task: int) -> list[dict[str, Any]]:
    """Flatten a run directory's jobs for ``task`` into prediction records.

    This is the inverse of :func:`chimera.contract.aggregate.write_predictions_dump`
    and a transcription of the evaluator's ``process_interf*`` handlers: same
    interface-key routing, same case-id recovery from the inline structured
    prompt, same file resolution. Jobs belonging to another task, and jobs with
    no case id, are dropped exactly as the evaluator drops them.

    Reading predictions back off disk -- rather than reusing the in-memory
    objects that wrote them -- is what makes a diff against the official
    ``metrics.json`` a test of the *files*, not just of the maths.
    """
    dump = run_dir / "predictions.json"
    if not dump.is_file():
        raise FileNotFoundError(f"no predictions.json under {run_dir}")
    jobs = json.loads(dump.read_text())
    if not isinstance(jobs, list):
        raise ValueError(f"predictions file is not a list of jobs: {dump}")

    slugs = _ACCEPTED_OUTPUT_SLUGS[task]
    clinical_slug = (spec.CLINICAL_SLUG_BY_TASK[task],)
    records: list[dict[str, Any]] = []

    for job in jobs:
        if not isinstance(job, dict):
            continue
        key = tuple(sorted(sv["socket"]["slug"] for sv in job["inputs"]))
        if spec.TASK_BY_INTERFACE_KEY.get(key) != task:
            continue

        prompt = _socket_value(job.get("inputs"), (spec.STRUCTURED_PROMPT_SLUG,))
        case_id = prompt.get("case_id") if isinstance(prompt, dict) else None
        if not case_id:
            continue

        decision = json.loads(_output_path(run_dir, job, slugs["decision"]).read_text())
        reasoning = json.loads(_output_path(run_dir, job, slugs["reasoning"]).read_text())
        clinical = _socket_value(job.get("inputs"), clinical_slug)

        if task == 3:
            outcome = decision if isinstance(decision, dict) else {}
            record = {
                "case_id": str(case_id),
                "months_to_recurrence": outcome.get("months_to_recurrence"),
                "event": outcome.get("event"),
                "free_text": reasoning if isinstance(reasoning, str) else None,
            }
        else:
            record = dict(reasoning) if isinstance(reasoning, dict) else {}
            if task == 1:
                record["biopsy_decision"] = decision
            else:
                record["treatment_recommendation"] = {"primary": decision}
            record["case_id"] = str(case_id)

        # Judge-only, but carried so a judge-on comparison stays possible.
        record["clinical_data"] = clinical if isinstance(clinical, dict) else {}
        records.append(record)

    return records


def pair_run_with_ground_truth(
    run_dir: Path, gt_root: Path, task: int
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Build the ``(gt, pred)`` pairs the evaluator's ``run()`` would score.

    Since 2026-08-16 ``run()`` derives the *phase* from the prediction dump --
    ``phase_case_ids`` -- and then keeps only the ground-truth cases in it. A
    ground-truth case the dump never mentions is therefore no longer scored as
    a missing candidate; it leaves the denominator entirely. That is a real
    change in what a dropped case costs, and copying it is what keeps this
    scorer's ``n_cases`` equal to the evaluator's.

    The residual ``None`` arm is kept because ``run()`` still has one: a job
    whose case id resolves but whose ground truth is absent is refused there
    with ``Missing ... ground truth for phase cases``, and a caller that filters
    the dump the way ``scripts/score.sh`` does can still hand us a target with
    no usable prediction.
    """
    targets = load_ground_truth(gt_root, task)
    by_case = {rec["case_id"]: rec for rec in targets}
    preds = predictions_from_run(run_dir, task)

    # The phase is the set of cases the dump predicts, exactly as run() defines it.
    phase = {pred["case_id"] for pred in preds}

    pairs: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    scored: set[str] = set()
    for pred in preds:
        gt = by_case.get(pred["case_id"])
        if gt is None:
            continue
        # ``_score_job`` backfills an empty inline clinical value from the
        # ground-truth copy. Judge-only, but the records must still match.
        if not pred.get("clinical_data"):
            pred["clinical_data"] = gt.get("clinical_data", {})
        pairs.append((gt, pred))
        scored.add(pred["case_id"])

    pairs.extend(
        (gt, None)
        for gt in targets
        if gt["case_id"] in phase and gt["case_id"] not in scored
    )
    return pairs
