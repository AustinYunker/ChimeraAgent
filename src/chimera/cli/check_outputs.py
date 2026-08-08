"""Validate container-written result sockets against the evaluator's schema gate.

The submission image writes two flat JSON files per case. Checking that they
merely *exist* is a weak test: the failure mode that actually costs points is a
file that exists, parses, and is then rejected or mis-scored by the evaluator --
a decision token outside the vocabulary, a reasoning object nested one level too
deep, a Task 3 outcome carrying a string where a number belongs.

So this reassembles each case's outputs into the flat record the evaluator
scores, exactly as :func:`chimera.scoring.records.predictions_from_run` does for
a full run directory, and runs the official gate
:func:`chimera.scoring.fast.validate_record` over it. Passing here means the
evaluator would score the case rather than zero it.

Used by ``scripts/smoke_test_image.sh`` in CI, which is the only place the image
can be executed at all -- the build host has no container runtime.

Usage::

    python -m chimera.cli.check_outputs --cases work/fixtures --outputs /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chimera.contract import spec
from chimera.scoring.fast import validate_record
from chimera.scoring.records import TASK_KIND


def record_from_output_dir(out_dir: Path, task: int) -> dict[str, Any]:
    """Rebuild the evaluator's flat record from one case's two socket files."""
    _, decision_name = spec.OUTPUT_SOCKETS[task]["decision"]
    _, reasoning_name = spec.OUTPUT_SOCKETS[task]["reasoning"]

    decision_path = out_dir / decision_name
    reasoning_path = out_dir / reasoning_name
    for path in (decision_path, reasoning_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing result socket: {path}")

    decision = json.loads(decision_path.read_text())
    reasoning = json.loads(reasoning_path.read_text())

    if task == 3:
        outcome = decision if isinstance(decision, dict) else {}
        return {
            "months_to_recurrence": outcome.get("months_to_recurrence"),
            "event": outcome.get("event"),
            "free_text": reasoning if isinstance(reasoning, str) else None,
        }

    record = dict(reasoning) if isinstance(reasoning, dict) else {}
    if task == 1:
        record["biopsy_decision"] = decision
    else:
        record["treatment_recommendation"] = {"primary": decision}
    return record


def check_case(out_dir: Path, task: int) -> list[str]:
    """Every problem with one case's outputs; empty means it would be scored."""
    try:
        record = record_from_output_dir(out_dir, task)
    except FileNotFoundError as exc:
        return [str(exc)]
    except json.JSONDecodeError as exc:
        return [f"{out_dir}: result socket is not valid JSON: {exc}"]

    problems: list[str] = []
    ok, reason = validate_record(record, TASK_KIND[task])
    if not ok:
        problems.append(f"{out_dir}: evaluator would reject this case: {reason}")

    if task == 3:
        if not isinstance(record.get("free_text"), str) or not record["free_text"].strip():
            problems.append(f"{out_dir}: task 3 reasoning must be a non-empty bare string")
        return problems

    # The gate above does not check these, but the evaluator scores them, and a
    # silently wrong shape here costs points rather than raising.
    expected_keys = {"confidence", "variable_weights", "reveal_sequence", "free_text"}
    reasoning_keys = set(record) - {"biopsy_decision", "treatment_recommendation", "case_id"}
    if reasoning_keys != expected_keys:
        problems.append(
            f"{out_dir}: reasoning keys {sorted(reasoning_keys)} != {sorted(expected_keys)}"
        )
    if record.get("confidence") not in spec.CONFIDENCE_LEVELS:
        problems.append(f"{out_dir}: confidence {record.get('confidence')!r} out of vocabulary")
    for section in record.get("reveal_sequence") or []:
        if section not in spec.REVEAL_SECTIONS:
            problems.append(f"{out_dir}: reveal {section!r} out of vocabulary")
    for var, weight in (record.get("variable_weights") or {}).items():
        if weight not in spec.WEIGHT_LEVELS:
            problems.append(f"{out_dir}: weight {weight!r} for {var!r} out of vocabulary")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True,
                        help="cohort root the container was run over (task<N>/<case_id>/)")
    parser.add_argument("--outputs", type=Path, required=True,
                        help="output root the container wrote (task<N>/<case_id>/)")
    parser.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    args = parser.parse_args()

    problems: list[str] = []
    checked = 0
    for task in args.tasks:
        task_dir = args.cases / f"task{task}"
        if not task_dir.is_dir():
            continue
        for case_dir in sorted(d for d in task_dir.iterdir() if d.is_dir()):
            out_dir = args.outputs / f"task{task}" / case_dir.name
            if not out_dir.is_dir():
                problems.append(f"{out_dir}: container wrote no output for this case")
                continue
            problems.extend(check_case(out_dir, task))
            checked += 1

    if not checked:
        print(f"!! no cases found under {args.cases}")
        return 1

    if problems:
        print(f"!! {len(problems)} problem(s) across {checked} case(s):")
        for problem in problems:
            print(f"   {problem}")
        return 1

    print(f"all {checked} case(s) produce sockets the evaluator would score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
