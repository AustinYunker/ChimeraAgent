"""Grand Challenge socket I/O.

On the platform each container invocation handles **one case through one
interface**: input sockets arrive as flat JSON files under ``/input``, described
by ``/input/inputs.json``, and result sockets are expected as flat JSON files
under ``/output``.

Locally we run whole cohorts, so :func:`write_case_outputs` also supports a
per-case directory layout that :mod:`chimera.contract.aggregate` turns back into
the ``predictions.json`` job dump the official evaluator consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chimera.contract import spec
from chimera.contract.types import Prediction, validate


@dataclass(slots=True)
class CaseInputs:
    """One case, as delivered by the platform."""

    task: int
    case_id: str
    structured_prompt: dict[str, Any]
    clinical_data: dict[str, Any]
    neural_representations: dict[str, list[list[float]]]

    def embeddings(self, origin: str) -> list[list[float]]:
        """Frozen embeddings for ``origin``; ``[]`` when the modality is absent.

        Origins are ``"MRI image"``, ``"Biopsy slide"``, ``"Prostatectomy slide"``.
        """
        value = self.neural_representations.get(origin) or []
        return value if isinstance(value, list) else []


def read_json(path: Path) -> Any:
    with path.open() as fh:
        return json.load(fh)


def write_json(path: Path, content: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(content, fh, indent=2)


def socket_paths(input_dir: Path) -> dict[str, Path]:
    """Map input socket slug -> file, via ``inputs.json``.

    Resolving through ``relative_path`` matters: Grand Challenge truncates slugs
    to 50 characters but leaves filenames intact, so the two differ for Task 3.
    """
    entries = read_json(input_dir / "inputs.json")
    return {e["socket"]["slug"]: input_dir / e["socket"]["relative_path"] for e in entries}


def detect_task(slugs: dict[str, Path] | set[str]) -> int:
    """Identify the task from the clinical-data socket slug."""
    present = set(slugs)
    for slug, task in spec.TASK_BY_CLINICAL_SLUG.items():
        if slug in present:
            return task
    raise ValueError(
        f"no known clinical-data socket in {sorted(present)}; "
        f"expected one of {sorted(spec.TASK_BY_CLINICAL_SLUG)}"
    )


def _read_dict(path: Path | None) -> dict[str, Any]:
    """Load a JSON object socket, tolerating absence and malformation.

    A socket declared in ``inputs.json`` whose file is missing or unreadable
    must not take the case down. Four Task 1 cases in the released training data
    have no neural-representations file at all, and on the platform a crashed
    case is not skipped -- it is scored against a sentinel label, costing the
    true class its recall. An empty dict is the same thing every model already
    has to handle for an absent modality.
    """
    if path is None or not path.is_file():
        return {}
    try:
        loaded = read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def read_case(input_dir: Path, *, fallback_case_id: str = "gc-case") -> CaseInputs:
    """Load a single case from a flat Grand Challenge ``/input`` directory."""
    paths = socket_paths(input_dir)
    task = detect_task(paths)

    prompt = _read_dict(paths.get(spec.STRUCTURED_PROMPT_SLUG))
    clinical = _read_dict(paths.get(spec.CLINICAL_SLUG_BY_TASK[task]))
    neural = _read_dict(paths.get(spec.NEURAL_REP_SLUG))

    # The platform anonymises cases, so case_id is often absent on the wire. The
    # evaluator reads it from the structured prompt to find ground truth, so
    # whatever is there wins.
    case_id = str(prompt.get("case_id") or fallback_case_id)

    return CaseInputs(
        task=task,
        case_id=case_id,
        structured_prompt=prompt,
        clinical_data=clinical,
        neural_representations=neural,
    )


def write_case_outputs(output_dir: Path, pred: Prediction) -> dict[str, Path]:
    """Validate then write the two result sockets flat into ``output_dir``.

    Validation happens *before* the first write so a bad prediction cannot leave
    a half-written case behind.
    """
    validate(pred)
    sockets = spec.OUTPUT_SOCKETS[pred.task]

    _, decision_name = sockets["decision"]
    _, reasoning_name = sockets["reasoning"]

    decision_path = output_dir / decision_name
    reasoning_path = output_dir / reasoning_name

    write_json(decision_path, pred.decision_json())
    write_json(reasoning_path, pred.reasoning_json())

    return {"decision": decision_path, "reasoning": reasoning_path}
