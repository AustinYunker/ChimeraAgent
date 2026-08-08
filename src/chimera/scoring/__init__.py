"""In-process scoring for cross-validation.

Per the challenge rules, only the official ``evaluation/evaluate.py`` may be
used to *report* performance. Everything in this package exists so that model
selection does not have to pay for a subprocess, a ``predictions.json`` round
trip and an Ollama judge on every fold. It is verified against the official
scorer by ``tests/test_scorer_parity.py`` and never substituted for it.
"""

from chimera.scoring.fast import score_case, score_cohort
from chimera.scoring.records import (
    gt_record_from_dir,
    load_ground_truth,
    pair_run_with_ground_truth,
    predictions_from_run,
    record_from_prediction,
)

__all__ = [
    "score_case",
    "score_cohort",
    "load_ground_truth",
    "gt_record_from_dir",
    "pair_run_with_ground_truth",
    "predictions_from_run",
    "record_from_prediction",
]
