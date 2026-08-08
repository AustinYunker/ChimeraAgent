"""The CHIMERA-agent submission contract: socket I/O, typed predictions, validation."""

from chimera.contract import spec
from chimera.contract.aggregate import build_job, job_pk, write_predictions_dump
from chimera.contract.io import (
    CaseInputs,
    detect_task,
    read_case,
    read_json,
    socket_paths,
    write_case_outputs,
    write_json,
)
from chimera.contract.types import (
    ContractError,
    DecisionPrediction,
    Prediction,
    Reasoning,
    RecurrencePrediction,
    validate,
)

__all__ = [
    "CaseInputs",
    "ContractError",
    "DecisionPrediction",
    "Prediction",
    "Reasoning",
    "RecurrencePrediction",
    "build_job",
    "detect_task",
    "job_pk",
    "read_case",
    "read_json",
    "socket_paths",
    "spec",
    "validate",
    "write_case_outputs",
    "write_json",
    "write_predictions_dump",
]
