"""The submission entrypoint and the prior it ships.

These guard the C1b container. The image cannot be built or run on the
development host -- there is no container runtime -- so everything checkable
without Docker is checked here, and the rest is left to the smoke-test job in
``.github/workflows/build-image.yml``.

The invariant with the longest reach is :func:`test_entrypoint_imports_only_the_stdlib`.
The Dockerfile installs the package with ``--no-deps`` and builds from
``python:3.11-slim``; the moment the entrypoint acquires a third-party import,
that image stops working and the failure appears in CI at build time rather than
here. This test moves it to the front.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chimera.cli.check_outputs import check_case
from chimera.contract import spec
from chimera.contract.io import CaseInputs
from chimera.contract.types import validate
from chimera.predictors.prior import PriorPredictor, load_params

REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "inference.py"
FIXTURE_ROOT = REPO_ROOT / "work" / "fixtures"

sys.path.insert(0, str(REPO_ROOT))


def _fixture_case(task: int) -> Path:
    candidates = sorted(FIXTURE_ROOT.glob(f"task{task}/*/inputs.json"))
    if not candidates:
        pytest.skip(f"no task{task} fixtures; run `python -m chimera.cli.make_fixtures`")
    return candidates[0].parent


def _run_entrypoint(input_dir: Path, output_dir: Path) -> subprocess.CompletedProcess:
    """Invoke the entrypoint the way the container does: as a script."""
    return subprocess.run(
        [sys.executable, str(ENTRYPOINT)],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "CHIMERA_INPUT": str(input_dir),
            "CHIMERA_OUTPUT": str(output_dir),
            "CHIMERA_MODEL": str(output_dir / "_nonexistent_model_mount"),
        },
        capture_output=True,
        text=True,
    )


# --------------------------------------------------------------------------- #
# The invariant that keeps the image small
# --------------------------------------------------------------------------- #

def test_entrypoint_imports_only_the_stdlib():
    """No third-party import may reach the entrypoint.

    ``Dockerfile`` installs with ``--no-deps`` onto ``python:3.11-slim``, so
    numpy, scikit-learn or pydantic appearing in this closure would not be a
    slow image -- it would be an ``ImportError`` on the platform.
    """
    probe = (
        "import sys, json\n"
        "before = set(sys.modules)\n"
        "import inference\n"
        "new = {m.split('.')[0] for m in set(sys.modules) - before}\n"
        "third = sorted(\n"
        "    m for m in new\n"
        "    if m not in sys.stdlib_module_names\n"
        "    and not m.startswith('_')\n"
        "    and m not in ('inference', 'chimera')\n"
        ")\n"
        "print(json.dumps(third))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    third_party = json.loads(proc.stdout.strip().splitlines()[-1])
    assert third_party == [], (
        f"entrypoint pulled in third-party modules {third_party}; "
        "the Dockerfile installs with --no-deps and they will not be present"
    )


# --------------------------------------------------------------------------- #
# End to end, without a container
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("task", [1, 2, 3])
def test_entrypoint_writes_sockets_the_evaluator_would_score(tmp_path, task):
    case_dir = _fixture_case(task)
    proc = _run_entrypoint(case_dir, tmp_path)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"

    decision_name = spec.OUTPUT_SOCKETS[task]["decision"][1]
    reasoning_name = spec.OUTPUT_SOCKETS[task]["reasoning"][1]
    assert (tmp_path / decision_name).is_file()
    assert (tmp_path / reasoning_name).is_file()

    assert check_case(tmp_path, task) == []


def test_entrypoint_fails_loudly_on_an_unrecognisable_interface(tmp_path):
    """No interface means no way to know which sockets to write."""
    (tmp_path / "inputs.json").write_text(json.dumps([
        {"socket": {"slug": "something-else", "relative_path": "x.json"}}
    ]))
    out = tmp_path / "out"
    out.mkdir()
    proc = _run_entrypoint(tmp_path, out)
    assert proc.returncode == 1
    assert list(out.iterdir()) == []


def test_entrypoint_survives_missing_payload_files(tmp_path):
    """A socket declared in inputs.json whose file is absent must not crash.

    Four Task 1 cases in the released data have no neural-representations file
    at all, and a crashed case is scored against a sentinel label rather than
    skipped -- so degrading is strictly better than raising.
    """
    slug = spec.CLINICAL_SLUG_BY_TASK[1]
    (tmp_path / "inputs.json").write_text(json.dumps([
        {"socket": {"slug": spec.STRUCTURED_PROMPT_SLUG,
                    "relative_path": "structured-prompt.json"}},
        {"socket": {"slug": spec.NEURAL_REP_SLUG,
                    "relative_path": "neural.json"}},
        {"socket": {"slug": slug, "relative_path": f"{slug}.json"}},
    ]))
    # Deliberately write none of the three payload files.
    out = tmp_path / "out"
    out.mkdir()

    proc = _run_entrypoint(tmp_path, out)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert check_case(out, 1) == []


def test_fallback_prediction_is_valid_for_every_task():
    import inference

    for task in (1, 2, 3):
        validate(inference.fallback_prediction(task))


# --------------------------------------------------------------------------- #
# The fitted prior
# --------------------------------------------------------------------------- #

def _case(task: int, clinical: dict) -> CaseInputs:
    return CaseInputs(
        task=task,
        case_id=f"T{task}-x",
        structured_prompt={"psa": 7.4, "age": 64, "pirads": "4"},
        clinical_data=clinical,
        neural_representations={},
    )


def test_prior_params_match_the_contract_vocabularies():
    """A stale parameter file must not be able to produce an invalid submission."""
    params = load_params()
    for task in (1, 2):
        entry = params[f"task{task}"]
        assert entry["decision"] in (
            spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS
        )
        assert entry["confidence"] in spec.CONFIDENCE_LEVELS
        assert set(entry["variable_weights"]) == set(spec.VARIABLES_BY_TASK[task])
        assert set(entry["variable_weights"].values()) <= set(spec.WEIGHT_LEVELS)
        assert set(entry["reveal_sequence"]) <= set(spec.REVEAL_SECTIONS)
    assert params["task3"]["event"] in (0, 1)
    assert float(params["task3"]["months_to_recurrence"]) >= 0


@pytest.mark.parametrize("task", [1, 2, 3])
def test_prior_output_passes_contract_validation(task):
    sections = {s: "content" for s in spec.REVEAL_SECTIONS}
    validate(PriorPredictor().predict(_case(task, sections)))


def test_prior_declares_only_sections_it_actually_read():
    """The reveal-honesty rule, as a test.

    ``docs/plan.md`` makes this a design commitment rather than an optimisation:
    the declared reveal_sequence must be exactly the evidence retrieved. A case
    missing a section the policy asks for must not have it declared anyway.
    """
    policy = load_params()["task1"]["reveal_sequence"]
    assert policy, "task 1's fitted policy should request at least one section"

    with_evidence = PriorPredictor().predict(_case(1, {s: "content" for s in policy}))
    assert list(with_evidence.reasoning.reveal_sequence) == list(policy)

    without = PriorPredictor().predict(_case(1, {}))
    assert without.reasoning.reveal_sequence == []

    # Present but empty is not evidence either.
    blank = PriorPredictor().predict(_case(1, {s: "   " for s in policy}))
    assert blank.reasoning.reveal_sequence == []


def test_prior_never_declares_a_section_outside_the_vocabulary():
    """Task 3's clinical data carries `surgical_pathology_report`, which is not
    a reveal name; nothing outside the six-name vocabulary may be declared."""
    predictor = PriorPredictor()
    case = _case(1, {"surgical_pathology_report": "x", "radiology_report": "y"})
    declared = predictor.predict(case).reasoning.reveal_sequence
    assert set(declared) <= set(spec.REVEAL_SECTIONS)


def test_prior_tolerates_a_corrupt_parameter_file():
    """Garbage in the parameters degrades to a valid prediction, never a crash."""
    predictor = PriorPredictor(params={"task1": {"decision": "definitely",
                                                 "confidence": "very",
                                                 "variable_weights": {"nonsense": "huge"},
                                                 "reveal_sequence": "not-a-list"}})
    validate(predictor.predict(_case(1, {"radiology_report": "x"})))


def test_prior_free_text_is_non_empty_and_case_specific():
    predictor = PriorPredictor()
    text = predictor.predict(_case(1, {"radiology_report": "x"})).reasoning.free_text
    assert text.strip()
    # It should quote something real from the case rather than being boilerplate.
    assert "7.4" in text and "PI-RADS 4" in text
    # ...but not `age`, which the judge cannot corroborate: its evidence context is
    # the clinical-data socket, and the reports state the age in 22% / 7% of Task 1
    # and Task 2 cases. See `chimera.predictors.rationale.CITABLE`.
    assert "64" not in text


# --------------------------------------------------------------------------- #
# The smoke-test checker itself
# --------------------------------------------------------------------------- #

def test_check_case_rejects_an_out_of_vocabulary_decision(tmp_path):
    """A negative control: the CI gate must actually catch a bad submission."""
    from chimera.contract.io import write_case_outputs

    write_case_outputs(tmp_path, PriorPredictor().predict(_case(1, {"radiology_report": "x"})))
    assert check_case(tmp_path, 1) == []

    (tmp_path / spec.OUTPUT_SOCKETS[1]["decision"][1]).write_text('"maybe"')
    problems = check_case(tmp_path, 1)
    assert problems and "invalid biopsy_decision" in problems[0]


def test_check_case_reports_a_missing_socket(tmp_path):
    problems = check_case(tmp_path, 1)
    assert problems and "missing result socket" in problems[0]


# --------------------------------------------------------------------------- #
# The C2 guideline predictor -- what the container now actually ships
# --------------------------------------------------------------------------- #

def test_guideline_params_are_well_formed():
    from chimera.models.guidelines import LEAVES_BY_TASK
    from chimera.predictors.guideline import GuidelinePredictor

    params = GuidelinePredictor().params
    for task in (1, 2):
        entry = params[f"task{task}"]
        allowed = spec.BIOPSY_DECISIONS if task == 1 else spec.TREATMENT_DECISIONS
        labels = entry["leaf_labels"]
        # Every guideline leaf must have a label; an unmapped leaf would fall back
        # to an arbitrary one at inference.
        assert set(labels) == set(LEAVES_BY_TASK[task])
        assert set(labels.values()) <= set(allowed)
        for reasoning in entry["reasoning"].values():
            assert reasoning["confidence"] in spec.CONFIDENCE_LEVELS
            assert set(reasoning["reveal_sequence"]) <= set(spec.REVEAL_SECTIONS)


@pytest.mark.parametrize("task", [1, 2, 3])
def test_guideline_output_passes_contract_validation(task):
    from chimera.predictors.guideline import GuidelinePredictor

    sections = {s: "content" for s in spec.REVEAL_SECTIONS}
    validate(GuidelinePredictor().predict(_case(task, sections)))


def test_guideline_routes_by_stratum_not_by_a_constant():
    """The whole point of C2: two cases in different strata get different answers."""
    from chimera.predictors.guideline import GuidelinePredictor

    predictor = GuidelinePredictor()
    low = CaseInputs(
        task=2, case_id="low",
        structured_prompt={"bx": "Negative", "bx_isup": 1, "psa": 5.0, "ct": "cT1c"},
        clinical_data={}, neural_representations={},
    )
    high = CaseInputs(
        task=2, case_id="high",
        structured_prompt={"bx": "Positive", "bx_isup": 5, "psa": 30.0, "ct": "cT3a"},
        clinical_data={}, neural_representations={},
    )
    assert predictor.predict(low).decision == "continued_surveillance"
    assert predictor.predict(high).decision == "active_treatment"


def test_guideline_task3_orders_by_risk():
    """Only the ordering of predicted months reaches the C-index."""
    from chimera.predictors.guideline import GuidelinePredictor

    benign = CaseInputs(
        task=3, case_id="b", structured_prompt={"psa": 4.0},
        clinical_data={"surgical_pathology_report":
                       "Gleason 3+3 (ISUP grade group 1). There was no extraprostatic "
                       "extension; surgical margins were negative; the seminal vesicles "
                       "were not invaded; there was no lymph node metastasis."},
        neural_representations={},
    )
    severe = CaseInputs(
        task=3, case_id="s", structured_prompt={"psa": 40.0},
        clinical_data={"surgical_pathology_report": SURGICAL_FOR_TEST},
        neural_representations={},
    )
    b = GuidelinePredictor().predict(benign)
    s = GuidelinePredictor().predict(severe)
    # Shorter predicted time = higher risk.
    assert s.months_to_recurrence < b.months_to_recurrence
    validate(b)
    validate(s)


SURGICAL_FOR_TEST = (
    "The prostatectomy specimen showed Gleason 4+5 (ISUP grade group 5), pathological "
    "stage pT3b. Extraprostatic extension was present; surgical margins were positive; "
    "the seminal vesicles were invaded; lymphovascular invasion was present; lymph node "
    "metastasis was present."
)


def test_guideline_degrades_to_a_valid_prediction_with_no_features():
    """A case with nothing readable must still produce a scoreable submission."""
    from chimera.predictors.guideline import GuidelinePredictor

    predictor = GuidelinePredictor()
    for task in (1, 2, 3):
        validate(predictor.predict(_case(task, {})))
