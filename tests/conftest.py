"""Shared fixtures.

``official_evaluator`` imports the challenge's ``evaluate.py`` as a module so
tests can assert our constants against *its* constants rather than against our
reading of the docs. If the organizers change a slug or a weight, the tests that
depend on this fixture fail immediately instead of on submission day.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = REPO_ROOT / "refs" / "challenge" / "evaluation" / "evaluate.py"
FIXTURE_ROOT = REPO_ROOT / "work" / "fixtures"

#: Locally a missing ``refs/`` or cohort is a reason to skip -- not everyone has
#: cloned them, and skipping keeps the fast tests usable. In CI it is a reason to
#: fail: the parity tests are the whole point of running the suite there, and a
#: green run that silently skipped them is worse than a red one.
REQUIRE_EVERYTHING = os.environ.get("CHIMERA_REQUIRE_REFS") == "1"

#: The one legitimate reason to skip in CI. ``data/train_release`` is the released
#: challenge data: not redistributable, gitignored, and so never present on a
#: runner. Tests marked with this are exempt from the rule above; everything else
#: skipping in CI means refs/ or a generated cohort is missing, which is a
#: misconfiguration and should fail loudly.
RELEASE_DATA_MARKER = "requires_release_data"


def pytest_sessionfinish(session, exitstatus):
    """Under ``CHIMERA_REQUIRE_REFS=1``, treat an unexpected skip as a failure."""
    if not REQUIRE_EVERYTHING or exitstatus != 0:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    unexpected = [r for r in skipped if RELEASE_DATA_MARKER not in r.keywords]
    if unexpected:
        print(f"\nCHIMERA_REQUIRE_REFS=1 but {len(unexpected)} test(s) skipped:")
        for report in unexpected:
            print(f"  {report.nodeid}")
        print(
            f"Only tests marked @pytest.mark.{RELEASE_DATA_MARKER} may skip in CI; "
            "anything else means refs/ or a cohort is missing."
        )
        session.exitstatus = 1


@pytest.fixture(scope="session")
def official_evaluator():
    """The challenge's ``evaluate.py``, imported as a module."""
    if not EVALUATOR_PATH.is_file():
        pytest.skip(f"reference evaluator not cloned at {EVALUATOR_PATH}")

    # The module reads configuration from the environment at import time. Pin
    # the judge off so importing never tries to reach Ollama.
    os.environ.setdefault("USE_RATIONALE_JUDGE", "0")

    spec_ = importlib.util.spec_from_file_location("_official_evaluate", EVALUATOR_PATH)
    assert spec_ and spec_.loader
    module = importlib.util.module_from_spec(spec_)
    sys.modules["_official_evaluate"] = module
    try:
        spec_.loader.exec_module(module)
    except ImportError as exc:
        # Importing the evaluator pulls in *its* dependencies, not ours. When one
        # is absent this otherwise surfaces as dozens of identical tracebacks
        # across every test that touches the fixture, which buries the one fact
        # that matters. `requests` is the usual culprit -- it is imported at the
        # evaluator's module top level even with the judge disabled.
        raise RuntimeError(
            f"the official evaluator needs a package that is not installed: {exc.name!r}.\n"
            f"Its dependencies are listed in "
            f"refs/challenge/evaluation/docker/requirements.txt; the ones needed with "
            f"USE_RATIONALE_JUDGE=0 belong in this project's `dev` extra.\n"
            f"Try: pip install -e '.[dev]'"
        ) from exc
    return module


@pytest.fixture
def tmp_fixture_case() -> Path:
    """One generated fixture case directory, in Grand Challenge flat layout."""
    candidates = sorted(FIXTURE_ROOT.glob("task*/*/inputs.json"))
    if not candidates:
        pytest.skip(
            f"no fixtures under {FIXTURE_ROOT}; "
            "run `python -m chimera.cli.make_fixtures` first"
        )
    return candidates[0].parent
