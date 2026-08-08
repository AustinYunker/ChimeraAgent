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


def pytest_sessionfinish(session, exitstatus):
    """Under ``CHIMERA_REQUIRE_REFS=1``, treat any skip as a failure."""
    if not REQUIRE_EVERYTHING or exitstatus != 0:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", []) if reporter else []
    if skipped:
        print(f"\nCHIMERA_REQUIRE_REFS=1 but {len(skipped)} test(s) skipped:")
        for report in skipped:
            print(f"  {report.nodeid}")
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
    spec_.loader.exec_module(module)
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
