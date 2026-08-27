"""C4a, over the wire: the masked documents are reached by tool call, or not at all.

The refactor that introduced this module was behaviour-preserving by
construction -- ``diff -r`` between a pre-change and a post-change run directory
is empty -- so nothing here re-checks the *predictions*. What it checks is the
property the refactor was for: that a section arrives because a tool returned
it, and that the ledger of what was returned is the thing a submission declares.

Everything runs against a real subprocess speaking real JSON-RPC over real
pipes. :class:`~chimera.mcp.client.DirectStore` appears only as the oracle in
:func:`test_the_wire_agrees_with_an_in_process_read` -- it is the fallback path,
so the one thing worth asserting about it is that it answers identically.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from chimera.contract import spec
from chimera.contract.io import read_case
from chimera.mcp import protocol
from chimera.mcp.client import DirectStore, McpSession
from chimera.mcp.tools import TOOL_BY_NAME, section_is_present, tools_for_task
from chimera.predictors import GuidelinePredictor

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "work" / "fixtures"


def _case_dirs() -> list[Path]:
    return sorted(p.parent for p in FIXTURES.glob("task*/*/inputs.json"))


CASE_DIRS = _case_dirs()


@pytest.fixture(scope="module")
def session():
    """One cohort server for the module, as the offline harness runs it."""
    if not CASE_DIRS:
        pytest.skip(
            f"no fixtures under {FIXTURES}; run `python -m chimera.cli.make_fixtures` first"
        )
    with McpSession.for_cohort(FIXTURES) as running:
        yield running


@pytest.fixture(scope="module")
def cases() -> list:
    """Every fixture case, read once."""
    if not CASE_DIRS:
        pytest.skip(f"no fixtures under {FIXTURES}")
    return [read_case(d) for d in CASE_DIRS]


def _ids(cases) -> list[str]:
    return [f"task{c.task}/{c.case_id}" for c in cases]


# -- the handshake and the surface ------------------------------------------ #


def test_the_server_announces_itself(session):
    assert session.server_info.get("name") == "chimera-clinical"


def test_a_cohort_server_publishes_the_union_of_its_tasks(session):
    """Fixtures span all three tasks, so every tool should be listed once."""
    published = session.list_tools()
    names = [t["name"] for t in published]
    assert len(names) == len(set(names)), f"a tool was published twice: {names}"

    expected = {tool.name for task in (1, 2, 3) for tool in tools_for_task(task)}
    assert set(names) == expected

    for tool in published:
        schema = tool["inputSchema"]
        assert schema["required"] == ["case_id"]
        assert tool["description"].strip(), f"{tool['name']} has no description"


def test_ping_is_answered(session):
    assert session._request("ping") == {}


# -- retrieval --------------------------------------------------------------- #


def test_the_wire_agrees_with_an_in_process_read(session, cases):
    """Every section of every fixture, fetched both ways, must match exactly.

    This is the fallback's justification: ``DirectStore`` is what a container
    drops to when the subprocess is lost, and it is only an acceptable
    degradation if it answers the same questions the same way.
    """
    for case in cases:
        over_wire = session.store_for(case)
        in_process = DirectStore(case)
        for section in spec.REVEAL_SECTIONS + ("surgical_pathology_report",):
            assert over_wire.section(section) == in_process.section(section), (
                f"task{case.task}/{case.case_id}: {section} differs between transports"
            )
        assert over_wire.retrieved == in_process.retrieved


def _case_without(tmp_path: Path, case_dir: Path, section: str) -> Path:
    """A copy of ``case_dir`` with one clinical section deleted.

    Every generated fixture carries every section of its task, so the
    biopsy-naive patient -- the case the "no data" path exists for -- has to be
    built. Deleting the key is exactly what the released cohort looks like for a
    patient who never had that document.
    """
    dest = tmp_path / case_dir.name
    shutil.copytree(case_dir, dest)
    task = read_case(case_dir).task
    clinical_path = dest / f"{spec.CLINICAL_SLUG_BY_TASK[task]}.json"
    clinical = json.loads(clinical_path.read_text())
    assert section in clinical, f"{case_dir.name} never had {section} to remove"
    del clinical[section]
    clinical_path.write_text(json.dumps(clinical))
    return dest


def test_a_section_the_case_lacks_is_absent_not_an_error(tmp_path):
    """A biopsy-naive patient has no pathology report; that is data, not failure.

    The tool is still published and still callable -- the task masks that
    document, this patient just has none -- so the server answers with a payload
    that omits the section, and the store reports ``None`` without recording a
    retrieval.
    """
    source = next((d for d in CASE_DIRS if d.parent.name == "task2"), None)
    if source is None:
        pytest.skip("no task2 fixtures")
    case_dir = _case_without(tmp_path, source, "pathology_report")
    case = read_case(case_dir)

    with McpSession.for_input(case_dir) as running:
        store = running.store_for(case)
        assert store.section("pathology_report") is None
        assert "pathology_report" not in store.retrieved
        # The neighbouring sections are untouched, so this is a missing
        # document and not a broken case.
        assert store.section("radiology_report") is not None

        payload = running.call_tool("get_pathology_report", {"case_id": case.case_id})
        assert payload == {"case_id": case.case_id}, "an absent section must be omitted, not null"


def test_a_section_outside_the_task_registry_is_refused(session, cases):
    """Task 1 is pre-biopsy: asking it for a pathology report yields nothing.

    The refusal is client-side -- ``tool_for_section`` returns ``None`` and no
    frame is sent -- and it must not enter the ledger, because a section that
    was never retrieved must never be declared.
    """
    task1 = [c for c in cases if c.task == 1]
    if not task1:
        pytest.skip("no task1 fixtures")
    for case in task1:
        store = session.store_for(case)
        assert store.section("pathology_report") is None
        assert "pathology_report" not in store.retrieved


def test_the_server_rejects_a_tool_the_case_task_does_not_mask(session, cases):
    """The same refusal held independently on the server side.

    The client is what normally stops this, so drive the tool call directly:
    a server that answered would let a buggy client read a document the form
    never offered for that task.
    """
    task1 = [c for c in cases if c.task == 1]
    if not task1:
        pytest.skip("no task1 fixtures")
    with pytest.raises(protocol.ProtocolError):
        session.call_tool("get_pathology_report", {"case_id": task1[0].case_id})


def test_an_unknown_case_is_an_error_not_an_empty_answer(session):
    with pytest.raises(protocol.ProtocolError):
        session.call_tool("get_mri_report", {"case_id": "no-such-case"})


def test_an_unknown_tool_is_an_error(session, cases):
    with pytest.raises(protocol.ProtocolError):
        session.call_tool("get_horoscope", {"case_id": cases[0].case_id})


# -- the ledger -------------------------------------------------------------- #


def test_the_ledger_holds_exactly_the_sections_that_returned_data(session, cases):
    for case in cases:
        store = session.store_for(case)
        asked = [t.section for t in tools_for_task(case.task)]
        for section in asked:
            store.section(section)
        expected = {
            s: case.clinical_data[s]
            for s in asked
            if section_is_present(case.clinical_data.get(s))
        }
        assert store.retrieved == expected


def test_the_ledger_is_in_call_order(session, cases):
    """Order is load-bearing: it is what a declared reveal sequence is built from."""
    case = next(c for c in cases if c.task == 2)
    store = session.store_for(case)
    wanted = [s for s in ("previous_notes", "radiology_report", "psa_trend")
              if store.section(s) is not None]
    assert list(store.retrieved) == wanted


def test_reading_a_section_twice_makes_one_tool_call(session, cases):
    """Memoisation, and not merely as an optimisation.

    Two extractors may want the same report. Without the cache the ledger would
    record one revealed document twice, and the declared sequence would repeat
    a section the clinician only ever opened once.
    """
    case = next(c for c in cases if c.task == 3)
    store = session.store_for(case)

    calls: list[str] = []
    real = session.call_tool

    def counting(name, arguments):
        calls.append(name)
        return real(name, arguments)

    session.call_tool = counting  # type: ignore[method-assign]
    try:
        first = store.section("radiology_report")
        for _ in range(4):
            assert store.section("radiology_report") == first
    finally:
        session.call_tool = real  # type: ignore[method-assign]

    assert calls == ["get_mri_report"]


def test_an_absent_section_is_also_asked_for_only_once(tmp_path):
    """The negative answer is cached too, or a missing report costs N round trips."""
    source = next((d for d in CASE_DIRS if d.parent.name == "task2"), None)
    if source is None:
        pytest.skip("no task2 fixtures")
    case_dir = _case_without(tmp_path, source, "pathology_report")
    case = read_case(case_dir)

    with McpSession.for_input(case_dir) as running:
        store = running.store_for(case)
        calls: list[str] = []
        real = running.call_tool

        def counting(name, arguments):
            calls.append(name)
            return real(name, arguments)

        running.call_tool = counting  # type: ignore[method-assign]
        try:
            for _ in range(3):
                assert store.section("pathology_report") is None
        finally:
            running.call_tool = real  # type: ignore[method-assign]

    assert calls == ["get_pathology_report"]


# -- reveal honesty ---------------------------------------------------------- #


def test_declared_reveals_are_retrieved(session, cases):
    """The invariant the whole access path exists to make structural.

    A declared ``reveal_sequence`` used to be a claim maintained by hand
    alongside a separate dict read. Now a section can only be declared if a tool
    call returned it, and this asserts it over every fixture: nothing declared
    is absent from the ledger, and nothing declared is outside the vocabulary.
    """
    predictor = GuidelinePredictor()
    for case in cases:
        if case.task == 3:
            continue  # Task 3's reasoning socket is prose, with no declaration
        store = session.store_for(case)
        prediction = predictor.predict(case, store)
        declared = prediction.reasoning.reveal_sequence
        ledger = store.retrieved

        assert len(set(declared)) == len(declared), f"{case.case_id}: duplicate declaration"
        for section in declared:
            assert section in spec.REVEAL_SECTIONS, f"{case.case_id}: {section} out of vocabulary"
            assert section in ledger, (
                f"{case.case_id}: declared {section} but no tool call returned it"
            )


# -- the transport itself ----------------------------------------------------- #


def test_the_server_speaks_line_delimited_json_by_hand():
    """No client involved: raw frames in, raw frames out.

    MCP stdio is newline-delimited JSON with no ``Content-Length`` framing, and
    a server that quietly required framing would still pass every test above --
    our client would be wrong in the same direction. This pins the wire format
    against the spec rather than against ourselves.
    """
    if not CASE_DIRS:
        pytest.skip(f"no fixtures under {FIXTURES}")
    case_dir = CASE_DIRS[0]
    case = read_case(case_dir)

    frames = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": protocol.PROTOCOL_VERSION,
                    "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "get_previous_notes", "arguments": {"case_id": case.case_id}}},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "chimera.mcp.server", "--input", str(case_dir)],
        input="".join(json.dumps(f) + "\n" for f in frames),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr

    replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert [r["id"] for r in replies] == [1, 2, 3], "the notification drew a reply"
    assert all(r["jsonrpc"] == "2.0" for r in replies)
    assert replies[0]["result"]["serverInfo"]["name"] == "chimera-clinical"

    listed = {t["name"] for t in replies[1]["result"]["tools"]}
    assert listed == {t.name for t in tools_for_task(case.task)}, (
        "a single-case server must publish exactly its own task's registry"
    )

    content = replies[2]["result"]["content"]
    assert content and content[0]["type"] == "text"
    assert json.loads(content[0]["text"])["case_id"] == case.case_id


def test_an_unparseable_frame_is_reported_and_the_server_keeps_serving():
    """A client error must not cost the session; the next frame is still answered."""
    if not CASE_DIRS:
        pytest.skip(f"no fixtures under {FIXTURES}")
    proc = subprocess.run(
        [sys.executable, "-m", "chimera.mcp.server", "--input", str(CASE_DIRS[0])],
        input='{"jsonrpc": "2.0", "id": 1, "meth\n{"jsonrpc":"2.0","id":2,"method":"ping"}\n',
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert any("error" in r for r in replies), "the malformed frame drew no error"
    assert any(r.get("id") == 2 and "result" in r for r in replies), (
        "the server stopped serving after a bad frame"
    )


def test_a_server_that_cannot_start_raises_rather_than_hanging(tmp_path):
    """The failure the container's fallback is written for.

    ``inference.py`` catches this and degrades to a direct read. That only works
    if the constructor actually raises -- promptly, and without leaving the
    child alive.
    """
    with pytest.raises(Exception):
        McpSession.for_cohort(tmp_path / "nothing-here", timeout=15.0, startup_timeout=15.0)


def test_a_failed_handshake_leaves_no_child_process(tmp_path):
    """A half-open session is worse than none: it orphans the subprocess."""
    script = tmp_path / "silent.py"
    script.write_text("import sys; sys.exit(3)\n")
    session = None
    try:
        session = McpSession([sys.executable, str(script)], timeout=15.0, startup_timeout=15.0)
    except Exception:
        pass
    else:  # pragma: no cover - a server that answered nothing must not succeed
        session.close()
        pytest.fail("a server that exits immediately produced a usable session")
    assert session is None


def test_startup_gets_a_longer_budget_than_a_tool_call(tmp_path):
    """A slow *start* is normal; a slow *call* is a hang. They need separate budgets.

    The server builds its case index before it can answer anything, and in cohort
    mode that is a recursive scan plus one JSON read per case. Charging it to the
    per-call budget killed a 40-minute fit on a busy host -- the scan was fine, the
    deadline was not. So the handshake waits longer, and the per-call timeout stays
    tight so a genuine hang is still caught quickly.

    Driven by a stub server that sleeps past the per-call timeout before answering
    ``initialize`` and is instant thereafter, which is exactly the real shape.
    """
    script = tmp_path / "slow_start.py"
    script.write_text(
        "import json, sys, time\n"
        "time.sleep(2.0)\n"  # longer than the per-call budget below
        "for line in sys.stdin:\n"
        "    line = line.strip()\n"
        "    if not line:\n"
        "        continue\n"
        "    msg = json.loads(line)\n"
        "    if 'id' not in msg:\n"
        "        continue\n"
        "    sys.stdout.write(json.dumps(\n"
        "        {'jsonrpc': '2.0', 'id': msg['id'],\n"
        "         'result': {'serverInfo': {'name': 'slow'}}}) + '\\n')\n"
        "    sys.stdout.flush()\n"
    )
    command = [sys.executable, str(script)]

    # One budget for both: the slow start is misread as a hang.
    with pytest.raises(Exception):
        McpSession(command, timeout=0.5, startup_timeout=0.5).close()

    # Separate budgets: the same server starts cleanly, and the per-call deadline
    # is still the tight one.
    session = McpSession(command, timeout=0.5, startup_timeout=30.0)
    try:
        assert session.server_info.get("name") == "slow"
        assert session.timeout == 0.5
    finally:
        session.close()
