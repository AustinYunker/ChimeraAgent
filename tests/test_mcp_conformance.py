"""Our hand-rolled server, driven by the official MCP client.

``tests/test_mcp.py`` proves our client and our server agree. That is not the
same as speaking MCP: two halves written together can be wrong in the same
direction and never notice. This module removes our client from the picture and
drives the server with the reference implementation instead, so "we speak MCP"
is a protocol claim rather than a naming convention.

The SDK cannot be used in the shipped path -- it pulls pydantic, anyio, httpx
and more, while ``Dockerfile`` installs with ``--no-deps`` and
``tests/test_entrypoint.py`` pins the entrypoint to the standard library. So it
lives in the ``dev`` extra: installed in CI, where this runs, and skipped on a
host that has not installed it.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from chimera.contract.io import read_case
from chimera.mcp import protocol
from chimera.mcp.tools import tools_for_task

mcp = pytest.importorskip(
    "mcp", reason="the official MCP SDK is not installed; it lives in the `dev` extra"
)

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.shared import exceptions as mcp_exceptions  # noqa: E402

#: The SDK renamed this between releases (``McpError`` -> ``MCPError``), and we
#: only need the type, so take whichever this installation calls it.
MCP_ERROR = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "work" / "fixtures"

TIMEOUT = 60


def _case_dir() -> Path:
    """One Task 3 fixture: the richest registry, including the undeclarable report."""
    candidates = sorted(FIXTURES.glob("task3/*/inputs.json"))
    if not candidates:
        pytest.skip(f"no task3 fixtures under {FIXTURES}")
    return candidates[0].parent


def _field(obj: object, *names: str) -> object:
    """The first attribute of ``obj`` that exists among ``names``.

    The SDK moved its models from camelCase to snake_case between releases. The
    wire names are fixed by the specification and are what our server emits;
    only the Python attribute is in question, so accepting either keeps this
    test about the protocol rather than about an SDK version.
    """
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{type(obj).__name__} has none of {names}")


async def _drive(case_dir: Path, case_id: str) -> dict:
    """Handshake, list, and call -- entirely through the reference client."""
    params = StdioServerParameters(
        # `sys.executable`, not "python": the interpreter running the tests is
        # the one the package is installed into, and it need not be on PATH.
        command=sys.executable,
        args=["-m", "chimera.mcp.server", "--input", str(case_dir)],
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            called = await session.call_tool(
                "get_surgical_pathology_report", {"case_id": case_id}
            )
            try:
                await session.call_tool("get_psa_trend", {"case_id": case_id})
            except MCP_ERROR as exc:
                refusal = str(exc)
            else:
                refusal = ""
            return {
                "refusal": refusal,
                "server_name": _field(_field(init, "server_info", "serverInfo"), "name"),
                "protocol_version": _field(init, "protocol_version", "protocolVersion"),
                "tools": sorted(t.name for t in listed.tools),
                "schemas": {
                    t.name: _field(t, "input_schema", "inputSchema") for t in listed.tools
                },
                "called_text": [c.text for c in called.content if c.type == "text"],
                "called_error": bool(_field(called, "is_error", "isError")),
            }


@pytest.fixture(scope="module")
def driven() -> dict:
    case_dir = _case_dir()
    case = read_case(case_dir)
    result = asyncio.run(asyncio.wait_for(_drive(case_dir, case.case_id), TIMEOUT))
    result["case_id"] = case.case_id
    return result


def test_the_official_client_completes_the_handshake(driven):
    """Negotiation included: the SDK's latest revision is newer than ours.

    The server answers with the newest revision it actually implements, and the
    reference client must accept that rather than hanging up -- which is the
    whole reason the server echoes from a supported set instead of asserting one
    version.
    """
    assert driven["server_name"] == "chimera-clinical"
    assert driven["protocol_version"] in protocol.SUPPORTED_PROTOCOL_VERSIONS
    # The SDK asks for a revision newer than any we implement, so agreement here
    # is the negotiation working rather than a coincidence of matching defaults.
    assert driven["protocol_version"] != mcp.types.LATEST_PROTOCOL_VERSION


def test_the_official_client_reads_our_tool_registry(driven):
    """A single-case server publishes exactly that task's tools, schemas parsed."""
    assert driven["tools"] == sorted(t.name for t in tools_for_task(3))
    for name, schema in driven["schemas"].items():
        assert schema["required"] == ["case_id"], f"{name} lost its argument schema"


def test_the_official_client_retrieves_a_document(driven):
    """The payload survives the SDK's own content parsing, not just ours."""
    assert not driven["called_error"]
    assert driven["called_text"], "the SDK saw no text content"
    payload = json.loads(driven["called_text"][0])
    assert payload["case_id"] == driven["case_id"]
    assert payload["surgical_pathology_report"].strip()


def test_a_refusal_reaches_the_official_client_as_an_error(driven):
    """Task 3 masks no PSA series, and the SDK must see the refusal as such.

    The refusal is a JSON-RPC error rather than a result carrying ``isError``,
    because the tool does not exist for this case's task -- that is a bad
    request, not a tool that ran and failed. The reference client raises on it,
    which is the behaviour that matters: a refusal delivered as a successful
    empty result would let an agent record a retrieval that never happened.
    """
    assert "not available for task 3" in driven["refusal"]
