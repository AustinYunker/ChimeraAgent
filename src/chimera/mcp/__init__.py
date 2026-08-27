"""Model Context Protocol access to the masked clinical documents.

The challenge masks the "Extended EHR view" documents -- reports, notes, labs,
PSA history, family history -- behind MCP tools, so that retrieving one is an
action the agent has to take rather than context it is handed. The organizers'
own submission entrypoint shows the intended shape: it reads the flat sockets
from ``/input`` (that is how the data arrives), re-materialises them for an MCP
server it spawns as a stdio subprocess, and lets the decision path reach the
documents only through tool calls.

This package is our equivalent, and nothing in it derives from theirs -- the
reference implementation is unlicensed. The tool names and the section
vocabulary are interface facts, and the six declarable sections were already
independently present as :data:`chimera.contract.spec.REVEAL_SECTIONS`.

**Pure standard library, deliberately.** ``Dockerfile`` installs the package
with ``--no-deps`` and ``tests/test_entrypoint.py`` pins the invariant, so the
official ``mcp`` SDK -- which pulls pydantic, anyio and httpx -- cannot ship in
the image. :mod:`chimera.mcp.protocol` therefore speaks JSON-RPC 2.0 over
newline-delimited stdio directly. That it really is the protocol and not a
naming convention is checked in CI by ``tests/test_mcp_conformance.py``, which
drives this server with the real SDK client.

Two stores implement :class:`~chimera.mcp.client.ClinicalStore`:
:class:`~chimera.mcp.client.McpStore` over the wire, and
:class:`~chimera.mcp.client.DirectStore` in process. The direct one exists only
as the transport-failure fallback -- a crashed case is scored against a sentinel
label rather than skipped, so losing the wire must cost provenance, not a case.
"""

from chimera.mcp.client import ClinicalStore, DirectStore, McpStore
from chimera.mcp.tools import TOOLS_BY_TASK, ToolSpec, tool_for_section

__all__ = [
    "TOOLS_BY_TASK",
    "ClinicalStore",
    "DirectStore",
    "McpStore",
    "ToolSpec",
    "tool_for_section",
]
