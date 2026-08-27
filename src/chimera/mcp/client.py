"""The client side: a session over the wire, and the per-case store it hands out.

:class:`ClinicalStore` is the whole interface the decision path sees. It replaces
the raw ``case.clinical_data`` dict every extractor used to read, and it differs
from that dict in two ways that matter:

* a section is reached by *asking* for it, which is the point of the exercise;
* every ask that returns data is recorded, in call order, in
  :attr:`~ClinicalStore.retrieved`.

That ledger is what makes reveal honesty structural. Under the old arrangement a
predictor declared a ``reveal_sequence`` and separately read some sections, and
the two agreeing was a convention maintained by hand. Now a section can only be
declared if it is in the ledger, and it can only be in the ledger if a tool call
returned it.

Two implementations, and the asymmetry between them is deliberate.
:class:`McpStore` goes over the real protocol and is the only route used in
normal operation. :class:`DirectStore` reads in process and exists solely as the
transport-failure fallback: a crashed case is not skipped by the evaluator, it
is scored against a sentinel label and costs the true class its recall, so
losing the subprocess has to cost provenance rather than a case. Anything
selecting :class:`DirectStore` logs loudly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from chimera.contract.io import CaseInputs
from chimera.mcp import protocol
from chimera.mcp.tools import section_is_present, tool_for_section

#: Seconds to wait for one ``tools/call``. Generous: the server does a JSON read
#: and a dict lookup, so anything approaching this is a hang, not slow work.
DEFAULT_TIMEOUT = 30.0

CLIENT_NAME = "chimera-agent"
CLIENT_VERSION = "1.0.0"


@runtime_checkable
class ClinicalStore(Protocol):
    """Ask for a masked section; get it, or ``None``, and leave a record."""

    task: int
    case_id: str

    def section(self, name: str) -> Any | None:
        """The section's value, or ``None`` if this case does not carry it."""

    @property
    def retrieved(self) -> dict[str, Any]:
        """Sections that returned data, in the order they were requested."""


class DirectStore:
    """In-process fallback. Same interface, same ledger, no wire."""

    def __init__(self, case: CaseInputs) -> None:
        self.task = case.task
        self.case_id = case.case_id
        self._clinical = case.clinical_data if isinstance(case.clinical_data, dict) else {}
        self._retrieved: dict[str, Any] = {}

    def section(self, name: str) -> Any | None:
        if tool_for_section(self.task, name) is None:
            # A section this task does not mask has no tool, so there is nothing
            # to retrieve and nothing to declare. Mirrors the server's refusal.
            return None
        if name in self._retrieved:
            return self._retrieved[name]
        value = self._clinical.get(name)
        if not section_is_present(value):
            return None
        self._retrieved[name] = value
        return value

    @property
    def retrieved(self) -> dict[str, Any]:
        return dict(self._retrieved)


class McpSession:
    """A running MCP server subprocess and the handshake with it.

    One session can serve many cases, which is what lets the offline harness pay
    a single process start for a whole cross-validation sweep. On the platform
    the session is per case because the container is.
    """

    def __init__(self, command: list[str], *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.timeout = timeout
        self._next_id = 0
        # bufsize=0: the reader works on the raw descriptor so it can enforce a
        # timeout, and a buffered writer would leave frames sitting unflushed.
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherited: an undrained stderr pipe can deadlock us
            bufsize=0,
        )
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._in_fd = self._proc.stdout.fileno()
        self._out_fd = self._proc.stdin.fileno()
        self._reader = protocol.LineReader(self._in_fd)
        self.server_info: dict[str, Any] = {}
        try:
            self._handshake()
        except BaseException:
            # A half-open session is worse than none: the caller will fall back
            # to a direct read and would otherwise leave the child running.
            self.close()
            raise

    # -- lifecycle ----------------------------------------------------------- #

    @classmethod
    def for_input(cls, input_dir: Path, **kwargs: Any) -> McpSession:
        """Serve the single case mounted at ``input_dir`` -- the container shape."""
        return cls(
            [sys.executable, "-m", "chimera.mcp.server", "--input", str(input_dir)], **kwargs
        )

    @classmethod
    def for_cohort(cls, cases_root: Path, **kwargs: Any) -> McpSession:
        """Serve every case under ``cases_root`` -- the offline-harness shape."""
        return cls(
            [sys.executable, "-m", "chimera.mcp.server", "--cases", str(cases_root)], **kwargs
        )

    def __enter__(self) -> McpSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Shut the server down by closing its stdin, then insist if it lingers."""
        if self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
                self._proc.wait(timeout=5)
        for stream in (self._proc.stdout, self._proc.stdin):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    # -- protocol ------------------------------------------------------------ #

    def _handshake(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": protocol.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        )
        self.server_info = result.get("serverInfo") or {}
        # Required by the spec and expected by conforming servers, even though
        # ours does not gate on it.
        protocol.write_message(self._out_fd, protocol.notification("notifications/initialized"))

    def _request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        msg_id = self._next_id
        protocol.write_message(self._out_fd, protocol.request(msg_id, method, params))

        while True:
            message = self._reader.read_message(timeout=self.timeout)
            if message is None:
                raise protocol.TransportError(f"server closed the stream during {method}")
            if message.get("id") != msg_id:
                continue  # a notification or a stale reply; not ours
            if "error" in message:
                err = message["error"] or {}
                raise protocol.ProtocolError(
                    str(err.get("message", "unknown error")), int(err.get("code", protocol.INTERNAL_ERROR))
                )
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def list_tools(self) -> list[dict[str, Any]]:
        tools = self._request("tools/list").get("tools")
        return tools if isinstance(tools, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke a tool and return its payload as a dict.

        Prefers ``structuredContent`` and falls back to parsing the text block,
        so this keeps working against a server that emits only one of the two.
        """
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise protocol.ProtocolError(f"{name} reported an error: {result}")

        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured

        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                try:
                    parsed = json.loads(block.get("text") or "")
                except json.JSONDecodeError as exc:
                    raise protocol.ProtocolError(f"{name} returned unparseable text: {exc}") from exc
                if isinstance(parsed, dict):
                    return parsed
        raise protocol.ProtocolError(f"{name} returned no readable content")

    # -- stores -------------------------------------------------------------- #

    def store(self, task: int, case_id: str) -> McpStore:
        return McpStore(self, task, case_id)

    def store_for(self, case: CaseInputs) -> McpStore:
        return McpStore(self, case.task, case.case_id)


class McpStore:
    """One case's view of a session: tool calls in, a retrieval ledger out."""

    def __init__(self, session: McpSession, task: int, case_id: str) -> None:
        self.session = session
        self.task = task
        self.case_id = case_id
        self._retrieved: dict[str, Any] = {}
        self._absent: set[str] = set()

    def section(self, name: str) -> Any | None:
        """Call the tool serving ``name``, once per case however often asked.

        Memoisation is not only an optimisation. Two extractors may want the
        same report, and without it the ledger would record one retrieval twice
        and the declared sequence would repeat a section that was revealed once.
        """
        if name in self._retrieved:
            return self._retrieved[name]
        if name in self._absent:
            return None

        tool = tool_for_section(self.task, name)
        if tool is None:
            self._absent.add(name)
            return None

        payload = self.session.call_tool(tool.name, {"case_id": self.case_id})
        value = payload.get(tool.section)
        if not section_is_present(value):
            self._absent.add(name)
            return None
        self._retrieved[name] = value
        return value

    @property
    def retrieved(self) -> dict[str, Any]:
        return dict(self._retrieved)


def spawn_store(input_dir: Path, case: CaseInputs, *, timeout: float = DEFAULT_TIMEOUT) -> tuple[McpSession, McpStore]:
    """A session and store for the single case mounted at ``input_dir``.

    Returned as a pair so the caller owns the session's lifetime; the container
    closes it after writing its outputs.
    """
    session = McpSession.for_input(input_dir, timeout=timeout)
    return session, session.store_for(case)
