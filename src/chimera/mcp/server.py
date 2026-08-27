"""The stdio MCP server over a cohort of flat-socket case directories.

Two modes, because the platform and the bench want different scopes::

    python -m chimera.mcp.server --input /input          # one case, as GC mounts it
    python -m chimera.mcp.server --cases work/train      # a whole cohort

``--input`` is the shape the container uses: Grand Challenge invokes the
algorithm once per case, so the server it spawns serves exactly that case.
``--cases`` exists so the offline harness can spawn **one** server for a whole
cross-validation sweep instead of one per case per fold -- the difference
between a single process start and several thousand, which is what makes it
affordable to put every offline path over the real wire rather than keeping a
faster direct route alongside it.

Case discovery reuses :mod:`chimera.contract.io` rather than re-deriving the
socket layout: any directory holding an ``inputs.json`` is a case, its task
comes from the clinical-data slug, and its identifier comes from the structured
prompt with the directory name as the fallback -- exactly as ``read_case`` does,
so the server and the predictor can never disagree about what a case is called.

Clinical data is read lazily and cached. A cohort server therefore pays for the
sections that are actually requested, which for our reveal policies is a small
fraction of the corpus.

Nothing here may write to stdout except protocol frames; stdout *is* the
transport. Diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from chimera.contract import spec
from chimera.contract.io import detect_task, read_dict, socket_paths
from chimera.mcp import protocol
from chimera.mcp.tools import TOOL_BY_NAME, section_is_present, tools_for_task

SERVER_NAME = "chimera-clinical"
SERVER_VERSION = "1.0.0"


class CaseIndex:
    """Cases discoverable under a root, keyed by the identifier they report."""

    def __init__(self, roots: list[Path]) -> None:
        self._task: dict[str, int] = {}
        self._clinical_path: dict[str, Path] = {}
        self._clinical: dict[str, dict[str, Any]] = {}
        for root in roots:
            self._index(root)

    def _index(self, root: Path) -> None:
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")

        # A root may itself be a single case (the /input mount) or contain many
        # at any depth (task<N>/<case_id>/ locally).
        manifests = [root / "inputs.json"] if (root / "inputs.json").is_file() else []
        if not manifests:
            manifests = sorted(root.rglob("inputs.json"))
        if not manifests:
            raise FileNotFoundError(f"no case directories (inputs.json) under {root}")

        for manifest in manifests:
            case_dir = manifest.parent
            try:
                paths = socket_paths(case_dir)
                task = detect_task(paths)
            except Exception as exc:  # a malformed case must not sink the cohort
                print(f"skipping {case_dir}: {exc}", file=sys.stderr)
                continue
            prompt = read_dict(paths.get(spec.STRUCTURED_PROMPT_SLUG))
            case_id = str(prompt.get("case_id") or case_dir.name)
            self._task[case_id] = task
            self._clinical_path[case_id] = case_dir / f"{spec.CLINICAL_SLUG_BY_TASK[task]}.json"
            clinical = paths.get(spec.CLINICAL_SLUG_BY_TASK[task])
            if clinical is not None:
                self._clinical_path[case_id] = clinical

    def __len__(self) -> int:
        return len(self._task)

    @property
    def tasks(self) -> set[int]:
        return set(self._task.values())

    def task_of(self, case_id: str) -> int | None:
        return self._task.get(case_id)

    def clinical(self, case_id: str) -> dict[str, Any]:
        """The case's clinical-data socket, read once and cached."""
        if case_id not in self._clinical:
            self._clinical[case_id] = read_dict(self._clinical_path.get(case_id))
        return self._clinical[case_id]

    def case_ids(self) -> list[str]:
        return sorted(self._task)


class Server:
    """Dispatch for the four MCP methods this tool surface needs."""

    def __init__(self, index: CaseIndex) -> None:
        self.index = index
        self.initialized = False

    # -- method surface ------------------------------------------------------ #

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        """Agree a protocol revision, preferring whatever the client proposed.

        Our tool surface is identical across every revision in
        :data:`~chimera.mcp.protocol.SUPPORTED_PROTOCOL_VERSIONS`, so echoing a
        supported request is both spec-conformant and the behaviour most likely
        to interoperate with an SDK client newer than this file.
        """
        requested = params.get("protocolVersion")
        version = (
            requested
            if isinstance(requested, str) and requested in protocol.SUPPORTED_PROTOCOL_VERSIONS
            else protocol.PROTOCOL_VERSION
        )
        self.initialized = True
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _tools_list(self) -> dict[str, Any]:
        """The registry for the served task.

        A cohort server may span tasks, in which case it publishes the union and
        rejects a call for a tool the individual case's task does not mask. A
        single-case server -- the container's shape -- publishes exactly that
        task's registry, which is what a client should see on the platform.
        """
        tasks = self.index.tasks
        seen: dict[str, Any] = {}
        for task in sorted(tasks):
            for tool in tools_for_task(task):
                seen[tool.name] = {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.input_schema(),
                }
        return {"tools": list(seen.values())}

    def _tools_call(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise protocol.ProtocolError("arguments must be an object", protocol.INVALID_PARAMS)

        tool = TOOL_BY_NAME.get(name) if isinstance(name, str) else None
        if tool is None:
            raise protocol.ProtocolError(f"unknown tool {name!r}", protocol.INVALID_PARAMS)

        case_id = arguments.get("case_id")
        if not isinstance(case_id, str):
            raise protocol.ProtocolError("case_id is required", protocol.INVALID_PARAMS)

        task = self.index.task_of(case_id)
        if task is None:
            raise protocol.ProtocolError(f"unknown case {case_id!r}", protocol.INVALID_PARAMS)
        if tool not in tools_for_task(task):
            raise protocol.ProtocolError(
                f"{tool.name} is not available for task {task}", protocol.INVALID_PARAMS
            )

        # A section the case has nothing for is omitted rather than returned as
        # null: "this patient never had a biopsy" and "this section is empty"
        # are the same answer to the agent, and neither is an error.
        payload: dict[str, Any] = {"case_id": case_id}
        value = self.index.clinical(case_id).get(tool.section)
        if section_is_present(value):
            payload[tool.section] = value
        return protocol.text_content(payload)

    # -- loop ---------------------------------------------------------------- #

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """One request in, at most one response out. ``None`` for notifications."""
        method = message.get("method")
        msg_id = message.get("id")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            params = {}

        if msg_id is None:
            # Notifications are acknowledged by silence. `initialized` is the
            # only one the client sends; anything else is harmlessly ignored.
            return None

        try:
            if method == "initialize":
                return protocol.result(msg_id, self._initialize(params))
            if method == "tools/list":
                return protocol.result(msg_id, self._tools_list())
            if method == "tools/call":
                return protocol.result(msg_id, self._tools_call(params))
            if method == "ping":
                return protocol.result(msg_id, {})
        except protocol.ProtocolError as exc:
            return protocol.error(msg_id, exc.code, exc.message)
        except Exception as exc:  # a bad case must not take the server down
            return protocol.error(msg_id, protocol.INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

        return protocol.error(msg_id, protocol.METHOD_NOT_FOUND, f"unknown method {method!r}")

    def serve(self, in_fd: int, out_fd: int) -> int:
        reader = protocol.LineReader(in_fd)
        while True:
            try:
                message = reader.read_message()
            except protocol.TransportError as exc:
                print(f"transport closed: {exc}", file=sys.stderr)
                return 1
            except protocol.ProtocolError as exc:
                # Unparseable frame: report it and keep serving, since the next
                # frame may be fine and the client is the one that erred.
                protocol.write_message(out_fd, protocol.error(None, exc.code, exc.message))
                continue
            if message is None:
                return 0
            response = self.handle(message)
            if response is not None:
                protocol.write_message(out_fd, response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chimera.mcp.server",
        description="Serve the masked clinical documents over MCP stdio.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input", type=Path, help="a single flat-socket case directory, as /input"
    )
    source.add_argument(
        "--cases", type=Path, help="a cohort root; any directory with inputs.json is a case"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.input if args.input is not None else args.cases
    try:
        index = CaseIndex([root])
    except Exception as exc:
        print(f"cannot serve {root}: {exc}", file=sys.stderr)
        return 2
    print(
        f"{SERVER_NAME} serving {len(index)} case(s) from {root} (tasks {sorted(index.tasks)})",
        file=sys.stderr,
    )
    return Server(index).serve(sys.stdin.fileno(), sys.stdout.fileno())


if __name__ == "__main__":
    raise SystemExit(main())
