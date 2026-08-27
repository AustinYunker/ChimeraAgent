"""JSON-RPC 2.0 over newline-delimited stdio -- the MCP wire, in the stdlib.

MCP's stdio transport is one JSON object per line in each direction: no
``Content-Length`` framing, no length prefix. That makes a conforming
implementation small enough to own rather than depend on, which is what keeps
the submission image installable with ``--no-deps``.

Only the four messages the clinical-document surface needs are modelled --
``initialize``, the ``notifications/initialized`` acknowledgement, ``tools/list``
and ``tools/call``. Both ends import this module, so the client and the server
cannot drift apart in their framing.

Reading is done against a raw file descriptor with an explicit buffer rather
than through :meth:`io.BufferedReader.readline`, because the client also needs a
timeout: mixing :func:`select.select` with a buffered reader loses any bytes
already sitting in the buffer, and the resulting hang is exactly the failure a
timeout was added to prevent.
"""

from __future__ import annotations

import json
import os
import select
from typing import Any

#: Protocol revisions this implementation is known to interoperate with. The
#: tool surface is identical across them, so the server accepts whichever the
#: client proposes from this set and echoes it back, per the spec's negotiation.
SUPPORTED_PROTOCOL_VERSIONS: tuple[str, ...] = (
    "2025-06-18",
    "2025-03-26",
    "2024-11-05",
)
PROTOCOL_VERSION: str = SUPPORTED_PROTOCOL_VERSIONS[0]

JSONRPC_VERSION = "2.0"

# JSON-RPC 2.0 reserved codes (spec section 5.1).
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class ProtocolError(Exception):
    """A JSON-RPC error, either received from a peer or about to be sent."""

    def __init__(self, message: str, code: int = INTERNAL_ERROR) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class TransportError(Exception):
    """The pipe failed: the peer died, timed out, or emitted a non-message."""


# --------------------------------------------------------------------------- #
# Message construction
# --------------------------------------------------------------------------- #

def request(msg_id: int, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def notification(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """A request with no ``id``; the peer must not reply to it."""
    msg: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def result(msg_id: Any, payload: Any) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "result": payload}


def error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": msg_id, "error": {"code": code, "message": message}}


def text_content(payload: Any) -> dict[str, Any]:
    """A ``tools/call`` result: JSON serialised into MCP's text content block.

    ``structuredContent`` carries the same payload natively for clients that
    read it, but the text block is what every client understands, so both are
    emitted rather than choosing between them.
    """
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "structuredContent": payload,
        "isError": False,
    }


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #

def encode(message: dict[str, Any]) -> bytes:
    """One message, one line. ``ensure_ascii`` keeps the frame byte-safe."""
    return (json.dumps(message, ensure_ascii=True) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict[str, Any]:
    text = line.decode("utf-8") if isinstance(line, bytes) else line
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"malformed JSON frame: {exc}", PARSE_ERROR) from exc
    if not isinstance(message, dict):
        raise ProtocolError("frame is not a JSON object", INVALID_REQUEST)
    return message


def write_message(fd: int, message: dict[str, Any]) -> None:
    """Write a whole frame, tolerating the partial writes a pipe may do."""
    data = encode(message)
    while data:
        try:
            written = os.write(fd, data)
        except OSError as exc:
            raise TransportError(f"write failed: {exc}") from exc
        if written <= 0:
            raise TransportError("write made no progress")
        data = data[written:]


class LineReader:
    """Newline-delimited frames off a raw fd, with an optional deadline."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buf = bytearray()

    def read_message(self, timeout: float | None = None) -> dict[str, Any] | None:
        """Next frame, or ``None`` at clean end of stream.

        Raises :class:`TransportError` on timeout or a broken pipe, and
        :class:`ProtocolError` on a frame that is not a JSON object.
        """
        while True:
            newline = self._buf.find(b"\n")
            if newline >= 0:
                line = bytes(self._buf[:newline])
                del self._buf[: newline + 1]
                if not line.strip():
                    continue  # blank keep-alive line, not a frame
                return decode(line)

            if timeout is not None:
                ready, _, _ = select.select([self._fd], [], [], timeout)
                if not ready:
                    raise TransportError(f"no response within {timeout:g}s")
            try:
                chunk = os.read(self._fd, 65536)
            except OSError as exc:
                raise TransportError(f"read failed: {exc}") from exc
            if not chunk:
                if self._buf.strip():
                    raise TransportError("stream ended mid-frame")
                return None
            self._buf.extend(chunk)
