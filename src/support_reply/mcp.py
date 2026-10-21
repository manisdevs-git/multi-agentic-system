"""JSON-RPC MCP client for workspace Docs / Gmail tools. Same pattern as GrowwAura."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

_ACCEPT = "application/json, text/event-stream"


class McpError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class ToolClient(Protocol):
    def initialize(self) -> dict[str, Any]: ...

    def close(self) -> None: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


def mcp_configured(url: str | None) -> bool:
    return bool((url or "").strip())


def _headers(*, auth_token: str | None, session_id: str | None) -> dict[str, str]:
    headers = {
        "Accept": _ACCEPT,
        "Content-Type": "application/json",
        "User-Agent": "support-reply/0.0.1",
    }
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    if session_id:
        headers["mcp-session-id"] = session_id
    return headers


def _sse_data_chunks(payload: str) -> list[str]:
    chunks: list[str] = []
    for line in payload.splitlines():
        if line.startswith("data:"):
            chunk = line[5:].strip()
            if chunk:
                chunks.append(chunk)
    return chunks


def _load_rpc_objects(payload: str) -> list[dict[str, Any]]:
    text = (payload or "").strip()
    if not text:
        raise McpError("Empty MCP response")
    blobs = _sse_data_chunks(text) if (text.startswith("event:") or "data:" in text) else [text]
    if not blobs:
        raise McpError("MCP SSE response had no data")
    objects: list[dict[str, Any]] = []
    for blob in blobs:
        try:
            body = json.loads(blob)
        except json.JSONDecodeError as exc:
            raise McpError("MCP response was not JSON") from exc
        if isinstance(body, dict):
            objects.append(body)
    if not objects:
        raise McpError("MCP response was not an object")
    return objects


def _parse_rpc(payload: str, rpc_id: int | None = None) -> dict[str, Any]:
    objects = _load_rpc_objects(payload)
    chosen: dict[str, Any] | None = None
    if rpc_id is not None:
        for body in objects:
            if body.get("id") == rpc_id:
                chosen = body
                break
    if chosen is None:
        for body in reversed(objects):
            if "result" in body or "error" in body:
                chosen = body
                break
    if chosen is None:
        chosen = objects[-1]
    error = chosen.get("error")
    if error:
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise McpError(str(message or "MCP error"))
    if "result" not in chosen:
        raise McpError("MCP response had no result")
    result = chosen.get("result")
    if not isinstance(result, dict):
        raise McpError("MCP result was not an object")
    return result


def _content_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            texts.append(str(item.get("text") or ""))
    return "\n".join(part for part in texts if part).strip()


def _raise_if_failed(payload: dict[str, Any]) -> None:
    if payload.get("success") is False:
        detail = payload.get("error") or payload.get("hint") or payload.get("text")
        raise McpError(str(detail or "MCP tool returned success=false"))


def _tool_payload(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("isError"):
        detail = result.get("error") or _content_text(result) or "MCP tool returned an error"
        raise McpError(str(detail))
    blob = _content_text(result)
    if blob:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            parsed = {"text": blob, **result}
        if isinstance(parsed, dict):
            _raise_if_failed(parsed)
            return parsed
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        _raise_if_failed(structured)
        return structured
    _raise_if_failed(result)
    return result


class McpClient:
    """Short-lived JSON-RPC session against one MCP HTTP endpoint."""

    def __init__(
        self,
        *,
        server_url: str,
        auth_token: str | None = None,
        timeout_seconds: float = 90.0,
        max_retries: int = 4,
        retry_wait: float = 3.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not mcp_configured(server_url):
            raise McpError("MCP_SERVER_URL is not configured", status_code=503)
        self.server_url = server_url.strip()
        self.auth_token = (auth_token or "").strip() or None
        self.max_retries = max(0, max_retries)
        self.retry_wait = max(0.0, retry_wait)
        self._own = client is None
        self._http = client or httpx.Client(timeout=timeout_seconds)
        self._session_id: str | None = None
        self._rpc_id = 0

    def close(self) -> None:
        if self._own:
            self._http.close()

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _error_for(self, response: httpx.Response) -> McpError:
        if response.status_code == 503:
            return McpError(
                "MCP Cloud Run is not available yet (HTTP 503). "
                "Point MCP_SERVER_URL at the local manismcpserver "
                "(http://127.0.0.1:8080/mcp) and click Send again.",
                status_code=503,
            )
        snippet = (response.text or "").replace("\n", " ")[:160]
        if "<html" in snippet.lower():
            snippet = f"HTTP {response.status_code} from Cloud Run"
        return McpError(f"MCP HTTP {response.status_code}: {snippet}", status_code=response.status_code)

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        last: McpError | None = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self._http.post(
                    self.server_url,
                    json=payload,
                    headers=_headers(auth_token=self.auth_token, session_id=self._session_id),
                )
            except httpx.TimeoutException as exc:
                last = McpError(f"MCP timed out: {exc}", status_code=504)
                logger.warning("MCP timeout attempt=%s", attempt + 1)
                if attempt + 1 < attempts and self.retry_wait:
                    time.sleep(self.retry_wait)
                continue
            session = response.headers.get("mcp-session-id")
            if session:
                self._session_id = session
            if response.status_code >= 500:
                last = self._error_for(response)
                logger.warning("MCP HTTP %s attempt=%s", response.status_code, attempt + 1)
                if attempt + 1 < attempts and self.retry_wait:
                    time.sleep(self.retry_wait)
                continue
            if response.status_code >= 400:
                raise self._error_for(response)
            return response
        raise last or McpError("MCP request failed")

    def initialize(self) -> dict[str, Any]:
        rpc_id = self._next_id()
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "support-reply", "version": "0.0.1"},
                },
            }
        )
        result = _parse_rpc(response.text, rpc_id=rpc_id)
        self._http.post(
            self.server_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=_headers(auth_token=self.auth_token, session_id=self._session_id),
        )
        return result

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        rpc_id = self._next_id()
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        return _tool_payload(_parse_rpc(response.text, rpc_id=rpc_id))
