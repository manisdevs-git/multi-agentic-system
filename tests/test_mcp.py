"""Cloud Run 503 should retry, then fail with a short message (no HTML dump)."""

import json

import httpx
import pytest

from support_reply.mcp import McpClient, McpError


def test_503_is_retried_then_succeeds():
    hits = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(
                503,
                text="<html><h2>The service you requested is not available yet.</h2>",
            )
        body = json.loads(request.content)
        if body.get("method") == "initialize":
            return httpx.Response(
                200,
                text='{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05"}}',
            )
        return httpx.Response(202, text="")

    client = McpClient(
        server_url="http://127.0.0.1:8080/mcp",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        retry_wait=0,
    )
    client.initialize()
    client.close()
    assert hits["n"] >= 2


def test_persistent_503_does_not_dump_html():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            text="<html><title>503 Server Error</title><h1>Error: Server Error</h1>",
        )

    client = McpClient(
        server_url="http://127.0.0.1:8080/mcp",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_retries=1,
        retry_wait=0,
    )
    with pytest.raises(McpError, match="Cloud Run is not available") as caught:
        client.initialize()
    client.close()
    assert "<html>" not in str(caught.value)
    assert caught.value.status_code == 503
