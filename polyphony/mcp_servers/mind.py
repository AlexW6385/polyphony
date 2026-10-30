"""Polyphony as an MCP server: any MCP-capable AI app can bring a question into
the room, read what the voices make of it, and start a cadence.

Implemented over the HTTP API (API_URL), so it works against a running
instance in Docker or locally.

    python -m polyphony.mcp_servers.mind                 # stdio (Claude Desktop, Claude Code)
    python -m polyphony.mcp_servers.mind --http          # streamable HTTP on MCP_HTTP_HOST:PORT

There is no auth: HTTP binds to 127.0.0.1 by default. Exposing it further is
an explicit choice (MCP_HTTP_HOST), and anyone who can reach it can speak in
the room.
"""

from __future__ import annotations

import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from ..config import get_settings

settings = get_settings()
mcp = FastMCP(
    "polyphony",
    instructions=("A room of voices with no chair. `say` drops a thought in; voices decide for "
                  "themselves whether to answer, so read the stream again after a while. "
                  "`start_cadence` makes the voices vote on a resolution."),
    host=settings.mcp_http_host,
    port=settings.mcp_http_port,
)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.api_url, timeout=30)


@mcp.tool()
async def say(content: str) -> dict[str, Any]:
    """Drop a thought into the room as the human participant. Returns the recorded thought."""
    async with _client() as c:
        r = await c.post("/thoughts", json={"content": content, "kind": "person"})
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def read_stream(after: int = 0, limit: int = 100) -> list[dict[str, Any]]:
    """Read thoughts with seq greater than `after`, oldest first (voices, world, cadences, digests)."""
    async with _client() as c:
        r = await c.get("/thoughts", params={"after": after, "limit": min(limit, 1000)})
        r.raise_for_status()
        return [{k: t[k] for k in ("seq", "author", "kind", "content", "mood")} for t in r.json()]


@mcp.tool()
async def start_cadence(resolution: str) -> dict[str, Any]:
    """Start a formal vote on a resolution. The result appears in the stream as a cadence thought."""
    async with _client() as c:
        r = await c.post("/cadence", json={"resolution": resolution})
        r.raise_for_status()
        return r.json()


@mcp.tool()
async def list_voices() -> list[dict[str, Any]]:
    """The voices in the room: name, enabled, whether forming, last listen."""
    async with _client() as c:
        r = await c.get("/voices")
        r.raise_for_status()
        return [{"name": v["name"], "enabled": v["enabled"], "forming": v["forming"],
                 "last_listen": v["last_listen"]} for v in r.json()]


def main() -> None:
    mcp.run("streamable-http" if "--http" in sys.argv else "stdio")


if __name__ == "__main__":
    main()
