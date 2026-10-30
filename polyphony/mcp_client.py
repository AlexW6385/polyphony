"""Sessions to MCP servers, for senses.

Servers are started lazily over stdio on first use inside an `async with
MCPClient(...)` block and shut down when it exits, so each sense run gets
fresh processes and a crashed server never outlives the run.

Only our own server, official reference servers and vendor-maintained servers
are configured: a server runs with whatever access it is given.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from .config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerSpec:
    command: str
    env: dict[str, str] | None = None


def server_specs(settings: Settings) -> dict[str, ServerSpec]:
    specs = {
        "feeds": ServerSpec(settings.mcp_feeds_command),
        "fetch": ServerSpec(settings.mcp_fetch_command),
    }
    if settings.exa_api_key:
        specs["search"] = ServerSpec(settings.mcp_exa_command, {"EXA_API_KEY": settings.exa_api_key})
    elif settings.tavily_api_key:
        specs["search"] = ServerSpec(settings.mcp_tavily_command,
                                     {"TAVILY_API_KEY": settings.tavily_api_key})
    return specs


def search_tool(settings: Settings) -> tuple[str, str] | None:
    """(tool name, query argument) of the configured search server."""
    if settings.exa_api_key:
        return settings.mcp_exa_tool, "query"
    if settings.tavily_api_key:
        return settings.mcp_tavily_tool, "query"
    return None


def result_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into text (structured content as JSON)."""
    structured = getattr(result, "structuredContent", None)
    if structured:
        inner = structured.get("result", structured) if isinstance(structured, dict) else structured
        return inner if isinstance(inner, str) else json.dumps(inner)
    parts = []
    for c in getattr(result, "content", []) or []:
        if getattr(c, "type", None) == "text":
            parts.append(c.text)
    return "\n".join(parts)


class MCPClient:
    def __init__(self, settings: Settings):
        self.specs = server_specs(settings)
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}

    def has(self, server: str) -> bool:
        return server in self.specs

    async def __aenter__(self) -> MCPClient:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        self._sessions.clear()
        if self._stack is not None:
            try:
                await self._stack.__aexit__(*exc)
            except Exception as e:  # a server that already died
                log.debug("mcp shutdown: %s", e)
            self._stack = None

    async def _session(self, server: str):
        if server in self._sessions:
            return self._sessions[server]
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        spec = self.specs[server]
        argv = shlex.split(spec.command)
        params = StdioServerParameters(command=argv[0], args=argv[1:],
                                       env={**os.environ, **(spec.env or {})})
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[server] = session
        return session

    async def call(self, server: str, tool: str, args: dict[str, Any]) -> str:
        if server not in self.specs:
            raise KeyError(f"MCP server {server!r} is not configured")
        session = await self._session(server)
        result = await session.call_tool(tool, args)
        if getattr(result, "isError", False):
            raise RuntimeError(f"{server}.{tool} failed: {result_text(result)[:300]}")
        return result_text(result)
