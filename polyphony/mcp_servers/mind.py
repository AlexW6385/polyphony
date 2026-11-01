"""Polyphony as an MCP server: any MCP-capable AI app can bring a question into
the stream and get back what the voices made of it.

Implemented over the HTTP API (API_URL), so it works against a running
instance in Docker or locally.

    python -m polyphony.mcp_servers.mind                 # stdio (Claude Desktop, Claude Code)
    python -m polyphony.mcp_servers.mind --http          # streamable HTTP on MCP_HTTP_HOST:PORT

There is no auth: HTTP binds to 127.0.0.1 by default. Exposing it further is
an explicit choice (MCP_HTTP_HOST), and anyone who can reach it can drop
thoughts into the stream.
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

from ..config import get_settings

settings = get_settings()
mcp = FastMCP(
    "polyphony",
    instructions=("The inner voices of one person, sharing one stream of thoughts. `ask` brings a "
                  "question in as the person's conscious part and returns at once; then call "
                  "`outcome` with its asked_seq, again and again, until finished is true (the "
                  "voices take a few minutes; each call waits up to about 50 s). `say` only drops "
                  "a thought in."),
    host=settings.mcp_http_host,
    port=settings.mcp_http_port,
)


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=settings.api_url, timeout=30)


OUTCOME_WAIT = 50  # seconds one `outcome` call waits at most: under any client's timeout


@mcp.tool()
async def ask(question: str) -> dict[str, Any]:
    """Bring a question in as conscious. Returns at once with its `asked_seq`.

    The voices then discuss it on their own, usually for a few minutes. Call
    `outcome(asked_seq)` until it says `finished`.
    """
    async with _client() as c:
        r = await c.post("/thoughts", json={"content": question, "kind": "conscious"})
        r.raise_for_status()
        asked = r.json()["seq"]
    return {"asked_seq": asked,
            "next": f"call outcome({asked}) until finished is true; the voices take a few minutes"}


@mcp.tool()
async def outcome(asked_seq: int) -> dict[str, Any]:
    """What the voices made of the question at `asked_seq`, waiting at most about 50 seconds.

    `finished` is true once the digest of the discussion is written (after the voices fall
    quiet): then it has whether it was decided, the outcome, each voice's position and what is
    still open. Until then it reports progress (thoughts so far, votes). Always has the votes.
    """
    deadline = time.monotonic() + OUTCOME_WAIT
    async with _client() as c:
        while True:
            r = await c.get("/thoughts", params={"after": asked_seq, "limit": 1000})
            r.raise_for_status()
            thoughts = r.json()
            digest = next((t for t in thoughts if t["kind"] == "digest"), None)
            if digest or time.monotonic() + 5 >= deadline:
                break
            await asyncio.sleep(5)
    votes = []
    for t in thoughts:
        m = t.get("meta") or {}
        if t["kind"] == "cadence" and m.get("options"):
            res = m.get("result") or {}
            votes.append({"question": m.get("question"), "options": m["options"],
                          "shares": res.get("shares"), "decided": res.get("decided"),
                          "winner": (m["options"][res["winner"]]
                                     if res.get("winner") is not None else None)})
    out: dict[str, Any] = {"asked_seq": asked_seq, "finished": digest is not None,
                           "voice_thoughts": sum(t["kind"] == "voice" for t in thoughts),
                           "votes": votes}
    summary = (digest.get("meta") or {}).get("summary") if digest else None
    if summary:
        out.update({k: summary.get(k) for k in ("decided", "outcome", "positions", "open")})
    elif digest:
        out["outcome"] = digest["content"]
    else:
        out["next"] = f"still discussing; call outcome({asked_seq}) again"
    return out


@mcp.tool()
async def say(content: str) -> dict[str, Any]:
    """Drop a thought into the stream as conscious, without waiting. Returns the recorded thought."""
    async with _client() as c:
        r = await c.post("/thoughts", json={"content": content, "kind": "conscious"})
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
async def list_voices() -> list[dict[str, Any]]:
    """The voices: name, enabled, whether listening or forming a thought now, last listen."""
    async with _client() as c:
        r = await c.get("/voices")
        r.raise_for_status()
        return [{"name": v["name"], "enabled": v["enabled"], "listening": v.get("listening", False), "forming": v["forming"],
                 "last_listen": v["last_listen"]} for v in r.json()]


def main() -> None:
    mcp.run("streamable-http" if "--http" in sys.argv else "stdio")


if __name__ == "__main__":
    main()
