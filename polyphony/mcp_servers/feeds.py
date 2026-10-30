"""polyphony-feeds: an MCP server that reads RSS/Atom feeds.

Stateless, no keys. SSRF-guarded: only http(s), and the host must resolve to
public addresses (no loopback, private, link-local or reserved ranges), so a
feed URL can never be used to reach services on the machine or the LAN.

    python -m polyphony.mcp_servers.feeds        # stdio
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx
from mcp.server.fastmcp import FastMCP

MAX_BYTES = 5_000_000
TIMEOUT = 20.0

mcp = FastMCP("polyphony-feeds")


class BlockedURL(ValueError):
    pass


async def check_url(url: str) -> None:
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise BlockedURL(f"only http(s) URLs are allowed: {url}")
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80),
                                       type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise BlockedURL(f"cannot resolve {p.hostname}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise BlockedURL(f"{p.hostname} resolves to a non-public address ({ip})")


async def fetch_bytes(url: str) -> bytes:
    """GET with redirects checked hop by hop, a size cap and a timeout."""
    async with httpx.AsyncClient(timeout=TIMEOUT, follow_redirects=False,
                                 headers={"User-Agent": "polyphony-feeds/1"}) as c:
        for _ in range(5):
            await check_url(url)
            r = await c.get(url)
            if r.is_redirect and r.headers.get("location"):
                url = str(r.url.join(r.headers["location"]))
                continue
            r.raise_for_status()
            if len(r.content) > MAX_BYTES:
                raise ValueError("feed too large")
            return r.content
    raise ValueError("too many redirects")


def parse_feed(raw: bytes, limit: int) -> dict[str, Any]:
    d = feedparser.parse(raw)
    entries = []
    for e in d.entries[:limit]:
        link = e.get("link") or ""
        title = (e.get("title") or "").strip()
        guid = e.get("id") or link or title
        entries.append({
            "id": hashlib.sha1(guid.encode()).hexdigest()[:16],
            "title": title,
            "link": link,
            "summary": (e.get("summary") or "")[:1500],
            "published": e.get("published") or e.get("updated") or "",
        })
    return {"feed": (d.feed.get("title") or "").strip(), "entries": entries}


@mcp.tool()
async def fetch_feed(url: str, limit: int = 30) -> dict[str, Any]:
    """Fetch an RSS or Atom feed and return its newest entries (title, link, summary, date)."""
    raw = await fetch_bytes(url)
    return parse_feed(raw, max(1, min(100, limit)))


if __name__ == "__main__":
    mcp.run()
