"""A sense: a standing instruction plus the feeds the person chose.

On a schedule, a sense runs one workflow (sense_graph.py): read the feeds over
MCP, let the model pick what matters, read the articles, summarise, and drop
each new item into the stream as a `world` thought. Items already seen (by URL
or by title) are never appended twice. Works with any provider: nothing here
depends on a vendor's built-in web search.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from .config import Settings
from .db import SenseLog, SenseRow
from .stream import Stream

log = logging.getLogger(__name__)

MIN_EVERY_MINUTES = 30
FIRST_RUN_DELAY = 10  # seconds after start, for a sense that has never run


@dataclass(frozen=True)
class SenseSpec:
    name: str
    instruction: str
    every_minutes: int
    feeds: tuple[str, ...] = ()
    search: bool = False

    @classmethod
    def from_row(cls, row: SenseRow) -> SenseSpec:
        return cls(row.name, row.instruction, max(MIN_EVERY_MINUTES, row.every_minutes),
                   tuple(row.feeds or ()), bool(row.search))

    def fingerprint(self) -> str:
        blob = json.dumps([self.instruction, self.every_minutes, list(self.feeds), self.search])
        return hashlib.sha256(blob.encode()).hexdigest()


def fingerprints(item: dict[str, Any]) -> list[str]:
    """One fingerprint per identifying field, so a headline is caught by URL *or* title."""
    out = []
    url = str(item.get("url") or "").strip().lower()
    url = re.sub(r"^https?://(www\.)?", "", url).rstrip("/")
    if url:
        out.append(hashlib.sha256(("url:" + url).encode()).hexdigest())
    title = re.sub(r"\s+", " ", str(item.get("title") or "").strip().lower())
    if title:
        out.append(hashlib.sha256(("title:" + title).encode()).hexdigest())
    return out


class Sense:
    def __init__(self, spec: SenseSpec, stream: Stream, settings: Settings):
        self.spec = spec
        self.stream = stream
        self.settings = settings

    @property
    def name(self) -> str:
        return self.spec.name

    async def _last_run(self) -> datetime | None:
        async with self.stream.sessions() as s:
            row = await s.get(SenseRow, self.name)
            return row.last_run if row else None

    async def run(self) -> None:
        while True:
            last = await self._last_run()
            every = timedelta(minutes=self.spec.every_minutes)
            if last is None:
                wait = FIRST_RUN_DELAY
            else:
                wait = max(0.0, (last + every - datetime.now(timezone.utc)).total_seconds())
            await asyncio.sleep(wait)
            try:
                await self.fetch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("sense %s: fetch failed", self.name)
                await self._touch()  # don't hammer a failing provider; try next period

    async def _touch(self) -> None:
        async with self.stream.sessions() as s:
            await s.execute(update(SenseRow).where(SenseRow.name == self.name)
                            .values(last_run=datetime.now(timezone.utc)))
            await s.commit()

    async def fetch_once(self, mcp=None) -> list[dict[str, Any]]:
        """One run of the sense workflow. Returns the thoughts appended.

        `mcp` is anything with `has()` and `call()`; default: real MCP servers over stdio.
        """
        from . import sense_graph
        from .frame import get_frame
        from .mcp_client import MCPClient, search_tool

        frame = await get_frame(self.stream.sessions, self.settings.ensemble)
        if not self.spec.feeds:
            log.info("sense %s: no feeds configured", self.name)
            await self._touch()
            return []

        async def go(client):
            graph = sense_graph.build(sense=self, settings=self.settings, frame=frame, mcp=client,
                                      search=search_tool(self.settings) if client.has("search")
                                      else None)
            return await graph.ainvoke({})

        if mcp is None:
            async with MCPClient(self.settings) as client:
                state = await go(client)
        else:
            state = await go(mcp)
        appended = state.get("appended") or []
        await self._touch()
        log.info("sense %s: %d new item(s)", self.name, len(appended))
        return appended

    async def seen(self, item: dict[str, Any]) -> bool:
        fps = fingerprints(item)
        return not fps or await self._seen(fps)

    async def mark_seen(self, item: dict[str, Any]) -> None:
        fps = fingerprints(item)
        if fps:
            await self._claim(fps)

    async def emit(self, item: dict[str, Any]) -> dict[str, Any] | None:
        """Dedup and append one item as a `world` thought. Returns it, or None if seen."""
        if not (item.get("title") or item.get("summary")):
            return None
        fps = fingerprints(item)
        if not fps or await self._seen(fps):
            return None
        title = str(item.get("title") or "").strip()
        summary = str(item.get("summary") or "").strip()
        content = f"{title}\n{summary}".strip()
        meta = {"url": item.get("url"), "title": title, "sense": self.name}
        # Claim the fingerprints first; if another process beat us, skip.
        if not await self._claim(fps):
            return None
        t = await self.stream.append(self.name, "world", content, meta=meta)
        await self._attach_seq(fps, t["seq"])
        return t

    async def _seen(self, fps: list[str]) -> bool:
        async with self.stream.sessions() as s:
            hit = await s.scalar(select(SenseLog.id).where(SenseLog.fingerprint.in_(fps)).limit(1))
            return hit is not None

    async def _claim(self, fps: list[str]) -> bool:
        async with self.stream.sessions() as s:
            res = await s.execute(
                insert(SenseLog)
                .values([{"sense": self.name, "fingerprint": fp} for fp in fps])
                .on_conflict_do_nothing(index_elements=["fingerprint"])
                .returning(SenseLog.id)
            )
            claimed = len(res.fetchall())
            await s.commit()
            return claimed == len(fps)

    async def _attach_seq(self, fps: list[str], seq: int) -> None:
        async with self.stream.sessions() as s:
            await s.execute(update(SenseLog).where(SenseLog.fingerprint.in_(fps)).values(seq=seq))
            await s.commit()
