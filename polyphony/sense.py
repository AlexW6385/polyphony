"""A sense: a standing instruction plus the feeds the conscious part chose.

Once a day at a fixed local time, a sense runs one workflow (sense_graph.py): read the feeds over
MCP, let the model pick what matters, read the articles, summarise, and drop
each new item into the score as a `world` thought. Items already seen (by URL
or by title) are never appended twice. Works with any provider: nothing here
depends on a vendor's built-in web search.

A fixed daily time (default 04:00) instead of an interval means a sense never runs just
because the room was resumed or restarted, and never cuts into a discussion at a random
moment. It can also be run once by hand (`POST /senses/{name}/run`).
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
from zoneinfo import ZoneInfo

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from .config import Settings
from .db import SenseLog, SenseRow
from .score import Score

log = logging.getLogger(__name__)

DEFAULT_DAILY_AT = "04:00"


def check_daily_at(value: str) -> str:
    """Normalise "H:MM" / "HH:MM" to "HH:MM"; ValueError if it is not a time of day."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(value or ""))
    if not m or int(m.group(1)) > 23 or int(m.group(2)) > 59:
        raise ValueError(f"not a time of day (HH:MM): {value!r}")
    return f"{int(m.group(1)):02d}:{m.group(2)}"


def next_run(daily_at: str, tz: str, now: datetime | None = None) -> datetime:
    """The next time `daily_at` (local to `tz`) comes round, strictly after `now`, in UTC."""
    zone = ZoneInfo(tz)
    local = (now or datetime.now(timezone.utc)).astimezone(zone)
    h, m = map(int, check_daily_at(daily_at).split(":"))
    target = local.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= local:
        target = (target + timedelta(days=1)).replace(hour=h, minute=m)
    return target.astimezone(timezone.utc)


@dataclass(frozen=True)
class SenseSpec:
    name: str
    instruction: str
    daily_at: str = DEFAULT_DAILY_AT
    feeds: tuple[str, ...] = ()
    search: bool = False

    @classmethod
    def from_row(cls, row: SenseRow) -> SenseSpec:
        return cls(row.name, row.instruction, row.daily_at or DEFAULT_DAILY_AT,
                   tuple(row.feeds or ()), bool(row.search))

    def fingerprint(self) -> str:
        blob = json.dumps([self.instruction, self.daily_at, list(self.feeds), self.search])
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
    def __init__(self, spec: SenseSpec, score: Score, settings: Settings):
        self.spec = spec
        self.score = score
        self.settings = settings

    @property
    def name(self) -> str:
        return self.spec.name

    async def run(self) -> None:
        while True:
            at = next_run(self.spec.daily_at, self.settings.timezone)
            await asyncio.sleep(max(0.0, (at - datetime.now(timezone.utc)).total_seconds()))
            try:
                await self.fetch_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("sense %s: fetch failed", self.name)
                await self._touch()

    async def _touch(self) -> None:
        async with self.score.sessions() as s:
            await s.execute(update(SenseRow).where(SenseRow.name == self.name)
                            .values(last_run=datetime.now(timezone.utc)))
            await s.commit()

    async def fetch_once(self, mcp=None) -> list[dict[str, Any]]:
        """One run of the sense workflow. Returns the thoughts appended.

        `mcp` is anything with `has()` and `call()`; default: real MCP servers over stdio.
        """
        from . import sense_graph
        from .key import get_key
        from .mcp_client import MCPClient, search_tool

        key = await get_key(self.score.sessions)
        if not self.spec.feeds:
            log.info("sense %s: no feeds configured", self.name)
            await self._touch()
            return []

        async def go(client):
            graph = sense_graph.build(sense=self, settings=self.settings, key=key, mcp=client,
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
        t = await self.score.append(self.name, "world", content, meta=meta)
        await self._attach_seq(fps, t["seq"])
        return t

    async def _seen(self, fps: list[str]) -> bool:
        async with self.score.sessions() as s:
            hit = await s.scalar(select(SenseLog.id).where(SenseLog.fingerprint.in_(fps)).limit(1))
            return hit is not None

    async def _claim(self, fps: list[str]) -> bool:
        async with self.score.sessions() as s:
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
        async with self.score.sessions() as s:
            await s.execute(update(SenseLog).where(SenseLog.fingerprint.in_(fps)).values(seq=seq))
            await s.commit()
