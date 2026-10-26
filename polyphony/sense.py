"""A sense: a standing instruction for what to bring in from the outside world.

On a schedule, a sense asks a model with web search to fetch and summarize,
then drops each new item into the stream as a `world` thought. Items already
seen (by URL or by title) are never appended twice.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from . import llm, prompts
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

    @classmethod
    def from_row(cls, row: SenseRow) -> SenseSpec:
        return cls(row.name, row.instruction, max(MIN_EVERY_MINUTES, row.every_minutes))

    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.instruction}\0{self.every_minutes}".encode()).hexdigest()


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

    async def fetch_once(self) -> list[dict[str, Any]]:
        """Fetch, dedup, append. Returns the thoughts appended."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        reply = await llm.search_json(
            self.settings.sense_tier,
            prompts.SENSE_SYSTEM,
            prompts.sense_user(self.spec.instruction, today),
        )
        items = reply.get("items") or []
        if not isinstance(items, list):
            items = []
        appended = []
        for item in items[:3]:
            if not isinstance(item, dict) or not (item.get("title") or item.get("summary")):
                continue
            fps = fingerprints(item)
            if not fps or await self._seen(fps):
                continue
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            content = f"{title}\n{summary}".strip()
            meta = {"url": item.get("url"), "title": title, "sense": self.name}
            # Claim the fingerprints first; if another process beat us, skip.
            if not await self._claim(fps):
                continue
            t = await self.stream.append(self.name, "world", content, meta=meta)
            await self._attach_seq(fps, t["seq"])
            appended.append(t)
        await self._touch()
        log.info("sense %s: %d new item(s)", self.name, len(appended))
        return appended

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
