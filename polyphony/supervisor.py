"""Supervisor: keeps one task per enabled voice and per enabled sense.

It reads the `voices` and `senses` tables, starts a task per row, and listens
on the control channel. On a change it diffs the tables against what is
running and starts, stops, or restarts (text changed) the affected tasks.
This is what makes "add a voice from the UI" work without a restart, and it
works across processes because the signal goes through Redis.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import select

from .cadence import propose_cadence
from .config import Settings
from .db import SenseRow
from .sense import Sense, SenseSpec
from .stream import K_CONTROL, Stream
from .voice import Voice, load_voice_specs

log = logging.getLogger(__name__)

RESYNC_SECONDS = 30  # safety net in case a control message is missed


@dataclass
class _Running:
    task: asyncio.Task
    fingerprint: str


class Supervisor:
    def __init__(self, stream: Stream, settings: Settings, *, run_senses: bool = True):
        self.stream = stream
        self.settings = settings
        self.run_senses = run_senses
        self.voices: dict[str, _Running] = {}
        self.senses: dict[str, _Running] = {}
        self._lock = asyncio.Lock()

    async def _propose(self, proposer: str, resolution: str):
        try:
            return await propose_cadence(self.stream, self.settings, proposer, resolution)
        except Exception:
            log.exception("cadence proposal by %s failed", proposer)

    async def sync(self) -> None:
        async with self._lock:
            specs = {s.name: s for s in await load_voice_specs(self.stream.sessions, self.settings)}
            await self._diff(
                self.voices,
                {n: s.fingerprint() for n, s in specs.items()},
                lambda n: Voice(specs[n], self.stream, self.settings, self._propose).run(),
                "voice",
            )
            if self.run_senses:
                async with self.stream.sessions() as s:
                    rows = (await s.scalars(select(SenseRow).where(SenseRow.enabled))).all()
                sspecs = {r.name: SenseSpec.from_row(r) for r in rows}
                await self._diff(
                    self.senses,
                    {n: s.fingerprint() for n, s in sspecs.items()},
                    lambda n: Sense(sspecs[n], self.stream, self.settings).run(),
                    "sense",
                )

    async def _diff(self, running: dict[str, _Running], wanted: dict[str, str], make, kind: str):
        for name in list(running):
            r = running[name]
            if name not in wanted or wanted[name] != r.fingerprint or r.task.done():
                r.task.cancel()
                await asyncio.gather(r.task, return_exceptions=True)
                del running[name]
                if name not in wanted:
                    log.info("stopped %s %s", kind, name)
        for name, fp in wanted.items():
            if name not in running:
                task = asyncio.create_task(make(name), name=f"{kind}:{name}")
                running[name] = _Running(task, fp)
                log.info("started %s %s", kind, name)

    async def run(self) -> None:
        await self.sync()
        pubsub = self.stream.redis.pubsub()
        await pubsub.subscribe(K_CONTROL)
        try:
            while True:
                try:
                    msg = await pubsub.get_message(ignore_subscribe_messages=True,
                                                   timeout=RESYNC_SECONDS)
                except Exception:
                    log.exception("control channel error")
                    await asyncio.sleep(1)
                    msg = None
                if msg:
                    log.info("control: %s", msg.get("data"))
                await self.sync()  # on a control message, or periodically as a safety net
        finally:
            await pubsub.unsubscribe(K_CONTROL)
            await pubsub.aclose()
            await self.stop()

    async def stop(self) -> None:
        tasks = [r.task for r in (*self.voices.values(), *self.senses.values())]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.voices.clear()
        self.senses.clear()
