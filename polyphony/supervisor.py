"""Supervisor: keeps one task per enabled voice and per enabled sense.

It reads the `voices` and `senses` tables, starts a task per row, and listens
on the control channel. On a change it diffs the tables against what is
running and starts, stops, or restarts (text changed) the affected tasks. A
frame edit changes every voice's fingerprint, so all voices restart with it.

It also runs the edge components that are not per-row: the memory embedder
and the digester.
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
from .digest import Digester
from .frame import get_frame
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
    def __init__(self, stream: Stream, settings: Settings, *, run_senses: bool | None = None,
                 memory=None):
        self.stream = stream
        self.settings = settings
        self.run_senses = settings.run_senses if run_senses is None else run_senses
        self.memory = memory
        self.voices: dict[str, _Running] = {}
        self.senses: dict[str, _Running] = {}
        self.edges: list[asyncio.Task] = []
        self._lock = asyncio.Lock()

    async def _propose(self, proposer: str, resolution: str):
        try:
            return await propose_cadence(self.stream, self.settings, proposer, resolution)
        except Exception:
            log.exception("cadence proposal by %s failed", proposer)

    async def sync(self) -> None:
        async with self._lock:
            frame = await get_frame(self.stream.sessions, self.settings.ensemble)
            ffp = repr(frame.to_dict())
            specs = {s.name: s for s in await load_voice_specs(self.stream.sessions, self.settings)}
            await self._diff(
                self.voices,
                {n: s.fingerprint() + ffp for n, s in specs.items()},
                lambda n: Voice(specs[n], self.stream, self.settings, self._propose, frame,
                                self.memory).run(),
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

    def _start_edges(self) -> None:
        if self.memory is not None and self.settings.run_memory and self.memory.enabled:
            self.edges.append(asyncio.create_task(self.memory.run(), name="memory"))
        if self.settings.run_digest:
            self.edges.append(asyncio.create_task(Digester(self.stream, self.settings).run(),
                                                  name="digest"))

    async def run(self) -> None:
        self._start_edges()
        await self.sync()
        pubsub = self.stream.redis.pubsub()
        await pubsub.subscribe(self.stream.key(K_CONTROL))
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
            await pubsub.unsubscribe(self.stream.key(K_CONTROL))
            await pubsub.aclose()
            await self.stop()

    async def stop(self) -> None:
        tasks = [r.task for r in (*self.voices.values(), *self.senses.values())] + self.edges
        self.edges = []
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.voices.clear()
        self.senses.clear()
