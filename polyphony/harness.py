"""Run an ensemble on a seed, in isolation, until it goes quiet.

Shared by the evals and the self-play simulator. Each run gets its own
database (created if missing) and its own Redis namespace, so it never sees or
disturbs a live room on the same servers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from .config import Settings
from .db import Base, init_db, make_engine, make_sessionmaker
from .stream import Stream

log = logging.getLogger(__name__)

RESET_TABLES = ["thoughts", "listens", "cadences", "voices", "senses", "frames", "sense_log"]


async def ensure_database(url: str) -> None:
    """CREATE DATABASE if it does not exist (connects to the server's `postgres` db)."""
    u = make_url(url)
    admin = create_async_engine(u.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(text("SELECT 1 FROM pg_database WHERE datname=:n"),
                                       {"n": u.database})
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{u.database}"'))
    finally:
        await admin.dispose()


def isolated(settings: Settings, suffix: str, **overrides) -> Settings:
    """A copy of `settings` pointing at `<db>_<suffix>` and namespace `<ns>-<suffix>`."""
    u = make_url(settings.database_url)
    return settings.model_copy(update={
        "database_url": u.set(database=f"{u.database}_{suffix}").render_as_string(
            hide_password=False),
        "namespace": f"{settings.namespace}-{suffix}",
        "run_senses": False, "run_digest": False,
        **overrides,
    })


@dataclass
class Run:
    settings: Settings
    stream: Stream
    engine: Any
    redis: Any
    memory: Any = None
    started: float = field(default_factory=time.monotonic)

    async def close(self) -> None:
        await self.redis.aclose()
        await self.engine.dispose()


async def open_run(settings: Settings, *, keep_samples: bool = True) -> Run:
    from .main import install_tracer, seed
    from .memory import Memory

    await ensure_database(settings.database_url)
    engine = make_engine(settings.database_url)
    await init_db(engine, embed_dim=settings.embed_dim)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("TRUNCATE " + ", ".join(RESET_TABLES) + " RESTART IDENTITY CASCADE"))
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    async for k in r.scan_iter(match=f"{settings.namespace}:*"):
        await r.delete(k)
    stream = Stream(make_sessionmaker(engine), r, window=settings.recent_window,
                    namespace=settings.namespace)
    await seed(stream, settings)
    install_tracer(stream)
    memory = Memory(stream, settings)
    return Run(settings, stream, engine, r, memory)


async def wait_quiet(stream: Stream, *, idle: float, timeout: float,
                     max_thoughts: int | None = None) -> tuple[list[dict], str]:
    """Wait until nothing is appended and nobody forms for `idle` seconds.

    Returns (thoughts, how): how is "quiet", "timeout" or "budget".
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last, since = None, loop.time()
    while True:
        ts = await stream.thoughts(limit=5000)
        if max_thoughts is not None and len(ts) >= max_thoughts:
            return ts, "budget"
        now = (len(ts), tuple(await stream.forming()), await stream.cadence_running())
        if now != last:
            last, since = now, loop.time()
        elif not now[1] and not now[2] and loop.time() - since >= idle:
            return ts, "quiet"
        if loop.time() >= deadline:
            return ts, "timeout"
        await asyncio.sleep(0.5)


async def play(run: Run, seeds: list[str], *, idle: float = 20, timeout: float = 600,
               max_thoughts: int | None = None, gap: float = 0.0) -> tuple[list[dict], str]:
    """Start the voices, drop the seed thoughts in, wait for quiet, stop."""
    from .supervisor import Supervisor

    sup = Supervisor(run.stream, run.settings, memory=run.memory)
    task = asyncio.create_task(sup.run())
    try:
        await asyncio.sleep(1.0)  # let voices subscribe
        for s in seeds:
            await run.stream.append("person", "person", s)
            if gap:
                await asyncio.sleep(gap)
        return await wait_quiet(run.stream, idle=idle, timeout=timeout, max_thoughts=max_thoughts)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
