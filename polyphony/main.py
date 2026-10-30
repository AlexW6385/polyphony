"""Wire everything together: database, Redis, stream, supervisor, API.

    python -m polyphony.main
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import redis.asyncio as aioredis
import uvicorn
import yaml
from sqlalchemy import func, select

from . import cadence_graph, llm
from .api import create_app
from .config import Settings, get_settings
from .db import FrameRow, LLMCall, SenseRow, VoiceRow, init_db, make_engine, make_sessionmaker
from .frame import Frame
from .memory import Memory
from .migrations import has_vector
from .stream import Stream
from .supervisor import Supervisor
from .voice import VoiceSpec

log = logging.getLogger("polyphony")


async def seed(stream: Stream, settings: Settings) -> None:
    """Load the ensemble's frame, voices and senses into the DB, only into empty tables.

    The DB is the record of truth: once seeded, deleting a voice in the UI
    keeps it deleted across restarts.
    """
    async with stream.sessions() as s:
        if await s.get(FrameRow, settings.ensemble) is None:
            f = Frame.load(settings.frame_file, settings.ensemble)
            s.add(FrameRow(name=f.name, label=f.label, person_label=f.person_label, text=f.text))
            log.info("seeded frame %s from %s", f.name, settings.frame_file)
        if not await s.scalar(select(func.count()).select_from(VoiceRow)):
            n = 0
            for path in sorted(settings.voices_dir.glob("*/voice.md")):
                spec = VoiceSpec.load(path)
                s.add(VoiceRow(name=spec.name, text=spec.text, overrides=spec.overrides or None))
                n += 1
            log.info("seeded %d voice(s) from %s", n, settings.voices_dir)
        if not await s.scalar(select(func.count()).select_from(SenseRow)):
            n = 0
            if settings.senses_file.exists():
                for entry in yaml.safe_load(settings.senses_file.read_text()) or []:
                    s.add(SenseRow(name=entry["name"], instruction=entry["instruction"],
                                   every_minutes=int(entry.get("every_minutes", 240)),
                                   feeds=list(entry.get("feeds") or []),
                                   search=bool(entry.get("search", False))))
                    n += 1
            log.info("seeded %d sense(s) from %s", n, settings.senses_file)
        await s.commit()


def install_tracer(stream: Stream) -> None:
    """Write one `llm_calls` row per model call, off the hot path."""

    async def trace(t: llm.Trace) -> None:
        try:
            async with stream.sessions() as s:
                s.add(LLMCall(tier=t.tier, provider=t.provider, model=t.model[:128],
                              purpose=t.purpose, voice=t.voice, latency_ms=t.latency_ms,
                              queue_ms=t.queue_ms, input_tokens=t.input_tokens,
                              output_tokens=t.output_tokens,
                              cached_input_tokens=t.cached_input_tokens, ok=t.ok, error=t.error))
                await s.commit()
        except Exception as e:  # tracing must never break a voice
            log.debug("trace dropped: %s", e)

    llm.set_tracer(trace)


def build(settings: Settings | None = None):
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    stream = Stream(make_sessionmaker(engine), redis, window=settings.recent_window,
                    namespace=settings.namespace)
    memory = Memory(stream, settings)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await init_db(engine, embed_dim=settings.embed_dim)
        if memory.enabled and not await has_vector(engine):
            log.warning("pgvector not available: memory is off")
            memory.enabled = False
        await seed(stream, settings)
        install_tracer(stream)
        await cadence_graph.setup(settings.database_url)
        supervisor = Supervisor(stream, settings, memory=memory)
        task = asyncio.create_task(supervisor.run(), name="supervisor")
        resume = asyncio.create_task(cadence_graph.resume_unfinished(stream, settings))
        app.state.supervisor = supervisor
        try:
            yield
        finally:
            task.cancel()
            resume.cancel()
            await asyncio.gather(task, resume, return_exceptions=True)
            llm.set_tracer(None)
            await cadence_graph.shutdown()
            await redis.aclose()
            await engine.dispose()

    return create_app(stream, settings, lifespan=lifespan, memory=memory)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    uvicorn.run(build(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
