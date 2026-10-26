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

from .api import create_app
from .config import Settings, get_settings
from .db import SenseRow, VoiceRow, init_db, make_engine, make_sessionmaker
from .stream import Stream
from .supervisor import Supervisor
from .voice import VoiceSpec

log = logging.getLogger("polyphony")


async def seed(stream: Stream, settings: Settings) -> None:
    """Load `voices/` and `senses.yaml` into the DB, but only into an empty table.

    The DB is the record of truth: once seeded, deleting a voice in the UI
    keeps it deleted across restarts.
    """
    async with stream.sessions() as s:
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
                                   every_minutes=int(entry.get("every_minutes", 240))))
                    n += 1
            log.info("seeded %d sense(s) from %s", n, settings.senses_file)
        await s.commit()


def build(settings: Settings | None = None):
    settings = settings or get_settings()
    engine = make_engine(settings.database_url)
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    stream = Stream(make_sessionmaker(engine), redis, window=settings.recent_window)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await init_db(engine)
        await seed(stream, settings)
        supervisor = Supervisor(stream, settings)
        task = asyncio.create_task(supervisor.run(), name="supervisor")
        app.state.supervisor = supervisor
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await redis.aclose()
            await engine.dispose()

    return create_app(stream, settings, lifespan=lifespan)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    uvicorn.run(build(settings), host="0.0.0.0", port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
