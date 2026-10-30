"""Embed every thought that has no embedding yet (e.g. after turning memory on).

    python scripts/backfill_embeddings.py
"""

from __future__ import annotations

import asyncio

import redis.asyncio as aioredis

from polyphony.config import get_settings
from polyphony.db import init_db, make_engine, make_sessionmaker
from polyphony.memory import Memory
from polyphony.stream import Stream


async def main() -> None:
    settings = get_settings()
    engine = make_engine(settings.database_url)
    await init_db(engine, embed_dim=settings.embed_dim)
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    memory = Memory(Stream(make_sessionmaker(engine), r, namespace=settings.namespace), settings)
    if not memory.enabled:
        raise SystemExit("EMBED_PROVIDER is empty: memory is off")
    total = 0
    while n := await memory.embed_pending():
        total += n
        print(f"embedded {total}")
    await r.aclose()
    await engine.dispose()
    print(f"done: {total}")


if __name__ == "__main__":
    asyncio.run(main())
