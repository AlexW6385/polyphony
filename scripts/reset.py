#!/usr/bin/env python3
"""Clear the mind's history: every thought, listen, vote, digest, memory and trace, and the
stream state in Redis. Voices, senses, the frame and saved settings stay.

Stop the server first (it keeps stream state in memory), then:

    python scripts/reset.py            # history only
    python scripts/reset.py --all      # also voices and senses: they reseed from mind/ on start
    python scripts/reset.py --yes      # skip the confirmation

Uses DATABASE_URL, REDIS_URL and NAMESPACE from the environment / .env, like the server.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis.asyncio as aioredis  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from polyphony.config import get_settings  # noqa: E402

HISTORY = ["thoughts", "listens", "cadences", "sense_log", "llm_calls", "thought_embeddings"]
# LangGraph's checkpoint tables (cadence resume state), if the Postgres saver created them
CHECKPOINTS = ["checkpoints", "checkpoint_blobs", "checkpoint_writes"]
CONFIG = ["voices", "senses"]


async def main(everything: bool) -> None:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    wanted = HISTORY + CHECKPOINTS + (CONFIG if everything else [])
    async with engine.begin() as conn:
        existing = set((await conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = current_schema()"))).scalars())
        tables = [t for t in wanted if t in existing]
        if tables:
            await conn.execute(text(f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE"))
    await engine.dispose()
    print("truncated:", ", ".join(tables) or "(nothing)")

    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    n = 0
    async for k in r.scan_iter(match=f"{settings.namespace}:*"):
        await r.delete(k)
        n += 1
    await r.aclose()
    print(f"redis: deleted {n} key(s) under {settings.namespace}:*")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--all", action="store_true", help="also clear voices and senses")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    a = p.parse_args()
    if not a.yes and input("Clear all history? Stop the server first. [y/N] ").lower() != "y":
        sys.exit("cancelled")
    asyncio.run(main(a.all))
