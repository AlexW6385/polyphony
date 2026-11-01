"""Ordered, idempotent SQL steps, run by `init_db` after `create_all`.

`create_all` creates tables that do not exist yet; it never alters existing
ones. Every change to an existing table (or anything `create_all` cannot
express) is a step here. Each step must be safe to run again.

"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger(__name__)

STEPS: list[tuple[str, str]] = [
    ("theme kind",
     "UPDATE thoughts SET kind = 'theme' WHERE kind IN ('person', 'conscious')"),
    ("conscious author", "UPDATE thoughts SET author = 'conscious' WHERE author = 'person'"),
    ("senses.feeds",
     "ALTER TABLE senses ADD COLUMN IF NOT EXISTS feeds JSONB NOT NULL DEFAULT '[]'"),
    ("senses.search",
     "ALTER TABLE senses ADD COLUMN IF NOT EXISTS search BOOLEAN NOT NULL DEFAULT false"),
    ("llm_calls.queue_ms", "ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS queue_ms INTEGER"),
    ("drop thought_embeddings", "DROP TABLE IF EXISTS thought_embeddings"),
    ("drop frames (now keys, reseeded from mind/key.md)", "DROP TABLE IF EXISTS frames"),
    ("listen_samples.ensemble",
     "ALTER TABLE listen_samples DROP COLUMN IF EXISTS ensemble"),
]


async def run_migrations(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for name, sql in STEPS:
            await conn.execute(text(sql))
