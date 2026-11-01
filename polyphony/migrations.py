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
    ("conscious kind",
     "UPDATE thoughts SET kind = 'conscious' WHERE kind IN ('person', 'theme')"),
    ("conscious author", "UPDATE thoughts SET author = 'conscious' WHERE author = 'person'"),
    ("senses.feeds",
     "ALTER TABLE senses ADD COLUMN IF NOT EXISTS feeds JSONB NOT NULL DEFAULT '[]'"),
    ("senses.search",
     "ALTER TABLE senses ADD COLUMN IF NOT EXISTS search BOOLEAN NOT NULL DEFAULT false"),
    ("llm_calls.queue_ms", "ALTER TABLE llm_calls DROP COLUMN IF EXISTS queue_ms"),
    ("drop listen_samples (no local listen)", "DROP TABLE IF EXISTS listen_samples"),
    ("drop thought_embeddings", "DROP TABLE IF EXISTS thought_embeddings"),
    ("senses.daily_at",
     "ALTER TABLE senses ADD COLUMN IF NOT EXISTS daily_at VARCHAR(5) NOT NULL DEFAULT '04:00'"),
    ("senses.every_minutes", "ALTER TABLE senses DROP COLUMN IF EXISTS every_minutes"),
    ("cadences.options",
     "ALTER TABLE cadences ADD COLUMN IF NOT EXISTS options JSONB NOT NULL DEFAULT '[]'"),
    ("drop keys (now frames, reseeded from mind/frame.md)", "DROP TABLE IF EXISTS keys"),
]


async def run_migrations(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        for name, sql in STEPS:
            await conn.execute(text(sql))
