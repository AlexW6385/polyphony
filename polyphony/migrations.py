"""Ordered, idempotent SQL steps, run by `init_db` after `create_all`.

`create_all` creates tables that do not exist yet; it never alters existing
ones. Every change to an existing table (or anything `create_all` cannot
express) is a step here. Each step must be safe to run again.

Optional steps (pgvector) log and continue when unavailable, so a plain
PostgreSQL still runs everything except memory.
"""

from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

log = logging.getLogger(__name__)

STEPS: list[tuple[str, str]] = [
    ("person kind", "UPDATE thoughts SET kind = 'person' WHERE kind = 'conscious'"),
    ("person author", "UPDATE thoughts SET author = 'person' WHERE author = 'conscious'"),
    ("senses.feeds",
     "ALTER TABLE senses ADD COLUMN IF NOT EXISTS feeds JSONB NOT NULL DEFAULT '[]'"),
    ("senses.search",
     "ALTER TABLE senses ADD COLUMN IF NOT EXISTS search BOOLEAN NOT NULL DEFAULT false"),
    ("llm_calls.queue_ms", "ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS queue_ms INTEGER"),
]

OPTIONAL_STEPS: list[tuple[str, str]] = [
    ("pgvector", "CREATE EXTENSION IF NOT EXISTS vector"),
    ("thought_embeddings", """
        CREATE TABLE IF NOT EXISTS thought_embeddings (
            seq BIGINT PRIMARY KEY REFERENCES thoughts(seq) ON DELETE CASCADE,
            model TEXT NOT NULL,
            embedding vector({dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )"""),
    ("thought_embeddings hnsw", """
        CREATE INDEX IF NOT EXISTS ix_thought_embeddings_hnsw
        ON thought_embeddings USING hnsw (embedding vector_cosine_ops)"""),
]


async def run_migrations(engine: AsyncEngine, embed_dim: int = 768) -> dict[str, bool]:
    """Run every step; return which optional ones succeeded."""
    async with engine.begin() as conn:
        for name, sql in STEPS:
            await conn.execute(text(sql))
    ok: dict[str, bool] = {}
    for name, sql in OPTIONAL_STEPS:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(sql.format(dim=int(embed_dim))))
            ok[name] = True
        except Exception as e:
            log.warning("optional migration %r skipped: %s", name, str(e).splitlines()[0])
            ok[name] = False
    return ok


async def has_vector(engine: AsyncEngine) -> bool:
    async with engine.connect() as conn:
        n = await conn.scalar(text(
            "SELECT count(*) FROM information_schema.tables WHERE table_name='thought_embeddings'"))
    return bool(n)
