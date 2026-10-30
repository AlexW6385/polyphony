"""Tests run against real PostgreSQL and Redis (no mocks for the parts that
matter: ordering, pub/sub, dedup), and a scripted fake LLM provider.

    TEST_DATABASE_URL=postgresql+asyncpg://postgres@localhost:5432/polyphony_test
    TEST_REDIS_URL=redis://localhost:6379/15
"""

from __future__ import annotations

import os
import re

import pytest
import redis.asyncio as aioredis

from polyphony import llm
from polyphony.config import Settings
from polyphony.db import Base, make_engine, make_sessionmaker
from polyphony.stream import Stream

DB_URL = os.environ.get("TEST_DATABASE_URL",
                        "postgresql+asyncpg://postgres@localhost:5432/polyphony_test")
REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/15")


class FakeProvider:
    """A deterministic stand-in for a model. `script(system, user, **kw) -> str`."""

    def __init__(self):
        self.calls: list[dict] = []
        self.script = None

    async def complete(self, *, model, system, user, json=False, schema=None, temperature=0.8,
                       max_tokens=1024):
        self.calls.append({"model": model, "system": system, "user": user, "json": json,
                           "schema": schema})
        return await self.script(system=system, user=user, json=json or bool(schema))


def voice_name(system: str) -> str:
    m = re.search(r'Your name is "([^"]+)"', system)
    return m.group(1) if m else "?"


@pytest.fixture
def fake():
    f = FakeProvider()
    llm.register_provider("fake", lambda base_url, api_key: f)
    return f


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=DB_URL, redis_url=REDIS_URL,
        listen_provider="fake", listen_model="fake-listen",
        speak_provider="fake", speak_model="fake-speak",
        ensembles_dir=tmp_path / "ensembles", namespace="ptest",
        embed_provider="", teacher_sample_rate=0.0,
        run_senses=False, run_memory=False, run_digest=False,
        _env_file=None,
    )


@pytest.fixture
async def stream(settings):
    engine = make_engine(DB_URL)
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:  # pragma: no cover
        await engine.dispose()
        pytest.skip(f"PostgreSQL not available at {DB_URL}: {e}")
    r = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await r.flushdb()
    except Exception as e:  # pragma: no cover
        await engine.dispose()
        pytest.skip(f"Redis not available at {REDIS_URL}: {e}")
    s = Stream(make_sessionmaker(engine), r, window=settings.recent_window,
               namespace=settings.namespace)
    yield s
    await r.flushdb()
    await r.aclose()
    await engine.dispose()
