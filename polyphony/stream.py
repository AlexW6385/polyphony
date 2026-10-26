"""The stream: the only shared thing.

PostgreSQL decides the order of record (`seq`). Redis holds a cache of the
recent window, the set of voices currently forming, every voice's latest
listen, and the pub/sub channels that wake voices up. Nothing here is held
in process memory, so voices can live in separate processes later.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from .db import Listen, Thought

log = logging.getLogger(__name__)

K_THOUGHTS = "polyphony:thoughts"  # pub/sub; payload = thought JSON
K_RECENT = "polyphony:recent"  # sorted set of thought JSON scored by seq, trimmed to the window
K_FORMING = "polyphony:forming"  # set of voice names currently composing
K_LISTEN = "polyphony:listen:"  # + voice name -> latest listen JSON
K_CONTROL = "polyphony:control"  # pub/sub; {"voices": "changed"} / {"senses": "changed"}
K_CADENCE = "polyphony:cadence:running"  # set while a cadence is running (with a TTL)

KINDS = {"voice", "conscious", "world", "cadence"}


class Stream:
    def __init__(self, sessions: async_sessionmaker, redis: aioredis.Redis, window: int = 40):
        self.sessions = sessions
        self.redis = redis
        self.window = window

    # ---------------------------------------------------------------- thoughts

    async def append(
        self,
        author: str,
        kind: str,
        content: str,
        mood: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit to PostgreSQL first (that assigns `seq`), then cache + publish."""
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        async with self.sessions() as s:
            t = Thought(author=author, kind=kind, content=content, mood=mood, meta=meta)
            s.add(t)
            await s.commit()
            await s.refresh(t)
            thought = t.to_dict()
        payload = json.dumps(thought)
        # A sorted set scored by seq: if two voices commit 5 and 6 but push 6
        # first, trimming still drops the oldest *seq*, not the earliest push.
        async with self.redis.pipeline(transaction=False) as p:
            p.zadd(K_RECENT, {payload: thought["seq"]})
            p.zremrangebyrank(K_RECENT, 0, -self.window - 1)
            p.publish(K_THOUGHTS, payload)
            await p.execute()
        return thought

    async def recent(self) -> list[dict[str, Any]]:
        """The recent window, ordered by `seq` (pushes can land out of order)."""
        raw = await self.redis.zrange(K_RECENT, 0, -1)
        if not raw:
            return await self._refill()
        by_seq = {t["seq"]: t for t in map(json.loads, raw)}
        return [by_seq[k] for k in sorted(by_seq)][-self.window :]

    async def _refill(self) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (
                await s.scalars(select(Thought).order_by(desc(Thought.seq)).limit(self.window))
            ).all()
        thoughts = [r.to_dict() for r in reversed(rows)]
        if thoughts:
            async with self.redis.pipeline(transaction=True) as p:
                p.zadd(K_RECENT, {json.dumps(t): t["seq"] for t in thoughts})
                p.zremrangebyrank(K_RECENT, 0, -self.window - 1)
                await p.execute()
        return thoughts

    async def strays(self, before_seq: int, n: int) -> list[dict[str, Any]]:
        """A few random older thoughts, for free association."""
        if n <= 0 or before_seq <= 1:
            return []
        async with self.sessions() as s:
            rows = (
                await s.scalars(
                    select(Thought)
                    .where(Thought.seq < before_seq)
                    .order_by(func.random())
                    .limit(n)
                )
            ).all()
        return sorted((r.to_dict() for r in rows), key=lambda t: t["seq"])

    async def thoughts(self, after: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (
                await s.scalars(
                    select(Thought).where(Thought.seq > after).order_by(Thought.seq).limit(limit)
                )
            ).all()
        return [r.to_dict() for r in rows]

    async def subscribe(self, ready: asyncio.Event | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yield every new thought as it is appended (any process).

        `ready` is set once the subscription is live, so callers can be sure
        nothing appended after that point is missed.
        """
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(K_THOUGHTS)
        if ready is not None:
            ready.set()
        try:
            async for msg in pubsub.listen():
                if msg.get("type") == "message":
                    yield json.loads(msg["data"])
        finally:
            await pubsub.unsubscribe(K_THOUGHTS)
            await pubsub.aclose()

    # ---------------------------------------------------------------- listens

    async def record_listen(self, voice: str, trigger_seq: int, decision: dict[str, Any]) -> dict:
        async with self.sessions() as s:
            row = Listen(
                voice=voice,
                trigger_seq=trigger_seq,
                speak=bool(decision.get("speak")),
                mood=str(decision.get("mood") or "")[:80],
                reason=str(decision.get("reason") or ""),
                pause_seconds=float(decision.get("pause_seconds") or 0),
            )
            s.add(row)
            await s.commit()
            await s.refresh(row)
            d = row.to_dict()
        await self.redis.set(K_LISTEN + voice, json.dumps(d))
        return d

    async def listens_latest(self) -> dict[str, dict[str, Any]]:
        """Every voice's latest listen, keyed by voice name."""
        keys = [k async for k in self.redis.scan_iter(match=K_LISTEN + "*")]
        if not keys:
            return {}
        values = await self.redis.mget(keys)
        out = {}
        for v in values:
            if v:
                d = json.loads(v)
                out[d["voice"]] = d
        return dict(sorted(out.items()))

    async def listens(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self.sessions() as s:
            rows = (await s.scalars(select(Listen).order_by(desc(Listen.id)).limit(limit))).all()
        return [r.to_dict() for r in rows]

    async def forget_voice(self, voice: str) -> None:
        await self.redis.delete(K_LISTEN + voice)
        await self.redis.srem(K_FORMING, voice)

    # ---------------------------------------------------------------- forming

    async def set_forming(self, voice: str, on: bool) -> None:
        if on:
            await self.redis.sadd(K_FORMING, voice)
        else:
            await self.redis.srem(K_FORMING, voice)

    async def forming(self) -> list[str]:
        return sorted(await self.redis.smembers(K_FORMING))

    # ---------------------------------------------------------------- control

    async def publish_control(self, what: str) -> None:
        await self.redis.publish(K_CONTROL, json.dumps({what: "changed"}))

    async def cadence_running(self) -> bool:
        async for _ in self.redis.scan_iter(match=K_CADENCE + ":*", count=100):
            return True
        return False

    async def set_cadence_running(self, cadence_id: int, on: bool, ttl: int = 900) -> None:
        """One key per running cadence; the TTL means a crash can't freeze the mind."""
        key = f"{K_CADENCE}:{cadence_id}"
        if on:
            await self.redis.set(key, "1", ex=ttl)
        else:
            await self.redis.delete(key)
