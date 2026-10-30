"""Memory: older thoughts related to what is being said now.

- A background embedder writes an embedding for every thought
  (`thought_embeddings`, pgvector). It never sits on the append path.
- Retrieval embeds the last few thoughts of the window as the query and
  returns the top-k most similar thoughts *older than the window*.
- The query depends only on the stream state, not the voice, so the result is
  cached in Redis under the window's tail `seq`: N voices listening against
  one state cost one embedding and one search, not N.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from sqlalchemy import text

from . import llm
from .config import Settings
from .stream import K_MEMQ, Stream

log = logging.getLogger(__name__)

EMBED_BATCH = 32
POLL_SECONDS = 2.0
CACHE_TTL = 300


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class Memory:
    def __init__(self, stream: Stream, settings: Settings):
        self.stream = stream
        self.settings = settings
        self.tier = settings.embed_tier
        self.enabled = self.tier is not None
        self._locks: dict[int, asyncio.Lock] = {}

    # ------------------------------------------------------------------ embedder

    async def pending(self, limit: int = EMBED_BATCH) -> list[dict[str, Any]]:
        async with self.stream.sessions() as s:
            rows = (await s.execute(text("""
                SELECT t.seq, t.content FROM thoughts t
                LEFT JOIN thought_embeddings e ON e.seq = t.seq
                WHERE e.seq IS NULL ORDER BY t.seq LIMIT :n"""), {"n": limit})).all()
        return [{"seq": r[0], "content": r[1]} for r in rows]

    async def embed_pending(self) -> int:
        """Embed one batch of thoughts that have no embedding yet. Returns how many."""
        batch = await self.pending()
        if not batch:
            return 0
        vectors = await llm.embed(self.tier, [b["content"][:4000] for b in batch])
        async with self.stream.sessions() as s:
            for b, v in zip(batch, vectors):
                await s.execute(text("""
                    INSERT INTO thought_embeddings (seq, model, embedding)
                    VALUES (:seq, :model, CAST(:v AS vector))
                    ON CONFLICT (seq) DO NOTHING"""),
                    {"seq": b["seq"], "model": self.tier.model, "v": _vec(v)})
            await s.commit()
        return len(batch)

    async def run(self) -> None:
        """Background embedder. Backs off on errors; never blocks a voice."""
        if not self.enabled:
            return
        delay = POLL_SECONDS
        while True:
            try:
                n = await self.embed_pending()
                delay = 0.1 if n == EMBED_BATCH else POLL_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("embedder: %s", e)
                delay = min(60.0, delay * 2)
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------ retrieval

    async def recall(self, recent: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Memories for this stream state. Cached by tail seq; never raises."""
        if not self.enabled or not recent:
            return []
        tail = recent[-1]["seq"]
        key = self.stream.key(K_MEMQ) + str(tail)
        try:
            cached = await self.stream.redis.get(key)
            if cached is not None:
                return json.loads(cached)
            lock = self._locks.setdefault(tail, asyncio.Lock())
            async with lock:  # in-process: voices waking together share one query
                cached = await self.stream.redis.get(key)
                if cached is not None:
                    return json.loads(cached)
                out = await self._search(recent)
                await self.stream.redis.set(key, json.dumps(out), ex=CACHE_TTL)
            if len(self._locks) > 256:
                self._locks.clear()
            return out
        except Exception as e:
            log.warning("memory recall failed: %s", e)
            return []

    async def _search(self, recent: list[dict[str, Any]]) -> list[dict[str, Any]]:
        oldest_in_window = recent[0]["seq"]
        if oldest_in_window <= 1:
            return []
        query = "\n".join(t["content"] for t in recent[-self.settings.memory_query_last:])
        [v] = await llm.embed(self.tier, [query[:4000]])
        async with self.stream.sessions() as s:
            rows = (await s.execute(text("""
                SELECT t.seq, t.author, t.kind, t.content, t.mood, t.created_at,
                       e.embedding <=> CAST(:v AS vector) AS distance
                FROM thought_embeddings e JOIN thoughts t ON t.seq = e.seq
                WHERE e.seq < :before AND t.kind <> 'digest'
                ORDER BY e.embedding <=> CAST(:v AS vector)
                LIMIT :k"""), {"v": _vec(v), "before": oldest_in_window,
                               "k": self.settings.memory_k})).all()
        out = [{"seq": r[0], "author": r[1], "kind": r[2], "content": r[3], "mood": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "distance": round(float(r[6]), 4)} for r in rows]
        return sorted(out, key=lambda t: t["seq"])

    async def recall_at(self, seq: int) -> list[dict[str, Any]]:
        """Debug: what memory would retrieve for the window ending at `seq`."""
        thoughts = await self.stream.thoughts(after=max(0, seq - self.stream.window), limit=1000)
        window = [t for t in thoughts if t["seq"] <= seq][-self.stream.window:]
        if not window:
            return []
        return await self._search(window)
