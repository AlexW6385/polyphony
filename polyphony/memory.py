"""Memory: older thoughts related to what is being said now, retrieved by meaning.

    write   every voice / conscious / world thought -> embedding model -> thought_embeddings
            (a background task, never on the append path; it also backfills anything missing)
    recall  the last few thoughts of the shown window -> embedding model -> query vector
            -> pgvector nearest neighbours among thoughts OLDER than the window
            -> drop the weak ones (cosine similarity below a floor) -> at most k memories

The embedding model is the only model involved; the search itself is plain SQL. The query
depends only on the stream state, not on the voice, so one recall per state is cached in Redis
under the window's last `seq`: N voices listening against one state cost one embedding and one
search.

Default model: Qwen3-Embedding-0.6B through Ollama (multilingual, Chinese and English in one
vector space, 1024 dimensions). It is instruction-aware: queries carry a one-line task,
documents are embedded as they are.

Memory is best-effort everywhere: if the embedding service is down or the pgvector extension is
missing, voices simply get no memories.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
from sqlalchemy import bindparam, text

from .config import Settings
from .stream import K_MEMORY, Stream

log = logging.getLogger(__name__)

TABLE = "thought_embeddings"
KINDS = ("voice", "conscious", "world")  # what is worth remembering (not votes or digests)
EMBED_BATCH = 32
POLL_SECONDS = 2.0
CACHE_TTL = 600
MAX_CHARS = 4000

QWEN_TASK = ("Given the latest thoughts of a discussion, retrieve earlier thoughts that are "
             "relevant to what is being discussed")


def _vec(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class Embedder:
    """A small client for an embedding endpoint: Ollama's native API or OpenAI-compatible."""

    def __init__(self, settings: Settings):
        self.provider = settings.embed_provider
        self.model = settings.embed_model
        self.dim = settings.embed_dim
        if self.provider == "ollama":
            base = settings.embed_base_url or "http://localhost:11434"
            self.base = base.rstrip("/").removesuffix("/v1")
        elif self.provider == "openai":
            self.base = (settings.embed_base_url or "https://api.openai.com/v1").rstrip("/")
        else:
            raise ValueError(f"unknown embed provider {self.provider!r} (ollama or openai)")
        headers = {"Authorization": f"Bearer {settings.embed_api_key}"} if settings.embed_api_key \
            else {}
        self.client = httpx.AsyncClient(timeout=60, headers=headers)

    @property
    def instruction_aware(self) -> bool:
        return "qwen3-embedding" in self.model.lower()

    def query_text(self, q: str) -> str:
        return f"Instruct: {QWEN_TASK}\nQuery:{q}" if self.instruction_aware else q

    async def embed(self, texts: list[str]) -> list[list[float]]:
        texts = [t[:MAX_CHARS] for t in texts]
        if self.provider == "ollama":
            r = await self.client.post(f"{self.base}/api/embed",
                                       json={"model": self.model, "input": texts})
            r.raise_for_status()
            vectors = r.json()["embeddings"]
        else:
            body: dict[str, Any] = {"model": self.model, "input": texts}
            if self.model.startswith("text-embedding-3"):
                body["dimensions"] = self.dim
            r = await self.client.post(f"{self.base}/embeddings", json=body)
            r.raise_for_status()
            vectors = [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"{self.model} returned {len(v)} dimensions; EMBED_DIM is "
                                 f"{self.dim}")
        return vectors

    async def aclose(self) -> None:
        await self.client.aclose()


class Memory:
    def __init__(self, stream: Stream, settings: Settings, embedder: Embedder | None = None):
        self.stream = stream
        self.settings = settings
        self.enabled = bool(settings.embed_provider) or embedder is not None
        self.embedder = embedder or (Embedder(settings) if self.enabled else None)
        self.model = self.embedder.model if self.embedder else ""
        self._locks: dict[int, asyncio.Lock] = {}
        self._failing = False

    # ------------------------------------------------------------------ table

    async def setup(self) -> bool:
        """Create the extension, table and index. A changed dimension rebuilds the table, a
        changed model drops the old vectors; the embedder then refills it in the background."""
        if not self.enabled:
            return False
        dim = self.settings.embed_dim
        try:
            async with self.stream.sessions() as s:
                await s.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                current = await s.scalar(text(
                    "SELECT a.atttypmod FROM pg_attribute a "
                    "WHERE a.attrelid = to_regclass(:t) AND a.attname = 'embedding'"),
                    {"t": TABLE})
                if current is not None and current != dim:
                    log.info("memory: vector size %s -> %s, rebuilding %s", current, dim, TABLE)
                    await s.execute(text(f"DROP TABLE {TABLE}"))
                await s.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {TABLE} (
                        seq BIGINT PRIMARY KEY REFERENCES thoughts(seq) ON DELETE CASCADE,
                        model TEXT NOT NULL,
                        embedding vector({dim}) NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now())"""))
                await s.execute(text(
                    f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_hnsw ON {TABLE} "
                    "USING hnsw (embedding vector_cosine_ops)"))
                await s.execute(text(f"DELETE FROM {TABLE} WHERE model <> :m"), {"m": self.model})
                await s.commit()
        except Exception as e:
            log.warning("memory off: could not set up pgvector (%s)", e)
            self.enabled = False
            return False
        log.info("memory on: %s via %s, %d dimensions", self.model, self.settings.embed_provider,
                 dim)
        return True

    # ------------------------------------------------------------------ embedder

    async def pending(self, limit: int = EMBED_BATCH) -> list[dict[str, Any]]:
        async with self.stream.sessions() as s:
            rows = (await s.execute(text(f"""
                SELECT t.seq, t.content FROM thoughts t
                LEFT JOIN {TABLE} e ON e.seq = t.seq
                WHERE e.seq IS NULL AND t.kind IN :kinds
                ORDER BY t.seq LIMIT :n""").bindparams(bindparam("kinds", expanding=True)),
                {"kinds": list(KINDS), "n": limit})).all()
        return [{"seq": r[0], "content": r[1]} for r in rows]

    async def embed_pending(self) -> int:
        """Embed one batch of thoughts that have no embedding yet. Returns how many."""
        batch = await self.pending()
        if not batch:
            return 0
        vectors = await self.embedder.embed([b["content"] for b in batch])
        async with self.stream.sessions() as s:
            for b, v in zip(batch, vectors):
                await s.execute(text(f"""
                    INSERT INTO {TABLE} (seq, model, embedding)
                    VALUES (:seq, :model, CAST(:v AS vector))
                    ON CONFLICT (seq) DO NOTHING"""),
                    {"seq": b["seq"], "model": self.model, "v": _vec(v)})
            await s.commit()
        return len(batch)

    async def run(self) -> None:
        """Background embedder. Backs off while the embedding service is down."""
        if not self.enabled:
            return
        delay = POLL_SECONDS
        while True:
            try:
                n = await self.embed_pending()
                if self._failing:
                    log.info("memory: embedding service is back")
                    self._failing = False
                delay = 0.05 if n == EMBED_BATCH else POLL_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self._failing:
                    log.warning("memory: embedding failed, retrying with backoff (%s: %s)",
                                type(e).__name__, e)
                    self._failing = True
                delay = min(60.0, max(POLL_SECONDS, delay * 2))
            await asyncio.sleep(delay)

    # ------------------------------------------------------------------ recall

    async def recall(self, shown: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Memories for the window a prompt shows. Cached by its last seq; never raises."""
        if not self.enabled or not shown:
            return []
        tail = shown[-1]["seq"]
        key = self.stream.redis_key(K_MEMORY) + str(tail)
        try:
            cached = await self.stream.redis.get(key)
            if cached is not None:
                return json.loads(cached)
            lock = self._locks.setdefault(tail, asyncio.Lock())
            async with lock:  # voices of this process waking together share one query
                cached = await self.stream.redis.get(key)
                if cached is not None:
                    return json.loads(cached)
                out = await self.search(shown)
                await self.stream.redis.set(key, json.dumps(out), ex=CACHE_TTL)
            if len(self._locks) > 256:
                self._locks.clear()
            return out
        except Exception as e:
            log.info("memory recall skipped: %s: %s", type(e).__name__, e)
            return []

    async def search(self, shown: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The search itself, uncached: thoughts older than `shown`, nearest to its tail."""
        first = shown[0]["seq"]
        talk = [t for t in shown if t["kind"] in KINDS]
        if first <= 1 or not talk:
            return []
        query = "\n".join(t["content"] for t in talk[-self.settings.memory_query_last:])
        [v] = await self.embedder.embed([self.embedder.query_text(query[-MAX_CHARS:])])
        k = self.settings.memory_k
        async with self.stream.sessions() as s:
            # The WHERE filter runs after the index scan; look wider than the default 40.
            await s.execute(text("SET LOCAL hnsw.ef_search = 100"))
            rows = (await s.execute(text(f"""
                SELECT t.seq, t.author, t.kind, t.content, t.mood, t.created_at,
                       1 - (e.embedding <=> CAST(:v AS vector)) AS similarity
                FROM {TABLE} e JOIN thoughts t ON t.seq = e.seq
                WHERE e.seq < :before AND e.model = :model
                ORDER BY e.embedding <=> CAST(:v AS vector)
                LIMIT :n"""), {"v": _vec(v), "before": first, "model": self.model,
                               "n": k * 3})).all()
        floor = self.settings.memory_min_similarity
        out = [{"seq": r[0], "author": r[1], "kind": r[2], "content": r[3], "mood": r[4],
                "created_at": r[5].isoformat() if r[5] else None,
                "similarity": round(float(r[6]), 4)}
               for r in rows if float(r[6]) >= floor][:k]
        return sorted(out, key=lambda t: t["seq"])

    async def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        async with self.stream.sessions() as s:
            embedded = await s.scalar(text(f"SELECT count(*) FROM {TABLE}"))
            total = await s.scalar(text("SELECT count(*) FROM thoughts WHERE kind IN "
                                        "('voice', 'conscious', 'world')"))
        return {"enabled": True, "provider": self.settings.embed_provider, "model": self.model,
                "dim": self.settings.embed_dim, "embedded": embedded, "pending": total - embedded,
                "failing": self._failing}

    async def aclose(self) -> None:
        if self.embedder:
            await self.embedder.aclose()
