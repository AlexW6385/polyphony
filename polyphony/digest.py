"""Digest: a readable outcome of a stretch of the score.

An edge component like a sense, never a voice. When the room has been quiet
for `DIGEST_IDLE_SECONDS` (no new thought, nobody forming, no cadence running)
and something was said since the last digest, it writes a summary of that
stretch — the question, each voice's position, what changed, the outcome —
and appends one `digest` thought. Voices do not wake on digest thoughts.

The quiet check is mechanical (time since the last thought); it does not
decide anything about the discussion, it only notices that nobody is talking.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import desc, select

from . import llm, prompts
from .config import Settings
from .db import Thought
from .key import get_key
from .score import Score

log = logging.getLogger(__name__)

CHECK_SECONDS = 5.0
MAX_THOUGHTS = 300


async def last_digest_seq(score: Score) -> int:
    async with score.sessions() as s:
        row = await s.scalar(select(Thought).where(Thought.kind == "digest")
                             .order_by(desc(Thought.seq)).limit(1))
    return int((row.meta or {}).get("to_seq", row.seq)) if row else 0


async def write_digest(score: Score, settings: Settings) -> dict[str, Any] | None:
    """Digest everything since the last digest. Returns the thought, or None if nothing new."""
    since = await last_digest_seq(score)
    thoughts = [t for t in await score.thoughts(after=since, limit=MAX_THOUGHTS)
                if t["kind"] != "digest"]
    if not any(t["kind"] == "voice" for t in thoughts):
        return None
    key = await get_key(score.sessions)
    user = prompts.render(thoughts, [], {}, [], None, tz=settings.timezone)
    text = await llm.complete_text(settings.speak_tier, prompts.digest_system(key), user,
                                   temperature=0.3, max_tokens=1200, tier_name="speak",
                                   purpose="digest")
    if not text:
        return None
    outcome = "decision" if text.rstrip().lower().endswith("outcome: decision") else "positions"
    body = "\n".join(l for l in text.rstrip().splitlines()
                     if not l.strip().lower().startswith("outcome:")).strip()
    meta = {"from_seq": thoughts[0]["seq"], "to_seq": thoughts[-1]["seq"], "outcome": outcome}
    return await score.append("digest", "digest", body, meta=meta)


async def quiet_for(score: Score) -> tuple[float, int] | None:
    """(seconds since the last non-digest thought, its seq), or None if anything is busy."""
    if await score.forming() or await score.cadence_running():
        return None
    recent = [t for t in await score.recent() if t["kind"] != "digest"]
    if not recent or not recent[-1].get("created_at"):
        return None
    last = datetime.fromisoformat(recent[-1]["created_at"])
    return (datetime.now(timezone.utc) - last).total_seconds(), recent[-1]["seq"]


async def digests(score: Score, limit: int = 20) -> list[dict[str, Any]]:
    async with score.sessions() as s:
        rows = (await s.scalars(select(Thought).where(Thought.kind == "digest")
                                .order_by(desc(Thought.seq)).limit(limit))).all()
    return [r.to_dict() for r in rows]


class Digester:
    def __init__(self, score: Score, settings: Settings):
        self.score = score
        self.settings = settings

    async def quiet_for(self) -> tuple[float, int] | None:
        return await quiet_for(self.score)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(CHECK_SECONDS)
            try:
                q = await self.quiet_for()
                if q is None or q[0] < self.settings.digest_idle_seconds:
                    continue
                if await last_digest_seq(self.score) >= q[1]:
                    continue
                t = await write_digest(self.score, self.settings)
                if t:
                    log.info("digest written for [%s..%s]", t["meta"]["from_seq"],
                             t["meta"]["to_seq"])
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("digest failed")
                await asyncio.sleep(60)
