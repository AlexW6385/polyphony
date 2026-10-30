"""Digest: a readable outcome of a stretch of the stream.

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
from .frame import get_frame
from .stream import Stream

log = logging.getLogger(__name__)

CHECK_SECONDS = 5.0
MAX_THOUGHTS = 300


async def last_digest_seq(stream: Stream) -> int:
    async with stream.sessions() as s:
        row = await s.scalar(select(Thought).where(Thought.kind == "digest")
                             .order_by(desc(Thought.seq)).limit(1))
    return int((row.meta or {}).get("to_seq", row.seq)) if row else 0


async def write_digest(stream: Stream, settings: Settings) -> dict[str, Any] | None:
    """Digest everything since the last digest. Returns the thought, or None if nothing new."""
    since = await last_digest_seq(stream)
    thoughts = [t for t in await stream.thoughts(after=since, limit=MAX_THOUGHTS)
                if t["kind"] != "digest"]
    if not any(t["kind"] == "voice" for t in thoughts):
        return None
    frame = await get_frame(stream.sessions, settings.ensemble)
    user = prompts.render(thoughts, [], {}, [], None, person_label=frame.person_label)
    text = await llm.complete_text(settings.speak_tier, prompts.digest_system(frame), user,
                                   temperature=0.3, max_tokens=1200, tier_name="speak",
                                   purpose="digest")
    if not text:
        return None
    outcome = "decision" if text.rstrip().lower().endswith("outcome: decision") else "positions"
    body = "\n".join(l for l in text.rstrip().splitlines()
                     if not l.strip().lower().startswith("outcome:")).strip()
    meta = {"from_seq": thoughts[0]["seq"], "to_seq": thoughts[-1]["seq"], "outcome": outcome}
    return await stream.append("digest", "digest", body, meta=meta)


async def digests(stream: Stream, limit: int = 20) -> list[dict[str, Any]]:
    async with stream.sessions() as s:
        rows = (await s.scalars(select(Thought).where(Thought.kind == "digest")
                                .order_by(desc(Thought.seq)).limit(limit))).all()
    return [r.to_dict() for r in rows]


class Digester:
    def __init__(self, stream: Stream, settings: Settings):
        self.stream = stream
        self.settings = settings

    async def quiet_for(self) -> tuple[float, int] | None:
        """(seconds since the last non-digest thought, its seq), or None if anything is busy."""
        if await self.stream.forming() or await self.stream.cadence_running():
            return None
        recent = [t for t in await self.stream.recent() if t["kind"] != "digest"]
        if not recent or not recent[-1].get("created_at"):
            return None
        last = datetime.fromisoformat(recent[-1]["created_at"])
        return (datetime.now(timezone.utc) - last).total_seconds(), recent[-1]["seq"]

    async def run(self) -> None:
        while True:
            await asyncio.sleep(CHECK_SECONDS)
            try:
                q = await self.quiet_for()
                if q is None or q[0] < self.settings.digest_idle_seconds:
                    continue
                if await last_digest_seq(self.stream) >= q[1]:
                    continue
                t = await write_digest(self.stream, self.settings)
                if t:
                    log.info("digest written for [%s..%s]", t["meta"]["from_seq"],
                             t["meta"]["to_seq"])
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("digest failed")
                await asyncio.sleep(60)
