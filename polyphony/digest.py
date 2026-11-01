"""Digest: a readable outcome of a stretch of the stream.

An edge component like a sense, never a voice. When the stream has been quiet
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
    frame = await get_frame(stream.sessions)
    user = prompts.render(thoughts, [], {}, [], None, tz=settings.timezone)
    reply = await llm.complete_json(settings.speak_tier, prompts.digest_system(frame), user,
                                    schema=prompts.DIGEST_SCHEMA, temperature=0.3,
                                    max_tokens=1200, tier_name="speak", purpose="digest")
    summary = _clean(reply)
    if not summary["outcome"]:
        return None
    meta = {"from_seq": thoughts[0]["seq"], "to_seq": thoughts[-1]["seq"],
            "outcome": "decision" if summary["decided"] else "positions", "summary": summary}
    return await stream.append("digest", "digest", _as_text(summary), meta=meta)


def _clean(reply: dict[str, Any]) -> dict[str, Any]:
    positions = [{"voice": str(p.get("voice") or "").strip(), "view": str(p.get("view") or "").strip()}
                 for p in reply.get("positions") or [] if isinstance(p, dict)]
    return {
        "decided": bool(reply.get("decided")),
        "outcome": str(reply.get("outcome") or "").strip(),
        "question": str(reply.get("question") or "").strip(),
        "positions": [p for p in positions if p["voice"] and p["view"]],
        "open": [str(o).strip() for o in reply.get("open") or [] if str(o).strip()],
    }


def _as_text(s: dict[str, Any]) -> str:
    """The digest as plain text: what voices read in the stream (the page uses the meta)."""
    lines = [("Decided: " if s["decided"] else "Not decided: ") + s["outcome"]]
    if s["question"]:
        lines.append(f"Question: {s['question']}")
    lines += [f"- {p['voice']}: {p['view']}" for p in s["positions"]]
    if s["open"]:
        lines.append("Still open: " + "; ".join(s["open"]))
    return "\n".join(lines)


async def quiet_for(stream: Stream) -> tuple[float, int] | None:
    """(seconds since the last non-digest thought, its seq), or None if anything is busy."""
    if await stream.forming() or await stream.cadence_running():
        return None
    recent = [t for t in await stream.recent() if t["kind"] != "digest"]
    if not recent or not recent[-1].get("created_at"):
        return None
    last = datetime.fromisoformat(recent[-1]["created_at"])
    return (datetime.now(timezone.utc) - last).total_seconds(), recent[-1]["seq"]


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
        return await quiet_for(self.stream)

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
