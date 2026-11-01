"""The clerk: decides when to settle something by a vote, and what the options are.

An edge component like the digest, never a voice: it does not speak in the stream. It looks
at the discussion every `CHECK_EVERY` new thoughts, and once when the stream has been quiet
for `QUIET_SECONDS` after something new was said. Each check is one model call that reads
the stream and answers whether a vote is due and, if so, the question and its options (at
least two) drawn from the discussion. The prompt depends on why it is looking: mid-discussion
it only calls a vote that is clearly due; after the stream has gone quiet the discussion is
over, so it calls one unless there is truly nothing to decide. A vote it calls runs as a
cadence (cadence.py) and its result lands in the stream; voices treat a decided cadence as
settled.

Only one cadence runs at a time, through the same slot as before, so a check that finds a
vote already under way does nothing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import llm, prompts
from .cadence import run_cadence
from .config import Settings
from .digest import quiet_for
from .frame import get_frame
from .stream import Stream

log = logging.getLogger(__name__)

CHECK_EVERY = 15  # new thoughts since the last check
QUIET_SECONDS = 30.0  # quiet this long after something new -> check once
TICK_SECONDS = 3.0
TALK = {"voice", "conscious", "world"}  # what counts as discussion (not votes or digests)


class Clerk:
    def __init__(self, stream: Stream, settings: Settings):
        self.stream = stream
        self.settings = settings
        self._checked_seq: int | None = None  # last thought already looked at

    async def tick(self) -> bool:
        """Check the discussion if it is due. Returns True if a check ran."""
        if await self.stream.cadence_running():
            return False
        talk = [t for t in await self.stream.recent() if t["kind"] in TALK]
        if not talk:
            return False
        last = talk[-1]["seq"]
        if self._checked_seq is None:  # just started: don't revisit what came before
            self._checked_seq = last
            return False
        new = [t for t in talk if t["seq"] > self._checked_seq]
        if not new:
            return False
        when = "count" if len(new) >= CHECK_EVERY else None
        if when is None and any(t["kind"] == "voice" for t in new):
            q = await quiet_for(self.stream)
            if q is not None and q[0] >= QUIET_SECONDS:
                when = "quiet"
        if when is None:
            return False
        self._checked_seq = last
        await self.check(when)
        return True

    async def check(self, when: str = "quiet") -> dict[str, Any] | None:
        """Ask the model whether a vote is due; if so, run it. Returns the cadence, if any.

        `when` is "count" (every CHECK_EVERY thoughts, mid-discussion) or "quiet" (the
        discussion is over).
        """
        frame = await get_frame(self.stream.sessions)
        recent = prompts.cache_window(await self.stream.recent())
        user = prompts.render(recent, [], {}, [], None, tz=self.settings.timezone)
        reply = await llm.complete_json(self.settings.speak_tier, prompts.clerk_system(frame, when),
                                        user, schema=prompts.CLERK_SCHEMA, temperature=0.2,
                                        tier_name="speak", purpose="clerk")
        question = str(reply.get("question") or "").strip()
        options = [str(o).strip() for o in (reply.get("options") or []) if str(o).strip()]
        if not reply.get("call_vote") or not question or len(options) < 2:
            log.info("clerk (%s): no vote now", when)
            return None
        if not await self.stream.claim_cadence():
            return None
        log.info("clerk (%s): vote on %r with %d options", when, question, len(options))
        return await run_cadence(self.stream, self.settings, question, options)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(TICK_SECONDS)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("clerk failed")
                await asyncio.sleep(30)
