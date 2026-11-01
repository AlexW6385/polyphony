"""Waker: when the room has gone quiet, wake one voice to look again.

Voices wake only when the stream moves, so once every voice has chosen not to
speak nothing would ever move the room again. After `IDLE_WAKE_SECONDS` of
quiet (no new thought, nobody forming, no cadence running) the waker wakes one
voice at random, once per quiet stretch: if that voice stays quiet too, the
room stays quiet until something new arrives. Like the digest's quiet check it
is mechanical; the woken voice still decides for itself whether to speak.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Callable
from typing import Any

from .config import Settings
from .digest import quiet_for
from .stream import Stream

log = logging.getLogger(__name__)

CHECK_SECONDS = 2.0


class Waker:
    def __init__(self, stream: Stream, settings: Settings, voices: Callable[[], dict[str, Any]]):
        self.stream = stream
        self.settings = settings
        self.voices = voices  # name -> running Voice; read fresh each tick
        self._woken_after: int | None = None  # seq of the last thought we already woke on

    async def tick(self) -> str | None:
        """Wake one voice if the room has been quiet long enough. Returns its name."""
        q = await quiet_for(self.stream)
        if q is None or q[0] < self.settings.idle_wake_seconds or q[1] == self._woken_after:
            return None
        voices = self.voices()
        if not voices:
            return None
        name = random.choice(sorted(voices))
        voices[name].nudge()
        self._woken_after = q[1]
        log.info("quiet for %.0fs after [%s]: woke %s", q[0], q[1], name)
        return name

    async def run(self) -> None:
        if self.settings.idle_wake_seconds <= 0:
            return
        while True:
            await asyncio.sleep(CHECK_SECONDS)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("waker failed")
                await asyncio.sleep(30)
