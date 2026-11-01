"""A voice: one asyncio task, defined entirely by a piece of text.

    idle --(stream moved)--> listen --(no)--> idle
                               |
                             (yes)
                               v
                        mark forming --> wait pause --> re-read --> speak --> append --> idle
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from . import llm, prompts
from .config import Settings
from .db import VoiceRow
from .frame import DEFAULT_FRAME, Frame
from .stream import Stream
from .text import parse_front_matter

log = logging.getLogger(__name__)

# Thoughts of these kinds never wake a voice: a digest is a summary for the
# human, not a contribution, so it must not restart the discussion.
SILENT_KINDS = {"digest"}

LISTEN_STAGGER = 8.0  # seconds: voices start listening at random within this after a new thought

@dataclass(frozen=True)
class VoiceSpec:
    name: str
    text: str
    overrides: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def parse(raw: str) -> tuple[str, dict[str, Any]]:
        return parse_front_matter(raw)

    @classmethod
    def load(cls, path: Path) -> VoiceSpec:
        """Load `voices/<name>/voice.md`."""
        text, overrides = cls.parse(path.read_text(encoding="utf-8"))
        return cls(name=path.parent.name, text=text, overrides=overrides)

    @classmethod
    def from_row(cls, row: VoiceRow) -> VoiceSpec:
        return cls(name=row.name, text=row.text, overrides=row.overrides or {})

    def fingerprint(self) -> str:
        blob = json.dumps([self.text, self.overrides], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


async def load_voice_specs(sessions: async_sessionmaker, settings: Settings) -> list[VoiceSpec]:
    """Every voice that should currently be running."""
    async with sessions() as s:
        rows = (await s.scalars(select(VoiceRow).where(VoiceRow.enabled).order_by(VoiceRow.name))).all()
    return [VoiceSpec.from_row(r) for r in rows]


class Voice:
    def __init__(
        self,
        spec: VoiceSpec,
        stream: Stream,
        settings: Settings,
        frame: Frame | None = None,
    ):
        self.spec = spec
        self.stream = stream
        self.settings = settings
        self.frame = frame or Frame(text=DEFAULT_FRAME)
        self.listen_tier = llm.resolve_tier(settings.listen_tier, spec.overrides.get("listen"))
        self.speak_tier = llm.resolve_tier(settings.speak_tier, spec.overrides.get("speak"))
        self._wake = asyncio.Event()
        self._ready = asyncio.Event()
        self._background: set[asyncio.Task] = set()

    @property
    def name(self) -> str:
        return self.spec.name

    def _spawn(self, coro) -> None:
        t = asyncio.create_task(coro)
        self._background.add(t)
        t.add_done_callback(self._background.discard)

    # ------------------------------------------------------------------ loop

    async def run(self) -> None:
        await self.stream.set_forming(self.name, False)  # clear anything a crash left behind
        listener = asyncio.create_task(self._listen_channel(), name=f"voice:{self.name}:channel")
        try:
            await self._ready.wait()
            while True:
                await self._wake.wait()
                # Don't all look at once: a voice that listens a few seconds later sees what
                # the first ones said, and anything that lands meanwhile joins this listen.
                await asyncio.sleep(random.uniform(0, LISTEN_STAGGER) * self.settings.pause_scale)
                self._wake.clear()
                while await self.stream.cadence_running():
                    # A cadence this voice is part of is running; the cadence
                    # thought will wake us again when it is over.
                    await asyncio.sleep(1)
                try:
                    await self.listen()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("voice %s: listen failed; staying quiet this time", self.name)
        finally:
            # Stopping a voice also stops anything it started, such as a cadence proposal.
            tasks = [listener, *self._background]
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self.stream.set_forming(self.name, False)
            except Exception:
                pass

    async def _listen_channel(self) -> None:
        while True:
            try:
                async for t in self.stream.subscribe(ready=self._ready):
                    if t["kind"] in SILENT_KINDS:
                        continue
                    if not (t["kind"] == "voice" and t["author"] == self.name):
                        self._wake.set()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("voice %s: subscription dropped; resubscribing", self.name)
                self._ready.set()
                await asyncio.sleep(1)

    # ------------------------------------------------------------------ listen / speak

    async def _context(self, recent):
        strays = await self.stream.strays(before_seq=recent[0]["seq"], n=self.settings.strays)
        others = await self.stream.listens_latest()
        forming = await self.stream.forming()
        return strays, others, forming

    def _render(self, recent, strays, others, forming, extra: str | None = None) -> str:
        return prompts.render(prompts.cache_window(recent), strays, others, forming, self.name,
                              voice_text=self.spec.text, extra=extra, tz=self.settings.timezone)

    async def listen(self) -> dict[str, Any] | None:
        recent = await self.stream.recent()
        if not recent:
            return None
        strays, others, forming = await self._context(recent)
        system = prompts.listen_system(self.frame)
        user = self._render(recent, strays, others, forming)
        decision = await llm.complete_json(
            self.listen_tier, system, user, schema=prompts.LISTEN_SCHEMA,
            tier_name="listen", purpose="listen", voice=self.name,
        )
        decision = normalize_decision(decision)
        await self.stream.record_listen(self.name, recent[-1]["seq"], decision)
        log.info("%s listened: %s — %s — %s", self.name,
                 "speak" if decision["speak"] else "rest", decision["mood"], decision["reason"])

        if not decision["speak"]:
            return decision

        await self.stream.set_forming(self.name, True)
        try:
            pause = max(0.0, min(60.0, decision["pause_seconds"])) + random.uniform(0, 2)
            await asyncio.sleep(pause * self.settings.pause_scale)
            recent = await self.stream.recent()  # the stream moved while we waited
            _, others, forming = await self._context(recent)
            text = await llm.complete_text(
                self.speak_tier,
                prompts.speak_system(self.frame),
                self._render(recent, strays, others, forming, extra=prompts.speak_mood(decision)),
                tier_name="speak", purpose="speak", voice=self.name,
                web_search=self.settings.speak_web_search,
            )
            if text:
                await self.stream.append(self.name, "voice", text, mood=decision["mood"])
            else:
                log.info("%s passed: nothing left to say after re-reading", self.name)
        finally:
            await self.stream.set_forming(self.name, False)
        # The wake flag is NOT cleared here: anything that landed while we were
        # forming triggers one fresh listen against the newest state.
        return decision

def normalize_decision(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce a listen reply into the exact schema; models are loose with types."""
    speak = d.get("speak")
    if isinstance(speak, str):
        speak = speak.strip().lower() in ("true", "yes", "1")
    try:
        pause = float(d.get("pause_seconds") or 0)
    except (TypeError, ValueError):
        pause = 0.0
    return {
        "speak": bool(speak),
        "mood": str(d.get("mood") or "")[:80],
        "reason": str(d.get("reason") or ""),
        "pause_seconds": pause,
    }
