"""A voice: one asyncio task, defined entirely by a piece of text.

    idle --(stream moved)--> stagger --> mark listening --> listen --(no)--> idle
                                                              |
                                                            (yes)
                                                              v
                        mark forming --> wait pause --> re-read --> speak --> append --> idle

The stagger is random within `listen_stagger_per_voice` x the number of enabled voices, so the
more voices there are, the more spread out they listen, and later ones see who is already
listening or forming. Listen goes to the listen model, or to Jev when `decider` is "jev".
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

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from . import jev, llm, prompts
from .config import Settings
from .db import VoiceRow
from .frame import DEFAULT_FRAME, Frame
from .stream import Stream
from .text import parse_front_matter

log = logging.getLogger(__name__)

# Thoughts of these kinds never wake a voice: a digest is a summary for the
# human, not a contribution, so it must not restart the discussion.
SILENT_KINDS = {"digest"}


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


async def count_enabled(sessions: async_sessionmaker) -> int:
    async with sessions() as s:
        return int(await s.scalar(select(func.count()).select_from(VoiceRow)
                                  .where(VoiceRow.enabled)) or 0)


def stagger_window(settings: Settings, voices: int) -> float:
    """Seconds within which voices start listening after a new thought."""
    return max(0.0, settings.listen_stagger_per_voice * voices) * settings.pause_scale


class Voice:
    def __init__(
        self,
        spec: VoiceSpec,
        stream: Stream,
        settings: Settings,
        frame: Frame | None = None,
        memory: Any = None,
    ):
        self.spec = spec
        self.memory = memory  # memory.Memory, or None for no memories
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
        # Clear anything a crash left behind.
        await self.stream.set_forming(self.name, False)
        await self.stream.set_listening(self.name, False)
        listener = asyncio.create_task(self._listen_channel(), name=f"voice:{self.name}:channel")
        try:
            await self._ready.wait()
            while True:
                await self._wake.wait()
                # Don't all look at once: a voice that listens a few seconds later sees what
                # the first ones said, and anything that lands meanwhile joins this listen.
                window = stagger_window(self.settings, await count_enabled(self.stream.sessions))
                await asyncio.sleep(random.uniform(0, window))
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
                await self.stream.set_listening(self.name, False)
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

    async def _context(self, shown):
        memories = await self.memory.recall(shown) if self.memory else None
        remembered = {m["seq"] for m in memories or []}
        strays = [t for t in await self.stream.strays(before_seq=shown[0]["seq"],
                                                      n=self.settings.strays)
                  if t["seq"] not in remembered]
        return {"strays": strays, "memories": memories,
                "others": await self.stream.listens_latest(),
                "listening": await self.stream.listening(),
                "forming": await self.stream.forming()}

    def _render(self, shown, ctx, extra: str | None = None) -> str:
        return prompts.render(shown, ctx["strays"], ctx["others"], ctx["forming"], self.name,
                              voice_text=self.spec.text, extra=extra, tz=self.settings.timezone,
                              listening=ctx["listening"], memories=ctx["memories"])

    async def _decide(self, shown, ctx) -> dict[str, Any]:
        if jev.enabled(self.settings):
            st = jev.state(self.frame, shown, me=self.name, voice_text=self.spec.text,
                           memories=ctx["memories"], others=ctx["others"],
                           listening=ctx["listening"], forming=ctx["forming"],
                           tz=self.settings.timezone)
            try:
                return await jev.listen(self.settings, st, self.name)
            except Exception as e:
                (log.debug if jev.cooling_down() else log.warning)(
                    "%s: Jev listen failed, using the listen model: %s", self.name, e)
        return await llm.complete_json(
            self.listen_tier, prompts.listen_system(self.frame), self._render(shown, ctx),
            schema=prompts.LISTEN_SCHEMA, tier_name="listen", purpose="listen", voice=self.name,
        )

    async def listen(self) -> dict[str, Any] | None:
        recent = await self.stream.recent()
        if not recent:
            return None
        await self.stream.set_listening(self.name, True)
        try:
            shown = prompts.cache_window(recent)
            ctx = await self._context(shown)
            decision = normalize_decision(await self._decide(shown, ctx))
            await self.stream.record_listen(self.name, recent[-1]["seq"], decision)
        finally:
            await self.stream.set_listening(self.name, False)
        log.info("%s listened: %s — %s — %s", self.name,
                 "speak" if decision["speak"] else "rest", decision["mood"], decision["reason"])

        if not decision["speak"]:
            return decision

        await self.stream.set_forming(self.name, True)
        try:
            pause = max(0.0, min(60.0, decision["pause_seconds"])) + random.uniform(0, 2)
            await asyncio.sleep(pause * self.settings.pause_scale)
            shown = prompts.cache_window(await self.stream.recent())  # moved while we waited
            strays = ctx["strays"]  # the same strays it listened with
            ctx = await self._context(shown)
            ctx["strays"] = strays
            text = await llm.complete_text(
                self.speak_tier,
                prompts.speak_system(self.frame),
                self._render(shown, ctx, extra=prompts.speak_mood(decision)),
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
