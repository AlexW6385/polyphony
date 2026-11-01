"""A voice: one asyncio task, defined entirely by a piece of text.

    idle --(score moved)--> listen --(no)--> idle
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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from . import llm, prompts
from .config import Settings
from .db import ListenSample, VoiceRow
from .key import DEFAULT_KEY, Key
from .score import Score
from .text import parse_front_matter

log = logging.getLogger(__name__)

ProposeFn = Callable[[str, str], Awaitable[Any]]  # (proposer, resolution)

# Thoughts of these kinds never wake a voice: a digest is a summary for the
# human, not a contribution, so it must not restart the discussion.
SILENT_KINDS = {"digest"}

# Small listen models copy schema placeholders into propose_cadence, or echo the
# proposal thought back; none of these is a resolution anyone could vote on.
PROPOSAL_PREFIX = "i think we should decide this:"
NOT_A_PROPOSAL = {
    "", "null", "none", "n/a", "no", "false", "cadence",
    "a resolution, phrased so it can be voted for or against",
    "a resolution",
}
PLACEHOLDER_FRAGMENT = "phrased so it can be voted"


def repeats_proposal(proposal: str, recent: list[dict[str, Any]]) -> bool:
    """True if this resolution was already proposed or voted on in the recent window."""
    needle = proposal.strip(" \"'.").casefold()
    for t in recent:
        content = t["content"].casefold()
        is_proposal = t["kind"] == "voice" and content.startswith(PROPOSAL_PREFIX)
        if (is_proposal or t["kind"] == "cadence") and needle in content:
            return True
    return False


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
        score: Score,
        settings: Settings,
        propose: ProposeFn | None = None,
        key: Key | None = None,
    ):
        self.spec = spec
        self.score = score
        self.settings = settings
        self.propose = propose
        self.key = key or Key(text=DEFAULT_KEY)
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
        await self.score.set_forming(self.name, False)  # clear anything a crash left behind
        listener = asyncio.create_task(self._listen_channel(), name=f"voice:{self.name}:channel")
        try:
            await self._ready.wait()
            while True:
                await self._wake.wait()
                self._wake.clear()
                while await self.score.cadence_running():
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
                await self.score.set_forming(self.name, False)
            except Exception:
                pass

    async def _listen_channel(self) -> None:
        while True:
            try:
                async for t in self.score.subscribe(ready=self._ready):
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
        strays = await self.score.strays(before_seq=recent[0]["seq"], n=self.settings.strays)
        others = await self.score.listens_latest()
        forming = await self.score.forming()
        return strays, others, forming

    def _render(self, recent, strays, others, forming) -> str:
        return prompts.render(recent, strays, others, forming, self.name)

    async def listen(self) -> dict[str, Any] | None:
        recent = await self.score.recent()
        if not recent:
            return None
        strays, others, forming = await self._context(recent)
        system = prompts.listen_system(self.key, self.name, self.spec.text)
        user = self._render(recent, strays, others, forming)
        decision = await llm.complete_json(
            self.listen_tier, system, user, schema=prompts.LISTEN_SCHEMA,
            tier_name="listen", purpose="listen", voice=self.name,
        )
        decision = normalize_decision(decision)
        if decision["propose_cadence"] and repeats_proposal(decision["propose_cadence"], recent):
            decision["propose_cadence"] = None
        row = await self.score.record_listen(self.name, recent[-1]["seq"], decision)
        log.info("%s listened: %s — %s — %s", self.name,
                 "speak" if decision["speak"] else "rest", decision["mood"], decision["reason"])
        self._sample(row["id"], system, user, decision)

        if decision["propose_cadence"] and self.propose:
            # Run the proposal vote in the background: this voice keeps living.
            self._spawn(self.propose(self.name, decision["propose_cadence"]))

        if not decision["speak"]:
            return decision

        await self.score.set_forming(self.name, True)
        try:
            pause = max(0.0, min(60.0, decision["pause_seconds"])) + random.uniform(0, 2)
            await asyncio.sleep(pause * self.settings.pause_scale)
            recent = await self.score.recent()  # the score moved while we waited
            _, others, forming = await self._context(recent)
            text = await llm.complete_text(
                self.speak_tier,
                prompts.speak_system(self.key, self.name, self.spec.text, decision),
                self._render(recent, strays, others, forming),
                tier_name="speak", purpose="speak", voice=self.name,
            )
            if text:
                await self.score.append(self.name, "voice", text, mood=decision["mood"])
            else:
                log.info("%s passed: nothing left to say after re-reading", self.name)
        finally:
            await self.score.set_forming(self.name, False)
        # The wake flag is NOT cleared here: anything that landed while we were
        # forming triggers one fresh listen against the newest state.
        return decision

    # ------------------------------------------------------------------ samples

    def _sample(self, listen_id: int, system: str, user: str, decision: dict[str, Any]) -> None:
        """Record listen samples for distillation, off the hot path.

        - Simulator (`record_all_listens`): every listen; the listen tier *is* the teacher.
        - Live: a `teacher_sample_rate` share also goes to the teacher in the background
          (the shadow teacher). The voice acts only on the student's decision.
        """
        if self.settings.record_all_listens:
            self._spawn(self._store_sample(listen_id, self.settings.record_all_listens, system,
                                           user, None, decision))
        elif self.settings.teacher_sample_rate > 0 and \
                random.random() < self.settings.teacher_sample_rate:
            self._spawn(self._shadow(listen_id, system, user, decision))

    async def _shadow(self, listen_id: int, system: str, user: str, student: dict) -> None:
        try:
            teacher = normalize_decision(await llm.complete_json(
                self.settings.teacher_tier, system, user, schema=prompts.LISTEN_SCHEMA,
                tier_name="teacher", purpose="shadow", voice=self.name,
            ))
        except Exception as e:
            log.warning("shadow teacher failed for %s: %s", self.name, e)
            return
        await self._store_sample(listen_id, "live", system, user, student, teacher)

    async def _store_sample(self, listen_id, source, system, user, student, teacher) -> None:
        t = self.settings.teacher_tier if source == "live" else self.listen_tier
        try:
            async with self.score.sessions() as s:
                s.add(ListenSample(
                    listen_id=listen_id, source=source,
                    voice=self.name, system=system, prompt=user,
                    student_provider=self.listen_tier.provider if student else None,
                    student_model=self.listen_tier.model if student else None,
                    student_decision=student,
                    teacher_provider=t.provider, teacher_model=t.model, teacher_decision=teacher,
                ))
                await s.commit()
        except Exception as e:
            log.warning("could not store listen sample: %s", e)


def normalize_decision(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce a listen reply into the exact schema; models are loose with types."""
    speak = d.get("speak")
    if isinstance(speak, str):
        speak = speak.strip().lower() in ("true", "yes", "1")
    try:
        pause = float(d.get("pause_seconds") or 0)
    except (TypeError, ValueError):
        pause = 0.0
    proposal = d.get("propose_cadence")
    if isinstance(proposal, str):
        proposal = proposal.strip()
        if proposal.lower().startswith(PROPOSAL_PREFIX):
            proposal = proposal[len(PROPOSAL_PREFIX):].strip()
    if (not isinstance(proposal, str) or proposal.strip(" \"'.").lower() in NOT_A_PROPOSAL
            or PLACEHOLDER_FRAGMENT in proposal.lower()):
        proposal = None
    return {
        "speak": bool(speak),
        "mood": str(d.get("mood") or "")[:80],
        "reason": str(d.get("reason") or ""),
        "pause_seconds": pause,
        "propose_cadence": proposal.strip() if proposal else None,
    }
