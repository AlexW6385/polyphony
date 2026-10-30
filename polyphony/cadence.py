"""Cadence: an explicit, optional decision procedure.

    propose (by a voice) -> yes/no "worth deciding now?" among all voices, simple majority
    start (by the person, or a passed proposal):
        round 1 standing: every voice weights every voice; averaged, normalised
        round 2 vote:     for | against | abstain, weighted
    -> a `cadence` thought in the stream with the full result

The procedure itself runs as a LangGraph workflow (cadence_graph.py) with a
checkpointer, so an interrupted cadence resumes from its last finished step.
This module holds the API surface, the proposal vote and the pure math.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from . import llm, prompts
from .config import Settings
from .db import Cadence
from .stream import Stream
from .frame import Frame, get_frame
from .voice import VoiceSpec, load_voice_specs

log = logging.getLogger(__name__)


async def _context(stream: Stream, frame: Frame, me: str) -> str:
    recent = await stream.recent()
    return prompts.render(recent, [], await stream.listens_latest(), await stream.forming(), me,
                          person_label=frame.person_label)


async def ask(stream: Stream, settings: Settings, frame: Frame, spec: VoiceSpec, tier_name: str,
              system: str, purpose: str) -> dict[str, Any] | None:
    base = settings.listen_tier if tier_name == "listen" else settings.speak_tier
    tier = llm.resolve_tier(base, spec.overrides.get(tier_name))
    try:
        return await llm.complete_json(tier, system, await _context(stream, frame, spec.name),
                                       temperature=0.5, tier_name=tier_name, purpose=purpose,
                                       voice=spec.name)
    except Exception:
        log.exception("cadence: %s did not answer", spec.name)
        return None


async def set_fields(stream: Stream, cadence_id: int, **fields) -> None:
    async with stream.sessions() as s:
        row = await s.get(Cadence, cadence_id)
        for k, v in fields.items():
            setattr(row, k, v)
        await s.commit()


async def get_cadence(stream: Stream, cadence_id: int) -> dict[str, Any] | None:
    async with stream.sessions() as s:
        row = await s.get(Cadence, cadence_id)
        return row.to_dict() if row else None


async def create_cadence(stream: Stream, resolution: str, proposed_by: str, status: str) -> int:
    async with stream.sessions() as s:
        row = Cadence(resolution=resolution, proposed_by=proposed_by, status=status)
        s.add(row)
        await s.commit()
        await s.refresh(row)
        return row.id


# --------------------------------------------------------------------------- math


def average_weights(names: list[str], proposals: list[dict[str, Any] | None]) -> dict[str, float]:
    """Average each voice's proposed weight across all proposals; normalise to sum 1."""
    sums = {n: 0.0 for n in names}
    counts = {n: 0 for n in names}
    for p in proposals:
        if not p:
            continue
        weights = p.get("weights", p)
        if not isinstance(weights, dict):
            continue
        for n, w in weights.items():
            if n not in sums:
                continue
            if isinstance(w, dict):
                w = w.get("weight")
            try:
                x = min(1.0, max(0.0, float(w)))
            except (TypeError, ValueError):
                continue
            sums[n] += x
            counts[n] += 1
    avg = {n: (sums[n] / counts[n]) if counts[n] else 0.0 for n in names}
    total = sum(avg.values())
    if total <= 0:
        return {n: 1.0 / len(names) for n in names} if names else {}
    return {n: v / total for n, v in avg.items()}


def tally(weights: dict[str, float], votes: dict[str, dict[str, str]]) -> dict[str, Any]:
    w_for = sum(weights.get(n, 0) for n, v in votes.items() if v["vote"] == "for")
    w_against = sum(weights.get(n, 0) for n, v in votes.items() if v["vote"] == "against")
    w_abstain = sum(weights.get(n, 0) for n, v in votes.items() if v["vote"] == "abstain")
    decided = w_for + w_against
    share = (w_for / decided) if decided > 0 else None
    return {
        "for": round(w_for, 4),
        "against": round(w_against, 4),
        "abstain": round(w_abstain, 4),
        "share_for": round(share, 4) if share is not None else None,
        "passed": share is not None and share > 0.5,
    }


def _normalize_vote(d: dict[str, Any] | None) -> dict[str, str]:
    if not d:
        return {"vote": "abstain", "reason": "(no answer)"}
    v = str(d.get("vote", "abstain")).strip().lower()
    if v not in ("for", "against", "abstain"):
        v = "abstain"
    return {"vote": v, "reason": str(d.get("reason") or "")}


def summarize(resolution: str, proposed_by: str, weights: dict[str, float],
              votes: dict[str, dict[str, str]], result: dict[str, Any]) -> str:
    verdict = "PASSED" if result["passed"] else "DID NOT PASS"
    share = "n/a" if result["share_for"] is None else f"{result['share_for']:.0%}"
    lines = [
        f'Cadence on: "{resolution}" (called by {proposed_by})',
        f"{verdict} — weighted share for: {share}",
        "",
        "Weights: " + ", ".join(f"{n} {w:.2f}" for n, w in
                                sorted(weights.items(), key=lambda kv: -kv[1])),
        "",
    ]
    for n, v in votes.items():
        lines.append(f"- {n} ({weights.get(n, 0):.2f}): {v['vote']} — {v['reason']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- procedures


async def run_cadence(stream: Stream, settings: Settings, resolution: str,
                      proposed_by: str = "person", cadence_id: int | None = None,
                      proposal: dict[str, Any] | None = None, confirm: bool = False) -> dict:
    """Run both rounds through the workflow and record the result.

    With `confirm=True` the workflow pauses before recording (status
    `awaiting_confirm`); `confirm_cadence()` resumes it.
    """
    from . import cadence_graph

    if cadence_id is None:
        cadence_id = await create_cadence(stream, resolution, proposed_by, "running")
    else:
        await set_fields(stream, cadence_id, status="running")
    await cadence_graph.start(stream, settings, cadence_id, resolution, proposed_by,
                              proposal=proposal, confirm=confirm)
    return await get_cadence(stream, cadence_id)


async def confirm_cadence(stream: Stream, settings: Settings, cadence_id: int) -> dict | None:
    from . import cadence_graph

    await cadence_graph.resume(stream, settings, cadence_id, confirmed=True)
    return await get_cadence(stream, cadence_id)


async def propose_cadence(stream: Stream, settings: Settings, proposer: str,
                          resolution: str) -> dict:
    """A voice proposes a cadence; all voices vote yes/no on whether to hold it now."""
    cadence_id = await create_cadence(stream, resolution, proposer, "proposed")
    await stream.append(proposer, "voice", f"I think we should decide this: {resolution}",
                        mood="proposing", meta={"proposes_cadence": cadence_id})

    frame = await get_frame(stream.sessions, settings.ensemble)
    specs = await load_voice_specs(stream.sessions, settings)
    answers = await asyncio.gather(*[
        ask(stream, settings, frame, s, "listen",
            prompts.proposal_system(frame, s.name, s.text, proposer, resolution), "proposal")
        for s in specs
    ])
    ballots = {}
    for s, a in zip(specs, answers):
        yes = bool(a and (a.get("yes") is True or str(a.get("yes")).lower() in ("true", "yes")))
        ballots[s.name] = {"yes": yes, "reason": str((a or {}).get("reason") or "(no answer)")}
    yes_count = sum(b["yes"] for b in ballots.values())
    passed = yes_count > len(specs) / 2

    if not passed:
        await set_fields(stream, cadence_id, status="rejected",
                   result={"passed": False, "proposal": ballots},
                   resolved_at=datetime.now(timezone.utc))
        await stream.append(
            "cadence", "cadence",
            f'Proposal by {proposer} not taken up ({yes_count} of {len(specs)} wanted to decide '
            f'now): "{resolution}"',
            meta={"cadence_id": cadence_id, "proposal": ballots, "passed": False},
        )
        return await get_cadence(stream, cadence_id)

    return await run_cadence(stream, settings, resolution, proposer, cadence_id, proposal=ballots)
