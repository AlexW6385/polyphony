"""Cadence: a vote on a question with two to four options, called by the clerk.

    round 1 standing: every voice weights every voice for this question; averaged, normalised
    round 2 vote:     every voice says which options it supports (any number), weighted
    -> each option's weighted support; the top one wins if it has at least 60% and is
       strictly ahead of the next, otherwise the question stays open
    -> a `cadence` thought in the stream with the full result

The procedure itself runs as a LangGraph workflow (cadence_graph.py) with a
checkpointer, so an interrupted cadence resumes from its last finished step.
This module holds the storage helpers and the pure math.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import jev, llm, prompts
from .config import Settings
from .db import Cadence
from .stream import Stream
from .frame import Frame
from .voice import VoiceSpec

log = logging.getLogger(__name__)

DECIDE_SHARE = 0.6  # weighted support an option needs to be decided


async def _context(stream: Stream, settings: Settings, spec: VoiceSpec) -> str:
    recent = prompts.cache_window(await stream.recent())
    return prompts.render(recent, [], await stream.listens_latest(), await stream.forming(),
                          spec.name, voice_text=spec.text, tz=settings.timezone)


async def ask_jev(stream: Stream, settings: Settings, frame: Frame, spec: VoiceSpec, call,
                  purpose: str) -> dict[str, Any] | None:
    """A cadence round on Jev: `call(state)` returns the answer; None if Jev failed."""
    recent = prompts.cache_window(await stream.recent())
    st = jev.state(frame, recent, me=spec.name, voice_text=spec.text, tz=settings.timezone)
    try:
        return await call(st)
    except Exception as e:
        (log.debug if jev.cooling_down() else log.warning)(
            "cadence: Jev %s for %s failed, using the listen model: %s", purpose, spec.name, e)
        return None


async def ask(stream: Stream, settings: Settings, frame: Frame, spec: VoiceSpec, tier_name: str,
              system: str, purpose: str,
              schema: dict[str, Any] | None = None) -> dict[str, Any] | None:
    base = settings.listen_tier if tier_name == "listen" else settings.speak_tier
    tier = llm.resolve_tier(base, spec.overrides.get(tier_name))
    try:
        return await llm.complete_json(tier, system, await _context(stream, settings, spec),
                                       temperature=0.5, tier_name=tier_name, purpose=purpose,
                                       voice=spec.name, schema=schema)
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


async def create_cadence(stream: Stream, question: str, options: list[str], called_by: str,
                         status: str) -> int:
    async with stream.sessions() as s:
        row = Cadence(resolution=question, options=options, proposed_by=called_by,
                      status=status)
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


def tally(weights: dict[str, float], votes: dict[str, dict[str, Any]],
          options: list[str]) -> dict[str, Any]:
    """Each option's weighted support among the voices that answered.

    A voice may support several options (approval voting). The top option wins if its
    support is at least DECIDE_SHARE and strictly above the next; otherwise nothing is decided.
    """
    answered = [n for n, v in votes.items() if v.get("answered", True)]
    total = sum(weights.get(n, 0) for n in answered)
    shares = []
    for i in range(len(options)):
        w = sum(weights.get(n, 0) for n in answered if i in votes[n]["support"])
        shares.append(round(w / total, 4) if total > 0 else 0.0)
    order = sorted(range(len(options)), key=lambda i: -shares[i])
    top = order[0] if order else None
    decided = (top is not None and shares[top] >= DECIDE_SHARE
               and (len(order) < 2 or shares[top] > shares[order[1]]))
    return {"shares": shares, "winner": top if decided else None, "decided": decided}


_PICK = re.compile(r"^\s*([A-Za-z]|\d+)\s*[.、:：)）-]?\s*(.*)$", re.S)


def _option_index(t: str, options: list[str]) -> int | None:
    """One pick as an option index. The option's text wins over its letter when both are given
    and disagree: a model copying the text means that option, a bare letter can slip."""
    t = t.strip()
    if t in options:
        return options.index(t)
    m = _PICK.match(t)
    if not m:
        return None
    head, rest = m.group(1), m.group(2).strip()
    if rest:
        if rest in options:
            return options.index(rest)
        hits = [i for i, o in enumerate(options) if rest.startswith(o) or o.startswith(rest)]
        if len(hits) == 1:
            return hits[0]
    if head.isalpha() and len(head) == 1 and ord(head.upper()) - 65 < len(options):
        return ord(head.upper()) - 65
    if head.isdigit() and 1 <= int(head) <= len(options):
        return int(head) - 1
    return None


def _normalize_vote(d: dict[str, Any] | None, options: list[str]) -> dict[str, Any]:
    """`support` as option indexes; "A", "A. <text>", numbers or the option text are accepted."""
    if not d:
        return {"support": [], "reason": "(no answer)", "answered": False}
    raw = d.get("support") or []
    if not isinstance(raw, list):
        raw = [raw]
    picked = {i for i in (_option_index(str(x), options) for x in raw) if i is not None}
    out = {"support": sorted(picked), "reason": str(d.get("reason") or ""), "answered": True}
    if d.get("by"):
        out["by"] = str(d["by"])
    if isinstance(d.get("p"), list):
        out["p"] = d["p"]
    return out


def summarize(question: str, options: list[str], weights: dict[str, float],
              votes: dict[str, dict[str, Any]], result: dict[str, Any]) -> str:
    letter = lambda i: chr(65 + i)
    if result["decided"]:
        w = result["winner"]
        verdict = f"DECIDED: {letter(w)}. {options[w]} ({result['shares'][w]:.0%})"
    else:
        verdict = f"UNDECIDED: no option had {DECIDE_SHARE:.0%} or more"
    lines = [f"Cadence on: {question}", verdict, ""]
    lines += [f"{letter(i)}. {o} — {result['shares'][i]:.0%}" for i, o in enumerate(options)]
    lines += ["", "Weights: " + ", ".join(f"{n} {w:.2f}" for n, w in
                                         sorted(weights.items(), key=lambda kv: -kv[1])), ""]
    for n, v in votes.items():
        picked = ", ".join(letter(i) for i in v["support"]) or "none"
        line = f"- {n} ({weights.get(n, 0):.2f}): {picked}"
        if v.get("by"):  # a decider that writes no reason (Jev)
            line += f" ({v['by']})"
        if v.get("reason"):
            line += f" — {v['reason']}"
        lines.append(line)
    return "\n".join(lines)


# --------------------------------------------------------------------------- procedures


async def run_cadence(stream: Stream, settings: Settings, question: str, options: list[str],
                      called_by: str = "clerk") -> dict:
    """Run both rounds through the workflow and record the result. The caller holds the slot."""
    from . import cadence_graph

    cadence_id = None
    try:
        cadence_id = await create_cadence(stream, question, options, called_by, "running")
        await cadence_graph.start(stream, settings, cadence_id, question, options, called_by)
    finally:
        await stream.release_cadence()
    return await get_cadence(stream, cadence_id)
