"""The cadence procedure as a LangGraph workflow with a checkpointer.

    standing (fan-out) -> weights -> vote (fan-out) -> tally -> [confirm] -> record

- Each node's output is checkpointed (PostgreSQL in production, memory in
  tests), keyed by thread `cadence-<id>`. After a crash or restart,
  `resume_unfinished()` continues every running cadence from its last
  finished step: voices that already answered a finished round are not asked
  again.
- `confirm`: the workflow interrupts before `record` until the person
  confirms (`POST /cadence/{id}/confirm`).

LangGraph is used here because this is a fixed, multi-step, stateful
procedure. It never coordinates the voices themselves.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from sqlalchemy import select

from . import prompts
from .cadence import (
    _normalize_vote,
    ask,
    average_weights,
    get_cadence,
    set_fields,
    summarize,
    tally,
)
from .config import Settings
from .db import Cadence
from .frame import get_frame
from .stream import Stream
from .voice import load_voice_specs

log = logging.getLogger(__name__)


class CadenceState(TypedDict, total=False):
    cadence_id: int
    resolution: str
    proposed_by: str
    proposal: dict[str, Any] | None
    confirm: bool
    names: list[str]
    proposals: dict[str, Any]
    weights: dict[str, float]
    votes: dict[str, dict[str, str]]
    result: dict[str, Any]
    recorded: bool


# One checkpointer per process. main.py installs a PostgreSQL one; tests and
# anything that never calls `setup()` get an in-memory one.
_checkpointer: Any = InMemorySaver()
_pg_cm: Any = None


async def setup(database_url: str) -> None:
    """Use a PostgreSQL checkpointer in the same database. Falls back to memory."""
    global _checkpointer, _pg_cm
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        conninfo = database_url.replace("postgresql+asyncpg://", "postgresql://")
        _pg_cm = AsyncPostgresSaver.from_conn_string(conninfo)
        saver = await _pg_cm.__aenter__()
        await saver.setup()
        _checkpointer = saver
        log.info("cadence checkpoints in PostgreSQL")
    except Exception as e:
        log.warning("cadence checkpoints in memory only (%s)", e)
        _checkpointer = InMemorySaver()
        _pg_cm = None


async def shutdown() -> None:
    global _pg_cm
    if _pg_cm is not None:
        await _pg_cm.__aexit__(None, None, None)
        _pg_cm = None


def _config(cadence_id: int) -> dict[str, Any]:
    return {"configurable": {"thread_id": f"cadence-{cadence_id}"}}


def build(stream: Stream, settings: Settings):
    async def standing(state: CadenceState) -> CadenceState:
        frame = await get_frame(stream.sessions, settings.ensemble)
        specs = await load_voice_specs(stream.sessions, settings)
        names = [s.name for s in specs]
        answers = await asyncio.gather(*[
            ask(stream, settings, frame, s, "speak",
                prompts.standing_system(frame, s.name, s.text, state["resolution"], names),
                "standing")
            for s in specs
        ])
        return {"names": names, "proposals": dict(zip(names, answers))}

    async def weights(state: CadenceState) -> CadenceState:
        w = average_weights(state["names"], list(state["proposals"].values()))
        await set_fields(stream, state["cadence_id"],
                         weights={"final": w, "proposals": state["proposals"]})
        return {"weights": w}

    async def vote(state: CadenceState) -> CadenceState:
        frame = await get_frame(stream.sessions, settings.ensemble)
        specs = [s for s in await load_voice_specs(stream.sessions, settings)
                 if s.name in state["names"]]
        answers = await asyncio.gather(*[
            ask(stream, settings, frame, s, "speak",
                prompts.vote_system(frame, s.name, s.text, state["resolution"], state["weights"]),
                "vote")
            for s in specs
        ])
        votes = {s.name: _normalize_vote(a) for s, a in zip(specs, answers)}
        for n in state["names"]:  # a voice removed mid-cadence abstains
            votes.setdefault(n, _normalize_vote(None))
        return {"votes": votes}

    async def count(state: CadenceState) -> CadenceState:
        result = tally(state["weights"], state["votes"])
        if state.get("proposal") is not None:
            result["proposal"] = state["proposal"]
        await set_fields(stream, state["cadence_id"], votes=state["votes"], result=result)
        return {"result": result}

    async def confirm(state: CadenceState) -> CadenceState:
        if state.get("confirm"):
            await set_fields(stream, state["cadence_id"], status="awaiting_confirm")
            await stream.set_cadence_running(state["cadence_id"], False)
            interrupt({"cadence_id": state["cadence_id"], "result": state["result"]})
        return {}

    async def record(state: CadenceState) -> CadenceState:
        cid = state["cadence_id"]
        await set_fields(stream, cid, status="resolved", resolved_at=datetime.now(timezone.utc))
        await stream.set_cadence_running(cid, False)
        content = summarize(state["resolution"], state["proposed_by"], state["weights"],
                            state["votes"], state["result"])
        meta = {"cadence_id": cid, "resolution": state["resolution"],
                "proposed_by": state["proposed_by"], "weights": state["weights"],
                "votes": state["votes"], "result": state["result"]}
        # Appended after the running flag is cleared, so this thought wakes the voices.
        await stream.append("cadence", "cadence", content, meta=meta)
        return {"recorded": True}

    g = StateGraph(CadenceState)
    for name, fn in [("standing", standing), ("weights", weights), ("vote", vote),
                     ("tally", count), ("confirm", confirm), ("record", record)]:
        g.add_node(name, fn)
    g.add_edge(START, "standing")
    g.add_edge("standing", "weights")
    g.add_edge("weights", "vote")
    g.add_edge("vote", "tally")
    g.add_edge("tally", "confirm")
    g.add_edge("confirm", "record")
    g.add_edge("record", END)
    return g.compile(checkpointer=_checkpointer)


async def _empty(stream: Stream, settings: Settings, cadence_id: int) -> bool:
    if await load_voice_specs(stream.sessions, settings):
        return False
    result = {"for": 0, "against": 0, "abstain": 0, "share_for": None, "passed": False}
    await set_fields(stream, cadence_id, status="resolved", weights={}, votes={}, result=result,
                     resolved_at=datetime.now(timezone.utc))
    return True


async def start(stream: Stream, settings: Settings, cadence_id: int, resolution: str,
                proposed_by: str, *, proposal: dict[str, Any] | None = None,
                confirm: bool = False) -> None:
    if await _empty(stream, settings, cadence_id):
        return
    await stream.set_cadence_running(cadence_id, True)
    try:
        await build(stream, settings).ainvoke(
            {"cadence_id": cadence_id, "resolution": resolution, "proposed_by": proposed_by,
             "proposal": proposal, "confirm": confirm},
            _config(cadence_id),
        )
    finally:
        c = await get_cadence(stream, cadence_id)
        if c and c["status"] != "running":
            await stream.set_cadence_running(cadence_id, False)


async def resume(stream: Stream, settings: Settings, cadence_id: int,
                 confirmed: bool = False) -> None:
    """Continue a cadence from its last checkpoint (after a crash, or after confirm)."""
    graph = build(stream, settings)
    snap = await graph.aget_state(_config(cadence_id))
    if not snap or not snap.next:
        return
    await set_fields(stream, cadence_id, status="running")
    await stream.set_cadence_running(cadence_id, True)
    try:
        payload = Command(resume=True) if confirmed else None
        await graph.ainvoke(payload, _config(cadence_id))
    finally:
        await stream.set_cadence_running(cadence_id, False)


async def resume_unfinished(stream: Stream, settings: Settings) -> None:
    """On startup: continue every cadence a crash left in `running`."""
    async with stream.sessions() as s:
        ids = (await s.scalars(select(Cadence.id).where(Cadence.status == "running"))).all()
    for cid in ids:
        log.info("resuming cadence %s", cid)
        try:
            await resume(stream, settings, cid)
        except Exception:
            log.exception("could not resume cadence %s", cid)
