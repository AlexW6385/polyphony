"""HTTP API: feed the room, read the record, watch it live, manage frame, voices and senses."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from . import digest as digest_mod
from .cadence import confirm_cadence, create_cadence, get_cadence, run_cadence
from .config import Settings
from .db import FrameRow, LLMCall, ListenSample, SenseRow, VoiceRow
from .frame import get_frame
from .sense import MIN_EVERY_MINUTES
from .stream import Stream
from .voice import VoiceSpec

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RESERVED = {"conscious", "person", "cadence", "world", "digest", "you"}
HEARTBEAT_SECONDS = 15


class ThoughtIn(BaseModel):
    content: str = Field(min_length=1)
    author: str = "person"
    kind: str = "person"  # "conscious" is accepted as the v0 name


class CadenceIn(BaseModel):
    resolution: str = Field(min_length=1)
    confirm: bool = False


class FramePatch(BaseModel):
    label: str | None = None
    person_label: str | None = None
    text: str | None = None


class VoiceIn(BaseModel):
    name: str
    text: str = Field(min_length=1)
    overrides: dict[str, Any] | None = None


class VoicePatch(BaseModel):
    text: str | None = None
    overrides: dict[str, Any] | None = None
    enabled: bool | None = None


class SenseIn(BaseModel):
    name: str
    instruction: str = Field(min_length=1)
    every_minutes: int = 240
    feeds: list[str] = Field(default_factory=list)
    search: bool = False


class SensePatch(BaseModel):
    instruction: str | None = None
    every_minutes: int | None = None
    feeds: list[str] | None = None
    search: bool | None = None
    enabled: bool | None = None


def _check_feeds(feeds: list[str]) -> list[str]:
    out = [f.strip() for f in feeds if f.strip()]
    for f in out:
        if not re.match(r"^https?://", f):
            raise HTTPException(422, f"feed must be an http(s) URL: {f}")
    return out


def _check_name(name: str) -> None:
    if not NAME_RE.match(name) or name in RESERVED:
        raise HTTPException(422, "name must be lowercase letters, digits, - or _, "
                                 f"at most 64 chars, and not one of {sorted(RESERVED)}")


def _split_voice_text(text: str, overrides: dict[str, Any] | None) -> tuple[str, dict]:
    """A voice's text may carry its own front matter; explicit overrides win."""
    try:
        body, meta = VoiceSpec.parse(text)
    except Exception as e:
        raise HTTPException(422, f"bad front matter: {e}")
    if not body:
        raise HTTPException(422, "voice text is empty")
    return body, {**meta, **(overrides or {})}


def create_app(stream: Stream, settings: Settings, lifespan=None, memory=None) -> FastAPI:
    app = FastAPI(title="Polyphony", version="1.0.0", lifespan=lifespan)
    background: set[asyncio.Task] = set()

    def spawn(coro) -> None:
        t = asyncio.create_task(coro)
        background.add(t)
        t.add_done_callback(background.discard)

    # ------------------------------------------------------------------ basics

    @app.get("/health")
    async def health():
        await stream.redis.ping()
        return {"ok": True}

    @app.get("/")
    async def index():
        return FileResponse(settings.static_dir / "index.html")

    # ------------------------------------------------------------------ thoughts

    @app.post("/thoughts")
    async def post_thought(t: ThoughtIn):
        kind = "person" if t.kind == "conscious" else t.kind
        if kind not in ("person", "world"):
            raise HTTPException(422, "kind must be 'person' (or 'conscious') or 'world'")
        author = "person" if kind == "person" else (t.author or "world")
        return await stream.append(author, kind, t.content.strip())

    @app.get("/thoughts")
    async def get_thoughts(after: int = 0, limit: int = Query(200, ge=1, le=1000)):
        return await stream.thoughts(after=after, limit=limit)

    @app.get("/stream")
    async def sse(after: int | None = None, last_event_id: str | None = Header(None)):
        if after is None and last_event_id and last_event_id.isdigit():
            after = int(last_event_id)

        async def events():
            ready = asyncio.Event()
            sub = stream.subscribe(ready=ready)
            live = asyncio.create_task(anext(sub))
            await ready.wait()  # subscribed before replaying: nothing falls in the gap
            last = 0
            try:
                if after is not None:
                    for t in await stream.thoughts(after=after, limit=1000):
                        last = t["seq"]
                        yield f"id: {t['seq']}\ndata: {json.dumps(t)}\n\n"
                while True:
                    done, _ = await asyncio.wait({live}, timeout=HEARTBEAT_SECONDS)
                    if not done:
                        yield ": keepalive\n\n"
                        continue
                    t = live.result()
                    live = asyncio.create_task(anext(sub))
                    if t["seq"] <= last:
                        continue
                    last = t["seq"]
                    yield f"id: {t['seq']}\ndata: {json.dumps(t)}\n\n"
            finally:
                live.cancel()
                await asyncio.gather(live, return_exceptions=True)
                await sub.aclose()

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ listens

    @app.get("/listens")
    async def get_listens(limit: int = Query(100, ge=1, le=1000)):
        return await stream.listens(limit=limit)

    # ------------------------------------------------------------------ cadence

    @app.post("/cadence")
    async def post_cadence(c: CadenceIn):
        resolution = c.resolution.strip()
        cid = await create_cadence(stream, resolution, "person", "running")
        spawn(run_cadence(stream, settings, resolution, "person", cid, confirm=c.confirm))
        return await get_cadence(stream, cid)

    @app.post("/cadence/{cid}/confirm")
    async def post_confirm(cid: int):
        c = await get_cadence(stream, cid)
        if c is None:
            raise HTTPException(404, "no such cadence")
        if c["status"] != "awaiting_confirm":
            raise HTTPException(409, f"cadence is {c['status']}, not awaiting_confirm")
        spawn(confirm_cadence(stream, settings, cid))
        return {**c, "status": "running"}

    @app.get("/cadence/{cid}")
    async def get_one_cadence(cid: int):
        c = await get_cadence(stream, cid)
        if c is None:
            raise HTTPException(404, "no such cadence")
        return c

    # ------------------------------------------------------------------ voices

    @app.get("/voices")
    async def list_voices():
        async with stream.sessions() as s:
            rows = (await s.scalars(select(VoiceRow).order_by(VoiceRow.name))).all()
        forming = set(await stream.forming())
        latest = await stream.listens_latest()
        return [
            {"name": r.name, "text": r.text, "overrides": r.overrides or {}, "enabled": r.enabled,
             "forming": r.name in forming, "last_listen": latest.get(r.name)}
            for r in rows
        ]

    @app.post("/voices", status_code=201)
    async def add_voice(v: VoiceIn):
        _check_name(v.name)
        text, overrides = _split_voice_text(v.text, v.overrides)
        async with stream.sessions() as s:
            if await s.get(VoiceRow, v.name):
                raise HTTPException(409, "a voice with that name exists")
            s.add(VoiceRow(name=v.name, text=text, overrides=overrides or None, enabled=True))
            await s.commit()
        await stream.publish_control("voices")
        return {"name": v.name, "text": text, "overrides": overrides, "enabled": True}

    @app.put("/voices/{name}")
    async def edit_voice(name: str, v: VoicePatch):
        async with stream.sessions() as s:
            row = await s.get(VoiceRow, name)
            if row is None:
                raise HTTPException(404, "no such voice")
            if v.text is not None:
                row.text, meta = _split_voice_text(v.text, None)
                if meta and v.overrides is None:
                    row.overrides = meta
            if v.overrides is not None:
                row.overrides = v.overrides or None
            if v.enabled is not None:
                row.enabled = v.enabled
            await s.commit()
            out = {"name": row.name, "text": row.text, "overrides": row.overrides or {},
                   "enabled": row.enabled}
        if v.enabled is False:
            await stream.set_forming(name, False)
        await stream.publish_control("voices")
        return out

    @app.delete("/voices/{name}", status_code=204)
    async def delete_voice(name: str):
        async with stream.sessions() as s:
            row = await s.get(VoiceRow, name)
            if row is None:
                raise HTTPException(404, "no such voice")
            await s.delete(row)
            await s.commit()
        await stream.forget_voice(name)
        await stream.publish_control("voices")

    # ------------------------------------------------------------------ frame

    @app.get("/frame")
    async def read_frame():
        return (await get_frame(stream.sessions, settings.ensemble)).to_dict()

    @app.put("/frame")
    async def edit_frame(f: FramePatch):
        async with stream.sessions() as s:
            row = await s.get(FrameRow, settings.ensemble)
            if row is None:
                cur = await get_frame(stream.sessions, settings.ensemble)
                row = FrameRow(name=cur.name, label=cur.label, person_label=cur.person_label,
                               text=cur.text)
                s.add(row)
            if f.label is not None:
                row.label = f.label.strip() or row.label
            if f.person_label is not None:
                row.person_label = f.person_label.strip() or row.person_label
            if f.text is not None:
                if not f.text.strip():
                    raise HTTPException(422, "frame text is empty")
                row.text = f.text.strip()
            await s.commit()
            out = row.to_dict()
        await stream.publish_control("frame")
        return out

    # ------------------------------------------------------------------ digest, memory

    @app.post("/digest")
    async def post_digest():
        t = await digest_mod.write_digest(stream, settings)
        if t is None:
            raise HTTPException(409, "nothing new to digest")
        return t

    @app.get("/digests")
    async def get_digests(limit: int = Query(20, ge=1, le=200)):
        return await digest_mod.digests(stream, limit)

    @app.get("/memories")
    async def get_memories(seq: int, k: int = Query(5, ge=1, le=50)):
        if memory is None or not memory.enabled:
            raise HTTPException(409, "memory is off (no embedder or no pgvector)")
        return (await memory.recall_at(seq))[:k]

    # ------------------------------------------------------------------ stats

    @app.get("/stats")
    async def stats(hours: float = Query(24, gt=0, le=24 * 30)):
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with stream.sessions() as s:
            rows = (await s.execute(
                select(LLMCall.tier, LLMCall.provider, LLMCall.model,
                       func.count(), func.count().filter(LLMCall.ok.is_(False)),
                       func.sum(LLMCall.input_tokens), func.sum(LLMCall.output_tokens),
                       func.sum(LLMCall.cached_input_tokens), func.avg(LLMCall.latency_ms),
                       func.percentile_cont(0.95).within_group(LLMCall.latency_ms),
                       func.avg(LLMCall.queue_ms),
                       func.percentile_cont(0.95).within_group(LLMCall.queue_ms))
                .where(LLMCall.created_at >= since)
                .group_by(LLMCall.tier, LLMCall.provider, LLMCall.model)
            )).all()
            samples = (await s.scalars(
                select(ListenSample).where(ListenSample.source == "live",
                                           ListenSample.created_at >= since,
                                           ListenSample.student_decision.isnot(None),
                                           ListenSample.teacher_decision.isnot(None))
            )).all()
        tiers = [{
            "tier": r[0], "provider": r[1], "model": r[2], "calls": r[3],
            "errors": int(r[4] or 0), "input_tokens": int(r[5] or 0),
            "output_tokens": int(r[6] or 0), "cached_input_tokens": int(r[7] or 0),
            "latency_ms_avg": round(float(r[8] or 0)), "latency_ms_p95": round(float(r[9] or 0)),
            "queue_ms_avg": None if r[10] is None else round(float(r[10])),
            "queue_ms_p95": None if r[11] is None else round(float(r[11])),
        } for r in rows]
        from .distill import agreement

        pairs = [(x.student_decision, x.teacher_decision) for x in samples]
        return {"hours": hours, "tiers": tiers, "student_teacher": agreement(pairs)}

    # ------------------------------------------------------------------ senses

    @app.get("/senses")
    async def list_senses():
        async with stream.sessions() as s:
            rows = (await s.scalars(select(SenseRow).order_by(SenseRow.name))).all()
        return [r.to_dict() for r in rows]

    @app.post("/senses", status_code=201)
    async def add_sense(x: SenseIn):
        _check_name(x.name)
        async with stream.sessions() as s:
            if await s.get(SenseRow, x.name):
                raise HTTPException(409, "a sense with that name exists")
            row = SenseRow(name=x.name, instruction=x.instruction.strip(),
                           every_minutes=max(MIN_EVERY_MINUTES, x.every_minutes),
                           feeds=_check_feeds(x.feeds), search=x.search, enabled=True)
            s.add(row)
            await s.commit()
            out = row.to_dict()
        await stream.publish_control("senses")
        return out

    @app.put("/senses/{name}")
    async def edit_sense(name: str, x: SensePatch):
        async with stream.sessions() as s:
            row = await s.get(SenseRow, name)
            if row is None:
                raise HTTPException(404, "no such sense")
            if x.instruction is not None:
                row.instruction = x.instruction.strip()
            if x.every_minutes is not None:
                row.every_minutes = max(MIN_EVERY_MINUTES, x.every_minutes)
            if x.feeds is not None:
                row.feeds = _check_feeds(x.feeds)
            if x.search is not None:
                row.search = x.search
            if x.enabled is not None:
                row.enabled = x.enabled
            await s.commit()
            out = row.to_dict()
        await stream.publish_control("senses")
        return out

    @app.delete("/senses/{name}", status_code=204)
    async def delete_sense(name: str):
        async with stream.sessions() as s:
            row = await s.get(SenseRow, name)
            if row is None:
                raise HTTPException(404, "no such sense")
            await s.delete(row)
            await s.commit()
        await stream.publish_control("senses")

    return app
