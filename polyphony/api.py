"""HTTP API: feed the room, read the record, watch it live, manage key, voices and senses."""

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
from . import llm, runtime_config
from .cadence import confirm_cadence, create_cadence, get_cadence, run_cadence
from .config import Settings
from .db import KeyRow, LLMCall, ListenSample, SenseRow, VoiceRow
from .key import CONSCIOUS, KEY_ROW, get_key
from .sense import DEFAULT_DAILY_AT, Sense, SenseSpec, check_daily_at
from .score import Score
from .voice import VoiceSpec

log = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RESERVED = {"conscious", "person", "theme", "cadence", "world", "digest", "you"}
HEARTBEAT_SECONDS = 15


class ThoughtIn(BaseModel):
    content: str = Field(min_length=1)
    author: str = CONSCIOUS
    kind: str = "theme"  # "person" and "conscious" are accepted as older names


class CadenceIn(BaseModel):
    resolution: str = Field(min_length=1)
    confirm: bool = False


class KeyPatch(BaseModel):
    text: str = Field(min_length=1)


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
    daily_at: str = DEFAULT_DAILY_AT
    feeds: list[str] = Field(default_factory=list)
    search: bool = False


class SensePatch(BaseModel):
    instruction: str | None = None
    daily_at: str | None = None
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


def create_app(score: Score, settings: Settings, lifespan=None) -> FastAPI:
    app = FastAPI(title="Polyphony", version="1.0.0", lifespan=lifespan)
    background: set[asyncio.Task] = set()

    def spawn(coro) -> None:
        t = asyncio.create_task(coro)
        background.add(t)
        t.add_done_callback(background.discard)

    # ------------------------------------------------------------------ basics

    @app.get("/health")
    async def health():
        await score.redis.ping()
        return {"ok": True}

    # ------------------------------------------------------------------ pause

    @app.get("/pause")
    async def get_pause():
        return {"paused": await score.paused()}

    @app.post("/pause")
    async def post_pause():
        """Stop everything that calls a model: voices, senses, digest, cadences."""
        await score.set_paused(True)
        for t in list(background):
            t.cancel()
        return {"paused": True}

    @app.post("/resume")
    async def post_resume():
        await score.set_paused(False)
        return {"paused": False}

    # ------------------------------------------------------------------ settings page

    @app.get("/settings")
    async def settings_page():
        return FileResponse(settings.static_dir / "settings.html")

    @app.get("/config")
    async def get_config():
        """Provider, models, and whether a key is set. Never returns the key."""
        return runtime_config.view(settings, await runtime_config.load(score.sessions))

    @app.put("/config")
    async def put_config(patch: dict[str, Any]):
        data = await runtime_config.load(score.sessions)
        try:
            data = runtime_config.merge(data, patch)
        except (ValueError, TypeError, AttributeError) as e:
            raise HTTPException(422, str(e))
        await runtime_config.save(score.sessions, data)
        runtime_config.apply(settings, data)
        await score.publish_control("config")  # voices restart on the new models
        return runtime_config.view(settings, data)

    @app.post("/config/test/{tier}")
    async def config_test(tier: str):
        """One tiny call on a tier, to check its provider, model and key."""
        if tier not in runtime_config.TIERS:
            raise HTTPException(404, "no such tier")
        t = getattr(settings, f"{tier}_tier")
        loop = asyncio.get_running_loop()
        start = loop.time()
        try:
            text = await llm.complete_text(t, "Reply with the single word: ok", "ping",
                                           max_tokens=64, tier_name=tier, purpose="test")
        except Exception as e:
            return {"ok": False, "provider": t.provider, "model": t.model, "error": str(e)[:300]}
        return {"ok": True, "provider": t.provider, "model": t.model, "reply": text[:80],
                "ms": round((loop.time() - start) * 1000)}

    async def refuse_if_paused() -> None:
        if await score.paused():
            raise HTTPException(409, "paused: resume first")

    @app.get("/")
    async def index():
        return FileResponse(settings.static_dir / "index.html")

    # ------------------------------------------------------------------ thoughts

    @app.post("/thoughts")
    async def post_thought(t: ThoughtIn):
        kind = "theme" if t.kind in ("person", "conscious") else t.kind
        if kind not in ("theme", "world"):
            raise HTTPException(422, "kind must be 'theme' or 'world'")
        author = CONSCIOUS if kind == "theme" else (t.author or "world")
        return await score.append(author, kind, t.content.strip())

    @app.get("/thoughts")
    async def get_thoughts(after: int = 0, limit: int = Query(200, ge=1, le=1000)):
        return await score.thoughts(after=after, limit=limit)

    @app.get("/score")
    async def sse(after: int | None = None, last_event_id: str | None = Header(None)):
        if after is None and last_event_id and last_event_id.isdigit():
            after = int(last_event_id)

        async def events():
            ready = asyncio.Event()
            sub = score.subscribe(ready=ready)
            live = asyncio.create_task(anext(sub))
            await ready.wait()  # subscribed before replaying: nothing falls in the gap
            last = 0
            try:
                if after is not None:
                    for t in await score.thoughts(after=after, limit=1000):
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
        return await score.listens(limit=limit)

    # ------------------------------------------------------------------ cadence

    @app.post("/cadence")
    async def post_cadence(c: CadenceIn):
        await refuse_if_paused()
        resolution = c.resolution.strip()
        if not await score.claim_cadence():
            raise HTTPException(409, "a cadence is already under way")
        who = CONSCIOUS
        try:
            cid = await create_cadence(score, resolution, who, "running")
        except BaseException:
            await score.release_cadence()
            raise
        spawn(run_cadence(score, settings, resolution, who, cid, confirm=c.confirm))
        return await get_cadence(score, cid)

    @app.post("/cadence/{cid}/confirm")
    async def post_confirm(cid: int):
        await refuse_if_paused()
        c = await get_cadence(score, cid)
        if c is None:
            raise HTTPException(404, "no such cadence")
        if c["status"] != "awaiting_confirm":
            raise HTTPException(409, f"cadence is {c['status']}, not awaiting_confirm")
        spawn(confirm_cadence(score, settings, cid))
        return {**c, "status": "running"}

    @app.get("/cadence/{cid}")
    async def get_one_cadence(cid: int):
        c = await get_cadence(score, cid)
        if c is None:
            raise HTTPException(404, "no such cadence")
        return c

    # ------------------------------------------------------------------ voices

    @app.get("/voices")
    async def list_voices():
        async with score.sessions() as s:
            rows = (await s.scalars(select(VoiceRow).order_by(VoiceRow.name))).all()
        forming = set(await score.forming())
        latest = await score.listens_latest()
        return [
            {"name": r.name, "text": r.text, "overrides": r.overrides or {}, "enabled": r.enabled,
             "forming": r.name in forming, "last_listen": latest.get(r.name)}
            for r in rows
        ]

    @app.post("/voices", status_code=201)
    async def add_voice(v: VoiceIn):
        _check_name(v.name)
        text, overrides = _split_voice_text(v.text, v.overrides)
        async with score.sessions() as s:
            if await s.get(VoiceRow, v.name):
                raise HTTPException(409, "a voice with that name exists")
            s.add(VoiceRow(name=v.name, text=text, overrides=overrides or None, enabled=True))
            await s.commit()
        await score.publish_control("voices")
        return {"name": v.name, "text": text, "overrides": overrides, "enabled": True}

    @app.put("/voices/{name}")
    async def edit_voice(name: str, v: VoicePatch):
        async with score.sessions() as s:
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
            await score.set_forming(name, False)
        await score.publish_control("voices")
        return out

    @app.delete("/voices/{name}", status_code=204)
    async def delete_voice(name: str):
        async with score.sessions() as s:
            row = await s.get(VoiceRow, name)
            if row is None:
                raise HTTPException(404, "no such voice")
            await s.delete(row)
            await s.commit()
        await score.forget_voice(name)
        await score.publish_control("voices")

    # ------------------------------------------------------------------ key

    @app.get("/key")
    async def read_key():
        return (await get_key(score.sessions)).to_dict()

    @app.put("/key")
    async def edit_key(k: KeyPatch):
        text = k.text.strip()
        if not text:
            raise HTTPException(422, "key text is empty")
        async with score.sessions() as s:
            row = await s.get(KeyRow, KEY_ROW)
            if row is None:
                s.add(KeyRow(id=KEY_ROW, text=text))
            else:
                row.text = text
            await s.commit()
        await score.publish_control("key")
        return {"text": text}

    # ------------------------------------------------------------------ digest

    @app.post("/digest")
    async def post_digest():
        await refuse_if_paused()
        t = await digest_mod.write_digest(score, settings)
        if t is None:
            raise HTTPException(409, "nothing new to digest")
        return t

    @app.get("/digests")
    async def get_digests(limit: int = Query(20, ge=1, le=200)):
        return await digest_mod.digests(score, limit)

    @app.get("/stats")
    async def stats(hours: float = Query(24, gt=0, le=24 * 30)):
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        async with score.sessions() as s:
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
            purposes = (await s.execute(
                select(LLMCall.purpose, func.count(), func.count().filter(LLMCall.ok.is_(False)),
                       func.sum(LLMCall.input_tokens), func.sum(LLMCall.output_tokens))
                .where(LLMCall.created_at >= since)
                .group_by(LLMCall.purpose).order_by(func.count().desc())
            )).all()
            unit = "minute" if hours <= 1 else "hour" if hours <= 48 else "day"
            bucket = func.date_trunc(unit, LLMCall.created_at)
            timeline = (await s.execute(
                select(bucket, func.count(), func.count().filter(LLMCall.ok.is_(False)))
                .where(LLMCall.created_at >= since).group_by(bucket).order_by(bucket)
            )).all()
            failed = (await s.scalars(
                select(LLMCall).where(LLMCall.created_at >= since, LLMCall.ok.is_(False))
                .order_by(LLMCall.id.desc()).limit(10)
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
        by_purpose = [{"purpose": r[0], "calls": r[1], "errors": int(r[2] or 0),
                       "tokens": int(r[3] or 0) + int(r[4] or 0)} for r in purposes]
        recent_errors = [{"at": f.created_at.isoformat() if f.created_at else None,
                          "tier": f.tier, "model": f.model, "purpose": f.purpose,
                          "voice": f.voice, "error": (f.error or "")[:300]} for f in failed]
        series = [{"t": r[0].isoformat(), "calls": r[1], "errors": int(r[2] or 0)}
                  for r in timeline]
        return {"hours": hours, "tiers": tiers, "by_purpose": by_purpose,
                "timeline": {"unit": unit, "since": since.isoformat(), "points": series},
                "recent_errors": recent_errors, "student_teacher": agreement(pairs)}

    # ------------------------------------------------------------------ senses

    def _daily_at(value: str) -> str:
        try:
            return check_daily_at(value)
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/senses")
    async def list_senses():
        async with score.sessions() as s:
            rows = (await s.scalars(select(SenseRow).order_by(SenseRow.name))).all()
        return [{**r.to_dict(), "timezone": settings.timezone} for r in rows]

    @app.post("/senses/{name}/run", status_code=202)
    async def run_sense(name: str):
        """Run a sense once now, enabled or not, in the background."""
        await refuse_if_paused()
        async with score.sessions() as s:
            row = await s.get(SenseRow, name)
            if row is None:
                raise HTTPException(404, "no such sense")
            spec = SenseSpec.from_row(row)
        spawn(Sense(spec, score, settings).fetch_once())
        return {"started": name}

    @app.post("/senses", status_code=201)
    async def add_sense(x: SenseIn):
        _check_name(x.name)
        async with score.sessions() as s:
            if await s.get(SenseRow, x.name):
                raise HTTPException(409, "a sense with that name exists")
            row = SenseRow(name=x.name, instruction=x.instruction.strip(),
                           daily_at=_daily_at(x.daily_at),
                           feeds=_check_feeds(x.feeds), search=x.search, enabled=True)
            s.add(row)
            await s.commit()
            out = row.to_dict()
        await score.publish_control("senses")
        return out

    @app.put("/senses/{name}")
    async def edit_sense(name: str, x: SensePatch):
        async with score.sessions() as s:
            row = await s.get(SenseRow, name)
            if row is None:
                raise HTTPException(404, "no such sense")
            if x.instruction is not None:
                row.instruction = x.instruction.strip()
            if x.daily_at is not None:
                row.daily_at = _daily_at(x.daily_at)
            if x.feeds is not None:
                row.feeds = _check_feeds(x.feeds)
            if x.search is not None:
                row.search = x.search
            if x.enabled is not None:
                row.enabled = x.enabled
            await s.commit()
            out = row.to_dict()
        await score.publish_control("senses")
        return out

    @app.delete("/senses/{name}", status_code=204)
    async def delete_sense(name: str):
        async with score.sessions() as s:
            row = await s.get(SenseRow, name)
            if row is None:
                raise HTTPException(404, "no such sense")
            await s.delete(row)
            await s.commit()
        await score.publish_control("senses")

    return app
