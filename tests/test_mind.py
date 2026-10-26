"""End-to-end behaviour with real PostgreSQL/Redis and a scripted model.

The fake model plays the part of judgement: a voice speaks once after the
conscious says something new, then gets bored. That the mind then goes quiet
is the point — there is no counter or cap in the code.
"""

import asyncio
import json
import re

import httpx
import pytest

from polyphony import cadence as cad
from polyphony.api import create_app
from polyphony.db import SenseRow, VoiceRow
from polyphony.sense import Sense, SenseSpec
from polyphony.supervisor import Supervisor

from .conftest import voice_name


def stream_lines(user: str) -> list[str]:
    return user.split("## The stream (oldest first)")[1].strip().splitlines()


def polite_script():
    """Speak once per new conscious thought, then get bored."""

    async def script(system, user, json, web_search):
        me = voice_name(system)
        if "Right now you are only listening" in system:
            lines = stream_lines(user)
            last_conscious = max((i for i, l in enumerate(lines) if "[conscious]" in l), default=-1)
            spoke_since = any(f"{me} (you)" in l for l in lines[last_conscious + 1:])
            if last_conscious >= 0 and not spoke_since:
                return _j(speak=True, mood="curious", reason="new thought", pause_seconds=0)
            return _j(speak=False, mood="bored", reason="said my piece", pause_seconds=0)
        if "You decided to speak" in system:
            topic = [l for l in stream_lines(user) if "[conscious]" in l][-1].split(": ", 1)[1]
            return f"{me} thinks about: {topic}"
        raise AssertionError("unexpected prompt")

    return script


def _j(**kw):
    kw.setdefault("propose_cadence", None)
    return json.dumps(kw)


async def add_voices(stream, *names):
    async with stream.sessions() as s:
        for n in names:
            s.add(VoiceRow(name=n, text=f"**Wants** — {n}."))
        await s.commit()


async def wait_quiet(stream, idle=1.5, timeout=20):
    """Wait until nothing new has been appended and nobody is forming for `idle` s."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last, since = None, loop.time()
    while loop.time() < deadline:
        ts = await stream.thoughts(limit=1000)
        now = (len(ts), tuple(await stream.forming()))
        if now != last:
            last, since = now, loop.time()
        elif not now[1] and loop.time() - since >= idle:
            return ts
        await asyncio.sleep(0.2)
    raise AssertionError("mind did not go quiet")


@pytest.fixture
async def mind(stream, settings, fake):
    fake.script = polite_script()
    await add_voices(stream, "ride", "rest", "make")
    sup = Supervisor(stream, settings, run_senses=False)
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.5)  # let voices subscribe
    yield sup
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_voices_speak_and_the_mind_goes_quiet(stream, mind, fake):
    await stream.append("conscious", "conscious", "should I take the specialization?")
    ts = await wait_quiet(stream)
    voices = [t for t in ts if t["kind"] == "voice"]
    assert sorted(t["author"] for t in voices) == ["make", "rest", "ride"]
    assert all("specialization" in t["content"] for t in voices)
    seqs = [t["seq"] for t in ts]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    listens = await stream.listens()
    assert {l["voice"] for l in listens} == {"ride", "rest", "make"}
    assert any(not l["speak"] and l["mood"] == "bored" for l in listens)


async def test_second_topic_mid_discussion(stream, mind, fake):
    await stream.append("conscious", "conscious", "first topic")
    await asyncio.sleep(0.3)
    await stream.append("conscious", "conscious", "unrelated second topic")
    ts = await wait_quiet(stream)
    assert any("second topic" in t["content"] for t in ts if t["kind"] == "voice")


async def test_voice_added_through_api_starts_listening(stream, settings, mind, fake):
    app = create_app(stream, settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/voices", json={"name": "learn", "text": "---\n---\n- \"wait, why?\""})
        assert r.status_code == 201
        for _ in range(50):
            if "learn" in mind.voices:
                break
            await asyncio.sleep(0.1)
        assert "learn" in mind.voices
        await asyncio.sleep(0.3)
        r = await c.post("/thoughts", json={"content": "a question for everyone"})
        assert r.status_code == 200 and r.json()["kind"] == "conscious"
        ts = await wait_quiet(stream)
        assert "learn" in {t["author"] for t in ts}
        voices = (await c.get("/voices")).json()
        assert next(v for v in voices if v["name"] == "learn")["last_listen"] is not None

        # disable -> the task stops
        await c.put("/voices/learn", json={"enabled": False})
        for _ in range(50):
            if "learn" not in mind.voices:
                break
            await asyncio.sleep(0.1)
        assert "learn" not in mind.voices


async def test_cadence_two_rounds(stream, settings, fake):
    await add_voices(stream, "ride", "rest")

    async def script(system, user, json, web_search):
        me = voice_name(system)
        if "First round: standing" in system:
            return _j(weights={"ride": {"weight": 0.9, "why": "stake"},
                               "rest": {"weight": 0.3, "why": "meh"}})
        if "Second round: vote" in system:
            return _j(vote="for" if me == "ride" else "against", reason=f"{me} says so")
        raise AssertionError(system[:80])

    fake.script = script
    out = await cad.run_cadence(stream, settings, "go cycling tomorrow")
    assert out["status"] == "resolved"
    assert out["result"]["passed"] is True
    assert out["result"]["share_for"] == pytest.approx(0.75)
    last = (await stream.thoughts())[-1]
    assert last["kind"] == "cadence" and last["meta"]["votes"]["rest"]["vote"] == "against"
    assert not await stream.cadence_running()


async def test_cadence_proposal_rejected(stream, settings, fake):
    await add_voices(stream, "ride", "rest", "make")

    async def script(system, user, json, web_search):
        assert "is this worth deciding now" in system
        return _j(yes=voice_name(system) == "ride", reason="r")

    fake.script = script
    out = await cad.propose_cadence(stream, settings, "ride", "go now")
    assert out["status"] == "rejected"
    kinds = [(t["author"], t["kind"]) for t in await stream.thoughts()]
    assert kinds == [("ride", "voice"), ("cadence", "cadence")]


async def test_api_cadence_runs_in_background(stream, settings, fake):
    await add_voices(stream, "ride")

    async def script(system, user, json, web_search):
        if "standing" in system:
            return _j(weights={"ride": 1})
        return _j(vote="for", reason="yes")

    fake.script = script
    app = create_app(stream, settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        cid = (await c.post("/cadence", json={"resolution": "ride"})).json()["id"]
        for _ in range(50):
            if (await c.get(f"/cadence/{cid}")).json()["status"] == "resolved":
                break
            await asyncio.sleep(0.1)
        assert (await c.get(f"/cadence/{cid}")).json()["result"]["passed"]


async def test_sense_appends_world_thought_once(stream, settings, fake):
    async with stream.sessions() as s:
        s.add(SenseRow(name="ai-policy", instruction="AI policy news", every_minutes=240))
        await s.commit()

    async def script(system, user, json, web_search):
        assert web_search
        return _j(items=[
            {"title": "Regulator publishes rules", "url": "https://ex.com/a", "summary": "S."},
            {"title": "Another thing", "url": "https://ex.com/b", "summary": "T."},
        ])

    fake.script = script
    sense = Sense(SenseSpec("ai-policy", "AI policy news", 240), stream, settings)
    first = await sense.fetch_once()
    second = await sense.fetch_once()
    assert len(first) == 2 and second == []
    ts = await stream.thoughts()
    assert [t["kind"] for t in ts] == ["world", "world"]
    assert ts[0]["author"] == "ai-policy" and ts[0]["meta"]["url"] == "https://ex.com/a"


async def test_sse_replays_and_streams(stream, settings, fake):
    # httpx's ASGI transport buffers whole responses, so serve for real.
    import socket

    import uvicorn

    await stream.append("conscious", "conscious", "before")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(stream, settings), host="127.0.0.1",
                                           port=port, log_level="warning"))
    serve = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    got = []

    async def read():
        async with httpx.AsyncClient(timeout=10) as c:
            async with c.stream("GET", f"http://127.0.0.1:{port}/stream?after=0") as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        got.append(json.loads(line[6:])["content"])
                        if len(got) == 2:
                            return

    try:
        task = asyncio.create_task(read())
        for _ in range(100):
            if got:
                break
            await asyncio.sleep(0.05)
        await stream.append("conscious", "conscious", "after")
        await asyncio.wait_for(task, 10)
        assert got == ["before", "after"]
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve, 10)


async def test_api_validation(stream, settings, fake):
    app = create_app(stream, settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post("/voices", json={"name": "Bad Name", "text": "x"})).status_code == 422
        assert (await c.post("/voices", json={"name": "conscious", "text": "x"})).status_code == 422
        assert (await c.post("/thoughts", json={"content": "x", "kind": "voice"})).status_code == 422
        r = await c.post("/senses", json={"name": "s", "instruction": "x", "every_minutes": 1})
        assert r.json()["every_minutes"] == 30
        assert (await c.get("/health")).json() == {"ok": True}
