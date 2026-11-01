"""End-to-end behaviour with real PostgreSQL/Redis and a scripted model.

The fake model plays the part of judgement: a voice speaks once after the
conscious part drops a thought in, then gets bored. That the mind then goes quiet
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
    """The stream part of a (plain-text) user message."""
    return user.split("## The stream (oldest first)")[1].split("## Now")[0].strip().splitlines()


def polite_script():
    """Speak once per new conscious thought, then get bored."""

    async def script(system, user, json):
        me = voice_name(user)
        if "Right now you are only listening" in system:
            lines = stream_lines(user)
            last_conscious = max((i for i, l in enumerate(lines) if "] conscious:" in l), default=-1)
            spoke_since = any(f"] {me}:" in l or f"] {me} (" in l
                              for l in lines[last_conscious + 1:])
            if last_conscious >= 0 and not spoke_since:
                return _j(speak=True, mood="curious", reason="new thought", pause_seconds=0)
            return _j(speak=False, mood="bored", reason="said my piece", pause_seconds=0)
        if "You decided to speak" in system:
            topic = [l for l in stream_lines(user) if "] conscious:" in l][-1].split(": ", 1)[1]
            return f"{me} thinks about: {topic}"
        raise AssertionError("unexpected prompt")

    return script


def _j(**kw):
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


def vote_script(supports):
    """Standing gives ride more weight; each voice supports the options in `supports`."""

    async def script(system, user, json):
        me = voice_name(user)
        if "First round: standing" in system:
            return _j(weights={"ride": {"weight": 0.9, "why": "stake"},
                               "rest": {"weight": 0.3, "why": "meh"}})
        if "Second round: vote" in system:
            return _j(support=supports[me], reason=f"{me} says so")
        raise AssertionError(system[:80])

    return script


async def test_cadence_with_options(stream, settings, fake):
    await add_voices(stream, "ride", "rest")
    fake.script = vote_script({"ride": ["A", "C"], "rest": ["B"]})
    assert await stream.claim_cadence()
    out = await cad.run_cadence(stream, settings, "Tomorrow?", ["ride", "study", "ride, then study"])
    assert out["status"] == "resolved" and out["options"][2] == "ride, then study"
    assert out["result"]["shares"] == [pytest.approx(0.75), pytest.approx(0.25), pytest.approx(0.75)]
    assert out["result"]["decided"] is False  # A and C tie
    last = (await stream.thoughts())[-1]
    assert last["kind"] == "cadence" and last["meta"]["votes"]["rest"]["support"] == [1]
    assert "UNDECIDED" in last["content"]
    assert not await stream.cadence_running()


async def test_clerk_calls_a_vote_when_due(stream, settings, fake):
    from polyphony import clerk as clerk_mod

    await add_voices(stream, "ride", "rest")
    votes = vote_script({"ride": ["A"], "rest": ["A"]})

    async def script(system, user, json):
        if "You are the clerk" in system:
            assert "still going" in system  # the every-15 check, mid-discussion
            return _j(call_vote=True, question="Tomorrow?", options=["ride", "study"])
        return await votes(system, user, json)

    fake.script = script
    clerk = clerk_mod.Clerk(stream, settings)
    await stream.append("conscious", "conscious", "ride or study tomorrow?")
    assert not await clerk.tick()  # first look only marks where it starts
    for i in range(clerk_mod.CHECK_EVERY - 1):
        await stream.append("ride", "voice", f"point {i}")
        assert not await clerk.tick()  # not quiet, not 15 new yet
    await stream.append("rest", "voice", "last point")
    assert await clerk.tick()  # 15 new thoughts: checked, and a vote ran
    last = (await stream.thoughts())[-1]
    assert last["kind"] == "cadence" and "DECIDED: A. ride" in last["content"]
    assert not await clerk.tick()  # nothing new since

    # quiet after new talk also triggers a check; the clerk may decline to call a vote
    async def no_vote(system, user, json):
        assert "You are the clerk" in system and "this discussion is over" in system
        return _j(call_vote=False, question="", options=[])

    fake.script = no_vote
    await stream.append("ride", "voice", "one more thing")
    monkey = clerk_mod.QUIET_SECONDS
    clerk_mod.QUIET_SECONDS = 0
    try:
        assert await clerk.tick()
    finally:
        clerk_mod.QUIET_SECONDS = monkey
    assert (await stream.thoughts())[-1]["kind"] == "voice"  # no vote this time


async def test_clerk_declining_mid_discussion_still_checks_once_quiet(stream, settings, fake):
    from polyphony import clerk as clerk_mod

    seen = []

    async def script(system, user, json):
        seen.append("still going" if "still going" in system else "over")
        return _j(call_vote=False, question="", options=[])

    fake.script = script
    clerk = clerk_mod.Clerk(stream, settings)
    await stream.append("conscious", "conscious", "which one?")
    assert not await clerk.tick()
    for i in range(clerk_mod.CHECK_EVERY):
        await stream.append("ride", "voice", f"point {i}")
    assert await clerk.tick() and seen == ["still going"]  # declined mid-discussion
    monkey = clerk_mod.QUIET_SECONDS
    clerk_mod.QUIET_SECONDS = 0
    try:
        assert await clerk.tick()  # nothing new, but the quiet check has not run yet
        assert not await clerk.tick()  # and it runs only once
    finally:
        clerk_mod.QUIET_SECONDS = monkey
    assert seen == ["still going", "over"]


async def test_clerk_waits_while_a_vote_is_under_way(stream, settings, fake):
    from polyphony import clerk as clerk_mod

    await add_voices(stream, "ride", "rest")

    async def script(system, user, json):
        return _j(call_vote=True, question="Tomorrow?", options=["ride", "study"])

    fake.script = script
    assert await stream.claim_cadence()  # something is already under way
    assert await clerk_mod.Clerk(stream, settings).check() is None
    assert not [t for t in await stream.thoughts() if t["kind"] == "cadence"]
    await stream.release_cadence()
    assert not await stream.cadence_running()


class FakeMCP:
    def __init__(self, feeds):
        self.feeds = feeds
        self.calls = []

    def has(self, server):
        return server in ("feeds", "fetch")

    async def call(self, server, tool, args):
        self.calls.append((server, tool, args))
        if server == "feeds":
            return json.dumps(self.feeds[args["url"]])
        if server == "fetch":
            return f"Full text of {args['url']}."
        raise AssertionError(server)


async def test_sense_reads_feeds_and_appends_world_thought_once(stream, settings, fake):
    feed = {"feed": "Ex", "entries": [
        {"id": "1", "title": "Regulator publishes rules", "link": "https://ex.com/a",
         "summary": "S.", "published": ""},
        {"id": "2", "title": "Celebrity news", "link": "https://ex.com/b", "summary": "T.",
         "published": ""},
    ]}
    mcp = FakeMCP({"https://ex.com/rss": feed})

    async def script(system, user, json):
        if '"pick"' in system:
            assert "Regulator" in user and "<entry" in user
            return _j(pick=[0])
        if '"items"' in system:
            assert "Full text of https://ex.com/a." in user and "<article" in user
            return _j(items=[{"title": "Regulator publishes rules", "url": "https://ex.com/a",
                              "summary": "Rules."}])
        raise AssertionError(system[:80])

    fake.script = script
    async with stream.sessions() as s:
        s.add(SenseRow(name="ai-policy", instruction="AI policy news",
                       feeds=["https://ex.com/rss"]))
        await s.commit()
    sense = Sense(SenseSpec("ai-policy", "AI policy news", "04:00", ("https://ex.com/rss",)),
                  stream, settings)
    first = await sense.fetch_once(mcp=mcp)
    second = await sense.fetch_once(mcp=mcp)
    assert len(first) == 1 and second == []
    ts = await stream.thoughts()
    assert [t["kind"] for t in ts] == ["world"]
    assert ts[0]["author"] == "ai-policy" and ts[0]["meta"]["url"] == "https://ex.com/a"


async def test_digest_summarises_and_does_not_wake_voices(stream, mind, fake, settings):
    from polyphony.digest import write_digest

    await stream.append("conscious", "conscious", "should I move?")
    await wait_quiet(stream)
    n_listens = len(await stream.listens())
    base = fake.script

    async def script(system, user, json):
        if "You are the digest" in system:
            return _j(decided=False, outcome="split.", question="move?",
                      positions=[{"voice": "ride", "view": "go"}], open=["when"])
        return await base(system=system, user=user, json=json)

    fake.script = script
    t = await write_digest(stream, settings)
    assert t["kind"] == "digest" and t["meta"]["outcome"] == "positions"
    assert t["meta"]["summary"]["positions"] == [{"voice": "ride", "view": "go"}]
    assert t["content"].startswith("Not decided: split.") and "Still open: when" in t["content"]
    await asyncio.sleep(1.0)
    assert len(await stream.listens()) == n_listens  # nobody woke up
    assert await write_digest(stream, settings) is None  # nothing new since


async def test_key_comes_from_the_file_and_is_read_only(stream, settings, fake):
    from polyphony.main import seed

    settings.frame_file.parent.mkdir(parents=True, exist_ok=True)
    settings.frame_file.write_text("Voices of one mind.")
    await seed(stream, settings)
    settings.frame_file.write_text("Voices of one mind, revised.")
    await seed(stream, settings)  # every start copies the file again
    app = create_app(stream, settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.get("/frame")).json()["text"] == "Voices of one mind, revised."
        assert (await c.put("/frame", json={"text": "x"})).status_code == 405


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
                assert r.headers["content-type"].startswith("text/event-stream")
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
        r = await c.post("/senses", json={"name": "s", "instruction": "x", "daily_at": "4:05"})
        assert r.json()["daily_at"] == "04:05" and r.json()["enabled"] is True
        assert (await c.post("/senses", json={"name": "t", "instruction": "x",
                                              "daily_at": "25:00"})).status_code == 422
        assert (await c.post("/senses/nope/run")).status_code == 404
        assert (await c.get("/health")).json() == {"ok": True}


async def test_pause_stops_everything_and_resume_restarts(stream, settings, mind, fake):
    app = create_app(stream, settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        assert (await c.post("/pause")).json() == {"paused": True}
        for _ in range(50):
            if not mind.voices and not mind.edges:
                break
            await asyncio.sleep(0.1)
        assert not mind.voices and not mind.edges
        assert (await c.get("/pause")).json() == {"paused": True}
        # Redis can be wiped (it has no volume); the flag lives in PostgreSQL.
        async for k in stream.redis.scan_iter(match=f"{stream.namespace}:*"):
            await stream.redis.delete(k)
        assert await stream.paused()

        # a thought still lands while paused, but nobody answers it
        await c.post("/thoughts", json={"content": "anyone there?"})
        await asyncio.sleep(0.5)
        assert not [t for t in await stream.thoughts() if t["kind"] == "voice"]

        assert (await c.post("/resume")).json() == {"paused": False}
        for _ in range(50):
            if len(mind.voices) == 3:
                break
            await asyncio.sleep(0.1)
        assert sorted(mind.voices) == ["make", "rest", "ride"]


async def test_config_api_saves_models_and_never_returns_keys(stream, settings):
    app = create_app(stream, settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        body = {"provider": "anthropic", "api_key": "sk-secret-9876",
                "listen_model": "claude-haiku-5-5", "speak_model": "claude-sonnet-5-5"}
        assert (await c.put("/config", json={**body, "provider": "fake"})).status_code == 422
        r = await c.put("/config", json=body)
        assert r.status_code == 200 and "sk-secret-9876" not in r.text
        got = (await c.get("/config")).json()
        assert got["api_key"]["hint"] == "…9876" and got["speak_model"] == "claude-sonnet-5-5"
        assert "sk-secret-9876" not in json.dumps(got)
        assert (await c.post("/config/test/nope")).status_code == 404
        await c.put("/config", json={"api_key": ""})
        assert (await c.get("/config")).json()["api_key"]["source"] != "page"
    from polyphony import runtime_config
    await runtime_config.save(stream.sessions, {})
    runtime_config.apply(settings, {})


async def test_running_vote_is_listed_for_the_page_until_resolved(stream, settings, fake):
    cid = await cad.create_cadence(stream, "Which one?", ["a", "b"], "clerk", "running")
    app = create_app(stream, settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        [r] = (await c.get("/cadences/running")).json()
        assert r["id"] == cid and r["options"] == ["a", "b"] and r["round"] == 1
        await cad.set_fields(stream, cid, weights={"final": {}})
        assert (await c.get("/cadences/running")).json()[0]["round"] == 2
        await cad.set_fields(stream, cid, status="resolved")
        assert (await c.get("/cadences/running")).json() == []
