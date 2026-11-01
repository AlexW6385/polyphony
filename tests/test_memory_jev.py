"""Memory (real pgvector, a deterministic fake embedder), the listening state, the stagger,
and the Jev decider (Jev's HTTP API faked; the fallback to the listen model is real)."""

import asyncio
import hashlib
import json
import math
import re

import pytest
from sqlalchemy import text

from polyphony import jev, prompts, runtime_config
from polyphony.frame import Frame
from polyphony.memory import Memory
from polyphony.voice import Voice, VoiceSpec, stagger_window

DIM = 16


class FakeEmbedder:
    """Bag of words hashed into DIM buckets: texts sharing words point the same way."""

    model = "fake-embed"

    def __init__(self):
        self.calls = 0

    def query_text(self, q):
        return q

    async def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in re.findall(r"\w+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    async def aclose(self):
        pass


@pytest.fixture
def mem_settings(settings):
    return settings.model_copy(update={"embed_dim": DIM, "memory_k": 2, "memory_query_last": 1,
                                       "memory_min_similarity": 0.3})


async def _pgvector_or_skip(stream):
    try:
        async with stream.sessions() as s:
            await s.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await s.commit()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pgvector not available: {e}")


@pytest.fixture
async def memory(stream, mem_settings):
    await _pgvector_or_skip(stream)
    m = Memory(stream, mem_settings, embedder=FakeEmbedder())
    assert await m.setup()
    return m


# --------------------------------------------------------------------------- memory


async def test_memory_recalls_related_older_thoughts_only(stream, memory):
    await stream.append("conscious", "conscious", "cycling along the river trail this weekend")
    await stream.append("ride", "voice", "badminton with friends is also fun")
    await stream.append("rest", "voice", "the resume deadline is on monday")
    await stream.append("cadence", "cadence", "Cadence on: cycling river trail")  # not embedded
    await stream.append("make", "voice", "what about cycling the river trail on saturday")
    while await memory.embed_pending():
        pass
    async with stream.sessions() as s:
        kinds = (await s.execute(text(
            "SELECT t.kind FROM thought_embeddings e JOIN thoughts t USING (seq)"))).scalars().all()
    assert "cadence" not in kinds and len(kinds) == 4

    shown = [t for t in await stream.recent() if t["seq"] >= 5]  # the window is the last one
    got = await memory.search(shown)
    best = max(got, key=lambda g: g["similarity"])
    assert best["seq"] == 1  # the cycling thought ranks first
    assert all(g["similarity"] >= 0.3 for g in got)  # weak matches are dropped
    assert all(g["seq"] < 5 and g["kind"] != "cadence" for g in got)

    # Older than the window only: with everything in the window there is nothing to recall.
    assert await memory.search(await stream.recent()) == []


async def test_recall_is_cached_per_stream_state(stream, memory):
    await stream.append("conscious", "conscious", "learn rust or go")
    await stream.append("ride", "voice", "rust is fun to learn")
    while await memory.embed_pending():
        pass
    shown = [t for t in await stream.recent() if t["seq"] == 2]
    calls = memory.embedder.calls
    a = await asyncio.gather(*[memory.recall(shown) for _ in range(5)])
    assert all(x == a[0] for x in a) and memory.embedder.calls == calls + 1


async def test_changed_dimension_rebuilds_and_changed_model_reembeds(stream, memory, mem_settings):
    await stream.append("conscious", "conscious", "hello there")
    await memory.embed_pending()
    other = FakeEmbedder()
    other.model = "other-embed"
    m2 = Memory(stream, mem_settings, embedder=other)
    assert await m2.setup()
    assert len(await m2.pending()) == 1  # old model's vectors were dropped
    m3 = Memory(stream, mem_settings.model_copy(update={"embed_dim": DIM * 2}),
                embedder=other)
    assert await m3.setup()
    async with stream.sessions() as s:
        size = await s.scalar(text("SELECT atttypmod FROM pg_attribute WHERE attrelid = "
                                   "to_regclass('thought_embeddings') AND attname = 'embedding'"))
    assert size == DIM * 2


async def test_recall_never_raises_when_embedding_fails(stream, memory):
    await stream.append("conscious", "conscious", "a")
    await stream.append("ride", "voice", "b")

    async def broken(texts):
        raise ConnectionError("ollama is down")

    memory.embedder.embed = broken
    assert await memory.recall([t for t in await stream.recent() if t["seq"] == 2]) == []


# --------------------------------------------------------------------------- listening, stagger


async def test_listening_set_and_pause_clears_it(stream):
    await stream.set_listening("ride", True)
    await stream.set_listening("rest", True)
    assert await stream.listening() == ["rest", "ride"]
    await stream.set_listening("ride", False)
    assert await stream.listening() == ["rest"]
    await stream.set_paused(True)
    assert await stream.listening() == []
    await stream.set_paused(False)


def test_stagger_scales_with_voice_count(settings):
    s = settings.model_copy(update={"pause_scale": 1.0})
    assert stagger_window(s, 10) == 5.0
    assert stagger_window(s, 4) == 2.0
    assert stagger_window(s.model_copy(update={"listen_stagger_per_voice": 0.2}), 10) == 2.0


def test_render_shows_listening_and_memory():
    t = {"seq": 3, "author": "ride", "kind": "voice", "content": "go cycling", "mood": None,
         "created_at": None}
    old = dict(t, seq=1, content="rode last week")
    out = prompts.render([t], [], {}, ["make"], "rest", voice_text="x", listening=["rest", "ride"],
                         memories=[old])
    assert "## Deciding right now whether to speak\nride" in out  # not yourself
    assert "## Memory" in out and "rode last week" in out
    plain = prompts.render([t], [], {}, [], "rest", voice_text="x")
    assert "## Memory" not in plain and "Deciding right now" not in plain


async def test_voice_marks_itself_listening_while_deciding(stream, settings, fake):
    seen = []

    async def script(system, user, json):
        seen.append(await stream.listening())
        return '{"speak": false, "mood": "calm", "reason": "-", "pause_seconds": 0}'

    fake.script = script
    await stream.append("conscious", "conscious", "hi")
    v = Voice(VoiceSpec("ride", "likes cycling"), stream, settings)
    await v.listen()
    assert seen == [["ride"]] and await stream.listening() == []


# --------------------------------------------------------------------------- Jev


def test_jev_pause_interpolates_levels():
    assert jev.pause_seconds(0) == 0.0
    assert jev.pause_seconds(1) == 5.0
    assert jev.pause_seconds(1.5) == 10.0
    assert jev.pause_seconds(3) == 30.0
    assert jev.pause_seconds(9) == 30.0


def test_jev_answers_map_to_the_existing_shapes():
    d = jev.listen_decision({"speak": {"noul": 0.7}, "pause": {"score": 2.0},
                             "mood": {"choice": "eager"}}, threshold=0.5)
    assert d == {"speak": True, "mood": "eager", "reason": "jev p(speak)=0.70",
                 "pause_seconds": 15.0}
    v = jev.vote_answer({"o0": {"noul": 0.9}, "o1": {"noul": 0.2}, "o2": {"noul": 0.5}},
                        ["a", "b", "c"], 0.5)
    assert v["support"] == ["A", "C"]
    w = jev.standing_answer({"w_ride": {"score": 2}, "w_rest": {"score": 1}}, ["ride", "rest"])
    assert w == {"weights": {"ride": 1.0, "rest": 0.5}}
    qs = jev.listen_questions("ride")
    assert {q["type"] for q in qs.values()} == {"noul", "score", "choice"}
    assert len(qs["pause"]["criteria"]) == len(jev.PAUSE_LEVELS)


async def test_voice_listens_through_jev_and_falls_back_to_the_model(stream, settings, fake,
                                                                     monkeypatch):
    s = settings.model_copy(update={"decider": "jev", "jev_api_key": "k"})
    sent = []

    async def fake_ask(settings_, state, questions, *, purpose, voice=None):
        sent.append((state, questions, purpose, voice))
        return {"speak": {"noul": 0.1}, "pause": {"score": 0}, "mood": {"choice": "bored"}}

    monkeypatch.setattr(jev, "ask", fake_ask)
    fake.script = None  # the listen model must not be called
    await stream.append("conscious", "conscious", "should I go cycling?")
    v = Voice(VoiceSpec("ride", "likes cycling"), stream, s, Frame(text="one mind"))
    d = await v.listen()
    assert d["speak"] is False and d["mood"] == "bored" and "0.10" in d["reason"]
    state, questions, purpose, voice = sent[0]
    assert purpose == "listen" and voice == "ride"
    assert state["you"] == "ride" and "should I go cycling?" in state["stream"][0]
    assert "who_you_are" in state and set(questions) == {"speak", "pause", "mood"}

    async def broken(*a, **kw):
        raise jev.JevError("HTTP 503")

    async def script(system, user, json):
        return '{"speak": false, "mood": "calm", "reason": "model", "pause_seconds": 0}'

    monkeypatch.setattr(jev, "ask", broken)
    fake.script = script
    d = await v.listen()
    assert d["reason"] == "model"


async def test_jev_request_shape(settings, monkeypatch):
    """The body sent to /v1/systemone and how answers come back."""
    import httpx

    captured = {}

    def handler(request):
        captured["auth"] = request.headers["authorization"]
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"model": "jev-latest", "usage": {"input_tokens": 9},
                                         "answers": {"t": {"type": "noul", "noul": 0.8}}})

    s = settings.model_copy(update={"decider": "jev", "jev_api_key": "secret"})
    client = httpx.AsyncClient(base_url="https://api.typesafe.ai",
                               transport=httpx.MockTransport(handler),
                               headers={"Authorization": "Bearer secret"})
    monkeypatch.setattr(jev, "_client", lambda settings_: client)
    a = await jev.ask(s, {"x": 1}, {"t": {"type": "noul", "instructions": "?"}}, purpose="test")
    assert a["t"]["noul"] == 0.8
    assert captured["path"] == "/v1/systemone" and captured["auth"] == "Bearer secret"
    assert captured["body"] == {"model": "jev-latest", "state": {"x": 1},
                                "questions": {"t": {"type": "noul", "instructions": "?"}}}


def test_runtime_config_switches_decider_and_keeps_the_jev_key_secret(settings):
    data = runtime_config.merge({}, {"provider": "anthropic", "listen_model": "a",
                                     "speak_model": "b", "decider": "jev",
                                     "jev_api_key": "ts-123456"})
    runtime_config.apply(settings, data)
    assert settings.decider == "jev" and settings.jev_api_key == "ts-123456"
    view = runtime_config.view(settings, data)
    assert view["decider"] == "jev" and view["jev_key"] == {"set": True, "source": "page",
                                                            "hint": "…3456"}
    assert "ts-123456" not in json.dumps(view)
    runtime_config.apply(settings, {})
    assert settings.decider == "llm"
    with pytest.raises(ValueError):
        runtime_config.merge(data, {"decider": "oracle"})
