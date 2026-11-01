"""Pure-function tests: JSON extraction, tier resolution, voice parsing, cadence math."""

import json

import pytest

from polyphony import prompts
from polyphony.cadence import average_weights, tally
from polyphony.config import TierConfig
from polyphony.llm import CACHE_BREAK, CHUNK, LLMError, extract_json, resolve_tier
from polyphony.sense import fingerprints
from polyphony.voice import VoiceSpec, normalize_decision


def test_extract_json_from_chatty_reply():
    text = 'Sure! Here it is:\n```json\n{"speak": false, "reason": "a {brace} in a string"}\n```'
    assert extract_json(text) == {"speak": False, "reason": "a {brace} in a string"}


def test_extract_json_skips_broken_object():
    assert extract_json('{not json} then {"ok": 1}') == {"ok": 1}


def test_extract_json_raises():
    with pytest.raises(LLMError):
        extract_json("no object here")


def test_resolve_tier_falls_back_and_switches_provider():
    base = TierConfig(provider="openai", model="gpt-4o-mini", api_key="k", base_url=None)
    assert resolve_tier(base, None) == base
    same = resolve_tier(base, {"model": "gpt-4o"})
    assert (same.provider, same.model, same.api_key) == ("openai", "gpt-4o", "k")
    other = resolve_tier(base, {"provider": "deepseek", "model": "deepseek-chat"})
    assert other.api_key is None  # never leak one vendor's key to another


def test_voice_described_with_front_matter():
    text, meta = VoiceSpec.parse("---\nlisten: { provider: deepseek, model: deepseek-chat }\n---\n"
                                 "**Wants** — rest.")
    assert text == "**Wants** — rest."
    assert meta == {"listen": {"provider": "deepseek", "model": "deepseek-chat"}}


def test_voice_shown_with_empty_front_matter():
    text, meta = VoiceSpec.parse('---\n---\n- 11pm: "ten more minutes"')
    assert text == '- 11pm: "ten more minutes"' and meta == {}


def test_voice_without_front_matter():
    assert VoiceSpec.parse("just text") == ("just text", {})


def test_normalize_decision_is_lenient():
    d = normalize_decision({"speak": "yes", "pause_seconds": "3"})
    assert d == {"speak": True, "mood": "", "reason": "", "pause_seconds": 3.0}

def test_average_weights_normalises():
    names = ["a", "b"]
    w = average_weights(names, [
        {"weights": {"a": {"weight": 1.0}, "b": {"weight": 0.0}}},
        {"weights": {"a": 0.5, "b": 0.5, "stranger": 1}},
        None,
    ])
    assert w == pytest.approx({"a": 0.75, "b": 0.25})


def test_average_weights_all_zero_is_equal():
    assert average_weights(["a", "b"], [None]) == {"a": 0.5, "b": 0.5}


def test_tally_multi_option_approval():
    from polyphony.cadence import _normalize_vote

    opts = ["ride", "study", "both"]
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    votes = {"a": _normalize_vote({"support": ["C"], "reason": "x"}, opts),
             "b": _normalize_vote({"support": ["b", "3"], "reason": "x"}, opts),
             "c": _normalize_vote(None, opts)}  # no answer: not counted
    r = tally(w, votes, opts)
    assert r["shares"] == [0.0, pytest.approx(0.375), 1.0] and r["winner"] == 2 and r["decided"]
    # a tie at the top, or nothing above half, decides nothing
    tie = {"a": {"support": [0], "reason": ""}, "b": {"support": [1], "reason": ""}}
    assert not tally({"a": 0.5, "b": 0.5}, tie, opts)["decided"]
    low = {"a": {"support": [0], "reason": ""}, "b": {"support": [], "reason": ""}}
    assert not tally({"a": 0.4, "b": 0.6}, low, opts)["decided"]

def test_fingerprints_match_on_url_or_title():
    a = fingerprints({"title": "Big News", "url": "https://www.x.com/a/"})
    b = fingerprints({"title": "big   news", "url": "http://x.com/other"})
    assert a[1] == b[1] and a[0] != b[0]


def test_render_marks_self_and_kinds():
    recent = [
        {"seq": 1, "author": "conscious", "kind": "conscious", "content": "hi", "mood": None},
        {"seq": 2, "author": "rest", "kind": "voice", "content": "tired", "mood": "flat"},
    ]
    listens = {"rest": {"speak": False, "mood": "tired", "reason": "third loop", "trigger_seq": 2}}
    out = prompts.render(recent, [], listens, ["make"], "rest", voice_text="**Wants** — rest.")
    assert "[1] conscious: hi" in out
    assert "[2] rest (flat): tired" in out  # the stream is the same for every voice
    assert "rest (you): quiet — tired (at [2])" in out
    assert 'Your name is "rest"' in out.split(CACHE_BREAK)[1]
    assert "third loop" not in out
    assert "make" in out.split("## Forming a thought right now")[1]


def test_prompt_prefix_is_shared_across_voices():
    from datetime import datetime, timezone

    from polyphony.frame import Frame

    k = Frame(text="A key.")
    assert "A key." in prompts.listen_system(k) and "only listening" in prompts.listen_system(k)
    now = datetime.now(timezone.utc).isoformat()
    recent = [{"seq": i, "author": "a", "kind": "voice", "content": str(i), "mood": None,
               "created_at": now} for i in range(1, 6)]
    a = prompts.render(recent, [], {}, [], "rest", voice_text="**Wants** — rest.")
    b = prompts.render(recent, [], {}, ["rest"], "make", voice_text="**Wants** — ship.")
    # system and stream are identical for every voice; only what follows the break differs
    assert a.split(CACHE_BREAK)[0] == b.split(CACHE_BREAK)[0]
    assert a.split(CACHE_BREAK)[1] != b.split(CACHE_BREAK)[1]


def test_cache_window_moves_in_steps():
    seqs = lambda n: [{"seq": i} for i in range(1, n + 1)]
    assert [t["seq"] for t in prompts.cache_window(seqs(30))] == list(range(1, 31))
    shown = [t["seq"] for t in prompts.cache_window(seqs(75))]
    assert shown[0] == 21 and shown[-1] == 75  # starts on a step, keeps at least 40
    assert prompts.cache_window(seqs(79))[0]["seq"] == 21  # same start: prefix still cached
    assert prompts.cache_window(seqs(80))[0]["seq"] == 41


def test_anthropic_content_marks_the_shared_prefix():
    from polyphony.llm import anthropic_content, plain

    user = prompts.render([{"seq": 1, "author": "a", "kind": "voice", "content": "x",
                            "mood": None}], [], {}, [], "a", voice_text="v")
    blocks = anthropic_content(user)
    assert blocks[-2]["cache_control"] == {"type": "ephemeral"} and "cache_control" not in blocks[-1]
    assert CACHE_BREAK not in plain(user) and CHUNK not in plain(user)


def test_engine_text_is_framing_neutral():
    for text in (prompts.LISTEN_ENGINE, prompts.SPEAK_ENGINE, prompts.CADENCE_ENGINE,
                 prompts.DIGEST_ENGINE, prompts.SENSE_ENGINE):
        low = text.lower()
        assert not any(w in low for w in (" mind", " head", "people", "groups")), text[:60]


def test_feeds_ssrf_guard_blocks_private_addresses():
    import asyncio

    from polyphony.mcp_servers.feeds import BlockedURL, check_url

    for url in ("http://127.0.0.1/rss", "http://localhost/x", "file:///etc/passwd",
                "http://169.254.169.254/latest"):
        with pytest.raises(BlockedURL):
            asyncio.run(check_url(url))


def test_render_shows_absolute_times_and_now():
    recent = [{"seq": 1, "author": "doubt", "kind": "voice", "content": "old", "mood": None,
               "created_at": "2026-10-29T12:40:00+00:00"}]
    out = prompts.render(recent, [], {}, [], None, tz="America/Toronto")
    assert "[1 · 2026-10-29 08:40] doubt: old" in out  # local time, never "3 days ago"
    assert "## Now" in out.split(CACHE_BREAK)[1] and "(America/Toronto)" in out

def test_runtime_config_overrides_env_and_restores_it(settings):
    from polyphony import runtime_config as rc

    env_speak = (settings.speak_provider, settings.speak_model)
    data = rc.merge({}, {"provider": "anthropic", "api_key": " sk-test-1234 ",
                         "listen_model": "claude-haiku-5-5", "speak_model": "claude-sonnet-5-5"})
    rc.apply(settings, data)
    assert (settings.listen_tier.provider, settings.listen_tier.model) == ("anthropic", "claude-haiku-5-5")
    assert settings.speak_tier.api_key == settings.listen_tier.api_key == "sk-test-1234"
    assert settings.sense_tier.model == "claude-sonnet-5-5"  # senses follow speak
    assert settings.sense_tier.provider == "anthropic"
    view = rc.view(settings, data)
    assert view["api_key"] == {"set": True, "source": "page", "hint": "…1234"}
    assert "sk-test-1234" not in json.dumps(view)
    # switching provider drops the old vendor's key
    assert "api_key" not in rc.merge(data, {"provider": "deepseek"})
    rc.apply(settings, {})
    assert (settings.speak_provider, settings.speak_model) == env_speak


@pytest.mark.parametrize("patch", [
    {"provider": "nope", "listen_model": "a", "speak_model": "b"},
    {"listen_model": "a", "speak_model": "b"},
    {"provider": "anthropic", "listen_model": "a"},
])
def test_runtime_config_rejects_bad_edits(patch):
    from polyphony import runtime_config as rc

    with pytest.raises(ValueError):
        rc.merge({}, patch)


def test_sense_runs_at_the_next_daily_time_only():
    from datetime import datetime, timezone

    from polyphony.sense import check_daily_at, next_run

    tz = "America/Toronto"
    # 03:00 Toronto (07:00 UTC in November): today's 04:00 is still ahead.
    assert next_run("04:00", tz, datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)) == \
        datetime(2026, 11, 2, 9, 0, tzinfo=timezone.utc)
    # just after 04:00 Toronto: tomorrow, never "right now"
    assert next_run("04:00", tz, datetime(2026, 11, 2, 9, 1, tzinfo=timezone.utc)) == \
        datetime(2026, 11, 3, 9, 0, tzinfo=timezone.utc)
    assert check_daily_at("4:05") == "04:05"
    with pytest.raises(ValueError):
        check_daily_at("24:00")


async def test_anthropic_request_is_cached_and_has_room_to_think():
    from types import SimpleNamespace

    from polyphony import llm

    sent = {}

    async def create(**kw):
        sent.update(kw)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"speak": false}')], stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0))

    p = llm.AnthropicProvider(None, "test-key")
    p.client.messages.create = create
    p.client.beta.messages.create = create
    user = prompts.render([{"seq": 1, "author": "a", "kind": "voice", "content": "x",
                            "mood": None}], [], {}, [], "a", voice_text="v")
    await p.complete(model="claude-haiku-5-5", system="sys", user=user, json=True)
    assert sent["max_tokens"] >= llm.ANTHROPIC_MIN_MAX_TOKENS
    assert sent["output_config"] == {"effort": "low"}
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert any("cache_control" in b for b in sent["messages"][0]["content"])
    assert "fallbacks" not in sent  # Haiku has no server-side fallback
    await p.complete(model="claude-sonnet-5-5", system="sys", user=user)
    assert sent["fallbacks"] == "default"
