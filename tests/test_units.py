"""Pure-function tests: JSON extraction, tier resolution, voice parsing, cadence math."""

import pytest

from polyphony import prompts
from polyphony.cadence import average_weights, tally
from polyphony.config import TierConfig
from polyphony.llm import LLMError, extract_json, resolve_tier
from polyphony.sense import fingerprints
from polyphony.voice import VoiceSpec, normalize_decision, repeats_proposal


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
    text, meta = VoiceSpec.parse("---\nlisten: { provider: ollama, model: qwen2.5:3b }\n---\n"
                                 "**Wants** — rest.")
    assert text == "**Wants** — rest."
    assert meta == {"listen": {"provider": "ollama", "model": "qwen2.5:3b"}}


def test_voice_shown_with_empty_front_matter():
    text, meta = VoiceSpec.parse('---\n---\n- 11pm: "ten more minutes"')
    assert text == '- 11pm: "ten more minutes"' and meta == {}


def test_voice_without_front_matter():
    assert VoiceSpec.parse("just text") == ("just text", {})


def test_normalize_decision_is_lenient():
    d = normalize_decision({"speak": "yes", "pause_seconds": "3", "propose_cadence": "null"})
    assert d == {"speak": True, "mood": "", "reason": "", "pause_seconds": 3.0,
                 "propose_cadence": None}


@pytest.mark.parametrize("raw", [
    "a resolution, phrased so it can be voted for or against",
    '"A resolution, phrased so it can be voted for or against."',
    "cadence",
    "I think we should decide this: a resolution, phrased so it can be voted for or against",
    "Cadence on: 'a resolution, phrased so it can be voted for or against'",
])
def test_normalize_decision_drops_placeholder_proposals(raw):
    assert normalize_decision({"propose_cadence": raw})["propose_cadence"] is None


def test_normalize_decision_keeps_real_proposals():
    d = normalize_decision({"propose_cadence": " I think we should decide this: 明早骑车 "})
    assert d["propose_cadence"] == "明早骑车"


def test_repeats_proposal_only_looks_at_proposals_and_cadences():
    recent = [
        {"seq": 1, "kind": "voice", "author": "doubt", "content": "Let's vote on: ride at 7."},
        {"seq": 2, "kind": "voice", "author": "ride",
         "content": "I think we should decide this: skip class and ride"},
        {"seq": 3, "kind": "cadence", "author": "cadence",
         "content": 'Proposal by learn not taken up (0 of 3): "study all Saturday"'},
    ]
    assert repeats_proposal("Skip class and ride.", recent)
    assert repeats_proposal("study all Saturday", recent)
    assert not repeats_proposal("ride at 7", recent)


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


def test_tally_ignores_abstain_in_share():
    r = tally({"a": 0.5, "b": 0.3, "c": 0.2},
              {"a": {"vote": "for"}, "b": {"vote": "against"}, "c": {"vote": "abstain"}})
    assert r["share_for"] == pytest.approx(0.5 / 0.8, abs=1e-3) and r["passed"]


def test_fingerprints_match_on_url_or_title():
    a = fingerprints({"title": "Big News", "url": "https://www.x.com/a/"})
    b = fingerprints({"title": "big   news", "url": "http://x.com/other"})
    assert a[1] == b[1] and a[0] != b[0]


def test_render_marks_self_and_kinds():
    recent = [
        {"seq": 1, "author": "person", "kind": "person", "content": "hi", "mood": None},
        {"seq": 2, "author": "rest", "kind": "voice", "content": "tired", "mood": "flat"},
    ]
    listens = {"rest": {"speak": False, "mood": "tired", "reason": "third loop", "trigger_seq": 2}}
    out = prompts.render(recent, [], listens, ["make"], "rest", person_label="conscious")
    assert "[1] [conscious] conscious: hi" in out
    assert "[2] rest (you) (flat): tired" in out
    assert 'rest (you): quiet — tired — "third loop"' in out
    assert "make" in out.split("## Forming a thought right now")[1]


def test_prompt_prefix_is_shared_across_voices():
    from polyphony.frame import Frame

    f = Frame("mind", "mind", "conscious", "A frame.")
    a = prompts.listen_system(f, "rest", "**Wants** — rest.")
    b = prompts.listen_system(f, "make", "**Wants** — ship.")
    prefix = a[: a.index('Your name is')]
    assert b.startswith(prefix) and "A frame." in prefix and "only listening" in prefix


def test_engine_text_is_framing_neutral():
    for text in (prompts.LISTEN_ENGINE, prompts.SPEAK_ENGINE, prompts.CADENCE_ENGINE,
                 prompts.DIGEST_ENGINE, prompts.SENSE_ENGINE):
        low = text.lower()
        assert not any(w in low for w in (" mind", " head", "people", "groups")), text[:60]


def test_agreement_metrics():
    from polyphony.distill import agreement

    pairs = [({"speak": True}, {"speak": True}), ({"speak": False}, {"speak": True}),
             ({"speak": False}, {"speak": False}), ({"speak": True}, {"speak": False}),
             (None, {"speak": True})]
    m = agreement(pairs)
    assert m["n"] == 4 and m["accuracy"] == 0.5
    assert m["precision_speak"] == 0.5 and m["recall_speak"] == 0.5


def test_feeds_ssrf_guard_blocks_private_addresses():
    import asyncio

    from polyphony.mcp_servers.feeds import BlockedURL, check_url

    for url in ("http://127.0.0.1/rss", "http://localhost/x", "file:///etc/passwd",
                "http://169.254.169.254/latest"):
        with pytest.raises(BlockedURL):
            asyncio.run(check_url(url))
