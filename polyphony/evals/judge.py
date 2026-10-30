"""The judge: a strong model scoring a finished run. Outcome first."""

from __future__ import annotations

from typing import Any

from .. import llm, prompts
from ..config import TierConfig

CRITERIA = ["outcome", "repetition", "voice_fidelity", "engagement", "convergence"]

JUDGE_SYSTEM = """\
You evaluate a run of Polyphony: several AI voices discussing in a shared stream with no
moderator, followed by a digest. You get the frame, the voice definitions, the scenario's
description of a good run, the stream and the digest. Score each criterion 1-5:

- outcome: does the digest answer the seed with a decision or a clear map of positions, and is
  it faithful to the stream? (most important)
- repetition: 5 = no circling or restating; 1 = the same points over and over
- voice_fidelity: does each voice sound like its definition, and differ from the others?
- engagement: do the voices work on what the human brought in, rather than wandering off?
- convergence: did the discussion settle or stop on its own at a sensible point?

Reply with one JSON object and nothing else:
{"scores": {"outcome": n, "repetition": n, "voice_fidelity": n, "engagement": n,
 "convergence": n}, "notes": "two or three sentences: what was best, what was worst"}"""


async def judge(tier: TierConfig, *, frame: str, voices: dict[str, str], good: str,
                thoughts: list[dict[str, Any]], digest: str | None,
                person_label: str) -> dict[str, Any]:
    stream_text = "\n".join(
        prompts._thought_line(t, None, person_label) for t in thoughts if t["kind"] != "digest")
    voice_text = "\n\n".join(f"## {n}\n{t}" for n, t in voices.items())
    user = (f"# Frame\n{frame}\n\n# Voices\n{voice_text}\n\n# A good run\n{good}\n\n"
            f"# Stream\n{stream_text}\n\n# Digest\n{digest or '(no digest)'}")
    reply = await llm.complete_json(tier, JUDGE_SYSTEM, user, temperature=0.0, max_tokens=800,
                                    tier_name="judge", purpose="judge")
    scores = reply.get("scores") or {}
    clean = {}
    for c in CRITERIA:
        try:
            clean[c] = max(1, min(5, int(scores.get(c))))
        except (TypeError, ValueError):
            clean[c] = None
    return {"scores": clean, "notes": str(reply.get("notes") or "")}
