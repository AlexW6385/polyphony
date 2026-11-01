"""Jev: TypeSafe AI's decision model, as an optional backend for the calls that only decide.

Jev does not write text. It takes a *state* (text or JSON) and typed *questions*, and returns
typed answers with probabilities, in well under a second:

    noul    is this statement true?           -> probability 0..1
    choice  which of these options?           -> the pick and a probability per option
    score   where on this ordered scale?      -> a fractional level and probabilities

With `DECIDER=jev` (or the switch on the settings page) these calls go to Jev instead of the
listen model:

    listen      speak? (noul), how long to wait (score), mood (choice)
    standing    each voice's weight on the question (one score per voice)
    vote        support for each option (one noul per option: approval voting stays)
    clerk       is a vote due? (noul) -- only a yes goes on to the speak model, which drafts
                the question and options (Jev cannot write them)

A listen decided by Jev has no written reason; its reason field records p(speak). Reasons are
not shown to other voices anyway, only on the page.

Every failure falls back to the listen model, so turning Jev on can never silence the mind.
Question text lives here rather than in prompts.py: it is Jev's own format (criteria, levels),
not a prompt for a language model.

API: POST {JEV_BASE_URL}/v1/systemone, Bearer key (JEV_API_KEY or TYPESAFE_API_KEY).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

import httpx

from . import llm
from .config import Settings
from .frame import Frame
from .prompts import _thought_line

log = logging.getLogger(__name__)

STREAM_LINES = 30  # Jev's accuracy drops with irrelevant context: show the recent part only

MOODS: dict[str, str | None] = {
    "curious": None, "eager": None, "calm": None, "doubtful": None, "worried": None,
    "amused": None, "bored": "the talk is going round or has nothing new",
    "irritated": "the talk is drifting, repeating, or ignoring something important",
    "satisfied": "the matter feels settled",
}

# Levels of the pause question and the seconds each one means; a fractional score
# interpolates between them.
PAUSE_LEVELS = [
    ("Speak right away", 0.0),
    ("Let a few seconds pass first", 5.0),
    ("Wait for the voices that are forming a thought to finish first", 15.0),
    ("Hold back well behind the others", 30.0),
]

WEIGHT_LEVELS = ["Should count very little", "Should count somewhat", "Should count a lot"]


class JevError(Exception):
    pass


class RateLimited(JevError):
    """The provider answered 429: Jev is skipped until its retry time has passed."""


def enabled(settings: Settings) -> bool:
    return settings.decider == "jev"


def api_key(settings: Settings) -> str:
    return settings.jev_api_key or os.environ.get("TYPESAFE_API_KEY", "")


_clients: dict[tuple[str, str], httpx.AsyncClient] = {}


def _client(settings: Settings) -> httpx.AsyncClient:
    key = api_key(settings)
    if not key:
        raise JevError("no Jev API key (JEV_API_KEY, TYPESAFE_API_KEY or the settings page)")
    k = (settings.jev_base_url, key)
    if k not in _clients:
        _clients[k] = httpx.AsyncClient(
            base_url=settings.jev_base_url.rstrip("/"), timeout=20,
            headers={"Authorization": f"Bearer {key}"})
    return _clients[k]


# After a 429 every call skips Jev until the provider's retry time has passed, so a free
# tier's few requests per minute are not hammered by every voice at once; skipped calls fall
# back to the listen model like any other failure and are not sent or traced.
DEFAULT_COOLDOWN = 60.0
MAX_COOLDOWN = 600.0
_cooldown_until = 0.0


def cooling_down() -> float:
    """Seconds left before Jev is tried again (0 if it is not cooling down)."""
    return max(0.0, _cooldown_until - time.monotonic())


def _retry_after(r: httpx.Response) -> float:
    """The provider's wait: the Retry-After header, or "retry after Ns" in the body."""
    header = r.headers.get("retry-after", "")
    try:
        if header:
            return min(MAX_COOLDOWN, max(1.0, float(header)))
    except ValueError:
        pass
    m = re.search(r"retry after (\d+(?:\.\d+)?)\s*s", r.text, re.IGNORECASE)
    return min(MAX_COOLDOWN, float(m.group(1))) if m else DEFAULT_COOLDOWN


def _cool_down(seconds: float) -> None:
    global _cooldown_until
    until = time.monotonic() + seconds
    if until > _cooldown_until:
        if cooling_down() == 0:
            log.warning("Jev is rate-limited; using the listen model for %.0f s", seconds)
        _cooldown_until = until


async def ask(settings: Settings, state: Any, questions: dict[str, dict[str, Any]], *,
              purpose: str, voice: str | None = None) -> dict[str, dict[str, Any]]:
    """One System One call; returns `answers` by question name. Traced like a model call."""
    left = cooling_down()
    if left > 0:
        raise JevError(f"rate-limited, cooling down for {left:.0f} s")
    body = {"model": settings.jev_model, "state": state, "questions": questions}
    for attempt in (1, 2):
        start = time.perf_counter()
        try:
            r = await _client(settings).post("/v1/systemone", json=body)
            if r.status_code == 429:
                _cool_down(_retry_after(r))
                _trace(settings, purpose, voice, start, ok=False,
                       error=f"HTTP 429: {r.text[:200]}")
                raise RateLimited(f"HTTP 429: {r.text[:200]}")
            if r.status_code == 408 or r.status_code >= 500:
                raise JevError(f"HTTP {r.status_code}: {r.text[:200]}")
            if r.status_code >= 400:
                raise ValueError(f"HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
        except RateLimited:
            raise
        except (JevError, httpx.TransportError) as e:
            _trace(settings, purpose, voice, start, ok=False, error=str(e))
            if attempt == 1:
                await asyncio.sleep(1.0)
                continue
            raise JevError(str(e)) from e
        except Exception as e:
            _trace(settings, purpose, voice, start, ok=False, error=str(e))
            raise JevError(str(e)) from e
        usage = data.get("usage") or {}
        _trace(settings, purpose, voice, start, input_tokens=usage.get("input_tokens"),
               output_tokens=usage.get("output_tokens"))
        answers = data.get("answers") or {}
        missing = set(questions) - set(answers)
        if missing:
            raise JevError(f"answers missing for {sorted(missing)}")
        return answers
    raise JevError("unreachable")


def _trace(settings: Settings, purpose: str, voice: str | None, start: float, *, ok: bool = True,
           error: str | None = None, input_tokens: int | None = None,
           output_tokens: int | None = None) -> None:
    llm._emit(llm.Trace("jev", "typesafe", settings.jev_model, purpose, voice,
                        int((time.perf_counter() - start) * 1000), input_tokens, output_tokens,
                        None, ok=ok, error=error[:500] if error else None))


# --------------------------------------------------------------------------- state


def state(frame: Frame, recent: list[dict[str, Any]], *, me: str | None = None,
          voice_text: str | None = None, memories: list[dict[str, Any]] | None = None,
          others: dict[str, dict[str, Any]] | None = None, listening: list[str] | None = None,
          forming: list[str] | None = None, tz: str = "UTC") -> dict[str, Any]:
    """What Jev reads: the same facts a listen prompt shows, as a JSON object, kept short."""
    s: dict[str, Any] = {"what_this_is": frame.text.strip()}
    if me is not None:
        s["you"] = me
        s["who_you_are"] = (voice_text or "").strip()
    s["stream"] = [_thought_line(t, tz) for t in recent[-STREAM_LINES:]] or ["(empty)"]
    if memories:
        s["older_related_thoughts"] = [_thought_line(t, tz) for t in memories]
    if others is not None:
        s["latest_listen_of_each_voice"] = {
            n: ("spoke" if l.get("speak") else "quiet") + f" — {l.get('mood') or '?'}"
            for n, l in others.items() if n != me}
    if listening is not None:
        s["deciding_right_now"] = [v for v in listening if v != me] or ["(nobody)"]
    if forming is not None:
        s["writing_a_thought_right_now"] = [v for v in forming if v != me] or ["(nobody)"]
    return s


def _letter(i: int) -> str:
    return chr(65 + i)


# --------------------------------------------------------------------------- listen


def listen_questions(me: str) -> dict[str, dict[str, Any]]:
    return {
        "speak": {
            "type": "noul",
            "instructions": (
                f"`you` is {me}, one voice in the discussion in `stream`, described in "
                f"`who_you_are`. Should {me} say something now? A voice speaks only to add "
                "something: a different view, a point not yet made, or agreement it feels "
                "strongly enough to give its own reason for. A [cadence] line records a vote "
                "that settles its question."),
            "criteria": {
                "true": f"{me} has something to add that the stream does not already say",
                "false": (f"what {me} thinks has already been said, the matter is settled, or "
                          f"{me} has nothing new to add"),
            },
        },
        "pause": {
            "type": "score",
            "instructions": (f"If {me} speaks, how long should it wait first? Voices listed in "
                             "`writing_a_thought_right_now` are about to speak."),
            "criteria": [label for label, _ in PAUSE_LEVELS],
        },
        "mood": {
            "type": "choice",
            "instructions": f"Which word best describes {me}'s mood about the stream right now?",
            "criteria": MOODS,
        },
    }


def pause_seconds(score: float) -> float:
    seconds = [s for _, s in PAUSE_LEVELS]
    x = max(0.0, min(float(len(seconds) - 1), float(score)))
    i = min(int(x), len(seconds) - 2)
    return round(seconds[i] + (seconds[i + 1] - seconds[i]) * (x - i), 1)


def listen_decision(answers: dict[str, dict[str, Any]], threshold: float) -> dict[str, Any]:
    p = float(answers["speak"].get("noul") or 0.0)
    return {
        "speak": p >= threshold,
        "mood": str(answers["mood"].get("choice") or ""),
        "reason": f"jev p(speak)={p:.2f}",
        "pause_seconds": pause_seconds(answers["pause"].get("score") or 0.0),
    }


async def listen(settings: Settings, st: dict[str, Any], me: str) -> dict[str, Any]:
    answers = await ask(settings, st, listen_questions(me), purpose="listen", voice=me)
    return listen_decision(answers, settings.jev_speak_threshold)


# --------------------------------------------------------------------------- cadence


def _question_block(question: str, options: list[str]) -> str:
    return question + "\n" + "\n".join(f"{_letter(i)}. {o}" for i, o in enumerate(options))


def standing_questions(me: str, question: str, options: list[str],
                       names: list[str]) -> dict[str, dict[str, Any]]:
    q = _question_block(question, options)

    def whose(n: str) -> str:
        return "your own" if n == me else f"{n}'s"

    return {
        f"w_{n}": {
            "type": "score",
            "instructions": (
                f"`you` is {me}. The voices are voting on:\n{q}\n\nAs {me}, judging fairly and "
                f"not in your own favour: how much should {whose(n)} vote count on THIS "
                "question, by expertise or by how much is at stake for it?"),
            "criteria": WEIGHT_LEVELS,
        }
        for n in names
    }


def standing_answer(answers: dict[str, dict[str, Any]], names: list[str]) -> dict[str, Any]:
    top = len(WEIGHT_LEVELS) - 1
    return {"weights": {n: max(0.0, min(1.0, float(answers[f"w_{n}"].get("score") or 0) / top))
                        for n in names}}


def vote_questions(me: str, question: str, options: list[str],
                   weights: dict[str, float]) -> dict[str, dict[str, Any]]:
    q = _question_block(question, options)
    w = ", ".join(f"{n} {x:.2f}" for n, x in weights.items())
    return {
        f"o{i}": {
            "type": "noul",
            "instructions": (
                f"`you` is {me}. The voices are voting on:\n{q}\n\n(Agreed weights: {w}.) A "
                "voice may support one option, several, or none. Does "
                f"{me} support option {_letter(i)}: {o}?"),
        }
        for i, o in enumerate(options)
    }


def vote_answer(answers: dict[str, dict[str, Any]], options: list[str],
                threshold: float) -> dict[str, Any]:
    """Jev writes no reason: the vote carries `by: "jev"` and its probabilities in `p`, which
    stay in the cadence record and are not printed into the stream as if they were words."""
    ps = [float(answers[f"o{i}"].get("noul") or 0.0) for i in range(len(options))]
    return {"support": [_letter(i) for i, p in enumerate(ps) if p >= threshold],
            "reason": "", "by": "jev", "p": [round(p, 2) for p in ps]}


async def standing(settings: Settings, st: dict[str, Any], me: str, question: str,
                   options: list[str], names: list[str]) -> dict[str, Any]:
    answers = await ask(settings, st, standing_questions(me, question, options, names),
                        purpose="standing", voice=me)
    return standing_answer(answers, names)


async def vote(settings: Settings, st: dict[str, Any], me: str, question: str,
               options: list[str], weights: dict[str, float]) -> dict[str, Any]:
    answers = await ask(settings, st, vote_questions(me, question, options, weights),
                        purpose="vote", voice=me)
    return vote_answer(answers, options, settings.jev_support_threshold)


# --------------------------------------------------------------------------- clerk


def clerk_questions(when_text: str) -> dict[str, dict[str, Any]]:
    return {
        "vote_due": {
            "type": "noul",
            "instructions": (
                "The `stream` is a discussion between voices; `conscious` is the person who "
                "brought the question in. Is it time to settle what conscious asked by a vote? "
                "Leave alone what a [cadence] line has already decided.\n\n" + when_text),
            "criteria": {
                "true": "there is a real choice to settle by a vote now",
                "false": "nothing to decide now: idle talk, still open, or already decided",
            },
        }
    }


async def vote_due(settings: Settings, st: dict[str, Any], when_text: str) -> float:
    answers = await ask(settings, st, clerk_questions(when_text), purpose="clerk")
    return float(answers["vote_due"].get("noul") or 0.0)
