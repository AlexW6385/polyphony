# Polyphony — design notes (v0)

> A mind is a polyphony: many voices, no conductor.

## The idea

Polyphony simulates the inside of a head. Thoughts appear; other thoughts
react to them; values argue, associate, wander, get tired, and eventually
settle on something — or let it go. Each voice is an independent AI agent
with its own values and temperament. Nobody coordinates them.

Humans and the outside world are not special participants. They are just
two more sources of thoughts that enter the same stream. Several topics can
be thrown in at once, one can be added halfway through, and any of them
can drift into something else or connect to something from long ago.

## The one rule: as little framework as possible

Everything that is *behaviour* is decided by a model, not by code:

- whether a voice speaks now, waits, or stays quiet
- whether it is bored, irritated, curious, or done
- whether the mind should decide and stop
- whether an old thought is worth pulling back in

Code owns only what a model cannot: the stream, its ordering, persistence,
fan-out, and concurrency. There is no topic object, no session, no turn
order, no loop cap, no coordinator. "Quiet" is not a state the code enters;
it is simply what happens when every voice looks at the stream and chooses
not to speak.

Voices are **stateless by design**. Each time a voice looks at the stream
it looks fresh: it has its values and temperament, and it sees what was
said (including what "it" said earlier), but it is not obliged to agree
with its past self. Continuity of personality comes from the prompt, not
from memory.

## Components

```text
human / world ──▶ POST /thoughts ──┐
                                   ▼
                              ┌─────────┐   append   ┌────────────┐
   voice A ──┐                │         │ ─────────▶ │ PostgreSQL │  thoughts (seq = order of record)
   voice B ──┼── append ────▶ │ Stream  │            │            │  attention (who considered speaking, why)
   voice C ──┘                │         │ ─────────▶ │ Redis      │  recent window (list), fan-out (pub/sub)
       ▲                      └─────────┘            └────────────┘
       │                           │
       └──────── subscribe ◀───────┘   every new thought wakes every voice
```

- **Stream** (`polyphony/stream.py`): the only shared thing. `append()`
  writes to PostgreSQL (which assigns `seq`), then pushes to the Redis
  recent-window list and publishes on a pub/sub channel. `recent()` reads
  the window; `stray_memories()` samples a few random older thoughts so
  voices can free-associate.
- **Voice** (`polyphony/voice.py`): one asyncio task per voice. It listens
  to the channel, and whenever anything new arrives it runs *attend*; if
  that says speak, it runs *speak* and appends the result. A voice is
  defined entirely by `voices/<name>/voice.md`, a free-text description of
  its values and temperament.
- **API** (`polyphony/api.py`): `POST /thoughts` to inject a thought,
  `GET /thoughts` to read the record, `GET /stream` (SSE) to watch live,
  `GET /attention` to see who considered speaking and why.
- **LLM** (`polyphony/llm.py`): two tiers through any OpenAI-compatible
  endpoint. *Attend* is cheap and can be a local model (Ollama exposes the
  same API); *speak* is the strong one.

## Two stages, both model-driven

1. **Attend** — the voice reads the recent window plus a few stray
   memories and returns `{speak, mood, reason, pause_seconds}`. Boredom,
   fatigue and "we are circling, decide already" are legitimate outputs
   here; this is where the mind's sense of *enough* lives. `pause_seconds`
   lets a voice hang back instead of answering instantly, which is what
   makes the pacing feel organic rather than round-robin.
2. **Speak** — only if attend said so. The strong model writes the thought
   in first person, with no name tag and no obligation to be consistent
   with earlier thoughts. Before speaking, the voice re-reads the stream so
   the thought responds to what is actually there after the pause.

## Concurrency and ordering

Voices run truly concurrently and may speak over each other. That is
allowed — people interrupt each other in their own heads too. What the
system guarantees is a single, unambiguous **order of record**:

- `seq` is a PostgreSQL `BIGSERIAL`; the database, not the application,
  decides the order.
- The Redis window is a cache; two voices can commit `seq` 5 and 6 and push
  6 before 5, so readers sort by `seq`. If the window is empty (cold
  start), it is refilled from PostgreSQL.
- Each voice coalesces wake-ups: if three thoughts land while it is busy,
  it reacts once to the newest state instead of three times. This keeps
  cost bounded under bursts without any rate-limiting rule.
- Nothing else is shared between voices, so moving voices into separate
  processes or containers later changes no code: they already coordinate
  only through PostgreSQL and Redis.

## Deliberately not built (yet)

- Topics or sessions: thoughts about several things can coexist; voices
  decide what to engage with.
- Loop limits: the stop condition is a model judgement, not a counter.
- Conductor, scheduler, turn order.
- Voice memory beyond the stream itself.

## Ideas for later

- Decision procedures as *voices*, not code: e.g. a "chair" voice that
  calls a vote, proposes weights (who is the expert, who is affected), and
  reads the result.
- A local small model for attend to make always-on attention nearly free.
- A web view of the stream with the attention log alongside it, so you can
  watch voices consider speaking and decline.
- Voices in separate containers (`docker compose --scale`).
