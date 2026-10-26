# Polyphony — design notes (v0)

> A mind is a polyphony: many voices, no conductor.

## The idea

Polyphony simulates the inside of a head. Thoughts appear; other thoughts
react to them; values argue, associate, wander, get tired, and eventually
settle on something — or let it go. Each voice is an independent AI agent
with its own values and temperament. Nobody coordinates them.

The person — the *conscious* — and the outside world are not special participants. They are two
more sources of thoughts entering the same stream. Several topics can be
thrown in at once, one can be added halfway through, and any of them can
drift into something else or connect to something from long ago.

## Vocabulary

| word | meaning |
|---|---|
| **voice** | one agent: a set of values and a temperament, defined by one text file |
| **thought** | one entry in the stream, from a voice, the conscious, or the world |
| **conscious** | the person. Thoughts they type in are tagged `conscious`: the deliberate, aware part of the mind, as opposed to the voices below the surface |
| **stream** | the single shared, ordered record of thoughts |
| **listen** | a voice reading the stream and deciding whether to speak; its result (mood, reason, speak or not) is itself recorded and visible to everyone |
| **forming** | the state of a voice that has decided to speak and is composing; visible to the others |
| **sense** | a standing instruction for what to bring in from the outside world (news, feeds); a sense periodically fetches and drops thoughts into the stream |
| **cadence** | a decision procedure: a resolution is posed, the voices vote, the mind resolves |

## Two principles

**Give as much information as possible.** Inside a head nothing is hidden.
Every voice sees the whole recent stream, a few random older thoughts, every
other voice's latest *listen* (whether it chose to speak, its mood, its
reason), and which voices are currently *forming* a thought. If everyone is
bored, each voice can see that everyone is bored. If three voices are
already forming replies, a fourth can see that and decide to wait.

**Prescribe as little as possible.** Everything that is *behaviour* is a
model judgement, not code:

- whether a voice speaks now, waits, or stays quiet
- whether it is bored, irritated, curious, or done
- whether the mind is repeating itself and should stop
- whether a thread has drifted so far from what the person or the world
  brought in that it is time to pause
- whether an old thought is worth pulling back in

Code owns only what a model cannot: the stream, its ordering, persistence,
fan-out, and concurrency. There is no topic object, no session, no turn
order, no loop cap, no coordinator. "Quiet" is not a state the code enters;
it is what happens when every voice listens and chooses not to speak.

Negative feedback against extremes (endless repetition, runaway tangents,
voices talking only to each other) is also left to the model, through the
listen prompt. The anchor is that original thoughts come only from the
person or from senses the person chose; the voices' job is to work on
those, not to entertain themselves. The prompt says so, and says that a
long drift far from any origin is a reason to stop; nothing in the code
enforces it.

Voices are **stateless by design**. Each time a voice listens it looks
fresh: it has its values and temperament, and it sees what was said
(including what "it" said earlier), but it is not obliged to agree with its
past self. Continuity comes from the voice file, not from memory.

## Components

```text
conscious ─▶ POST /thoughts ─┐
senses ──▶ (periodic) ──────┤
                            ▼
                      ┌──────────┐  append   ┌────────────┐
   voice A ──┐        │          │ ────────▶ │ PostgreSQL │ thoughts   (seq = order of record)
   voice B ──┼──────▶ │  Stream  │           │            │ listens    (every decision, spoken or not)
   voice C ──┘        │          │ ────────▶ │ Redis      │ recent window · forming set · pub/sub
       ▲              └──────────┘           └────────────┘
       │                   │
       └──── subscribe ◀───┘   every new thought wakes every listening voice
```

- **Stream**: the only shared thing. `append()` writes to PostgreSQL (which
  assigns `seq`), pushes to the Redis recent window, and publishes on a
  channel. `recent()` reads the window; `strays()` samples random older
  thoughts for free association; `listens()` returns each voice's latest
  decision; `forming()` returns who is composing right now.
- **Voice**: one asyncio task per voice, defined entirely by
  `voices/<name>/voice.md`. See the loop below.
- **Sense**: one task per entry in a list of plain-text instructions
  ("AI policy news from the last day", "anything about Waterloo co-op
  hiring"). On a schedule it asks a model (with search) to fetch and
  summarize, and appends the result as a `world` thought.
- **Cadence**: started by the conscious, or proposed by a voice. See below.
- **API**: inject thoughts, read the record, watch live (SSE), read the
  listen log, start a cadence.
- **LLM**: two tiers through any OpenAI-compatible endpoint. *Listen* is
  cheap and can be a local model (Ollama); *speak* is the strong one.

## The voice loop

```text
idle ──(stream moved)──▶ listen ──(no)──▶ idle
                            │
                           (yes)
                            ▼
                     mark forming ──▶ wait pause ──▶ re-read stream ──▶ speak ──▶ append ──▶ idle
```

1. **Listen** (cheap model). Input: recent window + strays + everyone's
   latest listen + who is forming. Output:
   `{speak, mood, reason, pause_seconds}`. Boredom, fatigue, "we are
   circling", "this has drifted too far", "three others are already on it"
   are all legitimate. `pause_seconds` lets a voice hang back instead of
   answering instantly.
2. **Speak** (strong model), only if listen said so. The voice is marked
   *forming* so the others can see it. After the pause it re-reads the
   stream, so the thought responds to what is actually there, then writes
   in first person with no name tag and no obligation to be consistent
   with earlier thoughts.

While a voice is forming it does not take new triggers; when its thought is
appended it goes back to idle and the next stream movement wakes it again.
This is the only throttle in the system, and it is a cost measure, not a
behaviour rule.

## Concurrency and ordering

Voices run truly concurrently and may speak over each other; that is
allowed. What the system guarantees is a single, unambiguous **order of
record**:

- `seq` is a PostgreSQL `BIGSERIAL`; the database decides the order.
- The Redis window is a cache. Two voices can commit 5 and 6 and push 6
  before 5, so readers sort by `seq`. A cold window is refilled from
  PostgreSQL.
- Wake-ups are coalesced: if several thoughts land while a voice is busy,
  it reacts once to the newest state.
- Nothing else is shared, so voices can later move to separate processes or
  containers without code changes; they already coordinate only through
  PostgreSQL and Redis.

## Cadence (decision)

An explicit, experimental procedure for settling a question. It can be
started two ways:

- **By the conscious.** The person presses a button and types a resolution.
  The cadence begins immediately; no permission needed.
- **By a voice.** A voice may, as part of speaking, propose a cadence with a
  resolution. That proposal is first put to a quick yes/no vote among all
  voices (one voice, one vote, simple majority: *is this worth deciding
  now?*). If it passes, the cadence begins; if not, the proposal stays in
  the stream as an ordinary thought and the discussion goes on.

Once begun, a cadence runs in two rounds so that the vote is weighted by
relevance rather than one-voice-one-vote:

1. **Standing.** Each voice reads the discussion and the resolution and
   proposes a weight for *every* voice (including itself) on this
   question: who has expertise here, who has the most at stake. Weights are
   averaged across proposals.
2. **Vote.** Each voice votes for / against / abstain with a one-line
   reason. The weighted result, the reasons, and the weights are appended
   to the stream as a `cadence` thought, visible to everyone afterwards
   like any other thought.

It is deliberately a separate, optional mechanism, not the normal way the
mind decides. Most of the time the mind should settle on its own.

## Deliberately not built

- Topics or sessions.
- Loop limits, cooldowns, speak quotas.
- Conductor, scheduler, turn order.
- Per-voice memory beyond the stream.

## Open questions

- Sense scheduling and dedup: how often, and how to avoid dropping the
  same headline in twice.
- How many strays to sample, and whether to bias toward older or toward
  thoughts the current voice wrote.
