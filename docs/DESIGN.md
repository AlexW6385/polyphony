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

---

# Implementation notes (v0)

This section is for whoever builds the first version. Everything above is
the *why*; this is the *what*, pinned down so the build does not drift.
Where this section and the prose above disagree, this section wins for v0.

## Stack

- Python 3.12, `asyncio` throughout. One process runs the API and every
  voice and sense as tasks; nothing requires multiple processes in v0, but
  nothing may prevent it later (no in-process shared state between voices).
- FastAPI + uvicorn for the API. SSE via `StreamingResponse`; no websockets.
- PostgreSQL 16 via SQLAlchemy 2.x async + asyncpg. Schema via
  `Base.metadata.create_all` on startup in v0; Alembic can come later.
- Redis 7 via `redis.asyncio`: recent window (list), forming set, pub/sub.
- `openai` client for both model tiers, each with its own
  `base_url`/`api_key`/`model` so the listen tier can point at Ollama.
- `pydantic-settings` for config from `.env`. Docker Compose for
  mind + postgres + redis. No frontend framework in v0: one static HTML
  page served by FastAPI is enough.

## Repository layout

```text
polyphony/
  config.py      settings (env)
  db.py          engine, session, ORM models, init_db()
  stream.py      Stream: append / recent / strays / listens / forming / subscribe
  llm.py         Tier(listen|speak), complete_json(), complete_text()
  prompts.py     all prompt text + render_stream()
  voice.py       VoiceSpec.load(), Voice.run()
  sense.py       Sense.run(): periodic fetch -> world thought
  cadence.py     run_cadence(), propose_cadence()
  api.py         create_app(stream)
  main.py        wires everything, starts tasks
voices/<name>/voice.md      one file per voice; the file is the voice
senses.yaml                 list of {name, instruction, every_minutes}
static/index.html           stream + listen log side by side (SSE)
scripts/say.py, scripts/tail.py
docs/DESIGN.md, README.md, .env.example, docker-compose.yml, Dockerfile
```

## Data model (PostgreSQL)

```sql
thoughts   seq BIGSERIAL PK, author VARCHAR(64), kind VARCHAR(16),
           content TEXT, mood VARCHAR(80) NULL, meta JSONB NULL,
           created_at TIMESTAMPTZ DEFAULT now()
           -- kind: voice | conscious | world | cadence
           -- meta: cadence result, sense source url, etc.

listens    id BIGSERIAL PK, voice VARCHAR(64), trigger_seq BIGINT,
           speak BOOL, mood VARCHAR(80), reason TEXT, pause_seconds REAL,
           created_at TIMESTAMPTZ DEFAULT now()
           -- one row per listen, spoken or not; index (voice, id desc)

cadences   id BIGSERIAL PK, resolution TEXT, proposed_by VARCHAR(64),
           status VARCHAR(16),   -- proposed | rejected | running | resolved
           weights JSONB, votes JSONB, result JSONB,
           created_at, resolved_at

sense_log  id BIGSERIAL PK, sense VARCHAR(64), fingerprint VARCHAR(128) UNIQUE,
           seq BIGINT NULL, fetched_at TIMESTAMPTZ
           -- fingerprint = sha256 of normalized title/url; dedup across runs
```

`seq` is the only ordering anywhere. Never order by `created_at`.

## Redis keys

```text
polyphony:thoughts            pub/sub channel; payload = thought JSON
polyphony:recent              list of thought JSON, LTRIM to RECENT_WINDOW
polyphony:forming             set of voice names currently composing
polyphony:listen:<voice>      string, latest listen JSON for that voice
```

`append()` order: commit to PostgreSQL first (get `seq`), then in one
pipeline RPUSH + LTRIM + PUBLISH. Readers of `recent` always sort by `seq`.
If `recent` is empty, refill from PostgreSQL (last RECENT_WINDOW rows).

## API

```text
GET  /health
POST /thoughts            {content, author="conscious", kind="conscious"|"world"} -> thought
GET  /thoughts?after=0&limit=200
GET  /stream              SSE, one event per thought (any kind)
GET  /listens?limit=100   latest listen rows
GET  /voices              [{name, forming: bool, last_listen}]
POST /cadence             {resolution} -> starts immediately (conscious)
GET  /cadence/{id}
GET  /                    static/index.html
```

## Config (.env)

```text
DATABASE_URL, REDIS_URL
OPENAI_API_KEY, OPENAI_BASE_URL            defaults for both tiers
LISTEN_MODEL, LISTEN_BASE_URL, LISTEN_API_KEY
SPEAK_MODEL,  SPEAK_BASE_URL,  SPEAK_API_KEY
RECENT_WINDOW=40   STRAYS=4   ENABLED_VOICES=   PORT=8000
```

## Voice loop (normative)

```python
async def run(self):
    listener = create_task(self._listen_channel())   # sets self._wake on any thought not by me
    while True:
        await self._wake.wait(); self._wake.clear()
        await self.attend()

async def attend(self):
    recent  = await stream.recent()
    if not recent: return
    strays  = await stream.strays(before_seq=recent[0]["seq"], n=STRAYS)
    others  = await stream.listens_latest()          # every voice's last listen
    forming = await stream.forming()
    decision = await llm.complete_json(LISTEN, listen_system(me), render(recent, strays, others, forming, me))
    await stream.record_listen(me, recent[-1]["seq"], decision)
    if not decision["speak"]: return
    await stream.set_forming(me, True)
    try:
        await sleep(clamp(decision["pause_seconds"], 0, 60) + uniform(0, 2))
        recent = await stream.recent()               # re-read: the stream moved while we waited
        text = await llm.complete_text(SPEAK, speak_system(me, decision), render(recent, strays, others, forming, me))
        if text: await stream.append(me, "voice", text, mood=decision["mood"])
    finally:
        await stream.set_forming(me, False)
    # wake flag is NOT cleared here: anything that landed while forming
    # triggers one fresh listen after we return
```

Wake-ups coalesce because `_wake` is an `asyncio.Event`: N thoughts during
one attend produce one further attend, against the newest state.

Listen output schema, exactly:
`{"speak": bool, "mood": str, "reason": str, "pause_seconds": number,
"propose_cadence": str | null}`. `propose_cadence`, when non-null, is a
resolution text and starts the proposal vote (see cadence).

## Prompts — required content

The exact wording is the implementer's, but the listen prompt must say all
of the following, and the speak prompt the parts marked *:

- You are one voice in a single mind; your name; the voice file verbatim.*
- You see: the stream, stray memories, every voice's latest listen (mood,
  reason, spoke or not), who is forming right now.
- You are never obliged to speak and never obliged to agree with what
  "you" said before; each look is fresh; only your values are constant.*
- Reasons to speak: new angle, disagreement, an association (including to
  a stray memory), a decision forming, a correction, the mind circling.
- Reasons not to: you would repeat, the thread is done, you're not
  interested, you're tired or irritated — these are legitimate.
- Original thoughts come only from the conscious or from the world. If the
  discussion has drifted far from any of them for a long time, that is a
  reason to stop. If the mind seems to be repeating itself, stop.
- Others already forming is a reason to wait or stay quiet; everyone bored
  is a reason to let it go quiet.
- You may propose a cadence (a resolution to vote on) if you think the
  mind should decide.*
- Speak*: first person, no name tag, no "as I said", any length that fits,
  may disagree with anyone including yourself, may wander, may concede, may
  propose to decide and stop. Output only the thought.

`render()` format: stray memories first (labelled as such), then the
latest listens as one line per voice (`rest: quiet — tired — "third loop"`),
then who is forming, then the stream as `[seq] author (mood): content`
with the current voice marked `(you)` and kinds `[conscious]` / `[world]`
labelled.

## Senses

`senses.yaml`:

```yaml
- name: ai-policy
  instruction: "Anything notable about AI regulation or export controls in the last day."
  every_minutes: 240
```

Each sense is a task: sleep, then call the speak tier with web search (or
a search tool the implementer picks) and the instruction, asking for 0–3
items as `{title, url, summary}`. For each item compute a fingerprint
(sha256 of lowercased title or url); skip if in `sense_log`; otherwise
append a `world` thought (`author=<sense name>`, content = title + summary,
meta.url) and insert into `sense_log`. A sense that returns nothing appends
nothing. No sense runs more often than every 30 minutes.

## Cadence

```text
propose(by voice)  -> cadences row status=proposed
                   -> every voice: listen-tier call "worth deciding now? yes/no + one line"
                   -> simple majority of all voices? no -> status=rejected, proposal text
                      remains in the stream as the voice's thought; yes -> start
start(by conscious or passed proposal) -> status=running
  round 1 (standing): every voice, speak tier: for each voice name, a weight
      0–1 with one line why (expertise / stake). Average per voice across
      all proposals; normalise to sum 1.
  round 2 (vote): every voice, speak tier: for | against | abstain + one
      line. Result = sum of weights for / sum of weights (for+against).
      Passes if > 0.5.
  append thought kind=cadence, author="cadence", content = human-readable
      summary (resolution, result, weights, each vote + reason);
      meta = full structured result. status=resolved.
```

Cadence calls run concurrently across voices (`asyncio.gather`). Voices do
not listen/speak on the stream while a cadence they are part of is
running; the cadence thought wakes them afterwards like any other thought.

## Starter voices

Seven value-voices, each a short paragraph (3–6 sentences): what it cares
about and how it tends to think. Ship: `freedom`, `rest`, `fairness`,
`curiosity`, `maker`, `beauty`, `skeptic`. No job titles, no roles.

## v0 is done when

1. `docker compose up` starts mind + postgres + redis with no manual step.
2. `scripts/say.py "..."` appends a conscious thought; within a minute
   several voices have listened (visible at `/listens`), some have spoken,
   and the stream goes quiet on its own without any counter or cap.
3. Two voices speaking at the same moment produce two thoughts with
   distinct, increasing `seq`, and `/thoughts` returns them in that order.
4. A second conscious thought on an unrelated subject can be dropped in
   mid-discussion and voices engage with either, by their own choice.
5. `POST /cadence` runs both rounds and appends a `cadence` thought.
6. One sense configured in `senses.yaml` appends a `world` thought and does
   not append the same item twice.
7. `/` shows the stream and the listen log side by side, live.

## Not in v0

Auth, multi-user, per-voice memory, voice-to-voice private channels,
Alembic migrations, metrics dashboards, any UI beyond the single page.
