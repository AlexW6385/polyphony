# Polyphony — design (v1)

> A polyphony of anything: many voices, no conductor.

This document replaces the v0 design notes. Everything v0 decided still holds unless a section
below says otherwise; the v0 build is the starting point, not something to throw away. The
first half is the *why*; the second half ("Implementation notes") is the *what*, pinned down
for whoever builds it. Where the two disagree, the implementation notes win.

---

## The idea

Polyphony is a room with no chair. Several voices share one stream of thoughts. Each voice is
an independent AI agent with its own wants and temperament; each one keeps looking at the
stream and decides for itself whether to add to it. Nobody schedules them, nobody moderates
them, nobody tells them when to stop. Thoughts appear, others react, values argue, associate,
wander, get tired, and eventually settle on something — or let it go.

v0 framed this as the inside of one head, and that remains the first and default use. But
nothing in the engine is specific to a mind. The same machinery is a polyphony of:

- **one mind** — the inner voices of a person (wants, fears, habits, stances);
- **several people** — a family deciding where to live, a team arguing about a launch, a
  council of characters;
- **groups of people** — residents of a city reacting to a policy, generations, professions,
  the stakeholders of a decision.

What a voice *is* lives entirely in its prompt. What the whole polyphony *is* lives in one
more piece of text, the **frame**. The engine does not change between them.

## What it is for

The point of a polyphony is its outcome, not the talk. A question goes in, from the person or
from the world, and one of two things comes out:

- **a decision**, when the voices settle it themselves or a cadence settles it formally;
- **a map of positions**, when they do not: who wanted what, who moved whom, where they still
  disagree and why.

Either way the full record stays: every thought, every listen, every weight and every vote.
That record is the evidence; the outcome is the product. A long stream is unreadable for a
human, so v1 adds a **digest**: when the room goes quiet, a summariser that is not a voice
writes what was asked, what each voice argued, what changed, and what was decided or left
open. The evals ask the same question of every run: *did the room produce a useful outcome for
what it was given?*

The human and the outside world are not special participants. They are two more sources of
thoughts entering the same stream. Several topics can be thrown in at once, one can be added
halfway through, and any of them can drift into something else or reconnect to something from
long ago.

## Vocabulary

| word | meaning |
|---|---|
| **ensemble** | one complete polyphony: a frame, its voices and its senses. One ensemble runs per instance |
| **frame** | text saying what this polyphony is, who the voices are to each other, and who the human is. Injected into every prompt |
| **voice** | one agent, defined by one text (described, or shown by example moments) |
| **thought** | one entry in the stream, from a voice, the person, the world, or a cadence |
| **person** | the human participant. The frame decides how their thoughts are labelled ("conscious" in a mind, "chair" in a council, "you" by default). v0 called this kind `conscious` |
| **stream** | the single shared, ordered record of thoughts |
| **listen** | a voice reading the stream and deciding whether to speak; its result (mood, reason, speak or not) is recorded and visible to everyone |
| **forming** | a voice that has decided to speak and is composing; visible to the others |
| **memory** | older thoughts beyond the recent window that are related to what is being said now, retrieved by meaning and shown to voices |
| **sense** | a standing instruction for what to bring in from the outside world; a sense periodically reads the feeds the person chose and drops `world` thoughts into the stream |
| **cadence** | a decision procedure: a resolution is posed, the voices weigh each other, vote, and the result is recorded |
| **digest** | a summary of a stretch of the stream, written by a non-voice summariser when the room goes quiet: the question, each voice's position, what changed, the outcome |
| **student / teacher** | the small local model that makes listen decisions, and the strong model whose decisions it learns from |

## Two principles

**Give as much information as possible.** Nothing in the room is hidden. Every voice sees the
whole recent stream, a few random older thoughts, older thoughts related to the present
(memories), every other voice's latest listen (whether it chose to speak, its mood, its
reason), and which voices are currently forming a thought. If everyone is bored, each voice can
see that everyone is bored. If three voices are already forming replies, a fourth can see that
and decide to wait.

**Prescribe as little as possible.** Everything that is *behaviour* is a model judgement, not
code:

- whether a voice speaks now, waits, or stays quiet;
- whether it is bored, irritated, curious, or done;
- whether the room is repeating itself and should stop;
- whether a thread has drifted so far from what the person or the world brought in that it is
  time to pause;
- whether an old thought is worth pulling back in.

Code owns only what a model cannot: the stream, its ordering, persistence, fan-out,
concurrency, retrieval, and calls to the outside world. There is no topic object, no session,
no turn order, no loop cap, no coordinator. "Quiet" is not a state the code enters; it is what
happens when every voice listens and chooses not to speak.

Negative feedback against extremes (endless repetition, runaway tangents, voices talking only
to each other) is also left to the model, through the listen prompt. The anchor is that
original topics come only from the person or from senses the person chose; the voices' job is
to work on those, not to entertain themselves. Nothing in the code enforces it.

A third principle follows from the first two and governs every addition in v1:

**Frameworks serve the edges, never the centre.** LangGraph orchestrates work *inside* a sense
run and *inside* a cadence. It never coordinates voices, never decides who speaks, never owns
the stream. MCP connects the room to the outside world and the outside world to the room; it
never becomes the room.

## Frames and ensembles

An ensemble is a directory:

```text
ensembles/<name>/
  frame.md
  voices/<voice>/voice.md
  senses.yaml
```

`frame.md`:

```markdown
---
label: mind               # shown in the UI
person_label: conscious   # tag for the human's thoughts
---
You are one voice inside a single human mind. The others are other voices of the same mind.
The person — the conscious, deliberate part — sometimes drops a thought in; the world comes in
through news the person chose to follow. ...
```

The frame is free text. It is placed first in every system prompt (listen, speak, cadence,
sense, digest), followed by the **engine text** — how the stream is rendered, when to speak,
boredom, cadence mechanics, output schemas — and only then, for voices, the voice text. Engine
text must be framing-neutral: it never mentions minds, heads, people, or groups.

Three ensembles ship as examples:

- `mind` — the v0 voices (`ride`, `rest`, `make`, `learn`, `people`, `fair`, `doubt`) and
  frame; reproduces v0 behaviour. These voices are placeholders: once phase 1 runs, the person
  replaces them with voices written from their own head (described or shown), and the
  shipped set becomes an example ensemble rather than the default.
- `council` — a handful of distinct people deciding something together (e.g. a family
  deciding whether to move cities).
- `public` — groups of people in one city reacting to the news its senses bring in.

One ensemble runs per instance (`ENSEMBLE=mind`). Switching means another instance or another
database. Running several ensembles inside one instance would require namespacing the stream,
Redis keys and SSE per ensemble; it is deliberately not built (see *Not built*).

## What a voice is

A voice is not a job title and not a bare abstraction. It is a *want*, rooted in something
concrete, that argues for a value when it talks.

- In a **mind**, "the one who wants to be on the bike tomorrow instead of at the desk" is a
  voice; so is "the one who keeps checking the group chat". When an abstract question lands,
  concrete voices drag it to the ground — *what does this do to my weekends?* — and that is
  what makes the discussion feel like a head rather than a seminar. Purely abstract voices
  (fairness, doubt) are allowed too.
- For **a person**, the voice is that person: what they want out of this, what they believe,
  how they talk, what they always bring up.
- For **a group**, the voice speaks for the group's typical concerns and stakes, in a register
  members of that group would recognise, and acknowledges that the group disagrees internally.
  It must not be a caricature. A group voice that only ever says one thing is badly written.

Voices are meant to be **personal**. The shipped ones are examples; the person is expected to
write their own. That is why a voice is nothing but text, with no logic attached.

A voice can be written two ways, and the prompts must accept both — the model interprets the
text, not code:

*Described* — the headings are a suggestion, not a schema:

```markdown
---
# optional model overrides
---
**Wants**   — what this voice is after, concretely.
**Believes** — the value underneath, in its own words.
**Sounds like** — how it talks.
**Shows up when** — what pulls it in; what bores it.
```

*Shown* — a few real moments, and nothing else, and the model works out the voice:

```markdown
---
---
- 11pm, laptop open, work done: "ten more minutes of this and then bed."
- A friend cancelled: relief first, guilt a second later.
```

Voices are **stateless by design**. Each listen looks fresh: the voice has its text, and it
sees what was said (including what "it" said earlier) and what memory retrieves, but it is not
obliged to agree with its past self. Memory in v1 is *shared* — retrieved from the one stream
everybody can see — not a private store per voice. Continuity of character comes from the
voice text.

## Components

```text
 person ─▶ POST /thoughts ───────────┐
 MCP clients (Claude, ...) ─▶ mind MCP server ─┤
 senses (LangGraph run) ─────────────┤
   ├─ feeds MCP (ours)                │
   ├─ Fetch MCP (official)            ▼
   └─ search MCP (Exa/Tavily, opt.) ┌──────────┐  append   ┌──────────────┐
                                    │          │ ────────▶ │ PostgreSQL   │ thoughts (seq = order of record)
   voice A ──┐                      │  Stream  │           │  + pgvector  │ listens · samples · embeddings
   voice B ──┼──────────────────▶   │          │ ────────▶ │ Redis        │ recent window · forming · pub/sub
   voice C ──┘                      └──────────┘           └──────────────┘
       ▲   listen: student (local)        │                       ▲
       │   speak:  strong model           │                       │
       └──────── subscribe ◀──────────────┘          embedder (background) ─┘
 cadence (LangGraph run, checkpointed) ──▶ appends a cadence thought
```

- **Stream** — unchanged from v0: the only shared thing. `append()` commits to PostgreSQL
  (which assigns `seq`), then pushes to the Redis window and publishes.
- **Voice** — one asyncio task per voice; the v0 loop, plus a memories block in its context.
- **Memory** — a background embedder writes an embedding for every thought; voices retrieve
  related older thoughts by similarity (pgvector).
- **Sense** — one task per sense; each run is a LangGraph workflow that reads feeds over MCP.
- **Cadence** — a LangGraph workflow with a checkpointer, started by the person or proposed by
  a voice.
- **Digest** — an edge component like a sense: it reads the stream when the room goes quiet
  (or on request) and appends one `digest` thought. Voices do not wake on `digest` thoughts,
  so a digest never restarts the discussion; they do see digests in their context afterwards.
- **MCP** — both directions: Polyphony is an MCP *client* (senses call feeds, fetch and
  search servers) and an MCP *server* (other AI apps can speak into the room, read it, and
  start cadences).
- **LLM layer** — provider-agnostic, two tiers (*listen*, *speak*) plus a *teacher* and an
  *embedder*; listen runs on a local small model by default.
- **Observability and evals** — every model call is traced; scenarios can be run and judged
  offline.

## The voice loop

```text
idle ──(stream moved)──▶ listen ──(no)──▶ idle
                            │
                           (yes)
                            ▼
                     mark forming ──▶ wait pause ──▶ re-read stream ──▶ speak ──▶ append ──▶ idle
```

1. **Listen** (student model). Input: frame, voice text, recent window, strays, memories,
   everyone's latest listen, who is forming. Output:
   `{speak, mood, reason, pause_seconds, propose_cadence}`.
2. **Speak** (strong model), only if listen said so. The voice is marked forming. After the
   pause it re-reads the stream, then writes its thought.

While forming, a voice takes no new triggers; afterwards the next stream movement wakes it
again. This is the only throttle, and it is a cost measure, not a behaviour rule.

## Concurrency and ordering

Unchanged from v0:

- `seq` is a PostgreSQL `BIGSERIAL`; the database decides the order of record.
- The Redis window is a cache: a sorted set scored by `seq` (pushes can arrive out of order);
  a cold window is refilled from PostgreSQL.
- Wake-ups coalesce: thoughts landing while a voice is busy cause one further listen against
  the newest state.
- Voices coordinate only through PostgreSQL and Redis, so they can move to separate processes
  without code changes.

v1 adds two background writers that must respect the same rule — the embedder and the shadow
teacher — and neither may block a voice or reorder anything.

## Memory

The recent window (default 40 thoughts) is what a voice "has in mind". Strays (a few random
older thoughts) give free association. v1 adds **memories**: the older thoughts most related
to what is being said now.

- Every thought gets an embedding, written by a background task (`thought_embeddings`,
  pgvector, HNSW cosine index). Embedding never sits on the append path.
- On listen and speak, the last few thoughts of the window are embedded as the query; the top
  k (default 5) most similar thoughts **older than the window** are rendered as a separate
  memories block, with their `seq` and how long ago they were.
- The query depends only on the stream state, not on the voice, so it is computed once per
  window tail: the result is cached in Redis under the tail `seq` (short TTL) and every voice
  listening against that state reuses it. N voices cost one embedding and one vector search
  per stream move, not N.
- Strays stay: memories are relevance, strays are serendipity. Both are labelled as such.
- Embeddings run locally by default (Ollama, `nomic-embed-text`); any embedding provider the
  LLM layer supports can be configured.

This is retrieval over the room's own history — the modern form of what used to be called
RAG — and it is how "topics can reconnect to something from long ago" becomes real rather than
dependent on a lucky stray.

## Senses

A sense is a standing instruction plus the sources the person chose:

```yaml
- name: world
  instruction: "The most important world news of the last day."
  every_minutes: 240
  feeds: [https://feeds.bbci.co.uk/news/world/rss.xml, https://www.cbc.ca/webfeed/rss/rss-world]
  search: false
```

v0 senses depended on a provider's built-in web search, so DeepSeek and Ollama could not run
them. v1 senses read the person's chosen feeds through MCP and work with any provider. Each
run is a small LangGraph workflow:

```text
collect ──▶ triage ──▶ [search] ──▶ read ──▶ summarize ──▶ emit
```

1. **collect** — fetch new entries from the sense's feeds (our feeds MCP server), drop any
   already seen.
2. **triage** — the model picks which entries matter for the instruction and, if search is
   allowed, which background queries to run. It answers in a JSON schema; this is how it
   chooses tools (see *Decisions*).
3. **search** — only if queries were requested and a search server is configured.
4. **read** — full text of the chosen articles through the official Fetch MCP server; on
   failure, the feed summary.
5. **summarize** — 0–3 items, grounded only in what was read.
6. **emit** — the v0 dedup (URL and title fingerprints, claim before append) and a `world`
   thought per new item.

Feeds are the primary source because they are free, they come from the publishers
themselves, and they are by definition "sources the person chose" — exactly the anchor the
first principle needs. Search is an optional extra for background, off without a key.

Fetched text is untrusted: it is delimited and marked as data in every prompt and can never
change which tools run beyond the triage schema.

## Cadence

Unchanged in meaning from v0: started by the person (immediately) or proposed by a voice
(first a simple-majority yes/no among all voices), then two rounds — **standing** (each voice
proposes a weight for every voice; averaged and normalised) and **vote** (for / against /
abstain with one line; weighted result, passes above 0.5) — and a `cadence` thought records
everything.

v1 runs the procedure as a LangGraph workflow with a PostgreSQL checkpointer:

```text
standing (fan-out) ──▶ weights ──▶ vote (fan-out) ──▶ tally ──▶ [confirm] ──▶ record
```

- An interrupted cadence (restart, crash) resumes from the last finished step instead of
  asking every voice again.
- Optional **confirm**: the person can ask for a pause before the result is recorded.
- The frame decides what a cadence *means* (the mind settles; the council decides; the city
  polls itself); the procedure is the same.

## Models

Two tiers, as in v0, each pointed at any provider (OpenAI, Anthropic, Gemini, DeepSeek,
Ollama), and per-voice overrides:

| tier | runs | default |
|---|---|---|
| **listen** (student) | every stream move, for every voice | a local small model via Ollama (`qwen2.5:3b`; `qwen2.5:1.5b` on 8 GB machines) |
| **speak** | only when a voice speaks; cadence rounds | a strong hosted model |
| **teacher** | sampled listens in the background; synthetic data generation | defaults to the speak model |
| **embed** | every thought, in the background | local `nomic-embed-text` via Ollama |

Listen is the hot path and the main token cost, so it runs locally. Its decision is a narrow,
fixed-schema judgement — exactly the kind of task a small fine-tuned model does well.

Local listen has a **latency budget**. Every stream move triggers one listen per voice, each a
prompt of a few thousand tokens. Ollama serves one request per model at a time unless
`OLLAMA_NUM_PARALLEL` is raised, so seven voices can queue for tens of seconds and the room
reacts in slow motion. Therefore:

- the README documents `OLLAMA_NUM_PARALLEL` (sized to the voice count and memory) and
  `OLLAMA_KEEP_ALIVE` (so the model stays loaded);
- tracing records queue wait separately from model time (client latency minus Ollama's
  reported `total_duration`), and prefill tokens separately from generated tokens;
- the target is a p95 listen latency under 10 s for the `mind` ensemble on the developer's
  machine, recorded in the README with the hardware it was measured on.

### Distilling the listen decision

The goal is a small local model fine-tuned to make listen decisions the way the teacher would,
replacing the off-the-shelf local model by changing one setting.

- **Training data: synthetic self-play.** An offline simulator runs the real engine — real
  prompt rendering, real ensembles — with the teacher on both tiers, accelerated, from a set
  of seed topics across all shipped ensembles. Every listen of every voice is recorded with
  its exact input and the teacher's decision. Labels are generated through providers' batch
  APIs where available. The speak/quiet balance is controlled at export time.
- **Real data, via a shadow teacher.** In normal runs a small share of listens (default 5%)
  is also sent to the teacher in the background; the voice acts only on the student's
  decision. Real data is what proves the model works; synthetic data alone would only prove it
  imitates the simulator.
- **Real data also trains.** Self-play under the teacher never reaches the streams the student
  itself produces; shadow pairs come from exactly those streams (the DAgger idea). Shadow
  pairs are split by time: pairs before a cutoff may join the training set; the block after
  the cutoff is the held-out test set and is never trained on. Each training run records its
  cutoff, so no pair is ever in both.
- **Training:** LoRA on a ~0.5–1.5B base, in a Colab notebook on a free GPU; export to GGUF and
  an Ollama model named `polyphony-listen`.
- **Switching:** `LISTEN_MODEL=polyphony-listen`. No code change.
- **Ceiling:** the teacher runs at non-zero temperature and does not agree with itself. A
  fixed sample of held-out inputs is labelled by the teacher twice, in independent calls; its
  self-agreement on `speak` is the ceiling. A student is judged against that ceiling, not
  against 100%.
- **Proof:** agreement with the teacher on `speak` (accuracy, precision and recall for
  `speak=true`, since most listens are quiet), JSON validity, latency and token cost —
  reported side by side for the off-the-shelf student, the fine-tuned student, running listen
  on the teacher, and the teacher's self-agreement ceiling.

## MCP, both directions

**Client.** Senses use three MCP servers:

| server | owner | role |
|---|---|---|
| `polyphony-feeds` | this repo | fetch RSS/Atom feeds; stateless, no keys, SSRF-guarded |
| `mcp-server-fetch` | official reference server | full article text as markdown |
| Exa or Tavily | the vendor | optional web search, only with an API key |

Community MCP servers are not used: a server runs with whatever access it is given, so only
our own code, official reference servers and vendor-maintained servers are allowed.

**Server.** Polyphony exposes itself as an MCP server (stdio and streamable HTTP) with
`say(content)`, `read_stream(after, limit)`, `start_cadence(resolution)` and `list_voices()`,
implemented over the HTTP API. Any MCP-capable AI app can then bring a question into the room
and read what the voices make of it.

There is no auth, so nothing listens beyond the machine by default: the streamable HTTP
transport binds to `127.0.0.1` (`MCP_HTTP_HOST`), and Docker Compose publishes the HTTP API
as `127.0.0.1:8000`. Exposing either further is an explicit choice, and the README says what
it means.

## Observability and evals

- **Tracing.** Every model call records tier, provider, model, purpose, voice, latency, input,
  output and cached tokens, and success. `GET /stats` and a UI panel summarise them, along with
  the live student–teacher agreement.
- **Evals.** Scenarios (an ensemble, a seed stream, a budget, a description of good) run
  against real models in their own namespace (see *Namespacing*); a judge model scores the
  outcome first — does the digest answer the seed with a decision or a clear map of
  positions — then repetition, voice fidelity, engagement with the seed, convergence, and for
  senses, grounding. Runs can be
  compared before a prompt or model change is kept. The same harness reports the distillation
  numbers above.

## Decisions

- **One ensemble per instance, namespaced from the start.** Multi-ensemble in one process
  would touch the stream core (queries, SSE); the reframing is achieved without it. But every
  Redis key carries a configurable prefix and the database is chosen per instance, so evals,
  the simulator and a live room can share one PostgreSQL server and one Redis without seeing
  each other. This costs one setting now and is most of the way to multi-ensemble later.
- **Outcome first.** A run is good if it leaves a decision or a clear map of positions; the
  digest makes that readable and the evals judge it.
- **Most-shared prompt text first.** frame → engine → voice → stream, so the frame and engine
  prefix is identical across voices and providers' prefix caches (and Ollama's KV cache) hit.
- **One memory query per stream state.** Retrieval depends on the stream, not the voice.
- **Shadow data trains and tests, split by time.** It covers the student's own distribution;
  the time split keeps the test set honest.
- **Frame text + neutral engine text.** The only thing that made v0 "a mind" was wording in
  the prompts; moving it into data generalises the product without touching the engine.
- **LangGraph only inside senses and cadence.** Both are fixed, multi-step, stateful
  workflows that benefit from steps, budgets, resumability and confirmation. The voices are
  the opposite — decentralised and event-driven — and a graph over them would be a conductor.
- **Tools chosen through JSON, not native tool-calling.** Small local models often lack tool
  calling; a schema-constrained JSON choice works on every provider and keeps tool execution
  in code we control.
- **Feeds before search.** Free, attributable, chosen by the person, and true to the first
  principle; search is an add-on.
- **Local listen, strong speak.** Listen is frequent and narrow; speak is rare and needs
  quality.
- **Synthetic data to train, real data to judge.** Fast to produce, honest to evaluate.
- **Shared memory, stateless voices.** Retrieval reads the room's history; voices still look
  fresh each time.

### Decisions made while building v1

- **`mcp>=1.12,<2`.** The 2.x SDK renamed FastMCP and changed client APIs; v1 targets the stable
  1.x API until 2.x settles.
- **Ollama through its native API**, not the OpenAI-compatible one: it accepts a JSON schema as
  `format`, takes `keep_alive`, and reports `total_duration`, which is what makes queue wait
  measurable.
- **Senses mark passed-over entries as seen.** Triage judges each feed entry once; otherwise
  every run would pay to re-triage the same unpicked headlines.
- **Per-run MCP sessions.** Feed, fetch and search servers start inside a sense run and stop
  with it, so a crashed server never outlives a run and nothing needs restarting.
- **Shared isolation harness** (`harness.py`) for evals and the simulator: database
  `<db>_<suffix>` (created if missing) plus namespace `<ns>-<suffix>`.
- **A digest thought never wakes a voice** (`SILENT_KINDS`), but voices see digests in context.
- **Cadence confirm clears the running flag** while waiting, so voices are not frozen while the
  person decides whether to record the result.

## Not built (deliberately)

- Topics or sessions; loop limits, cooldowns, speak quotas; conductor, scheduler, turn order.
- Any framework orchestration of voices.
- Private per-voice memory stores.
- Several ensembles in one process (namespacing exists; per-ensemble routing does not).
- Native tool-calling in the LLM layer.
- Auth, multi-user, voice-to-voice private channels.

## Open questions

- How many strays versus memories, and whether memories should favour thoughts by the current
  voice.
- Whether group voices need a frame-level reminder about internal diversity beyond the voice
  text.
- Whether the student should also take over the cadence proposal yes/no vote.

---

# Implementation notes (v1)

## Stack

v0 stack, plus:

- PostgreSQL image `pgvector/pgvector:pg16` (extension `vector`).
- `langgraph` (+ its PostgreSQL checkpointer for cadences), `mcp` (official Python SDK),
  `feedparser`, `httpx`.
- Ollama on the host for listen and embeddings. On macOS, Docker cannot use the GPU, so Ollama
  runs natively and the app reaches it at `http://host.docker.internal:11434`. An optional
  `ollama` service (compose profile `ollama`) serves Linux hosts.
- Still no migration tool: `polyphony/migrations.py` holds an ordered list of idempotent SQL
  steps run by `init_db` after `create_all`.

## Repository layout

```text
polyphony/
  config.py  db.py  migrations.py  stream.py  prompts.py  voice.py  api.py  main.py  supervisor.py
  llm.py              providers, tiers, structured output, embeddings, tracing hook
  frame.py            load/seed frames, frame text for prompts
  memory.py           embedder task, retrieval
  sense.py            scheduling + dedup (v0) ; delegates a run to sense_graph
  sense_graph.py      LangGraph sense run
  cadence.py          API surface (v0) ; delegates to cadence_graph
  cadence_graph.py    LangGraph cadence with checkpointer
  digest.py           quiet detection, digest writing (edge component, not a voice)
  mcp_client.py       sessions to MCP servers, call(), restart, shutdown
  mcp_servers/
    feeds.py          polyphony-feeds (FastMCP, stdio)
    mind.py           Polyphony as an MCP server
  evals/              scenarios, runner, judge, report
ensembles/
  mind/  council/  public/      frame.md, voices/, senses.yaml
polyphony/harness.py            isolated runs (own DB + namespace) for evals and the simulator
polyphony/distill.py            agreement metrics, time split, chat-format export
simulate/                       offline self-play for synthetic listen data
scripts/                        say.py, tail.py, export_listen_samples.py, eval.py,
                                eval_listen_student.py, backfill_embeddings.py
training/distill_listen.ipynb   Colab notebook (Unsloth, LoRA, GGUF + Modelfile)
static/index.html
```

## Data model changes

```sql
-- thoughts: kind 'conscious' renamed to 'person' (migration UPDATE; API accepts the old name);
--           new kind 'digest' (meta: from_seq, to_seq, outcome = decision | positions)

frames            name VARCHAR(64) PK, label TEXT, person_label TEXT, text TEXT, updated_at

senses            + feeds JSONB NOT NULL DEFAULT '[]', + search BOOL NOT NULL DEFAULT false

thought_embeddings seq BIGINT PK REFERENCES thoughts(seq), model TEXT,
                   embedding vector(N), created_at        -- HNSW (vector_cosine_ops)

llm_calls         id BIGSERIAL PK, tier, provider, model, purpose, voice NULL,
                  latency_ms INT, queue_ms INT NULL, input_tokens INT, output_tokens INT,
                  cached_input_tokens INT,
                  ok BOOL, error TEXT NULL, created_at

listen_samples    id BIGSERIAL PK, listen_id BIGINT NULL, source VARCHAR(16),  -- live | synthetic
                  ensemble VARCHAR(64), voice VARCHAR(64), system TEXT, prompt TEXT,
                  student_provider, student_model, student_decision JSONB NULL,
                  teacher_provider, teacher_model, teacher_decision JSONB NULL, created_at
```

Cadence checkpoints live in LangGraph's own tables in the same database.

## Namespacing

`NAMESPACE` (default `polyphony`) replaces the fixed `polyphony:` prefix on every Redis key,
and `DATABASE_URL` selects the database. Evals and the simulator set their own `NAMESPACE`
and database automatically, so they never touch a live room.

## Redis keys

v0 keys under `<NAMESPACE>:`, plus `<NAMESPACE>:control` messages `{"frame": "changed"}` and
`<NAMESPACE>:memq:<seq>` (cached memory retrieval for a window tail, TTL 5 min).

## API additions

```text
GET  /frame                    {name, label, person_label, text}
PUT  /frame                    {label?, person_label?, text?}       -> publishes control
POST /thoughts                 kind "person" (alias "conscious") | "world"
GET  /senses ... POST/PUT      + feeds[], search
POST /cadence                  {resolution, confirm?: bool}
POST /cadence/{id}/confirm     resumes a cadence paused before recording
GET  /stats?hours=24           calls, tokens, latency per tier/provider; student–teacher agreement
GET  /memories?seq=<n>&k=5     debug: what memory retrieves for a point in the stream
POST /digest                   write a digest now for everything since the last one
GET  /digests?limit=20         recent digests
```

## UI additions

Frame editor (label, person label, text); feeds and search fields on senses; a "confirm
before recording" checkbox on Decide; digests rendered as cards in the stream with a "digest
now" button; a stats panel with listen queue wait; a memories line under each thought in a
debug view. Still one page, no build step.

## Config additions (.env)

```text
ENSEMBLE=mind
LISTEN_PROVIDER=ollama  LISTEN_MODEL=qwen2.5:3b  LISTEN_BASE_URL=   # host.docker.internal on macOS
TEACHER_PROVIDER=  TEACHER_MODEL=  TEACHER_SAMPLE_RATE=0.05          # empty = speak tier
EMBED_PROVIDER=ollama  EMBED_MODEL=nomic-embed-text  EMBED_DIM=768  MEMORY_K=5
MCP_FETCH_COMMAND="uvx mcp-server-fetch"
MCP_FEEDS_COMMAND="python -m polyphony.mcp_servers.feeds"
EXA_API_KEY=  TAVILY_API_KEY=
SENSE_MAX_ITEMS=3  SENSE_READ_MAX_ARTICLES=3
NAMESPACE=polyphony
DIGEST_IDLE_SECONDS=90       # quiet this long after a thought with nobody forming -> digest
MCP_HTTP_HOST=127.0.0.1
```

If Ollama is unreachable, fail loudly naming the setting; never fall back silently to a paid
tier.

## Prompts — required content

Every prompt is ordered from most shared to least shared, for prompt caching:
`frame text` → `engine text` (identical for every voice) → `voice text` (where relevant) →
the rendered stream. Frame and engine text form a prefix shared by every voice in the
ensemble; putting the voice text earlier would break that prefix at its first byte. Engine
text keeps all v0 requirements, reworded neutrally:

- You see: the stream, stray older thoughts, memories (related older thoughts), every voice's
  latest listen, who is forming.
- You are never obliged to speak or to agree with what "you" said before.
- Reasons to speak; reasons not to (repetition, done, not interested, tired, irritated).
- Original topics come only from the human or the world; long drift away from them, or
  repetition, is a reason to stop.
- Others forming is a reason to wait; everyone bored is a reason to let it go quiet.
- You may propose a cadence.
- Speak: no name tag, no "as I said", any length, may disagree with anyone including yourself.

`render()` adds a memories block after strays (`[seq] author, 3 days ago: content`) and tags
the human's thoughts with the frame's `person_label`. Structured output: pass the JSON schema
to providers that support it (OpenAI, Gemini, Ollama `format`); always keep extraction and
normalisation as the fallback.

## Build order

Each phase ends with all tests green and its own commits.

1. **Frames and ensembles** — frame loading/seeding, neutral engine prompts in shared-prefix
   order, `person` rename, migrations module, `NAMESPACE`, three ensembles, localhost
   binding, docs.
2. **Instrumentation and local listen** — `llm_calls` with queue wait, `/stats`, local listen
   defaults and Ollama parallelism docs, structured outputs, prompt caching, shadow teacher
   and `listen_samples`.
3. **Senses** — feeds MCP server, MCP client, sense graph, sense config, UI.
4. **Memory** — pgvector, embedder, retrieval, backfill.
5. **Digest and evals** — digest component and UI, then scenarios, runner, outcome-first
   judge, compare.
6. **Cadence graph** — checkpointer, resume, confirm.
7. **MCP server** — `mind.py`, docs for Claude Desktop / Claude Code.
8. **Distillation** — simulator, time-split export of synthetic and shadow data, teacher
   self-agreement, Colab notebook, student eval, `polyphony-listen`.

## v1 is done when

1. Everything in v0's "done" list still holds with `ENSEMBLE=mind`.
2. Each example ensemble starts and produces a discussion; editing the frame takes effect live;
   no engine prompt mentions minds or heads.
3. With only Ollama and one hosted key, listen runs locally, voices speak, and `/stats` shows
   calls, tokens, queue wait and student–teacher agreement; the p95 listen latency for `mind`
   is measured and recorded in the README.
4. A feed-based sense appends `world` thoughts with any provider; with an Exa or Tavily key and
   `search: true`, searches inform summaries; without, they are skipped.
5. A thought from far outside the window that is related to the present appears as a memory,
   and one stream move with N voices listening performs one retrieval, not N.
6. A cadence interrupted mid-way resumes without re-asking finished voices; confirm mode
   pauses before recording.
7. An MCP client can `say`, `read_stream` and `start_cadence`; by default neither the MCP
   server nor the API is reachable from another machine.
8. When the room goes quiet, a `digest` thought appears that states the question, each
   voice's position and the outcome, and no voice wakes because of it.
9. An eval run and a live room share one PostgreSQL server and Redis without seeing each
   other's thoughts.
10. The simulator produces a labelled dataset; the notebook trains and exports
    `polyphony-listen`; switching to it is one setting; the student report compares the three
    students against the teacher's self-agreement ceiling on time-held-out real data.

## Working rules

- Tests keep v0's conventions: real PostgreSQL and Redis, a scripted fake LLM provider, plus a
  fake MCP client. Every phase adds tests for its own behaviour.
- `docker compose up` still starts everything with no manual step; optional features stay off
  without their keys.
- Every new setting is in `.env.example` with a safe default; README updated per phase; this
  document's *Decisions* section gains an entry when a phase makes a new choice.
- All code, comments, docs and commit messages in English. Small, focused commits authored as
  `Alex Wang <alex119122114@gmail.com>`, with no AI co-author trailers or "Generated with"
  lines.
- If existing code blocks the plan, make the smallest change and record it here.
