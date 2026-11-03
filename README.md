# Polyphony

> Many voices of one mind, no conductor.

Polyphony is the inside of a head, played out. Several AI agents, the *voices* of one mind,
each with its own wants and temperament, read one shared *stream* of thoughts and each decides
for itself whether to add to it. Nobody schedules them, moderates them or tells them when to
stop. You, the conscious part, drop a thought in (**Think**); *senses* bring in news from feeds you chose;
the *clerk* (not a voice) calls a *cadence*, a weighted vote between options drawn from the
discussion; when the stream goes quiet, a *digest* says what came out of it: a decision, or a
clear map of where the voices stand.

What a voice is lives in a sentence or two of text; what the whole mind is lives in one more
text, the *frame*. There is no topic object, no turn order, no loop cap. A voice that listens and
chooses not to speak stays quiet; when every voice does, the stream is quiet.

Design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Run it

You need Docker and one API key (Anthropic by default). For memory, also
[Ollama](https://ollama.com) on the host with the embedding model (optional: without it,
everything runs and voices just have no memories):

```bash
ollama pull qwen3-embedding:0.6b      # ~640 MB, Chinese and English
docker compose up                     # app + PostgreSQL (pgvector) + Redis
open http://localhost:8000/settings   # pick the provider, paste the key, name the two models
open http://localhost:8000            # think something
```

From a terminal:

```bash
python scripts/say.py "should I take the AI specialization next term?"
python scripts/tail.py          # watch the stream live
```

Without Docker: Python 3.12+, PostgreSQL 16, Redis 7, then
`pip install -e ".[dev]" && python -m polyphony.main`.

## Words

| word | meaning |
|---|---|
| **voice** | one agent of the mind, defined by a sentence or two of text (`mind/voices/<name>/voice.md`) |
| **frame** | what this mind is and how its voices relate (`mind/frame.md`); first in every prompt; part of the mechanism, reloaded from the file on every start, not editable from the page |
| **stream** | the one shared, ordered record of thoughts |
| **conscious** | you: the conscious part of the mind; what you type under **Think** |
| **listen** | a voice reading the stream and deciding whether to speak, or staying **quiet**; while it decides it shows as *listening* |
| **memory** | older thoughts related to what is being said now, found by meaning and shown to the voices |
| **clerk** | not a voice: every 5 new thoughts, and after 30 s of quiet, decides whether a vote is due and drafts its question and options; after quiet the discussion is over, so it calls a vote unless there is truly nothing to decide |
| **cadence** | the vote: voices weigh each other, then support any of the options; the top option wins with at least 60%; one at a time, everyone else waits |
| **digest** | the outcome of a stretch of the stream, written after 90 s of quiet |
| **sense** | a standing instruction that brings news from your feeds in as `world` thoughts, once a day |

The voices and senses under `mind/` seed the database on first start; after that, edit them live
from the page. The frame is always taken from `mind/frame.md`.

## Your voices

A voice is one part of you: a want, a worry, a way of seeing things. The voices are not
characters or experts; they all speak as "I", and "I" is you. What makes a discussion worth
reading is that they pull in different directions, the way your own thoughts do.

Each voice is one or two sentences: what it cares about and how it talks. That is the whole
definition. For example (`mind/voices/restraint/voice.md`):

```
---
---
觉得越少越好，能删就删，多一个步骤、多一句话都嫌多。说话很短。
```

The eleven voices in `mind/voices/` are the author's own, drawn from their style. To make your own:

1. **Find your recurring pulls.** Read things you wrote yourself (chats with an assistant,
   notes, a journal), or ask a model that knows your history to list the tensions it sees:
   what you keep wanting, what you worry about, how you argue with yourself.
2. **Merge and cut to about eight to twelve.** Fewer than six rarely disagree; many more and
   each one gets little to say. Keep pairs that pull against each other (move fast vs. do it
   right, more vs. less).
3. **Write each in one or two sentences, without examples.** Describe the stance and the
   manner of speaking. Leave out sample lines and personal facts: a model treats them as
   things to talk about, and the voice keeps bringing them up.
4. **Name each with one lowercase word** (`a-z`, digits, `-`, `_`; not `conscious`, `cadence`, `world` or `digest`).

Add them on the page (Voices → Add a voice) or as `mind/voices/<name>/voice.md`. Files seed the
database only on first start, when it has no voices yet; after that, the page is where voices
are added, edited, disabled or deleted. The front matter can set a different model for one
voice, e.g. `speak: {model: claude-opus-5-5}`; leave it empty otherwise.

## Models

| tier | called | default |
|---|---|---|
| `listen` | every new thought, once per voice; the cadence rounds | `claude-haiku-5-5` |
| `speak` | when a voice speaks; the clerk; the digest; judges evals | `claude-sonnet-5-5` |
| `sense` | picks and summarises news | the speak model |
| `embed` | every thought once, in the background; one query per stream state | `qwen3-embedding:0.6b` in Ollama |
| `jev` | instead of listen, the cadence rounds and the clerk's "vote due?", when Use Jev is on | off |

Prompts are laid out for the provider's prompt cache: the frame and rules first, then the stream
(the same for every voice, absolute times, one cache block per thought), and only then what is
particular to one call (the voice, the time now, strays, who is listening). A new thought
appends to a prefix that is already cached instead of changing it.

Set the provider, key and the two models on the settings page, or in `.env`
(see `.env.example`). Providers: `anthropic`, `openai` (and OpenAI-compatible endpoints),
`deepseek`, `gemini`.

## Memory

Every voice, conscious and world thought is embedded in the background (never on the append
path) and stored in PostgreSQL with pgvector. When a voice listens or speaks, the last few
thoughts are embedded as the query and the nearest older thoughts (outside the window the
voice already sees, cosine similarity at least `MEMORY_MIN_SIMILARITY`, at most `MEMORY_K`) are
shown as *Memory*, next to the random *Stray thoughts*. One query per stream state, cached in
Redis, serves every voice.

The default model, Qwen3-Embedding-0.6B, puts Chinese and English in one space, so a thought in
one language can recall one in the other. Run Ollama natively, not in Docker: on a Mac only the
native app uses the GPU. Any OpenAI-compatible endpoint works too
(`EMBED_PROVIDER=openai`, `EMBED_MODEL`, `EMBED_DIM`, `EMBED_API_KEY`). Changing the model or
dimension re-embeds everything in the background. `GET /memory?seq=123` shows what would be
recalled at any point of the stream; `GET /memory/status` shows how far the embedder is.

## Decisions: listen model or Jev

Listening, the two cadence rounds and the clerk's "is a vote due?" are decisions, not writing.
By default the listen model makes them. Turn on **Use Jev** in the Jev section of the settings page
(or `DECIDER=jev`) to send them to [Jev](https://typesafe.ai), TypeSafe AI's decision model,
which answers typed questions with probabilities instead of text: speak or not as a
probability, the pause as a score, the mood as a choice, each voice's weight as a score, and
support for each option as a probability. It needs a TypeSafe API key (early access), or a Vercel
AI Gateway key with `JEV_BASE_URL=https://ai-gateway.vercel.sh/typesafe` and
`JEV_MODEL=typesafe-ai/jev` (paid credits only; free-tier keys get a 403). A Jev
listen has no written reason; anything Jev fails on falls back to the listen model.

## Senses

```yaml
- name: world
  instruction: "The most important world news of the last day."
  daily_at: "04:00"        # once a day, local time (TIMEZONE, default America/Toronto)
  enabled: false           # the shipped senses are off; enable on the page or press Run now
  feeds: [https://feeds.bbci.co.uk/news/world/rss.xml]
```

A sense never runs because everything was resumed or restarted: only at its daily time, or once
by hand with **Run now**.

Each run is a small LangGraph workflow (collect, triage, read, summarise, emit) over MCP: our
`polyphony-feeds` server (RSS/Atom, SSRF-guarded) and the official `mcp-server-fetch` for
article text. There is no search step: voices search the web themselves when they speak.

## Polyphony as an MCP server

Other AI apps (Claude Code, Claude Desktop) can bring a question to your voices.

```bash
claude mcp add polyphony -- docker compose -f /path/to/polyphony/docker-compose.yml \
  exec -T mind python -m polyphony.mcp_servers.mind
```

| Tool | What it does |
|---|---|
| `ask` | brings a question in as conscious and returns at once with its `asked_seq` |
| `outcome` | waits up to ~50 s per call; once the voices are done (a few minutes), decided or not, the result, each voice's position, what is open, the votes; until then, progress |
| `say` | drops a thought in without waiting |
| `read_stream` | the thoughts after a given `seq` |
| `list_voices` | the voices, whether each is forming a thought, its last listen |

Then, in Claude Code: "ask polyphony whether I should send out my resume tomorrow morning or go
play badminton". `python -m polyphony.mcp_servers.mind --http` serves streamable HTTP on
127.0.0.1:8001 instead of stdio.

**No auth.** The HTTP API (via Compose) and the MCP HTTP transport listen on `127.0.0.1` only.
Publishing either elsewhere lets anyone who can reach it drop thoughts in.

## Evals

```bash
python scripts/eval.py                      # every scenario in polyphony/evals/scenarios
python scripts/eval.py --compare evals/results/A.json evals/results/B.json
```

Each scenario runs in its own database and Redis namespace against the real models; the speak
model judges the outcome first (does the digest give a decision or a clear map of positions), then
repetition, voice fidelity, engagement and convergence.

## API

| | |
|---|---|
| `POST /thoughts` | `{content}`: a thought from conscious (or `kind: "world"`) |
| `GET /thoughts?after=&limit=` · `GET /stream?after=` | the record; SSE replay then live |
| `GET /listens` | every listen, spoken or rested |
| `GET /cadences/running` | votes under way (the page's "voting" card) |
| `GET /cadence/{id}` | a vote the clerk called: question, options, weights, votes, result |
| `GET /frame` | what this mind is (read-only) |
| `GET /digests` | outcome summaries (written after 90 s of quiet) |
| `POST /pause` · `POST /resume` · `GET /pause` | stop / restart everything that calls a model (the **Stop all** button) |
| `GET /settings` | provider, API key, listen and speak models, monitoring charts |
| `GET/PUT /config` · `POST /config/test/{listen\|speak}` | what the page reads and saves; the key is stored in the database and never returned |
| `GET /stats?hours=` | calls, tokens, latency, errors, calls over time |
| `/voices`, `/senses` · `POST /senses/{name}/run` | manage live; run a sense once now |

## Tests

```bash
TEST_DATABASE_URL=postgresql+asyncpg://postgres@localhost:5432/polyphony_test \
TEST_REDIS_URL=redis://localhost:6379/15 pytest
```

Real PostgreSQL and Redis, a scripted fake model, a fake MCP client.

Grew out of [multi-agent-chat-system](https://github.com/AlexW6385/multi-agent-chat-system).

MIT
