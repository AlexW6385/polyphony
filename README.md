# Polyphony

> Many voices of one mind, no conductor.

Polyphony is the inside of a head, played out. Several AI agents, the *voices* of one mind,
each with its own wants and temperament, read one shared *score* of thoughts and each decides
for itself whether to add to it. Nobody schedules them, moderates them or tells them when to
stop. You, the conscious part, drop a *theme* in; *senses* bring in news from feeds you chose;
a *cadence* puts a resolution to a weighted vote; when the score goes quiet, a *digest* says what
came out of it: a decision, or a clear map of where the voices stand.

What a voice is lives in a sentence or two of text; what the whole mind is lives in one more
text, the *key*. There is no topic object, no turn order, no loop cap. A voice that listens and
chooses not to speak *rests*; when every voice rests, the score is quiet.

Design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Run it

You need Docker and one API key (Anthropic by default).

```bash
docker compose up                     # app + PostgreSQL + Redis
open http://localhost:8000/settings   # pick the provider, paste the key, name the two models
open http://localhost:8000            # drop a theme in
```

From a terminal:

```bash
python scripts/say.py "should I take the AI specialization next term?"
python scripts/tail.py          # watch the score live
```

Without Docker: Python 3.12+, PostgreSQL 16, Redis 7, then
`pip install -e ".[dev]" && python -m polyphony.main`.

## Words

| word | meaning |
|---|---|
| **voice** | one agent of the mind, defined by a sentence or two of text (`mind/voices/<name>/voice.md`) |
| **key** | what this mind is and how its voices relate (`mind/key.md`); first in every prompt |
| **score** | the one shared, ordered record of thoughts |
| **theme** | a thought you drop in, as the conscious part |
| **listen** | a voice reading the score and deciding whether to speak; it either speaks or **rests** |
| **cadence** | a weighted vote on a resolution; one at a time, and everyone else waits |
| **digest** | the outcome of a stretch of the score, written after 90 s of quiet |
| **sense** | a standing instruction that brings news from your feeds in as `world` thoughts, once a day |

The files under `mind/` seed the database on first start; after that, edit the key, voices and
senses live from the page.

## Models

| tier | called | default |
|---|---|---|
| `listen` | every new thought, once per voice | `claude-haiku-5-5` |
| `speak` | when a voice speaks; cadences; digest | `claude-sonnet-5-5` |
| `teacher` | labels 5% of listens for training a local listen model later; judges evals | the speak model |
| `sense` | picks and summarises news | the speak model |

Set the provider, key and the two models on the settings page, or in `.env`
(see `.env.example`). Providers: `anthropic`, `openai` (and OpenAI-compatible endpoints),
`deepseek`, `gemini`.

## Senses

```yaml
- name: world
  instruction: "The most important world news of the last day."
  daily_at: "04:00"        # once a day, local time (TIMEZONE, default America/Toronto)
  enabled: false           # the shipped senses are off; enable on the page or press Run now
  feeds: [https://feeds.bbci.co.uk/news/world/rss.xml]
  search: false            # true + EXA_API_KEY or TAVILY_API_KEY: background search
```

A sense never runs because the room was resumed or restarted: only at its daily time, or once
by hand with **Run now**.

Each run is a small LangGraph workflow (collect, triage, [search], read, summarise, emit) over
MCP: our `polyphony-feeds` server (RSS/Atom, SSRF-guarded), the official `mcp-server-fetch` for
article text, optionally Exa or Tavily.

## Polyphony as an MCP server

```bash
python -m polyphony.mcp_servers.mind           # stdio
python -m polyphony.mcp_servers.mind --http    # streamable HTTP on 127.0.0.1:8001
```

Tools: `say`, `read_score`, `start_cadence`, `list_voices` (over the HTTP API at `API_URL`).
Claude Code: `claude mcp add polyphony -- python -m polyphony.mcp_servers.mind`.

**No auth.** The HTTP API (via Compose) and the MCP HTTP transport listen on `127.0.0.1` only.
Publishing either elsewhere lets anyone who can reach it drop themes in and start cadences.

## Evals

```bash
python scripts/eval.py                      # every scenario in polyphony/evals/scenarios
python scripts/eval.py --compare evals/results/A.json evals/results/B.json
```

Each scenario runs in its own database and Redis namespace against the real models; the teacher
judges the outcome first (does the digest give a decision or a clear map of positions), then
repetition, voice fidelity, engagement and convergence.

## Distilling listen (later)

The tooling for replacing the listen model with a small local one is in place but not in use:
self-play data (`python -m simulate.run`), the 5% teacher labels on live listens, a time-split
export (`scripts/export_listen_samples.py`), a Colab notebook
([`training/distill_listen.ipynb`](training/distill_listen.ipynb)) and a student report
(`scripts/eval_listen_student.py`).

## API

| | |
|---|---|
| `POST /thoughts` | `{content}`: a theme (or `kind: "world"`) |
| `GET /thoughts?after=&limit=` · `GET /score?after=` | the record; SSE replay then live |
| `GET /listens` | every listen, spoken or rested |
| `POST /cadence` `{resolution}` · `GET /cadence/{id}` | weighted vote; 409 while one is under way |
| `GET/PUT /key` | what this mind is |
| `POST /digest` · `GET /digests` | outcome summaries |
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

Grew out of [Multi-Agent-Chat-System](https://github.com/AlexW6385/Multi-Agent-Chat-System).

MIT
