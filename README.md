# Polyphony

> A mind is a polyphony: many voices, no conductor.

Polyphony simulates the inside of a head: several AI agents — *voices*, each
with its own wants and temperament — watch one shared stream of thoughts and
independently decide whether to add to it. Nobody schedules them, nobody
moderates them, and nobody tells them when to stop. You are the *conscious*:
you drop thoughts in; *senses* bring in news you chose to follow; and when it
is time to settle something, a *cadence* puts it to a weighted vote.

There is no topic object, no turn order, no loop cap. "Quiet" is not a state
in the code — it is what happens when every voice listens and chooses not to
speak.

Design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Run it

```bash
cp .env.example .env      # add at least one API key
docker compose up         # mind + PostgreSQL + Redis
open http://localhost:8000
```

From a terminal:

```bash
python scripts/say.py "should I take the AI specialization next term?"
python scripts/tail.py    # watch the stream live
```

Without Docker: Python 3.12+, a PostgreSQL 16 and a Redis 7, then

```bash
pip install -e ".[dev]"
python -m polyphony.main
```

## Models

Two tiers, each pointed at any provider independently:

| tier | called | good choice |
|---|---|---|
| `listen` | every time the stream moves, by every voice | small and cheap, or local via Ollama |
| `speak` | only when a voice decides to speak, and for cadences | a strong model |

Providers: `openai` (and anything OpenAI-compatible via `*_BASE_URL`),
`anthropic`, `gemini`, `deepseek`, `ollama`. A voice can override either tier
in its front matter, so one mind can be a mixture of models:

```yaml
---
listen: { provider: ollama, model: qwen2.5:3b }
speak:  { provider: deepseek, model: deepseek-chat }
---
```

Senses need web search, so they use `openai`, `anthropic` or `gemini`
(`SENSE_PROVIDER`, defaulting to the speak tier).

## Voices

A voice is nothing but text — [`voices/<name>/voice.md`](voices) seeds the
database on first start, and after that you add, edit, disable and delete
voices from the page, live. Write one either way:

- *Described*: **Wants** / **Believes** / **Sounds like** / **Shows up when**
  (see [`voices/ride`](voices/ride/voice.md)).
- *Shown*: a few real moments when this voice spoke in your head, and
  nothing else (see [`voices/learn`](voices/learn/voice.md)).

The seven shipped voices are examples. Rewrite them to match your own head.

## API

| | |
|---|---|
| `POST /thoughts` | `{content}` — a conscious thought (or `kind: "world"`) |
| `GET /thoughts?after=&limit=` | the record, in order of `seq` |
| `GET /stream?after=` | SSE: replay from `after`, then live |
| `GET /listens` | every listen, spoken or not, with mood and reason |
| `POST /cadence` · `GET /cadence/{id}` | start a weighted vote on a resolution |
| `GET/POST /voices` · `PUT/DELETE /voices/{name}` | manage voices, live |
| `GET/POST /senses` · `PUT/DELETE /senses/{name}` | manage senses, live |

## Layout

```text
polyphony/   stream · voice · llm · prompts · cadence · sense · supervisor · api · main
voices/      seed voices          senses.yaml   seed senses
static/      the one page         scripts/      say.py, tail.py
tests/       run against real PostgreSQL + Redis with a scripted model
```

## Tests

```bash
TEST_DATABASE_URL=postgresql+asyncpg://postgres@localhost:5432/polyphony_test \
TEST_REDIS_URL=redis://localhost:6379/15 pytest
```

Grew out of [Multi-Agent-Chat-System](https://github.com/AlexW6385/Multi-Agent-Chat-System).

MIT
