# Polyphony

> A polyphony of anything: many voices, no conductor.

Polyphony is a room with no chair. Several AI agents — *voices*, each with its own wants and
temperament — watch one shared stream of thoughts and independently decide whether to add to
it. Nobody schedules them, nobody moderates them, nobody tells them when to stop. You drop a
question in; *senses* bring in news from feeds you chose; a *cadence* puts a resolution to a
weighted vote; when the room goes quiet, a *digest* tells you what came out of it: a decision,
or a clear map of positions.

The voices can be the inner voices of one mind, the members of a family, or the groups of a
city. What a voice is lives in its text; what the whole polyphony is lives in one more text, the
*frame*. The engine is the same.

There is no topic object, no turn order, no loop cap. "Quiet" is not a state in the code — it is
what happens when every voice listens and chooses not to speak.

Design: [`docs/DESIGN.md`](docs/DESIGN.md).

## Run it

You need [Ollama](https://ollama.com) (listen and memory run locally) and one hosted API key
(speak).

```bash
ollama pull qwen2.5:3b          # qwen2.5:1.5b on 8 GB machines
ollama pull nomic-embed-text
cp .env.example .env            # set SPEAK_PROVIDER / SPEAK_API_KEY
docker compose up               # mind + PostgreSQL (pgvector) + Redis
open http://localhost:8000
```

Pick the ensemble with `ENSEMBLE=mind|council|public` (one ensemble per instance).

From a terminal:

```bash
python scripts/say.py "should I take the AI specialization next term?"
python scripts/tail.py          # watch the stream live
```

Without Docker: Python 3.12+, PostgreSQL 16 with pgvector, Redis 7, then
`pip install -e ".[dev]" && python -m polyphony.main`. Without pgvector everything runs except
memory.

### Local listen: latency

Every stream move triggers one listen per voice, each a prompt of a few thousand tokens. Ollama
serves one request per model at a time unless told otherwise, so seven voices can queue for tens
of seconds. Set, where Ollama runs:

```bash
OLLAMA_NUM_PARALLEL=4     # ~ number of voices, as memory allows
OLLAMA_KEEP_ALIVE=30m     # keep the model loaded between bursts
```

`/stats` (and the panel on the page) shows p95 latency and **queue wait** per tier — queue wait
is client latency minus Ollama's own `total_duration`. Target: p95 listen latency under 10 s for
the `mind` ensemble. Measured on: *(fill in your hardware and number)*.

## Models

| tier | called | default |
|---|---|---|
| `listen` (student) | every stream move, by every voice | local `qwen2.5:3b` via Ollama |
| `speak` | when a voice speaks; cadences; digest | a strong hosted model |
| `sense` | triage + summarise feed articles | the speak tier |
| `teacher` | shadow-labels 5% of listens; judges evals; plays both tiers in self-play | the speak tier |
| `embed` | every thought, in the background | local `nomic-embed-text` |

Providers: `openai` (and anything OpenAI-compatible via `*_BASE_URL`), `anthropic`, `gemini`,
`deepseek`, `ollama`. If Ollama is unreachable Polyphony fails loudly — it never falls back to a
paid tier. A voice can override its tiers in front matter:

```yaml
---
listen: { provider: ollama, model: qwen2.5:1.5b }
speak:  { provider: deepseek, model: deepseek-chat }
---
```

## Ensembles

```text
ensembles/<name>/
  frame.md                 what this polyphony is; label; the human's label
  voices/<voice>/voice.md  one text per voice
  senses.yaml              instructions + RSS/Atom feeds
```

Shipped: `mind` (inner voices — placeholders, rewrite them as your own), `council` (a family
deciding whether to move), `public` (groups of residents reacting to city news). The files seed
the database on first start; after that, edit the frame, voices and senses live from the page.

A voice is *described* (**Wants** / **Believes** / **Sounds like** / **Shows up when**) or
*shown* (a few real moments and nothing else; see `ensembles/council/voices/maya`).

## Senses

```yaml
- name: world
  instruction: "The most important world news of the last day."
  every_minutes: 240
  feeds: [https://feeds.bbci.co.uk/news/world/rss.xml]
  search: false            # true + EXA_API_KEY or TAVILY_API_KEY: background search
```

Each run is a small LangGraph workflow — collect → triage → [search] → read → summarise → emit —
over MCP: our `polyphony-feeds` server (RSS/Atom, SSRF-guarded), the official
`mcp-server-fetch` for article text, optionally Exa or Tavily. Works with any provider.

## Polyphony as an MCP server

```bash
python -m polyphony.mcp_servers.mind           # stdio
python -m polyphony.mcp_servers.mind --http    # streamable HTTP on 127.0.0.1:8001
```

Tools: `say`, `read_stream`, `start_cadence`, `list_voices` (over the HTTP API at `API_URL`).
Claude Code: `claude mcp add polyphony -- python -m polyphony.mcp_servers.mind`. Claude Desktop:
add `{"command": "python", "args": ["-m", "polyphony.mcp_servers.mind"]}` under `mcpServers`.

**No auth.** The HTTP API (via Compose) and the MCP HTTP transport listen on `127.0.0.1` only.
Setting `MCP_HTTP_HOST=0.0.0.0` or publishing the port elsewhere lets anyone who can reach it
speak in the room and start cadences.

## Evals

```bash
python scripts/eval.py                      # every scenario in polyphony/evals/scenarios
python scripts/eval.py --compare evals/results/A.json evals/results/B.json
```

Each scenario runs in its own database and Redis namespace against the real models; the teacher
judges the outcome first (does the digest give a decision or a clear map of positions), then
repetition, voice fidelity, engagement and convergence.

## Distilling listen

1. **Synthetic data** — self-play with the teacher on both tiers, in `<db>_sim`:
   `python -m simulate.run --ensembles mind,council,public --repeats 2`
2. **Real data** — happens by itself: 5% of live listens are also sent to the teacher
   (`listen_samples`, source `live`).
3. **Export, split by time** — live samples before the cutoff may train; after it they are the
   held-out test set:
   `python scripts/export_listen_samples.py --cutoff 2026-11-15T00:00:00Z --live-db $DATABASE_URL --synthetic-db ${DATABASE_URL}_sim --speak-share 0.35`
4. **Train** — [`training/distill_listen.ipynb`](training/distill_listen.ipynb) on Colab (LoRA,
   Unsloth) → `polyphony-listen.gguf` + `Modelfile` → `ollama create polyphony-listen -f Modelfile`.
5. **Switch** — `LISTEN_MODEL=polyphony-listen`. No code change.
6. **Judge** — against the teacher's own self-agreement (the ceiling):
   `python scripts/eval_listen_student.py --db $DATABASE_URL --cutoff ... --model ollama:qwen2.5:3b --model ollama:polyphony-listen --model teacher`

## API

| | |
|---|---|
| `POST /thoughts` | `{content}` — the human's thought (`kind: "person"`, or `"world"`) |
| `GET /thoughts?after=&limit=` · `GET /stream?after=` | the record; SSE replay then live |
| `GET /listens` | every listen, spoken or not |
| `POST /cadence` `{resolution, confirm?}` · `GET /cadence/{id}` · `POST /cadence/{id}/confirm` | weighted vote |
| `GET/PUT /frame` | what this polyphony is |
| `POST /digest` · `GET /digests` | outcome summaries |
| `GET /stats?hours=` | calls, tokens, latency, queue wait; student–teacher agreement |
| `GET /memories?seq=&k=` | debug: what memory retrieves at a point |
| `/voices`, `/senses` | manage live |

## Tests

```bash
TEST_DATABASE_URL=postgresql+asyncpg://postgres@localhost:5432/polyphony_test \
TEST_REDIS_URL=redis://localhost:6379/15 pytest
```

Real PostgreSQL and Redis, a scripted fake model, a fake MCP client.

Grew out of [Multi-Agent-Chat-System](https://github.com/AlexW6385/Multi-Agent-Chat-System).

MIT
