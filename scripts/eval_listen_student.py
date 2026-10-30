"""Compare listen students against the teacher on time-held-out real data.

    python scripts/eval_listen_student.py --db postgresql+asyncpg://.../polyphony \
        --cutoff 2026-11-15T00:00:00Z \
        --model ollama:qwen2.5:3b --model ollama:polyphony-listen --model teacher \
        --ceiling 200 --out reports/listen.md

For each model, every held-out live sample (created at/after --cutoff) is run
again with the exact stored system prompt and input; its decision is compared
with the stored teacher decision. `teacher` runs listen on the teacher tier.

Ceiling: the first --ceiling samples are labelled by the teacher once more, in
independent calls; that self-agreement is the most any student can be expected
to reach. Reports agreement on `speak`, JSON validity, latency and tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from polyphony import llm, prompts
from polyphony.config import TierConfig, get_settings
from polyphony.db import ListenSample, make_engine, make_sessionmaker
from polyphony.distill import agreement
from polyphony.voice import normalize_decision

CONCURRENCY = 4


def tier_of(spec: str) -> TierConfig:
    if spec == "teacher":
        return get_settings().teacher_tier
    provider, model = spec.split(":", 1)
    return TierConfig(provider=provider, model=model)


async def run_model(tier: TierConfig, rows: list[ListenSample]) -> dict:
    sem = asyncio.Semaphore(CONCURRENCY)
    calls: list[llm.Trace] = []

    async def tracer(t: llm.Trace) -> None:
        calls.append(t)

    llm.set_tracer(tracer)

    async def one(r: ListenSample):
        async with sem:
            try:
                d = await llm.complete_json(tier, r.system, r.prompt, schema=prompts.LISTEN_SCHEMA,
                                            tier_name="eval", purpose="eval_listen")
                return normalize_decision(d), True
            except Exception:
                return None, False

    start = time.perf_counter()
    results = await asyncio.gather(*[one(r) for r in rows])
    wall = time.perf_counter() - start
    await asyncio.sleep(0.1)
    llm.set_tracer(None)
    lat = [c.latency_ms for c in calls if c.ok]
    return {
        **agreement([(d, r.teacher_decision) for (d, _), r in zip(results, rows)]),
        "json_valid": round(sum(ok for _, ok in results) / len(rows), 4) if rows else None,
        "latency_ms_p50": statistics.median(lat) if lat else None,
        "latency_ms_p95": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
        "input_tokens": sum(c.input_tokens or 0 for c in calls),
        "output_tokens": sum(c.output_tokens or 0 for c in calls),
        "wall_s": round(wall, 1),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="provider:model, or 'teacher'; repeatable")
    ap.add_argument("--ceiling", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="reports/listen.md")
    args = ap.parse_args()

    cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
    engine = make_engine(args.db)
    async with make_sessionmaker(engine)() as s:
        q = (select(ListenSample).where(ListenSample.source == "live",
                                        ListenSample.created_at >= cutoff,
                                        ListenSample.teacher_decision.isnot(None))
             .order_by(ListenSample.id))
        rows = list((await s.scalars(q)).all())
    await engine.dispose()
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("no held-out live samples after the cutoff")

    report: dict[str, dict] = {}
    ceiling_rows = rows[: args.ceiling]
    report["teacher self-agreement (ceiling)"] = await run_model(tier_of("teacher"), ceiling_rows)
    for spec in args.model:
        report[spec] = await run_model(tier_of(spec), rows)

    cols = ["n", "accuracy", "precision_speak", "recall_speak", "f1_speak", "json_valid",
            "latency_ms_p50", "latency_ms_p95", "input_tokens", "output_tokens"]
    lines = [f"# Listen students vs teacher\n\nHeld out: live samples since {cutoff.isoformat()}"
             f" ({len(rows)}). Reference speak rate: "
             f"{report[args.model[0]]['reference_speak_rate']}.\n",
             "| model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, m in report.items():
        lines.append(f"| {name} | " + " | ".join(str(m.get(c)) for c in cols) + " |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
