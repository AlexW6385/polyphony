"""Run scenarios, judge them, write a report, compare reports.

A scenario (polyphony/evals/scenarios/*.yaml): an ensemble, seed thoughts, a
budget, and a description of a good run. Each run happens in its own database
and Redis namespace (see harness.py), against the configured real models.
Results go to evals/results/<timestamp>.json; `--compare A B` shows the
difference per scenario and criterion, to decide whether to keep a prompt or
model change.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import func, select

from ..config import get_settings
from ..db import LLMCall, VoiceRow
from ..digest import write_digest
from ..frame import get_frame
from ..harness import isolated, open_run, play
from .judge import CRITERIA, judge

log = logging.getLogger("evals")
SCENARIOS = Path(__file__).resolve().parent / "scenarios"
RESULTS = Path.cwd() / "evals" / "results"


def load_scenarios(only: list[str] | None = None) -> dict[str, dict[str, Any]]:
    out = {}
    for p in sorted(SCENARIOS.glob("*.yaml")):
        if only and p.stem not in only:
            continue
        out[p.stem] = yaml.safe_load(p.read_text())
    return out


async def run_scenario(name: str, sc: dict[str, Any]) -> dict[str, Any]:
    base = get_settings()
    settings = isolated(base, "eval", ensemble=sc["ensemble"], teacher_sample_rate=0.0)
    budget = sc.get("budget") or {}
    run = await open_run(settings)
    try:
        start = time.monotonic()
        thoughts, how = await play(run, sc["seed"], idle=budget.get("idle_s", 30),
                                   timeout=budget.get("timeout_s", 600),
                                   max_thoughts=budget.get("max_thoughts"),
                                   gap=sc.get("gap_s", 0))
        digest = await write_digest(run.stream, settings)
        frame = await get_frame(run.stream.sessions, settings.ensemble)
        async with run.stream.sessions() as s:
            voices = {v.name: v.text for v in (await s.scalars(select(VoiceRow))).all()}
            cost = (await s.execute(select(func.count(), func.sum(LLMCall.input_tokens),
                                           func.sum(LLMCall.output_tokens))
                                    .where(LLMCall.purpose != "judge"))).one()
        verdict = await judge(base.teacher_tier, frame=frame.text, voices=voices,
                              good=sc.get("good", ""), thoughts=thoughts,
                              digest=digest["content"] if digest else None,
                              person_label=frame.person_label)
        return {
            "scenario": name, "ensemble": sc["ensemble"], "ended": how,
            "seconds": round(time.monotonic() - start, 1),
            "thoughts": len(thoughts),
            "voice_thoughts": sum(t["kind"] == "voice" for t in thoughts),
            "calls": cost[0], "input_tokens": int(cost[1] or 0), "output_tokens": int(cost[2] or 0),
            "digest": digest["content"] if digest else None,
            "outcome": (digest or {}).get("meta", {}).get("outcome"),
            **verdict,
        }
    finally:
        await run.close()


def compare(a_path: str, b_path: str) -> str:
    a = {r["scenario"]: r for r in json.loads(Path(a_path).read_text())["results"]}
    b = {r["scenario"]: r for r in json.loads(Path(b_path).read_text())["results"]}
    lines = ["| scenario | " + " | ".join(CRITERIA) + " | tokens |",
             "|---" * (len(CRITERIA) + 2) + "|"]
    for name in sorted(set(a) & set(b)):
        cells = []
        for c in CRITERIA:
            x, y = a[name]["scores"].get(c), b[name]["scores"].get(c)
            cells.append(f"{x}→{y}" if x != y else str(x))
        ta = a[name]["input_tokens"] + a[name]["output_tokens"]
        tb = b[name]["input_tokens"] + b[name]["output_tokens"]
        cells.append(f"{ta}→{tb}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


async def cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        print(compare(*args.compare))
        return
    results = []
    for name, sc in load_scenarios(args.only).items():
        log.info("scenario %s", name)
        r = await run_scenario(name, sc)
        log.info("%s: %s — %s", name, r["scores"], r["notes"])
        results.append(r)
    RESULTS.mkdir(parents=True, exist_ok=True)
    s = get_settings()
    out = RESULTS / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(),
        "models": {"listen": s.listen_tier.model, "speak": s.speak_tier.model,
                   "judge": s.teacher_tier.model},
        "results": results,
    }, indent=2))
    print(f"wrote {out}")
    for r in results:
        print(f"- {r['scenario']}: {r['scores']} ({r['ended']}, {r['thoughts']} thoughts)")
