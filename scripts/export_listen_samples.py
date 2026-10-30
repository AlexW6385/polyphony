"""Export listen samples as a fine-tuning dataset, split by time.

    python scripts/export_listen_samples.py --cutoff 2026-11-15T00:00:00Z \
        --live-db postgresql+asyncpg://.../polyphony \
        --synthetic-db postgresql+asyncpg://.../polyphony_sim \
        --speak-share 0.35 --out data/listen

Writes:
  train.jsonl  synthetic samples + live (shadow) samples created before --cutoff
  test.jsonl   live samples created at/after --cutoff — held out, never trained on
  meta.json    cutoff, counts, sources, speak rates

Labels are the teacher's decisions. The time split keeps the test set honest:
every training run records its cutoff, so no pair is in both sets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import select

from polyphony.db import ListenSample, make_engine, make_sessionmaker
from polyphony.distill import balance, time_split, to_chat_example


async def load(url: str, source: str) -> list[ListenSample]:
    engine = make_engine(url)
    try:
        async with make_sessionmaker(engine)() as s:
            return list((await s.scalars(
                select(ListenSample).where(ListenSample.source == source,
                                           ListenSample.teacher_decision.isnot(None))
                .order_by(ListenSample.id))).all())
    finally:
        await engine.dispose()


def write(path: Path, rows: list[ListenSample]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(to_chat_example(r.system, r.prompt, r.teacher_decision),
                               ensure_ascii=False) + "\n")


def speak_rate(rows) -> float | None:
    return round(sum(bool(r.teacher_decision.get("speak")) for r in rows) / len(rows), 4) \
        if rows else None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, help="ISO time; live samples after it are test")
    ap.add_argument("--live-db", required=True)
    ap.add_argument("--synthetic-db")
    ap.add_argument("--speak-share", type=float, default=None,
                    help="downsample quiet examples in train to this speak share")
    ap.add_argument("--out", default="data/listen")
    args = ap.parse_args()

    cutoff = datetime.fromisoformat(args.cutoff.replace("Z", "+00:00"))
    live = await load(args.live_db, "live")
    synthetic = await load(args.synthetic_db, "synthetic") if args.synthetic_db else []
    live_train, live_test = time_split(live, cutoff)
    train = balance(synthetic, args.speak_share) + balance(live_train, args.speak_share)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "train.jsonl", train)
    write(out / "test.jsonl", live_test)
    meta = {
        "cutoff": cutoff.isoformat(), "exported_at": datetime.now().astimezone().isoformat(),
        "train": {"n": len(train), "synthetic": len(synthetic), "live_before_cutoff":
                  len(live_train), "speak_rate": speak_rate(train)},
        "test": {"n": len(live_test), "speak_rate": speak_rate(live_test),
                 "ids": [r.id for r in live_test]},
        "speak_share": args.speak_share,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "test"} | {
        "test": {k: v for k, v in meta["test"].items() if k != "ids"}}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
