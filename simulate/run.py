"""Self-play: generate synthetic listen data for distillation.

Runs the real engine — real prompt rendering, real ensembles — with the
teacher on both tiers, accelerated (shorter pauses), from seed topics across
ensembles. Every listen of every voice is stored in `listen_samples` with
source `synthetic` and the teacher's decision as the label.

    python -m simulate.run --ensembles mind,council,public --repeats 2

Runs in its own database (`<db>_sim`) and Redis namespace, never in a live room.
Labels come from live API calls; for very large runs, export the prompts and
label them through a provider's batch API instead (see README).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import yaml

from polyphony.config import get_settings
from polyphony.harness import isolated, open_run, play

log = logging.getLogger("simulate")
ROOT = Path(__file__).resolve().parent


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensembles", default="mind,council,public")
    ap.add_argument("--seeds", default=str(ROOT / "seeds.yaml"))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--max-thoughts", type=int, default=60)
    ap.add_argument("--pause-scale", type=float, default=0.1)
    args = ap.parse_args()

    base = get_settings()
    teacher = base.teacher_tier
    seeds = yaml.safe_load(Path(args.seeds).read_text())
    total = 0
    for ensemble in args.ensembles.split(","):
        for rep in range(args.repeats):
            for seed in seeds.get(ensemble, []):
                settings = isolated(
                    base, "sim", ensemble=ensemble, record_all_listens="synthetic",
                    teacher_sample_rate=0.0, pause_scale=args.pause_scale,
                    listen_provider=teacher.provider, listen_model=teacher.model,
                    listen_base_url=teacher.base_url or "", listen_api_key=teacher.api_key or "",
                    speak_provider=teacher.provider, speak_model=teacher.model,
                    speak_base_url=teacher.base_url or "", speak_api_key=teacher.api_key or "",
                )
                run = await open_run(settings)
                try:
                    ts, how = await play(run, [seed], idle=10, timeout=args.timeout,
                                         max_thoughts=args.max_thoughts)
                    n = len(await run.stream.listens(limit=10000))
                    total += n
                    log.info("%s #%d %r: %d thoughts, %d listens (%s)", ensemble, rep, seed[:50],
                             len(ts), n, how)
                finally:
                    await asyncio.sleep(1)  # let sample writes land
                    await run.close()
    log.info("done: %d synthetic listens", total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    asyncio.run(main())
