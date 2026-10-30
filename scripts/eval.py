"""Run eval scenarios against real models and report; optionally compare two runs.

    python scripts/eval.py                          # every scenario in evals/scenarios
    python scripts/eval.py --only mind-specialization
    python scripts/eval.py --compare evals/results/A.json evals/results/B.json
"""

from __future__ import annotations

import asyncio

from polyphony.evals.runner import cli

if __name__ == "__main__":
    asyncio.run(cli())
