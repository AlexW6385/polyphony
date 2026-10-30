"""Distilling the listen decision: metrics and dataset export.

The student (small local model) is judged by agreement with the teacher on
`speak`. Most listens are quiet, so accuracy alone flatters a student that
never speaks: precision and recall for `speak=true` are reported too.

The ceiling is the teacher's agreement with itself (two independent labels of
the same input at the same temperature). A student is measured against that,
not against 100%.

Shadow pairs (live) are split by time: before a cutoff they may train, after it
they are held out. Each export records its cutoff, so no pair is in both.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any


def agreement(pairs: Iterable[tuple[dict[str, Any] | None, dict[str, Any] | None]]) -> dict:
    """Agreement of `a` with reference `b` on `speak`. Pairs with a missing side are skipped."""
    tp = fp = fn = tn = 0
    for a, b in pairs:
        if not a or not b:
            continue
        pa, pb = bool(a.get("speak")), bool(b.get("speak"))
        if pa and pb:
            tp += 1
        elif pa and not pb:
            fp += 1
        elif not pa and pb:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else None)
    return {
        "n": n,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "precision_speak": None if precision is None else round(precision, 4),
        "recall_speak": None if recall is None else round(recall, 4),
        "f1_speak": None if f1 is None else round(f1, 4),
        "reference_speak_rate": round((tp + fn) / n, 4) if n else None,
    }


def to_chat_example(system: str, prompt: str, decision: dict[str, Any]) -> dict[str, Any]:
    """One supervised example in the common chat format (works with Unsloth / TRL)."""
    target = {k: decision.get(k) for k in
              ("speak", "mood", "reason", "pause_seconds", "propose_cadence")}
    return {"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]}


def time_split(rows: list[Any], cutoff: datetime) -> tuple[list[Any], list[Any]]:
    """(before cutoff, at/after cutoff) by `created_at`."""
    before = [r for r in rows if r.created_at < cutoff]
    after = [r for r in rows if r.created_at >= cutoff]
    return before, after


def balance(rows: list[Any], speak_share: float | None, key=lambda r: r.teacher_decision) -> list:
    """Downsample quiet examples so speak=true is about `speak_share` of the set."""
    if not speak_share:
        return rows
    speak = [r for r in rows if bool((key(r) or {}).get("speak"))]
    quiet = [r for r in rows if not bool((key(r) or {}).get("speak"))]
    if not speak:
        return rows
    want_quiet = int(len(speak) * (1 - speak_share) / speak_share)
    step = max(1, len(quiet) // max(1, want_quiet))
    return sorted(speak + quiet[::step][:want_quiet], key=lambda r: r.id)
