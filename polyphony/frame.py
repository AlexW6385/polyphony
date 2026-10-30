"""The frame: text saying what this polyphony is.

It is placed first in every system prompt. Everything after it is engine text
and is framing-neutral. The frame lives in the `frames` table (seeded from
`ensembles/<name>/frame.md`) and can be edited live.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from .db import FrameRow
from .text import parse_front_matter

DEFAULT_FRAME = """\
You are one voice in a polyphony: several voices share one stream of thoughts and each decides
for itself whether to add to it. The human participant sometimes drops a thought in; the
outside world comes in through sources the human chose."""


@dataclass(frozen=True)
class Frame:
    name: str
    label: str
    person_label: str
    text: str

    @classmethod
    def load(cls, path: Path, name: str) -> Frame:
        if not path.exists():
            return cls(name=name, label=name, person_label="you", text=DEFAULT_FRAME)
        body, meta = parse_front_matter(path.read_text(encoding="utf-8"))
        return cls(
            name=name,
            label=str(meta.get("label") or name),
            person_label=str(meta.get("person_label") or "you"),
            text=body or DEFAULT_FRAME,
        )

    @classmethod
    def from_row(cls, row: FrameRow) -> Frame:
        return cls(row.name, row.label, row.person_label, row.text)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "label": self.label, "person_label": self.person_label,
                "text": self.text}


async def get_frame(sessions: async_sessionmaker, name: str) -> Frame:
    async with sessions() as s:
        row = await s.get(FrameRow, name)
    if row is None:
        return Frame(name=name, label=name, person_label="you", text=DEFAULT_FRAME)
    return Frame.from_row(row)
