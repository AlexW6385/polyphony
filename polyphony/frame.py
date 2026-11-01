"""The frame: text saying what this mind is and how its voices relate.

It is placed first in every system prompt. Everything after it is engine text
and is framing-neutral. The frame is part of the mechanism, not a setting: it is
copied from `mind/frame.md` into the `frames` table on every start and is not
editable from the page.

The human participant is the conscious part of the mind: its thoughts are
labelled `conscious` in every prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from .db import FrameRow
from .text import parse_front_matter

CONSCIOUS = "conscious"  # the human's thoughts: their kind and author
FRAME_ROW = 1  # the frames table has one row

DEFAULT_FRAME = """\
You are one voice among several voices of one mind; they share one stream of thoughts and each
decides for itself whether to add to it. The conscious part sometimes drops a thought in; the
outside world comes in through sources it chose."""


@dataclass(frozen=True)
class Frame:
    text: str

    @classmethod
    def load(cls, path: Path) -> Frame:
        if not path.exists():
            return cls(text=DEFAULT_FRAME)
        body, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        return cls(text=body or DEFAULT_FRAME)

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text}


async def get_frame(sessions: async_sessionmaker) -> Frame:
    async with sessions() as s:
        row = await s.get(FrameRow, FRAME_ROW)
    return Frame(text=row.text) if row is not None else Frame(text=DEFAULT_FRAME)
