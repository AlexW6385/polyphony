"""The key: text saying what this mind is and how its voices relate.

It is placed first in every system prompt. Everything after it is engine text
and is framing-neutral. The key lives in the `keys` table (seeded from
`mind/key.md`) and can be edited live.

The human participant is the conscious part of the mind: what it drops in is a
theme, labelled `conscious` in every prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker

from .db import KeyRow
from .text import parse_front_matter

CONSCIOUS = "conscious"  # who themes come from, in prompts and on the page
KEY_ROW = 1  # the keys table has one row

DEFAULT_KEY = """\
You are one voice among several voices of one mind; they share one score of thoughts and each
decides for itself whether to add to it. The conscious part sometimes drops a theme in; the
outside world comes in through sources it chose."""


@dataclass(frozen=True)
class Key:
    text: str

    @classmethod
    def load(cls, path: Path) -> Key:
        if not path.exists():
            return cls(text=DEFAULT_KEY)
        body, _ = parse_front_matter(path.read_text(encoding="utf-8"))
        return cls(text=body or DEFAULT_KEY)

    def to_dict(self) -> dict[str, str]:
        return {"text": self.text}


async def get_key(sessions: async_sessionmaker) -> Key:
    async with sessions() as s:
        row = await s.get(KeyRow, KEY_ROW)
    return Key(text=row.text) if row is not None else Key(text=DEFAULT_KEY)
