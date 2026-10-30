"""Engine, session factory, ORM models, init_db().

`thoughts.seq` is the only ordering in the system. Never order by `created_at`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class Thought(Base):
    __tablename__ = "thoughts"

    seq: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    author: Mapped[str] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16))  # voice | person | world | cadence | digest
    content: Mapped[str] = mapped_column(Text)
    mood: Mapped[str | None] = mapped_column(String(80), nullable=True)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "author": self.author,
            "kind": self.kind,
            "content": self.content,
            "mood": self.mood,
            "meta": self.meta,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Listen(Base):
    __tablename__ = "listens"
    __table_args__ = (Index("ix_listens_voice_id", "voice", "id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    voice: Mapped[str] = mapped_column(String(64))
    trigger_seq: Mapped[int] = mapped_column(BigInteger)
    speak: Mapped[bool] = mapped_column(Boolean)
    mood: Mapped[str] = mapped_column(String(80))
    reason: Mapped[str] = mapped_column(Text)
    pause_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "voice": self.voice,
            "trigger_seq": self.trigger_seq,
            "speak": self.speak,
            "mood": self.mood,
            "reason": self.reason,
            "pause_seconds": self.pause_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Cadence(Base):
    __tablename__ = "cadences"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    resolution: Mapped[str] = mapped_column(Text)
    proposed_by: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))  # proposed | rejected | running | resolved
    weights: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    votes: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "resolution": self.resolution,
            "proposed_by": self.proposed_by,
            "status": self.status,
            "weights": self.weights,
            "votes": self.votes,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class VoiceRow(Base):
    """The record of truth for voices; `voices/` on disk only seeds this."""

    __tablename__ = "voices"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text)
    overrides: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class SenseRow(Base):
    __tablename__ = "senses"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    instruction: Mapped[str] = mapped_column(Text)
    every_minutes: Mapped[int] = mapped_column(Integer)
    feeds: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default="[]")
    search: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    last_run: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instruction": self.instruction,
            "every_minutes": self.every_minutes,
            "feeds": list(self.feeds or []),
            "search": bool(self.search),
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
        }


class SenseLog(Base):
    __tablename__ = "sense_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    sense: Mapped[str] = mapped_column(String(64))
    fingerprint: Mapped[str] = mapped_column(String(128), unique=True)
    seq: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class FrameRow(Base):
    """What this polyphony is. One row per ensemble; the running one is used."""

    __tablename__ = "frames"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str] = mapped_column(Text)
    person_label: Mapped[str] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "person_label": self.person_label,
                "text": self.text}


class LLMCall(Base):
    """One row per model call: the raw material of /stats and the evals."""

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_created", "created_at"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tier: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    purpose: Mapped[str] = mapped_column(String(32))
    voice: Mapped[str | None] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer)
    queue_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ok: Mapped[bool] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ListenSample(Base):
    """A listen input with the student's and/or the teacher's decision.

    `source` is `live` (shadow teacher on real runs) or `synthetic` (self-play).
    """

    __tablename__ = "listen_samples"
    __table_args__ = (Index("ix_listen_samples_source_id", "source", "id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    listen_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    source: Mapped[str] = mapped_column(String(16))
    ensemble: Mapped[str] = mapped_column(String(64))
    voice: Mapped[str] = mapped_column(String(64))
    system: Mapped[str] = mapped_column(Text)
    prompt: Mapped[str] = mapped_column(Text)
    student_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    student_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    student_decision: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    teacher_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    teacher_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    teacher_decision: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


def make_engine(url: str) -> AsyncEngine:
    return create_async_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=20)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine, attempts: int = 30, embed_dim: int = 768) -> None:
    """Create tables, then run migrations. Retries so `docker compose up` needs no ordering."""
    from .migrations import run_migrations

    for i in range(attempts):
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            await run_migrations(engine, embed_dim=embed_dim)
            return
        except (OSError, ConnectionError) as e:  # DB not up yet
            log.info("waiting for database (%s/%s): %s", i + 1, attempts, e)
            await asyncio.sleep(1)
        except Exception as e:  # asyncpg raises its own types while starting up
            if "starting up" in str(e) or "Connect" in type(e).__name__:
                log.info("waiting for database (%s/%s): %s", i + 1, attempts, e)
                await asyncio.sleep(1)
            else:
                raise
    raise RuntimeError("database did not become available")
