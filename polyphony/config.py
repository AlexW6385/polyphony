"""Settings, read from the environment / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class TierConfig(BaseModel):
    """Which provider/model a tier (listen, speak, sense) talks to."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://polyphony:polyphony@localhost:5432/polyphony"
    redis_url: str = "redis://localhost:6379/0"

    listen_provider: str = "openai"
    listen_model: str = "gpt-4o-mini"
    listen_base_url: str = ""
    listen_api_key: str = ""

    speak_provider: str = "anthropic"
    speak_model: str = "claude-sonnet-4-5"
    speak_base_url: str = ""
    speak_api_key: str = ""

    # Senses need a model with web search. Empty = use the speak tier.
    sense_provider: str = ""
    sense_model: str = ""
    sense_base_url: str = ""
    sense_api_key: str = ""

    recent_window: int = 40
    strays: int = 4
    enabled_voices: str = ""  # comma-separated; empty = every enabled voice in the DB
    port: int = 8000

    voices_dir: Path = ROOT / "voices"
    senses_file: Path = ROOT / "senses.yaml"
    static_dir: Path = ROOT / "static"

    @property
    def listen_tier(self) -> TierConfig:
        return TierConfig(
            provider=self.listen_provider,
            model=self.listen_model,
            base_url=self.listen_base_url or None,
            api_key=self.listen_api_key or None,
        )

    @property
    def speak_tier(self) -> TierConfig:
        return TierConfig(
            provider=self.speak_provider,
            model=self.speak_model,
            base_url=self.speak_base_url or None,
            api_key=self.speak_api_key or None,
        )

    @property
    def sense_tier(self) -> TierConfig:
        if not self.sense_provider:
            return self.speak_tier
        return TierConfig(
            provider=self.sense_provider,
            model=self.sense_model or self.speak_model,
            base_url=self.sense_base_url or None,
            api_key=self.sense_api_key or None,
        )

    @property
    def voice_allowlist(self) -> set[str] | None:
        names = {n.strip() for n in self.enabled_voices.split(",") if n.strip()}
        return names or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
