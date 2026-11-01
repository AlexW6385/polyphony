"""Settings, read from the environment / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, PrivateAttr
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
    # The values read from .env, kept by runtime_config before the settings page overrides them.
    _env_values: Settings | None = PrivateAttr(default=None)

    database_url: str = "postgresql+asyncpg://polyphony:polyphony@localhost:5432/polyphony"
    redis_url: str = "redis://localhost:6379/0"
    namespace: str = "polyphony"  # prefix of every Redis key; isolates eval runs

    # The key, voices and senses the database is seeded from on first start.
    mind_dir: Path = ROOT / "mind"

    # listen: hot path, every new thought for every voice, and the cadence rounds.
    listen_provider: str = "anthropic"
    listen_model: str = "claude-haiku-5-5"
    listen_base_url: str = ""
    listen_api_key: str = ""

    speak_provider: str = "anthropic"
    speak_model: str = "claude-sonnet-5-5"
    speak_base_url: str = ""
    speak_api_key: str = ""

    # Senses: summarise what was read. Empty = speak tier.
    sense_provider: str = ""
    sense_model: str = ""
    sense_base_url: str = ""
    sense_api_key: str = ""

    # Digest: quiet this long (no new thought, nobody forming) -> write a digest.
    digest_idle_seconds: float = 90.0

    # Senses run once a day at a fixed time in this timezone.
    timezone: str = "America/Toronto"

    # Senses over MCP.
    mcp_fetch_command: str = "uvx mcp-server-fetch"
    mcp_feeds_command: str = "python -m polyphony.mcp_servers.feeds"
    sense_max_items: int = 3
    sense_read_max_articles: int = 3

    # Polyphony as an MCP server.
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = 8001
    api_url: str = "http://127.0.0.1:8000"

    # Memory: older thoughts related to what is being said now, by meaning (pgvector).
    # Empty provider = memory off. "ollama" (native API) or "openai" (any OpenAI-compatible
    # /v1/embeddings endpoint). The dimension fixes the table's vector size; changing the
    # model or dimension re-embeds everything in the background.
    embed_provider: str = "ollama"
    embed_model: str = "qwen3-embedding:0.6b"
    embed_base_url: str = ""  # ollama: http://localhost:11434; compose points at the host
    embed_api_key: str = ""
    embed_dim: int = 1024
    memory_k: int = 4  # at most this many memories per prompt
    memory_query_last: int = 3  # the query is the last few thoughts of the stream
    memory_min_similarity: float = 0.5  # cosine similarity below this is not a memory

    # Decisions: "llm" (the listen model, as before) or "jev" (TypeSafe's decision model) for
    # the calls that are pure decisions: listen, the cadence rounds, and whether the clerk
    # calls a vote. Jev returns typed answers with probabilities, no text.
    decider: str = "llm"
    jev_api_key: str = ""  # or TYPESAFE_API_KEY
    jev_model: str = "jev-latest"
    jev_base_url: str = "https://api.typesafe.ai"
    jev_speak_threshold: float = 0.5  # p(speak) at or above this -> the voice speaks
    jev_support_threshold: float = 0.5  # p(supports option) at or above this -> supports it
    jev_clerk_threshold: float = 0.5  # p(a vote is due) at or above this -> the clerk drafts one

    # Voices start listening at random within (this x number of voices) seconds of a new
    # thought, so later ones see what the first ones decided. 10 voices -> within 5 s.
    listen_stagger_per_voice: float = 0.5

    pause_scale: float = 1.0  # tests: shrink voices' pauses and listen stagger to run faster
    recent_window: int = 60  # what Redis keeps; prompts show 40-59 (see prompts.cache_window)
    strays: int = 4
    host: str = "127.0.0.1"  # compose sets 0.0.0.0 inside the container, published on 127.0.0.1
    port: int = 8000

    # Background components (tests and the simulator switch some off).
    run_senses: bool = True
    run_digest: bool = True
    run_clerk: bool = True
    run_memory: bool = True
    speak_web_search: bool = True  # voices may search the web once when they speak (Anthropic)

    static_dir: Path = ROOT / "static"

    # ------------------------------------------------------------------ paths

    @property
    def voices_dir(self) -> Path:
        return self.mind_dir / "voices"

    @property
    def senses_file(self) -> Path:
        return self.mind_dir / "senses.yaml"

    @property
    def frame_file(self) -> Path:
        return self.mind_dir / "frame.md"

    # ------------------------------------------------------------------ tiers

    def _tier(self, name: str, fallback: TierConfig | None = None) -> TierConfig:
        provider = getattr(self, f"{name}_provider")
        if not provider and fallback is not None:
            model = getattr(self, f"{name}_model")
            return fallback.model_copy(update={"model": model}) if model else fallback
        return TierConfig(
            provider=provider,
            model=getattr(self, f"{name}_model"),
            base_url=getattr(self, f"{name}_base_url") or None,
            api_key=getattr(self, f"{name}_api_key") or None,
        )

    @property
    def listen_tier(self) -> TierConfig:
        return self._tier("listen")

    @property
    def speak_tier(self) -> TierConfig:
        return self._tier("speak")

    @property
    def sense_tier(self) -> TierConfig:
        return self._tier("sense", self.speak_tier)


@lru_cache
def get_settings() -> Settings:
    return Settings()
