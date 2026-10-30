"""Settings, read from the environment / `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class TierConfig(BaseModel):
    """Which provider/model a tier (listen, speak, sense, teacher, embed) talks to."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://polyphony:polyphony@localhost:5432/polyphony"
    redis_url: str = "redis://localhost:6379/0"
    namespace: str = "polyphony"  # prefix of every Redis key; isolates evals/simulator runs

    # Which ensemble (frame + voices + senses) this instance runs.
    ensemble: str = "mind"
    ensembles_dir: Path = ROOT / "ensembles"

    # listen: hot path, local small model by default (the "student").
    listen_provider: str = "ollama"
    listen_model: str = "qwen2.5:3b"
    listen_base_url: str = ""
    listen_api_key: str = ""

    speak_provider: str = "anthropic"
    speak_model: str = "claude-sonnet-4-5"
    speak_base_url: str = ""
    speak_api_key: str = ""

    # Senses: summarise what was read. Empty = speak tier.
    sense_provider: str = ""
    sense_model: str = ""
    sense_base_url: str = ""
    sense_api_key: str = ""

    # Teacher: labels a sample of listens in the background. Empty = speak tier.
    teacher_provider: str = ""
    teacher_model: str = ""
    teacher_base_url: str = ""
    teacher_api_key: str = ""
    teacher_sample_rate: float = 0.05
    # Simulator: record every listen as a sample with this source (e.g. "synthetic").
    record_all_listens: str = ""

    # Embeddings for memory. Empty provider = memory off.
    embed_provider: str = "ollama"
    embed_model: str = "nomic-embed-text"
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_dim: int = 768
    memory_k: int = 5
    memory_query_last: int = 3

    # Digest: quiet this long (no new thought, nobody forming) -> write a digest.
    digest_idle_seconds: float = 90.0

    # Senses over MCP.
    mcp_fetch_command: str = "uvx mcp-server-fetch"
    mcp_feeds_command: str = "python -m polyphony.mcp_servers.feeds"
    exa_api_key: str = ""
    tavily_api_key: str = ""
    mcp_exa_command: str = "npx -y exa-mcp-server"
    mcp_exa_tool: str = "web_search_exa"
    mcp_tavily_command: str = "npx -y tavily-mcp"
    mcp_tavily_tool: str = "tavily-search"
    sense_max_items: int = 3
    sense_read_max_articles: int = 3

    # Polyphony as an MCP server.
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = 8001
    api_url: str = "http://127.0.0.1:8000"

    pause_scale: float = 1.0  # simulator: shrink voices' pause_seconds to run faster
    recent_window: int = 40
    strays: int = 4
    enabled_voices: str = ""  # comma-separated; empty = every enabled voice in the DB
    host: str = "127.0.0.1"  # compose sets 0.0.0.0 inside the container, published on 127.0.0.1
    port: int = 8000

    # Background components (tests and the simulator switch some off).
    run_senses: bool = True
    run_memory: bool = True
    run_digest: bool = True

    static_dir: Path = ROOT / "static"

    # ------------------------------------------------------------------ paths

    @property
    def ensemble_dir(self) -> Path:
        return self.ensembles_dir / self.ensemble

    @property
    def voices_dir(self) -> Path:
        return self.ensemble_dir / "voices"

    @property
    def senses_file(self) -> Path:
        return self.ensemble_dir / "senses.yaml"

    @property
    def frame_file(self) -> Path:
        return self.ensemble_dir / "frame.md"

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

    @property
    def teacher_tier(self) -> TierConfig:
        return self._tier("teacher", self.speak_tier)

    @property
    def embed_tier(self) -> TierConfig | None:
        return self._tier("embed") if self.embed_provider else None

    @property
    def voice_allowlist(self) -> set[str] | None:
        names = {n.strip() for n in self.enabled_voices.split(",") if n.strip()}
        return names or None


@lru_cache
def get_settings() -> Settings:
    return Settings()
